#!/usr/bin/env python3
"""Standalone test-only runner using vLLM servers as the backend.

All three model roles (judge, partner, policy) are served by independent
vLLM processes started beforehand (steps 1-3). This script is step 4:
it connects to the three servers and runs the concurrent dialogue test.

Steps:
  1. bash run_judge_server.sh    – judge model
  2. bash run_partner_server.sh  – partner model
  3. bash run_policy_server.sh   – policy model (LoRA-enabled)
  4. bash run_test_only_vllm.sh  – calls this script

No torchrun / DDP / GPU required here; all inference is remote over HTTP.
LoRA adapters are forwarded via --peft-checkpoint-path as a lora_request.
"""
from __future__ import annotations

import argparse
import datetime
import os
import random
from typing import List

import numpy as np

from rl_dataset import expand_scenes_with_role_swap, load_and_split_backgrounds
from rl_testing_loop_goal_single import testing_loop_goal_single
from utils import logger
from vllm_policy import VllmPolicy


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="vLLM-backed standalone test-only runner for dialogue policy checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- policy (vLLM server) ---
    parser.add_argument(
        "--policy-base-url",
        type=str,
        default=os.environ.get("POLICY_BASE_URL", "http://127.0.0.1:9000/v1"),
        help="Base URL of the vLLM policy server.",
    )
    parser.add_argument(
        "--policy-model",
        type=str,
        default=os.environ.get("POLICY_MODEL", ""),
        help="Served model name on the policy vLLM server (--served-model-name).",
    )
    parser.add_argument(
        "--policy-api-key",
        type=str,
        default=os.environ.get("POLICY_API_KEY", "EMPTY"),
    )
    parser.add_argument(
        "--peft-checkpoint-path",
        type=str,
        default=os.environ.get("PEFT_CHECKPOINT_PATH", ""),
        help="Optional LoRA adapter dir. Forwarded to the vLLM server via lora_request.",
    )
    parser.add_argument(
        "--policy-request-timeout",
        type=float,
        default=float(os.environ.get("POLICY_REQUEST_TIMEOUT", "120")),
        help="Per-request HTTP timeout (seconds) for policy server calls.",
    )
    parser.add_argument(
        "--policy-server-ready-timeout",
        type=float,
        default=float(os.environ.get("POLICY_SERVER_READY_TIMEOUT", "300")),
        help=(
            "Seconds to wait for the policy vLLM server before aborting. "
            "Set to 0 to skip the health check (server already running)."
        ),
    )

    # --- data ---
    parser.add_argument(
        "--backgrounds",
        type=str,
        default=os.path.join(
            os.environ.get("PROJECT_PATH", "."), "data", "backgrounds_true_data_subgoal.jsonl"
        ),
    )
    parser.add_argument(
        "--test-split",
        choices=["val", "train", "all"],
        default=os.environ.get("TEST_SPLIT", "val"),
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=float(os.environ.get("VAL_RATIO", "0.146")),
    )
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")))

    # --- dialogue ---
    parser.add_argument("--max-rounds", type=int, default=int(os.environ.get("MAX_ROUNDS", "20")))
    parser.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("MAX_NEW_TOKENS", "128")))
    parser.add_argument("--scene-batch-size", type=int, default=int(os.environ.get("SCENE_BATCH_SIZE", "48")))
    parser.add_argument("--num-groups", type=int, default=int(os.environ.get("NUM_GROUPS", "2")))
    parser.add_argument("--test-max-scenes", type=int, default=int(os.environ.get("TEST_MAX_SCENES", "0")))
    parser.add_argument(
        "--which-agent-large",
        type=str,
        default=os.environ.get("WHICH_AGENT_LARGE", "agent0"),
        choices=["agent0", "agent1"],
    )

    # --- partner ---
    parser.add_argument("--partner-base-url", type=str, default=os.environ.get("PARTNER_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--partner-model", type=str, default=os.environ.get("PARTNER_MODEL", "Qwen3.5-35B-A3B"))
    parser.add_argument("--partner-api-key", type=str, default=os.environ.get("PARTNER_API_KEY", "EMPTY"))

    # --- judge (step 1 server) ---
    parser.add_argument(
        "--judge-base-url",
        type=str,
        default=os.environ.get("JUDGE_BASE_URL", "http://127.0.0.1:8001/v1"),
        help="Base URL of the judge vLLM server (step 1).",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=os.environ.get("JUDGE_MODEL", "Qwen3.5-35B-A3B"),
        help="Served model name on the judge server.",
    )
    parser.add_argument(
        "--judge-api-key",
        type=str,
        default=os.environ.get("JUDGE_API_KEY", "EMPTY"),
    )

    # --- output ---
    parser.add_argument("--output-dir", type=str, default=os.environ.get("TEST_OUTPUT_DIR", "./checkpoints"))

    # --- role swap ---
    parser.add_argument("--swap-roles", action="store_true", default=True)
    parser.add_argument("--disable-swap-roles", dest="swap_roles", action="store_false")

    # --- wandb ---
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--wandb-project", type=str, default=os.environ.get("PROJECT_NAME", "sotopia-test-only-vllm"))

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Scene loading
# ---------------------------------------------------------------------------

def _select_test_scenes(args: argparse.Namespace) -> List[dict]:
    train_scenes, val_scenes = load_and_split_backgrounds(
        args.backgrounds,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    if args.test_split == "train":
        return train_scenes
    if args.test_split == "all":
        return train_scenes + val_scenes
    return val_scenes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    wandb_run = None
    try:
        args = parse_args()

        if not args.policy_model:
            raise ValueError(
                "--policy-model (or POLICY_MODEL) is required. "
                "It must match the --served-model-name of the vLLM policy server."
            )

        random.seed(args.seed)
        np.random.seed(args.seed)

        run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = os.path.abspath(args.output_dir)
        run_dir = os.path.join(output_root, f"test_only_{run_ts}")
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs("logs", exist_ok=True)

        metrics_path = os.path.join(run_dir, f"testing_metrics_summary_{run_ts}.jsonl")
        histories_path = os.path.join(run_dir, "test_histories.jsonl")

        # --- scenes ---
        test_scenes = _select_test_scenes(args)
        logger.info("Loaded %d scenes for split=%s", len(test_scenes), args.test_split)
        if not test_scenes:
            raise RuntimeError("No scenes found. Check --backgrounds and --test-split.")

        if getattr(args, "swap_roles", True):
            scenes_before = len(test_scenes)
            test_scenes = expand_scenes_with_role_swap(
                test_scenes, shuffle=True, seed=args.seed
            )
            logger.info("Role-swap expansion: %d -> %d scenes", scenes_before, len(test_scenes))
        else:
            logger.info("Role-swap expansion disabled.")

        # --- policy ---
        peft_path = (args.peft_checkpoint_path or "").strip() or None
        if peft_path:
            peft_path = os.path.abspath(peft_path)
            if not os.path.isdir(peft_path):
                raise FileNotFoundError(f"PEFT checkpoint not found: {peft_path}")

        policy = VllmPolicy(
            base_url=args.policy_base_url,
            model_name=args.policy_model,
            lora_path=peft_path,
            api_key=args.policy_api_key,
            request_timeout=args.policy_request_timeout,
        )

        if args.policy_server_ready_timeout > 0:
            logger.info(
                "Waiting up to %.0fs for policy server at %s …",
                args.policy_server_ready_timeout,
                args.policy_base_url,
            )
            if not policy.wait_until_ready(timeout=args.policy_server_ready_timeout):
                raise RuntimeError(
                    f"Policy vLLM server at {args.policy_base_url} did not become ready. "
                    "Start it first or pass --policy-server-ready-timeout 0 to skip."
                )

        # --- wandb ---
        if args.wandb:
            import wandb as _wandb
            wandb_run = _wandb.init(
                project=args.wandb_project,
                config=vars(args),
                reinit=True,
            )

        logger.info("[Step 1] judge  : %s @ %s", args.judge_model, args.judge_base_url)
        logger.info("[Step 2] partner: %s @ %s", args.partner_model, args.partner_base_url)
        logger.info("[Step 3] policy : %s @ %s  peft=%s",
                    args.policy_model, args.policy_base_url, peft_path or "(none)")
        logger.info("[Step 4] split=%s  scenes=%d  batch=%d  groups=%d  rounds=%d  tokens=%d",
                    args.test_split, len(test_scenes),
                    args.scene_batch_size, args.num_groups,
                    args.max_rounds, args.max_new_tokens)
        logger.info("output=%s", run_dir)

        # --- run ---
        result = testing_loop_goal_single(
            policy=policy,
            val_scenes=test_scenes,
            which_agent_large=args.which_agent_large,
            max_rounds=args.max_rounds,
            max_new_tokens=args.max_new_tokens,
            scene_batch_size=args.scene_batch_size,
            num_groups=args.num_groups if args.num_groups > 0 else None,
            epoch_idx=-1,
            rank=0,
            world_size=1,
            wandb_run=wandb_run,
            local_metrics_path=metrics_path,
            test_histories_path=histories_path,
            test_max_scenes=args.test_max_scenes,
            partner_base_url=args.partner_base_url,
            partner_model=args.partner_model,
            partner_api_key=args.partner_api_key,
            judge_model=args.judge_model,
            judge_base_url=args.judge_base_url,
            judge_api_key=args.judge_api_key,
        )

        logger.info("Test-only run finished. result=%s", result)
        logger.info("Artifacts saved under: %s", run_dir)

    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
