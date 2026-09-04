#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime
import os
import random
from typing import List

import numpy as np
import torch
import wandb

from rl_dataset import expand_scenes_with_role_swap, load_and_split_backgrounds
from rl_policy_model import QwenPolicyTorch
from rl_testing_loop_goal_single import testing_loop_goal_single
from utils import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone test-only runner for dialogue policy checkpoints."
    )
    parser.add_argument(
        "--policy-model-path",
        type=str,
        default=os.environ.get("MODEL_PATH", ""),
        help="Base policy model path.",
    )
    parser.add_argument(
        "--peft-checkpoint-path",
        type=str,
        default=os.environ.get("RESUME_FROM", ""),
        help="Optional PEFT adapter checkpoint path for evaluation.",
    )
    parser.add_argument(
        "--backgrounds",
        type=str,
        default=os.path.join(
            os.environ.get("PROJECT_PATH", "."), "data", "backgrounds_mid.jsonl"
        ),
        help="Path to backgrounds jsonl.",
    )
    parser.add_argument(
        "--test-split",
        choices=["val", "train", "all"],
        default="val",
        help="Which split to evaluate: val/train/all.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio used when splitting backgrounds.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max-rounds", type=int, default=10, help="Max dialogue rounds.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Max new tokens for policy generation.",
    )
    parser.add_argument(
        "--scene-batch-size",
        type=int,
        default=64,
        help="How many scenes to evaluate in parallel per batch.",
    )
    parser.add_argument(
        "--num-groups",
        type=int,
        default=4,
        help="Number of sampled groups per scene (0 -> 1).",
    )
    parser.add_argument(
        "--test-max-scenes",
        type=int,
        default=0,
        help="Max scenes to test (0 means all selected split scenes).",
    )
    parser.add_argument(
        "--which-agent-large",
        type=str,
        default="agent0",
        choices=["agent0", "agent1"],
        help="Which agent is controlled by partner model in simulation.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.environ.get("TEST_OUTPUT_DIR", "./checkpoints"),
        help="Directory to save test artifacts.",
    )
    parser.add_argument(
        "--partner-base-url",
        type=str,
        default=os.environ.get("PARTNER_BASE_URL", "http://127.0.0.1:8000/v1"),
        help="Partner model base URL.",
    )
    parser.add_argument(
        "--partner-model",
        type=str,
        default=os.environ.get("PARTNER_MODEL", "Qwen3.5-35B-A3B"),
        help="Partner model name.",
    )
    parser.add_argument(
        "--partner-api-key",
        type=str,
        default=os.environ.get("PARTNER_API_KEY", "EMPTY"),
        help="Partner model API key.",
    )
    parser.add_argument(
        "--judge-base-url",
        type=str,
        default=os.environ.get("JUDGE_BASE_URL", ""),
        help="Judge model base URL (empty -> fallback to reward base URL).",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=os.environ.get("JUDGE_MODEL", ""),
        help="Judge model name (empty -> fallback to reward model).",
    )
    parser.add_argument(
        "--judge-api-key",
        type=str,
        default=os.environ.get("JUDGE_API_KEY", ""),
        help="Judge model API key (empty -> fallback to reward API key).",
    )
    parser.add_argument(
        "--reward-base-url",
        type=str,
        default=os.environ.get("REWARD_BASE_URL", "http://127.0.0.1:8000/v1"),
        help="Fallback reward model base URL.",
    )
    parser.add_argument(
        "--reward-model",
        type=str,
        default=os.environ.get("REWARD_MODEL", "Qwen3.5-35B-A3B"),
        help="Fallback reward model name.",
    )
    parser.add_argument(
        "--reward-api-key",
        type=str,
        default=os.environ.get("REWARD_API_KEY", "EMPTY"),
        help="Fallback reward model API key.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for loading policy model.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=os.environ.get("PROJECT_NAME", "sotopia-test-only"),
        help="WandB project name.",
    )
    parser.add_argument(
        "--swap-roles",
        action="store_true",
        default=True,
        help="启用 policy/partner 角色互换扩展：每个测试场景产出两份样本，policy 模型在同一场景中扮演两种角色（默认开启）。",
    )
    parser.add_argument(
        "--disable-swap-roles",
        dest="swap_roles",
        action="store_false",
        help="关闭 policy/partner 角色互换扩展，恢复每个场景仅一种角色绑定的旧行为。",
    )
    return parser.parse_args()


def _select_test_scenes(args: argparse.Namespace) -> List[dict]:
    train_scenes, val_scenes = load_and_split_backgrounds(
        args.backgrounds,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    if args.test_split == "train":
        scenes = train_scenes
    elif args.test_split == "all":
        scenes = train_scenes + val_scenes
    else:
        scenes = val_scenes
    return scenes


def main() -> None:
    args = parse_args()
    if not args.policy_model_path:
        raise ValueError("`--policy-model-path` (or MODEL_PATH) is required.")

    peft_path = (args.peft_checkpoint_path or "").strip() or None
    if peft_path:
        peft_path = os.path.abspath(peft_path)
        if not os.path.isdir(peft_path):
            raise FileNotFoundError(f"PEFT checkpoint not found: {peft_path}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = os.path.abspath(args.output_dir)
    run_dir = os.path.join(output_root, f"test_only_{run_ts}")
    os.makedirs(run_dir, exist_ok=True)
    metrics_path = os.path.join(run_dir, f"testing_metrics_summary_{run_ts}.jsonl")
    histories_path = os.path.join(run_dir, "test_histories.jsonl")

    test_scenes = _select_test_scenes(args)
    logger.info("Loaded %d scenes for split=%s", len(test_scenes), args.test_split)
    if not test_scenes:
        raise RuntimeError("No scenes found for testing. Check `--backgrounds` and split.")
    if bool(getattr(args, "swap_roles", True)):
        before = len(test_scenes)
        test_scenes = expand_scenes_with_role_swap(
            test_scenes, shuffle=True, seed=args.seed
        )
        logger.info(
            "Role-swap expansion enabled for testing: %d -> %d scenes",
            before,
            len(test_scenes),
        )
    else:
        logger.info("Role-swap expansion disabled for testing.")

    device = torch.device(args.device)
    policy = QwenPolicyTorch(
        model_path=args.policy_model_path,
        device=device,
        use_lora=False,
        lora_config=None,
        enable_gradient_checkpointing=False,
        enable_kv_cache=True,
        use_device_map_auto=True,
        peft_checkpoint_path=peft_path,
    )

    wandb_run = None
    try:
        if args.wandb:
            wandb_run = wandb.init(
                project=args.wandb_project,
                config=vars(args),
                reinit=True,
            )

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
