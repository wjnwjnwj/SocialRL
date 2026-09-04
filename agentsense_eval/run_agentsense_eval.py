#!/usr/bin/env python3
"""
AgentSense evaluation entry point.

Usage
-----
  python eval/run_agentsense_eval.py \
      --policy-base-url  http://127.0.0.1:9000/v1 \
      --policy-model     <served-model-name> \
      --partner-base-url http://127.0.0.1:9001/v1 \
      --partner-model    <partner-model-name> \
      --judge-base-url   http://127.0.0.1:8001/v1 \
      --judge-model      <judge-model-name> \
      --data-path        ./SENSE/data/final_data.jsonl \
      --output-dir       ./eval/output

Or via the shell wrapper:
  bash eval/run_agentsense_eval.sh
"""
from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agentsense_model import AgentModel
from agentsense_runner import run_agentsense_evaluation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_agentsense_eval")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AgentSense multi-turn social dialogue evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- policy server ---
    p.add_argument("--policy-base-url", default=os.environ.get("POLICY_BASE_URL", "http://127.0.0.1:9000/v1"))
    p.add_argument("--policy-model",    default=os.environ.get("POLICY_MODEL", ""),
                   help="Served model name (must match --served-model-name on the vLLM server).")
    p.add_argument("--policy-api-key",  default=os.environ.get("POLICY_API_KEY", "EMPTY"))
    p.add_argument("--policy-temperature", type=float,
                   default=float(os.environ.get("POLICY_TEMPERATURE", "0.7")))
    p.add_argument("--policy-max-tokens", type=int,
                   default=int(os.environ.get("POLICY_MAX_TOKENS", "128")))
    p.add_argument("--policy-server-ready-timeout", type=float,
                   default=float(os.environ.get("POLICY_SERVER_READY_TIMEOUT", "300")),
                   help="Seconds to wait for the policy server. Set 0 to skip.")

    # --- partner server ---
    p.add_argument("--partner-base-url", default=os.environ.get("PARTNER_BASE_URL", "http://127.0.0.1:9001/v1"))
    p.add_argument("--partner-model",    default=os.environ.get("PARTNER_MODEL", ""),
                   help="Partner model name. Defaults to policy model if not set.")
    p.add_argument("--partner-api-key",  default=os.environ.get("PARTNER_API_KEY", "EMPTY"))
    p.add_argument("--partner-temperature", type=float,
                   default=float(os.environ.get("PARTNER_TEMPERATURE", "0.7")))
    p.add_argument("--partner-max-tokens", type=int,
                   default=int(os.environ.get("PARTNER_MAX_TOKENS", "128")))

    # --- judge server ---
    p.add_argument("--judge-base-url", default=os.environ.get("JUDGE_BASE_URL", "http://127.0.0.1:8001/v1"))
    p.add_argument("--judge-model",    default=os.environ.get("JUDGE_MODEL", ""),
                   help="Judge model name.")
    p.add_argument("--judge-api-key",  default=os.environ.get("JUDGE_API_KEY", "EMPTY"))

    # --- data ---
    _default_data = os.path.join(
        os.path.dirname(__file__), "..", "SENSE", "data", "final_data.jsonl"
    )
    p.add_argument("--data-path", default=os.environ.get("DATA_PATH", _default_data),
                   help="Path to AgentSense final_data.jsonl.")
    p.add_argument("--max-scenes", type=int,
                   default=int(os.environ.get("MAX_SCENES", "0")),
                   help="Limit to first N scenes before expansion (0 = all).")

    # --- evaluation ---
    p.add_argument("--eval-workers", type=int,
                   default=int(os.environ.get("EVAL_WORKERS", "32")),
                   help="Number of run items evaluated concurrently.")

    # --- output ---
    _default_out = os.path.join(os.path.dirname(__file__), "output")
    p.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", _default_out))

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.policy_model:
        logger.error("--policy-model is required.")
        sys.exit(1)
    if not args.judge_model:
        logger.error("--judge-model is required.")
        sys.exit(1)

    # Partner model defaults to policy model if not explicitly set.
    partner_model_name = args.partner_model or args.policy_model
    partner_base_url   = args.partner_base_url or args.policy_base_url
    partner_api_key    = args.partner_api_key  or args.policy_api_key

    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    logger.info("=" * 60)
    logger.info("  AgentSense Evaluation")
    logger.info("  policy : %s @ %s", args.policy_model, args.policy_base_url)
    logger.info("  partner: %s @ %s", partner_model_name, partner_base_url)
    logger.info("  judge  : %s @ %s", args.judge_model, args.judge_base_url)
    logger.info("  data   : %s", args.data_path)
    logger.info("  max_scenes=%d  eval_workers=%d", args.max_scenes, args.eval_workers)
    logger.info("  output : %s", args.output_dir)
    logger.info("=" * 60)

    policy = AgentModel(
        base_url=args.policy_base_url,
        model_name=args.policy_model,
        api_key=args.policy_api_key,
        temperature=args.policy_temperature,
        max_tokens=args.policy_max_tokens,
    )
    partner = AgentModel(
        base_url=partner_base_url,
        model_name=partner_model_name,
        api_key=partner_api_key,
        temperature=args.partner_temperature,
        max_tokens=args.partner_max_tokens,
    )

    if args.policy_server_ready_timeout > 0:
        logger.info("Waiting for policy server at %s …", args.policy_base_url)
        if not policy.wait_until_ready(timeout=args.policy_server_ready_timeout):
            logger.error("Policy server not ready. Aborting.")
            sys.exit(1)

    summary = run_agentsense_evaluation(
        data_path=args.data_path,
        output_dir=args.output_dir,
        policy_model=policy,
        partner_model=partner,
        judge_model=args.judge_model,
        judge_base_url=args.judge_base_url,
        judge_api_key=args.judge_api_key,
        max_scenes=args.max_scenes,
        eval_workers=args.eval_workers,
        run_timestamp=run_ts,
    )

    overall_goal = (summary.get("overall_goal") or {})
    overall_rel = (summary.get("overall_relation_delta") or {})
    logger.info(
        "Done. scenes=%d  run_items=%d  goal_mean=%.2f  rel_delta_mean=%.2f",
        summary.get("num_scenes", 0),
        summary.get("num_run_items", 0),
        overall_goal.get("mean", 0),
        overall_rel.get("mean", 0),
    )
    logger.info("Artifacts saved under: %s", args.output_dir)


if __name__ == "__main__":
    main()
