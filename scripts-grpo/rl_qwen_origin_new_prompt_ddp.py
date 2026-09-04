#!/usr/bin/env python3
"""
Reinforcement-learning loop to fine-tune a small Qwen-style model using
responses scored against a large model's replies. This is a scaffold that:

- runs a loop for `max_rounds`
- obtains a reply from a (placeholder) large model
- generates multiple candidates from a Qwen policy
- scores each candidate to produce rewards
- runs a simple GRPO-style policy update on the Qwen policy

Replace the placeholder model functions with real model calls (Qwen APIs,
large-model APIs) when integrating into a real training pipeline.
"""

from __future__ import annotations

import os
# import weave

# 环境变量（如 OPENAI_API_KEY, WANDB_API_KEY, MODEL_PATH, SAVE_PATH）请通过 .env 或 export 配置，参见 env.example

import argparse
import json
import random
from contextlib import nullcontext
import datetime

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
import wandb
from types import MethodType

from utils import (
    logger,
)
from rl_dataset import expand_scenes_with_role_swap, load_and_split_backgrounds
from rl_policy_model import QwenPolicyTorch
from rl_training_loop_goal_single import training_loop_goal_single
from rl_testing_loop_goal_single import testing_loop_goal_single


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RL fine-tune Qwen small model scaffold (GRPO-like)."
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-6, help="Policy learning rate"
    )
    parser.add_argument(
        "--kl-constraint", type=float, default=0.02, help="KL constraint for GRPO step"
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.0,
        help="Entropy regularization coefficient (higher -> more exploration).",
    )
    parser.add_argument(
        "--ppo-clip",
        type=float,
        default=0.2,
        help="Clipping epsilon for importance ratio in Process-GRPO objective (ε).",
    )
    parser.add_argument(
        "--process-reward-weight",
        type=float,
        default=0.1,
        help="Scale factor applied to per-turn rubric process rewards.",
    )
    parser.add_argument(
        "--goal-reward-weight",
        type=float,
        default=1.0,
        help="Scale factor applied to terminal goal-achievement reward.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Discount factor γ for backward-cumulative advantage.",
    )
    parser.add_argument(
        "--expected-horizon",
        type=int,
        default=5,
        help="Expected number of small-model turns per episode (used for rubric phase scheduling).",
    )
    parser.add_argument(
        "--dynamic-weights-from-judge",
        action="store_true",
        default=False,
        help="Let the judge LLM propose rubric dimension weights (blended with schedule baseline).",
    )
    parser.add_argument(
        "--dynamic-weights-mix-ratio",
        type=float,
        default=0.5,
        help="Blend ratio for LLM-proposed vs schedule-baseline rubric weights (0=baseline only).",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument(
        "--save-path",
        type=str,
        default=os.environ.get("SAVE_PATH", ""),
        help="Where to save the final policy (or set SAVE_PATH)",
    )
    parser.add_argument(
        "--policy-model-path",
        type=str,
        default=os.environ.get("MODEL_PATH", ""),
        help="Path to local Qwen model (or set MODEL_PATH)",
    )
    parser.add_argument(
        "--reward-model",
        type=str,
        default=os.environ.get("REWARD_MODEL", "Qwen3.5-35B-A3B"),
        help="Model used for training-time reward scoring.",
    )
    parser.add_argument(
        "--reward-base-url",
        type=str,
        default=os.environ.get("REWARD_BASE_URL", "http://127.0.0.1:8000/v1"),
        help="OpenAI-compatible base URL for reward model service.",
    )
    parser.add_argument(
        "--reward-api-key",
        type=str,
        default=os.environ.get("REWARD_API_KEY", "EMPTY"),
        help="API key for reward model service.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=os.environ.get("JUDGE_MODEL", ""),
        help="Optional model used only for test-time judging (empty -> fallback to reward-model).",
    )
    parser.add_argument(
        "--judge-base-url",
        type=str,
        default=os.environ.get("JUDGE_BASE_URL", ""),
        help="Optional base URL used only for test-time judging (empty -> fallback to reward-base-url).",
    )
    parser.add_argument(
        "--judge-api-key",
        type=str,
        default=os.environ.get("JUDGE_API_KEY", ""),
        help="Optional API key used only for test-time judging (empty -> fallback to reward-api-key).",
    )
    parser.add_argument(
        "--partner-base-url",
        type=str,
        default=os.environ.get("PARTNER_BASE_URL", "http://127.0.0.1:8000/v1"),
        help="OpenAI-compatible base URL for local inference service (vLLM/TGI gateway).",
    )
    parser.add_argument(
        "--partner-model",
        type=str,
        default=os.environ.get("PARTNER_MODEL", "Qwen2.5-7B-Instruct"),
        help="Model name served by local inference service (empty -> fallback to --llm-model).",
    )
    parser.add_argument(
        "--partner-api-key",
        type=str,
        default=os.environ.get("PARTNER_API_KEY", "EMPTY"),
        help="API key for local inference service (vLLM commonly accepts 'EMPTY').",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to load Qwen model onto",
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        default=True,
        help="Use LoRA/PEFT adapters for fine-tuning the base model",
    )
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated target modules for LoRA (model-specific)",
    )
    parser.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout")
    parser.add_argument(
        "--backgrounds",
        type=str,
        default=os.path.join(os.environ.get("PROJECT_PATH"), "data", "backgrounds_short.jsonl"),
        help="Path to backgrounds.jsonl",
    )
    parser.add_argument(
        "--scene-index",
        type=int,
        default=0,
        help="Index of scene from backgrounds to use for initialization",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Save checkpoint every N batches (0 disables)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--use-ddp",
        action="store_true",
        default=True,
        help="Use DistributedDataParallel for multi-GPU training",
    )
    parser.add_argument(
        "--ddp-backend",
        type=str,
        default="nccl",
        help="DDP backend, e.g. nccl/gloo",
    )
    parser.add_argument(
        "--nccl-timeout-minutes",
        type=int,
        default=120,
        help="NCCL collective operation timeout in minutes (default 120).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum new tokens to generate for each candidate",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="sotopia-" + os.environ["TRAIN_NAME"],
        help="WandB project name",
    )
    parser.add_argument(
        "--max-rounds", type=int, default=10, help="Maximum dialogue rounds"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Held-out validation/test split ratio.",
    )
    parser.add_argument(
        "--disable-test-eval",
        action="store_true",
        default=False,
        help="Disable epoch-end test evaluation on held-out split.",
    )
    parser.add_argument(
        "--test-max-scenes",
        type=int,
        default=0,
        help="Max held-out scenes to evaluate each epoch (0 = all).",
    )
    parser.add_argument(
        "--test-every-batches",
        type=int,
        default=1,
        help="Run test every N training batches (1 = every batch, 0 = disable in-epoch tests).",
    )
    parser.add_argument(
        "--test-scene-batch-size",
        type=int,
        default=int(os.environ.get("TEST_SCENE_BATCH_SIZE", "64")),
        help="Number of scenes to process in parallel during test evaluation.",
    )
    parser.add_argument(
        "--test-num-groups",
        type=int,
        default=int(os.environ.get("TEST_NUM_GROUPS", "4")),
        help="Number of sampled groups per scene during test evaluation (0 = 1).",
    )
    parser.add_argument(
        "--scene-batch-size",
        type=int,
        default=2,
        help="Number of scenes to process in parallel per policy update (multi-batch).",
    )
    parser.add_argument(
        "--num-groups",
        type=int,
        default=8,
        help="Number of sampled groups G per scene (0 = 1).",
    )
    parser.add_argument(
        "--gradient-batch-size",
        type=int,
        default=0,
        help="Chunk size for backward pass in step_policy_gradient (0 = no chunking).",
    )
    parser.add_argument(
        "--disable-gradient-checkpointing",
        action="store_true",
        default=False,
        help="Disable gradient checkpointing (faster, more memory).",
    )
    parser.add_argument(
        "--enable-kv-cache",
        action="store_true",
        default=True,
        help="Enable KV cache during generation (faster, more memory).",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        type=str,
        default=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        help="Comma-separated GPU ids to use (empty keeps current env).",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume training (loads PEFT adapter and continues from next unfinished batch).",
    )
    parser.add_argument(
        "--swap-roles",
        action="store_true",
        default=True,
        help="启用 policy/partner 角色互换扩展：每个场景产出两份样本，policy 模型在同一场景中扮演两种角色，避免只学一种角色（默认开启）。",
    )
    parser.add_argument(
        "--disable-swap-roles",
        dest="swap_roles",
        action="store_false",
        help="关闭 policy/partner 角色互换扩展，恢复每个场景仅一种角色绑定的旧行为。",
    )
    return parser.parse_args()


def _is_main_process(rank: int) -> bool:
    return rank == 0


def _setup_distributed(args: argparse.Namespace) -> tuple[bool, int, int, int]:
    if not args.use_ddp:
        return False, 0, 0, 1

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 1:
        return False, rank, local_rank, world_size

    if not torch.cuda.is_available():
        raise RuntimeError("DDP with WORLD_SIZE > 1 requires CUDA in this script.")

    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        timeout = datetime.timedelta(minutes=getattr(args, "nccl_timeout_minutes", 120))
        dist.init_process_group(
            backend=args.ddp_backend,
            init_method="env://",
            timeout=timeout,
        )
    return True, rank, local_rank, world_size


def _cleanup_distributed(is_ddp: bool) -> None:
    if is_ddp and dist.is_initialized():
        dist.destroy_process_group()


TRAIN_STATE_FILENAME = "train_state.json"


def _load_train_state(ckpt_dir: str) -> tuple[int, int]:
    """Load train_state.json. Returns (last_completed_epoch, last_completed_batch), both 0-based."""
    path = os.path.join(ckpt_dir, TRAIN_STATE_FILENAME)
    if not os.path.isfile(path):
        return -1, -1
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return int(data.get("epoch_index", -1)), int(data.get("batch_index", -1))
        return int(data), -1
    except Exception as e:
        logger.warning("Could not load train_state from %s: %s", path, e)
        return -1, -1


def _save_train_state(ckpt_dir: str, epoch_index: int, batch_index: int = -1) -> None:
    """Write train_state.json with last completed epoch/batch (0-based)."""
    path = os.path.join(ckpt_dir, TRAIN_STATE_FILENAME)
    try:
        os.makedirs(ckpt_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"epoch_index": int(epoch_index), "batch_index": int(batch_index)},
                f,
                indent=2,
            )
    except Exception as e:
        logger.warning("Could not save train_state to %s: %s", path, e)


def _patch_compute_logprob_for_ddp(policy_obj: QwenPolicyTorch) -> None:
    # DDP 同步依赖走 wrapper 的 forward；这里覆盖实例方法，避免取 self.model.module。
    def _compute_logprob_of_sequences_ddp(
        self, input_prompts, sequences, return_entropy=False
    ):
        batch_log_probs = []
        batch_entropies = []
        model_for_comp = self.model
        for prompt, seq in zip(input_prompts, sequences):
            combined = (prompt + " " + seq).strip()
            if hasattr(model_for_comp, "device"):
                device = model_for_comp.device
            else:
                device = next(model_for_comp.parameters()).device
            tokens = self.tokenizer(combined, return_tensors="pt").to(device)
            input_ids = tokens["input_ids"]

            with torch.enable_grad():
                with (
                    torch.amp.autocast(device_type="cuda")
                    if torch.cuda.is_available()
                    else nullcontext()
                ):
                    outputs = model_for_comp(input_ids)
                    logits = outputs.logits
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                    probs = torch.exp(log_probs)

                seq_len = input_ids.shape[1]
                if seq_len <= 1:
                    batch_log_probs.append(torch.tensor(0.0, device=device))
                    batch_entropies.append(torch.tensor(0.0, device=device))
                    continue

                token_log_probs = []
                token_entropies = []
                for t in range(seq_len - 1):
                    target_id = input_ids[0, t + 1]
                    token_log_probs.append(log_probs[0, t, target_id])
                    token_entropy = -(probs[0, t] * log_probs[0, t]).sum()
                    token_entropies.append(token_entropy)
                batch_log_probs.append(torch.stack(token_log_probs).sum())
                batch_entropies.append(torch.stack(token_entropies).mean())
        stacked_log_probs = torch.stack(batch_log_probs)
        if return_entropy:
            return stacked_log_probs, torch.stack(batch_entropies)
        return stacked_log_probs

    policy_obj.compute_logprob_of_sequences = MethodType(
        _compute_logprob_of_sequences_ddp, policy_obj
    )


def main():
    args = parse_args()

    if args.wandb_project is None:
        args.wandb_project = "sotopia-" + args.train_name

    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        logger.info("Set CUDA_VISIBLE_DEVICES=%s", args.cuda_visible_devices)

    is_ddp = False
    rank = 0
    local_rank = 0
    world_size = 1
    wandb_run = None
    try:
        is_ddp, rank, local_rank, world_size = _setup_distributed(args)
        is_main = _is_main_process(rank)
        run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if is_ddp and dist.is_initialized():
            ts_holder = [run_timestamp if is_main else ""]
            dist.broadcast_object_list(ts_holder, src=0)
            run_timestamp = str(ts_holder[0])

        save_root = os.path.abspath(args.save_path)
        save_prefix = os.path.basename(save_root.rstrip(os.sep)) or "checkpoints"
        # Intuitive layout: use save_path as root directory, each run gets a timestamp subdir.
        run_ckpt_dir = os.path.join(save_root, run_timestamp)
        os.makedirs(run_ckpt_dir, exist_ok=True)

        if is_main:
            print("your policy will be saved to %s" % args.save_path)
            logger.info("Run checkpoint root: %s", run_ckpt_dir)

        # 设置随机种子
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        # Propagate dynamic-rubric config to child code via env vars.
        os.environ["EXPECTED_HORIZON"] = str(getattr(args, "expected_horizon", 5))
        os.environ["DYNAMIC_WEIGHTS_FROM_JUDGE"] = (
            "1" if getattr(args, "dynamic_weights_from_judge", False) else "0"
        )
        os.environ["DYNAMIC_WEIGHTS_MIX_RATIO"] = str(
            getattr(args, "dynamic_weights_mix_ratio", 0.5)
        )
        # Always use rubric_dynamic for process rewards in this setup.
        os.environ.setdefault("PROCESS_REWARD_KIND", "rubric_dynamic")

        # 在入口脚本中统一切分 train/test，避免训练/测试循环各自重复切分
        train_scenes, val_scenes = load_and_split_backgrounds(
            args.backgrounds,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        if is_main:
            logger.info(
                "Dataset split done in main: train=%d, test=%d (val_ratio=%.4f)",
                len(train_scenes), len(val_scenes), float(args.val_ratio),
            )

        if bool(getattr(args, "swap_roles", True)):
            train_before = len(train_scenes)
            val_before = len(val_scenes)
            train_scenes = expand_scenes_with_role_swap(
                train_scenes, shuffle=True, seed=args.seed
            )
            val_scenes = expand_scenes_with_role_swap(
                val_scenes, shuffle=True, seed=args.seed
            )
            if is_main:
                logger.info(
                    "Role-swap expansion enabled: train %d -> %d, val %d -> %d "
                    "(每个场景生成 _swap_agents=False/True 两个变体)",
                    train_before, len(train_scenes), val_before, len(val_scenes),
                )
        elif is_main:
            logger.info(
                "Role-swap expansion disabled via --disable-swap-roles; policy 仅扮演单一角色。"
            )

        # 检查GPU数量
        num_gpus = torch.cuda.device_count()
        logger.info(
            "Found %d GPUs (rank=%d local_rank=%d world_size=%d, ddp=%s)",
            num_gpus, rank, local_rank, world_size, is_ddp,
        )

        if is_ddp:
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device(args.device)

        # 加载模型
        logger.info("Loading Qwen model from %s", args.policy_model_path)

        resume_from = getattr(args, "resume_from", None) or os.environ.get("RESUME_FROM", "").strip()
        if resume_from:
            resume_from = os.path.abspath(resume_from)
            if not os.path.isdir(resume_from):
                raise FileNotFoundError("Resume checkpoint path is not a directory: %s" % resume_from)
            logger.info("Resuming from checkpoint: %s", resume_from)

        lora_conf = None
        if args.use_lora and not resume_from:
            lora_conf = {
                "r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "target_modules": [m.strip() for m in args.lora_target_modules.split(",") if m.strip()],
                "lora_dropout": args.lora_dropout,
            }

        # 创建策略（续跑时从 checkpoint 加载 PEFT，否则新建 LoRA）
        qwen_policy = QwenPolicyTorch(
            args.policy_model_path,
            device=device,
            use_lora=args.use_lora,
            lora_config=lora_conf,
            enable_gradient_checkpointing=not args.disable_gradient_checkpointing,
            enable_kv_cache=args.enable_kv_cache,
            use_device_map_auto=not is_ddp,
            peft_checkpoint_path=resume_from if resume_from else None,
        )

        # 使用 DDP 包装底层语言模型参数
        if is_ddp:
            qwen_policy.model = DDP(
                qwen_policy.model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
                broadcast_buffers=False,
            )
            _patch_compute_logprob_for_ddp(qwen_policy)

        # 创建优化器
        optimizer = qwen_policy.create_optimizer(lr=args.learning_rate)
        qwen_policy.optimizer = optimizer

        # 运行训练循环
        trained_policy = None
        if args.wandb and is_main:
            wandb_run = wandb.init(
                project=getattr(args, "wandb_project", "sotopia-rl"),
                config=vars(args),
                reinit=True,
            )
        last_completed_epoch, last_completed_batch = (
            _load_train_state(resume_from) if resume_from else (-1, -1)
        )
        if last_completed_epoch < 0:
            start_epoch = 0
            start_batch_idx = 0
        elif last_completed_batch >= 0:
            start_epoch = last_completed_epoch
            start_batch_idx = last_completed_batch + 1
        else:
            start_epoch = last_completed_epoch + 1
            start_batch_idx = 0
        if resume_from:
            logger.info(
                "Resume state: last_completed_epoch=%d, last_completed_batch=%d -> start from epoch=%d, batch=%d",
                last_completed_epoch,
                last_completed_batch,
                start_epoch,
                start_batch_idx,
            )
            logger.info(
                "Will run epochs %d..%d%s",
                start_epoch + 1,
                args.epochs,
                (
                    f" (epoch {start_epoch + 1} starts at batch {start_batch_idx + 1})"
                    if start_batch_idx > 0
                    else ""
                ),
            )
        metrics_summary_path = os.path.join(
            run_ckpt_dir,
            f"training_metrics_summary_{run_timestamp}.jsonl",
        )

        # 训练之前先进行一次测试，作为baseline
        if not args.disable_test_eval:
            testing_loop_goal_single(
                policy=qwen_policy,
                val_scenes=val_scenes,
                which_agent_large="agent0",
                max_rounds=args.max_rounds,
                max_new_tokens=args.max_new_tokens,
                scene_batch_size=getattr(args, "test_scene_batch_size", 64),
                num_groups=getattr(args, "test_num_groups", 4) or None,
                epoch_idx=-1,
                rank=rank,
                world_size=world_size,
                wandb_run=wandb_run,
                local_metrics_path=metrics_summary_path,
                test_histories_path=os.path.join(run_ckpt_dir, "test_histories.jsonl"),
                test_max_scenes=getattr(args, "test_max_scenes", 0),
                partner_base_url=args.partner_base_url,
                partner_model=args.partner_model,
                partner_api_key=args.partner_api_key,
                judge_model=args.judge_model,
                judge_base_url=args.judge_base_url,
                judge_api_key=args.judge_api_key,
            )

        for epoch in range(start_epoch, args.epochs):
            logger.info("Starting epoch %d / %d", epoch + 1, args.epochs)
            epoch_start_batch_idx = start_batch_idx if epoch == start_epoch else 0
            epoch_ckpt_dir = os.path.join(run_ckpt_dir, f"{save_prefix}_epoch{epoch + 1}")
            
            trained_policy = training_loop_goal_single(
                qwen_policy,
                max_rounds=args.max_rounds,
                kl_constraint=args.kl_constraint,
                entropy_coef=args.entropy_coef,
                ppo_clip=args.ppo_clip,
                process_reward_weight=args.process_reward_weight,
                goal_reward_weight=args.goal_reward_weight,
                gamma=args.gamma,
                max_new_tokens=args.max_new_tokens,
                checkpoint_every=args.checkpoint_every,
                save_path=epoch_ckpt_dir,
                backgrounds_path=args.backgrounds,
                val_ratio=args.val_ratio,
                seed=args.seed,
                wandb_run=wandb_run,
                scene_batch_size=getattr(args, "scene_batch_size", 1),
                gradient_batch_size=getattr(args, "gradient_batch_size", 0) or None,
                num_groups=getattr(args, "num_groups", 0) or None,
                epoch_idx=epoch,
                rank=rank,
                world_size=world_size,
                run_timestamp=run_timestamp,
                local_metrics_path=metrics_summary_path,
                train_scenes=train_scenes,
                start_batch_idx=epoch_start_batch_idx,
                partner_base_url=args.partner_base_url,
                partner_model=args.partner_model,
                partner_api_key=args.partner_api_key,
                reward_model=args.reward_model,
                reward_base_url=args.reward_base_url,
                reward_api_key=args.reward_api_key,
                val_scenes=None if args.disable_test_eval else val_scenes,
                test_every_batches=getattr(args, "test_every_batches", 1),
                test_max_scenes=getattr(args, "test_max_scenes", 0),
                test_scene_batch_size=getattr(args, "test_scene_batch_size", 64),
                test_num_groups=getattr(args, "test_num_groups", 4),
                judge_model=args.judge_model or None,
                judge_base_url=args.judge_base_url or None,
                judge_api_key=args.judge_api_key or None,
            )

            if is_ddp:
                dist.barrier()

            if is_ddp:
                dist.barrier()

            # 每个 epoch 结束后保存（仅 rank0）
            if trained_policy is not None and is_main:
                trained_policy.save(epoch_ckpt_dir)
                _save_train_state(epoch_ckpt_dir, epoch, -1)
                logger.info(
                    "Saved epoch %d model to %s",
                    epoch + 1,
                    epoch_ckpt_dir,
                )

        # 最终保存（仅 rank0，使用格式化时间以避免类型问题）
        if trained_policy is not None and is_main:
            final_ckpt_dir = os.path.join(run_ckpt_dir, f"{save_prefix}_final")
            trained_policy.save(final_ckpt_dir)
            logger.info("Saved final Qwen policy to %s", final_ckpt_dir)
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        _cleanup_distributed(is_ddp)


if __name__ == "__main__":
    main()
