#!/usr/bin/env python3
from __future__ import annotations

import gc
import os
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import wandb

import numpy as np

from rl_dataset import DialogueDataLoader
from utils import (
    _append_local_metrics,
    _ddp_aggregate_stats,
    _ddp_reduce_scalar,
    append_histories_to_file,
    append_histories_with_rewards,
    clone_history,
    collect_rewards,
    logger,
    run_dialogue_round,
)


def _rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _ddp_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _ddp_weighted_mean(value: float, weight: float) -> float:
    if not (dist.is_available() and dist.is_initialized()):
        return float(value)
    device = _ddp_device()
    t = torch.tensor(
        [float(value) * float(weight), float(weight)],
        device=device,
        dtype=torch.float64,
    )
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    denom = float(t[1].item())
    return float(t[0].item() / denom) if denom > 0 else 0.0


def build_wandb_dialogue_tables(
    epoch_idx: int,
    batch_idx: int,
    scene_count: int,
    rollout: dict,
    group_index_map: Dict[tuple, int],
    goal_scores_per_group: List[float],
    goal_reasons_per_group: List[str],
) -> Dict[str, object]:
    tables: Dict[str, object] = {}
    scene_groups = rollout.get("scene_groups") or []
    small_agent_list = rollout.get("small_agent_list") or []
    for scene_idx in range(scene_count):
        groups = scene_groups[scene_idx] if scene_idx < len(scene_groups) else []
        if not groups:
            continue
        group_cols = [f"group_{g}" for g in range(len(groups))]
        table = wandb.Table(columns=group_cols)
        max_rounds_in_scene = max(
            (len(group.get("all_replies", [])) for group in groups),
            default=0,
        )
        small_agent_name = (
            small_agent_list[scene_idx] if scene_idx < len(small_agent_list) else None
        )
        large_system_row = []
        small_system_row = []
        for group in groups:
            histories_all = group.get("histories_all") or {}
            large_system_msg = ""
            small_system_msg = ""
            for speaker, history in histories_all.items():
                system_msg = ""
                for turn in history:
                    if turn.get("role") == "system":
                        system_msg = (turn.get("content") or "").strip()
                        break
                if not system_msg:
                    continue
                if small_agent_name is not None and speaker == small_agent_name:
                    if not small_system_msg:
                        small_system_msg = f"SMALL_SYSTEM: {speaker}: {system_msg}"
                else:
                    if not large_system_msg:
                        large_system_msg = f"LARGE_SYSTEM: {speaker}: {system_msg}"
            large_system_row.append(large_system_msg)
            small_system_row.append(small_system_msg)
        table.add_data(*large_system_row)
        table.add_data(*small_system_row)
        for round_idx in range(max_rounds_in_scene):
            row = []
            for group_idx, group in enumerate(groups):
                replies = group.get("all_replies", [])
                speakers = group.get("all_speakers", [])
                small_indices = group.get("all_small_indices", [])
                if round_idx < len(replies):
                    speaker = (
                        speakers[round_idx] if round_idx < len(speakers) else "unknown"
                    )
                    content = (replies[round_idx] or "").strip()
                    row.append(f"{speaker}: {content}")
                else:
                    row.append("")
            table.add_data(*row)
        goal_row = []
        total_row = []
        goal_reason_row = []
        for group_idx, _group in enumerate(groups):
            global_idx = group_index_map.get((batch_idx, scene_idx, group_idx))
            if global_idx is None:
                goal_row.append("")
                total_row.append("")
                goal_reason_row.append("")
                continue
            goal_score = goal_scores_per_group[global_idx]
            goal_row.append(f"goal_reward={goal_score:.2f}")
            total_row.append(f"total_reward={goal_score:.2f}")
            goal_reason = goal_reasons_per_group[global_idx]
            goal_reason_row.append(f"goal_reason={goal_reason}" if goal_reason else "")
        table.add_data(*goal_row)
        table.add_data(*total_row)
        table.add_data(*goal_reason_row)
        tables[f"dialogue_table_epoch{epoch_idx}_batch{batch_idx}_scene{scene_idx}"] = (
            table
        )
    return tables


def _standardize_flat(scores: List[float]) -> List[float]:
    """Global z-score normalisation across all scores (mean=0, std=1)."""
    if not scores:
        return []
    arr = np.array(scores, dtype=np.float64)
    if arr.std() < 1e-8:
        return [0.0] * len(scores)
    return ((arr - arr.mean()) / (arr.std() + 0.01)).tolist()


def compute_advantages(
    all_groups: List[tuple],
    goal_scores_per_group: List[float],
    process_reward_weight: float = 0.1,
    goal_reward_weight: float = 1.0,
    gamma: float = 1.0,
) -> tuple:
    """Process-GRPO advantage computation.

    For each group we build a reward sequence r_1 … r_n where:
      - r_t  = process_reward_weight * rubric_score_t   for t < n
      - r_n += goal_reward_weight * terminal_goal_score (added to last step)

    All reward values are globally standardised across every (group, step)
    tuple in the batch.  Then for each token t the advantage is the
    backward-cumulative discounted sum of standardised rewards at t and
    all later steps (Equation in the paper):
        Â_{i,t} = Σ_{index(j) ≥ t}  r̃_i^{index(j)}

    Returns:
        batch_input_prompts  – flat list of prompts
        batch_sequences      – flat list of responses
        batch_advantages     – per-response advantage (A_{i,t} for the whole turn)
        batch_old_logprobs   – old log-probs (collected during rollout)
        batch_raw_rewards    – raw (un-normalised) shaped rewards for logging
        process_stats        – dict with per-group mean/sum for wandb
    """
    # ── Step 1: collect raw shaped rewards per (group, step) ────────────────
    # Each entry: (group_idx, step_idx, raw_reward)
    all_step_rewards: List[float] = []
    group_step_map: List[Tuple[int, int]] = []   # maps flat index -> (group_idx, step_idx)
    group_horizons: List[int] = []

    for group_idx, (_batch_idx, _b, _g, group, _scene, _agent) in enumerate(all_groups):
        prompts = list(group.get("prompts") or [])
        responses = list(group.get("responses") or [])
        horizon = min(len(prompts), len(responses))
        group_horizons.append(horizon)
        process_rewards = list(group.get("process_rewards") or [])
        if len(process_rewards) < horizon:
            process_rewards.extend([0.0] * (horizon - len(process_rewards)))
        terminal = float(goal_scores_per_group[group_idx]) * float(goal_reward_weight)
        for step_idx in range(horizon):
            raw = float(process_reward_weight) * float(process_rewards[step_idx])
            if step_idx == horizon - 1:
                raw += terminal
            all_step_rewards.append(raw)
            group_step_map.append((group_idx, step_idx))

    # ── Step 2: global standardisation across all (group, step) ─────────────
    norm_rewards = _standardize_flat(all_step_rewards)

    # Rebuild per-group normalised reward sequences.
    group_norm_rewards: List[List[float]] = [[] for _ in all_groups]
    for flat_idx, (group_idx, step_idx) in enumerate(group_step_map):
        while len(group_norm_rewards[group_idx]) <= step_idx:
            group_norm_rewards[group_idx].append(0.0)
        group_norm_rewards[group_idx][step_idx] = float(norm_rewards[flat_idx])

    # ── Step 3: backward-cumulative advantage per step ────────────────────
    # Â_{i,t} = Σ_{j: index(j) >= t} γ^{index(j)-t} * r̃_i^{index(j)}
    group_advantages: List[List[float]] = []
    for group_idx, norm_seq in enumerate(group_norm_rewards):
        n = len(norm_seq)
        adv_seq = [0.0] * n
        running = 0.0
        for t in reversed(range(n)):
            running = float(norm_seq[t]) + float(gamma) * running
            adv_seq[t] = running
        group_advantages.append(adv_seq)

    # ── Step 4: flatten into batch lists ────────────────────────────────────
    batch_input_prompts: List[str] = []
    batch_sequences: List[str] = []
    batch_advantages: List[float] = []
    batch_old_logprobs: List[float] = []
    batch_raw_rewards: List[float] = []

    process_mean_per_group: List[float] = []
    process_sum_per_group: List[float] = []
    process_step_count_per_group: List[int] = []

    for group_idx, (_batch_idx, _b, _g, group, _scene, _agent) in enumerate(all_groups):
        prompts = list(group.get("prompts") or [])
        responses = list(group.get("responses") or [])
        old_lps = list(group.get("old_logprobs") or [])
        horizon = group_horizons[group_idx]
        adv_seq = group_advantages[group_idx]
        raw_seq = all_step_rewards[
            sum(group_horizons[:group_idx]) : sum(group_horizons[:group_idx]) + horizon
        ]
        pr = list(group.get("process_rewards") or [])[:horizon]
        process_mean_per_group.append(float(np.mean(pr)) if pr else 0.0)
        process_sum_per_group.append(float(np.sum(pr)) if pr else 0.0)
        process_step_count_per_group.append(int(horizon))

        for step_idx in range(horizon):
            batch_input_prompts.append(prompts[step_idx])
            batch_sequences.append(responses[step_idx])
            batch_advantages.append(adv_seq[step_idx])
            batch_old_logprobs.append(
                float(old_lps[step_idx]) if step_idx < len(old_lps) else 0.0
            )
            batch_raw_rewards.append(raw_seq[step_idx] if step_idx < len(raw_seq) else 0.0)

    process_stats = {
        "process_mean_per_group": process_mean_per_group,
        "process_sum_per_group": process_sum_per_group,
        "process_step_count_per_group": process_step_count_per_group,
    }
    return (
        batch_input_prompts,
        batch_sequences,
        batch_advantages,
        batch_old_logprobs,
        batch_raw_rewards,
        process_stats,
    )


def apply_policy_update(
    batch_input_prompts: List[str],
    batch_sequences: List[str],
    batch_advantages: List[float],
    batch_old_logprobs: List[float],
    gradient_batch_size: Optional[int],
    kl_constraint: float,
    entropy_coef: float,
    ppo_clip: float,
    policy: object,
) -> Optional[object]:
    if not batch_sequences:
        logger.warning(
            "[rank %d] No small-model responses collected; skipped policy update.",
            _rank(),
        )
        return None

    grad_bs = (
        gradient_batch_size if gradient_batch_size and gradient_batch_size > 0 else None
    )
    policy_impl = policy.module if hasattr(policy, "module") else policy
    metrics = policy_impl.step_policy_gradient(
        batch_input_prompts,
        batch_sequences,
        advantages=batch_advantages,
        old_logprobs=batch_old_logprobs if batch_old_logprobs else None,
        kl_constraint=kl_constraint,
        entropy_coef=entropy_coef,
        ppo_clip=ppo_clip,
        batch_size=grad_bs,
    )
    return metrics


def log_policy_update(
    batch_raw_rewards: List[float],
    metrics: Optional[object],
    stats: Dict[str, object],
    groups_per_scene: int,
    wandb_run: object,
) -> Optional[Dict[str, object]]:
    if not isinstance(metrics, dict):
        total_groups = _ddp_reduce_scalar(
            float(stats.get("total_groups", 0)),
            dist.ReduceOp.SUM,
        )
        logger.info(
            "[rank %d] Dialogue policy update (groups=%d) completed.",
            _rank(),
            int(total_groups),
        )
        return None

    update_sample_count = int(stats.get("update_sample_count", 0))
    total_groups = int(
        _ddp_reduce_scalar(float(stats.get("total_groups", 0)), dist.ReduceOp.SUM)
    )
    total_scenes = int(
        _ddp_reduce_scalar(float(stats.get("total_scenes", 0)), dist.ReduceOp.SUM)
    )
    max_round_idx = int(
        _ddp_reduce_scalar(float(stats.get("max_round_idx", 0)), dist.ReduceOp.MAX)
    )
    loss_global = _ddp_weighted_mean(
        float(metrics.get("loss", 0.0)),
        max(1, update_sample_count),
    )
    kl_global = _ddp_weighted_mean(
        float(metrics.get("kl", 0.0)),
        max(1, update_sample_count),
    )
    entropy_global = _ddp_weighted_mean(
        float(metrics.get("entropy", 0.0)),
        max(1, update_sample_count),
    )
    grad_norm_global = _ddp_weighted_mean(
        float(metrics.get("grad_norm", 0.0)),
        max(1, update_sample_count),
    )
    raw_reward_mean, raw_reward_std, raw_reward_min, raw_reward_max, _ = (
        _ddp_aggregate_stats(batch_raw_rewards)
    )
    goal_scores = stats["goal_scores_per_group"]
    goal_reward_mean, goal_reward_std, goal_reward_min, goal_reward_max, _ = (
        _ddp_aggregate_stats(goal_scores)
    )
    history_round_counts = stats.get("history_round_counts", [])
    history_round_mean, _history_round_std, _history_round_min, _history_round_max, _ = (
        _ddp_aggregate_stats(history_round_counts)
    )

    logger.info(
        "[rank %d] Dialogue policy update (global groups=%d): loss=%.6f kl=%.6e entropy=%.6f",
        _rank(),
        total_groups,
        loss_global,
        kl_global,
        entropy_global,
    )
    log_dict = {
        "round": max_round_idx,
        "scene_batch_size": total_scenes,
        "num_groups": groups_per_scene,
        "total_groups": total_groups,
        "loss": loss_global,
        "kl": kl_global,
        "approx_kl": float(metrics.get("approx_kl", kl_global)),
        "clipfrac": float(metrics.get("clipfrac", 0.0)),
        "entropy": entropy_global,
        "raw_reward_mean": raw_reward_mean,
        "raw_reward_std": raw_reward_std,
        "raw_reward_min": raw_reward_min,
        "raw_reward_max": raw_reward_max,
        "goal_reward_mean": goal_reward_mean,
        "goal_reward_std": goal_reward_std,
        "goal_reward_min": goal_reward_min,
        "goal_reward_max": goal_reward_max,
        "history_avg_rounds": history_round_mean,
    }
    process_stats = stats.get("process_stats") or {}
    if process_stats:
        proc_means = process_stats.get("process_mean_per_group", [])
        proc_sums = process_stats.get("process_sum_per_group", [])
        proc_steps = process_stats.get("process_step_count_per_group", [])
        if proc_means:
            pm, ps, pmin, pmax, _ = _ddp_aggregate_stats(proc_means)
            log_dict["process_reward_mean"] = pm
            log_dict["process_reward_std"] = ps
        if proc_sums:
            sm, ss, smin, smax, _ = _ddp_aggregate_stats(proc_sums)
            log_dict["process_reward_sum_mean"] = sm
        if proc_steps:
            stm, _, _, _, _ = _ddp_aggregate_stats(proc_steps)
            log_dict["process_step_count_mean"] = stm
    if "grad_norm" in metrics:
        log_dict["grad_norm"] = grad_norm_global
    return log_dict


def update_policy_global(
    completed: Dict[int, object],
    rollouts: Dict[int, dict],
    groups_per_scene: int,
    gradient_batch_size: Optional[int],
    kl_constraint: float,
    entropy_coef: float,
    ppo_clip: float,
    policy: object,
    llm_model: str,
    llm_base_url: str,
    llm_api_key: str,
    wandb_run: object,
    epoch_idx: int,
    process_reward_weight: float = 0.1,
    goal_reward_weight: float = 1.0,
    gamma: float = 1.0,
    local_metrics_path: Optional[str] = None,
    histories_path: Optional[str] = None,
) -> None:
    all_groups = []
    group_index_map: Dict[tuple, int] = {}
    batch_scene_counts: Dict[int, int] = {}
    batch_step = (max(completed.keys()) + 1) if completed else None
    total_scenes = 0
    max_round_idx = 0
    for batch_idx, batch_state in completed.items():
        rollout = rollouts.get(batch_idx)
        if not rollout:
            continue
        B = len(batch_state.prompts)
        total_scenes += B
        batch_scene_counts[batch_idx] = B
        max_round_idx = max(max_round_idx, int(getattr(batch_state, "round_idx", 0)))
        scene_groups = rollout["scene_groups"]
        small_agent_list = rollout["small_agent_list"]
        scenes = rollout["scenes"]
        for b in range(B):
            for g in range(groups_per_scene):
                group_index_map[(batch_idx, b, g)] = len(all_groups)
                all_groups.append(
                    (
                        batch_idx,
                        b,
                        g,
                        scene_groups[b][g],
                        scenes[b],
                        small_agent_list[b],
                    )
                )

    if not all_groups:
        logger.warning(
            "[rank %d] No completed rollouts available; skipped policy update.",
            _rank(),
        )
        return

    history_round_counts = [
        float(len((group or {}).get("all_replies", [])))
        for (_batch_idx, _b, _g, group, _scene, _agent) in all_groups
    ]

    goal_scores_per_group, goal_reasons_per_group = collect_rewards(
        all_groups, llm_model, llm_base_url, llm_api_key
    )
    table_payload: Dict[str, object] = {}
    for batch_idx, batch_state in completed.items():
        rollout = rollouts.get(batch_idx)
        if not rollout:
            continue
        scene_count = batch_scene_counts.get(batch_idx, len(batch_state.prompts))
        table_payload.update(
            build_wandb_dialogue_tables(
                epoch_idx,
                batch_idx,
                scene_count,
                rollout,
                group_index_map,
                goal_scores_per_group,
                goal_reasons_per_group,
            )
        )
    if wandb_run is not None and table_payload:
        wandb_run.log(table_payload, commit=False)

    if histories_path:
        try:
            append_histories_with_rewards(
                histories_path, all_groups, goal_scores_per_group, goal_reasons_per_group
            )
        except Exception:
            pass

    # ── Collect old log-probs for importance-ratio computation ───────────────
    # We compute them here (no grad) so the rollout buffer is frozen before the
    # gradient update loop starts.
    policy_impl = policy.module if hasattr(policy, "module") else policy
    for _batch_idx, _b, _g, group, _scene, _agent in all_groups:
        prompts = list(group.get("prompts") or [])
        responses = list(group.get("responses") or [])
        if prompts and responses:
            with torch.no_grad():
                old_lp, _ = policy_impl.compute_response_logprob_and_entropy(
                    prompts, responses, require_grad=False
                )
            group["old_logprobs"] = [float(v) for v in old_lp.cpu().tolist()]
        else:
            group["old_logprobs"] = []

    (
        batch_input_prompts,
        batch_sequences,
        batch_advantages,
        batch_old_logprobs,
        batch_raw_rewards,
        process_stats,
    ) = compute_advantages(
        all_groups,
        goal_scores_per_group,
        process_reward_weight=process_reward_weight,
        goal_reward_weight=goal_reward_weight,
        gamma=gamma,
    )

    metrics = apply_policy_update(
        batch_input_prompts,
        batch_sequences,
        batch_advantages,
        batch_old_logprobs,
        gradient_batch_size,
        kl_constraint,
        entropy_coef,
        ppo_clip,
        policy,
    )
    log_dict = log_policy_update(
        batch_raw_rewards,
        metrics,
        {
            "max_round_idx": max_round_idx,
            "total_scenes": total_scenes,
            "total_groups": len(all_groups),
            "goal_scores_per_group": goal_scores_per_group,
            "history_round_counts": history_round_counts,
            "batch_step": batch_step,
            "update_sample_count": len(batch_sequences),
            "process_stats": process_stats,
        },
        groups_per_scene,
        wandb_run,
    )
    if wandb_run is not None and isinstance(log_dict, dict):
        wandb_run.log(log_dict)
    if isinstance(log_dict, dict):
        local_payload = dict(log_dict)
        local_payload["epoch_idx"] = int(epoch_idx) if epoch_idx is not None else -1
        local_payload["batch_step"] = int(batch_step) if batch_step is not None else -1
        local_payload["logged_at"] = int(time.time())
        _append_local_metrics(local_metrics_path, local_payload)
    return

def _save_train_state(ckpt_dir: str, epoch_index: int, batch_index: int = -1) -> None:
    """Write train_state.json with last completed epoch/batch (0-based)."""
    import os
    import json
    TRAIN_STATE_FILENAME = "train_state.json"
    path = os.path.join(ckpt_dir, TRAIN_STATE_FILENAME)
    try:
        os.makedirs(ckpt_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "epoch_index": int(epoch_index)
                    if epoch_index is not None
                    else -1,
                    "batch_index": int(batch_index)
                    if batch_index is not None
                    else -1,
                },
                f,
                indent=2,
            )
    except Exception as e:
        logger.warning("Could not save train_state to %s: %s", path, e)


def init_dialogue_round(
    B,
    batch_state,
    which_agent_large,
    groups_per_scene,
    per_batch_rollouts,
    batch_idx,
):
    small_agent_list = []
    scene_groups = []
    for b in range(B):
        small_agent = (
            batch_state.agent1_list[b]
            if which_agent_large == "agent0"
            else batch_state.agent0_list[b]
        )
        small_agent_list.append(small_agent)
        base_history = batch_state.history_messages_all[b]
        groups = []
        for _ in range(groups_per_scene):
            history_clone = {
                batch_state.agent0_list[b]: clone_history(
                    base_history[batch_state.agent0_list[b]]
                ),
                batch_state.agent1_list[b]: clone_history(
                    base_history[batch_state.agent1_list[b]]
                ),
            }
            groups.append(
                {
                    "histories_all": history_clone,
                    "last_reply": None,
                    "prompts": [],
                    "responses": [],
                    "turn_histories": [],
                    "all_replies": [],
                    "all_speakers": [],
                    "all_small_indices": [],
                }
            )
        scene_groups.append(groups)
    group_active_mask = [
        [True] * len(scene_groups[b]) for b in range(len(scene_groups))
    ]
    per_batch_rollouts[batch_idx] = {
        "scene_groups": scene_groups,
        "small_agent_list": small_agent_list,
        "scenes": batch_state.scenes,
        "active_mask": [True] * B,
        "group_active_mask": group_active_mask,
    }


def _sync_should_update_all_ranks(
    local_should_update: bool,
    batch_idx: int,
    world_size: int,
    rank: int,
) -> bool:
    should_update_all_ranks = local_should_update
    if dist.is_available() and dist.is_initialized():
        sync_device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        flag = torch.tensor([1 if local_should_update else 0], device=sync_device)
        dist.all_reduce(flag, op=dist.ReduceOp.SUM)
        ready_count = int(flag.item())
        if ready_count not in (0, world_size):
            if rank == 0:
                logger.warning(
                    "[rank %d] DDP update-step mismatch at batch %d: %d/%d ranks ready; skip this update.",
                    _rank(),
                    batch_idx + 1,
                    ready_count,
                    world_size,
                )
            return False
        should_update_all_ranks = ready_count == world_size
    return should_update_all_ranks


def run_single_train_batch(
    *,
    batch_idx: int,
    loader: DialogueDataLoader,
    loader_iter,
    policy: object,
    per_batch_rollouts: Dict[int, dict],
    saved_batches: set,
    histories_path: str,
    which_agent_large: str,
    groups_per_scene: int,
    max_rounds: int,
    max_new_tokens: int,
    gradient_batch_size: Optional[int],
    kl_constraint: float,
    entropy_coef: float,
    ppo_clip: float,
    process_reward_weight: float,
    goal_reward_weight: float,
    gamma: float,
    reward_model: str,
    reward_base_url: str,
    reward_api_key: str,
    wandb_run: Optional[object],
    epoch_idx: Optional[int],
    local_metrics_path: Optional[str],
    partner_base_url: str,
    partner_model: str,
    partner_api_key: str,
    world_size: int,
    rank: int,
) -> Dict[str, bool]:
    completed_batch_state = None
    for round_idx in range(max_rounds):
        try:
            batch_state = next(loader_iter)
        except StopIteration:
            break
        B = len(batch_state.prompts)

        if batch_idx not in per_batch_rollouts:
            init_dialogue_round(
                B,
                batch_state,
                which_agent_large,
                groups_per_scene,
                per_batch_rollouts,
                batch_idx,
            )

        rollout = per_batch_rollouts[batch_idx]
        group_active_mask = rollout.get("group_active_mask")
        if group_active_mask is None:
            group_active_mask = [[True] * len(rollout["scene_groups"][i]) for i in range(B)]
        active_mask = [any(group_active_mask[i]) for i in range(B)]
        active_group_counts = [
            sum(1 for flag in masks if flag) for masks in group_active_mask
        ]
        scene_active_groups = ",".join(str(c) for c in active_group_counts)
        finished_scene_indices = [i for i, is_active in enumerate(active_mask) if not is_active]
        logger.info(
            "[rank %d] Batch %d/%d Round %d/%d (batch_size=%d, groups=%d, active_groups=%d, scene_active_groups=%s, finished_scenes=%s)",
            _rank(),
            batch_idx + 1,
            loader.num_batches,
            round_idx + 1,
            max_rounds,
            B,
            groups_per_scene,
            sum(active_group_counts),
            scene_active_groups,
            (
                ",".join(str(i) for i in finished_scene_indices)
                if finished_scene_indices
                else "-"
            ),
        )
        reply_texts, has_none_reply, stop_flags = run_dialogue_round(
            policy,
            batch_state,
            rollout,
            max_new_tokens,
            active_mask=active_mask,
            group_active_mask=group_active_mask,
            partner_base_url=partner_base_url,
            partner_model=partner_model,
            partner_api_key=partner_api_key,
        )

        if has_none_reply:
            reply_texts = [r if r is not None else None for r in reply_texts]

        missing_active_reply = any(
            active_mask[i] and (reply_texts[i] is None) for i in range(B)
        )
        if has_none_reply or missing_active_reply:
            if batch_idx not in saved_batches:
                try:
                    append_histories_to_file(histories_path, batch_state)
                    saved_batches.add(batch_idx)
                except Exception:
                    pass
            for _ in range(round_idx + 1, max_rounds):
                try:
                    next(loader_iter)
                except StopIteration:
                    break
            break

        # Commit only active samples; inactive ones are frozen.
        for i in range(B):
            if not active_mask[i]:
                continue
            r = (reply_texts[i] or "").strip()
            batch_state.last_reply[i] = r
            speaker = batch_state.speakers[i]
            batch_state.history_messages_all[i][speaker].append(
                {"role": "assistant", "content": r}
            )

        # A group should stop immediately once stop_token is generated.
        for i in range(B):
            for g in range(len(group_active_mask[i])):
                if group_active_mask[i][g] and stop_flags[i][g]:
                    group_active_mask[i][g] = False
            if not any(group_active_mask[i]):
                active_mask[i] = False
                # Prevent dataloader from appending this scene as next-round user input.
                batch_state.last_reply[i] = None
            else:
                active_mask[i] = True
        rollout["group_active_mask"] = group_active_mask
        rollout["active_mask"] = active_mask

        if not any(active_mask):
            completed_batch_state = batch_state
            for _ in range(round_idx + 1, max_rounds):
                try:
                    next(loader_iter)
                except StopIteration:
                    break
            break

        if (round_idx + 1) >= max_rounds:
            completed_batch_state = batch_state

        time.sleep(0.01)

    local_should_update = completed_batch_state is not None
    should_update_all_ranks = _sync_should_update_all_ranks(
        local_should_update=local_should_update,
        batch_idx=batch_idx,
        world_size=world_size,
        rank=rank,
    )
    if should_update_all_ranks and batch_idx in per_batch_rollouts:
        update_policy_global(
            {batch_idx: completed_batch_state},
            {batch_idx: per_batch_rollouts[batch_idx]},
            groups_per_scene,
            gradient_batch_size,
            kl_constraint,
            entropy_coef,
            ppo_clip,
            policy,
            reward_model,
            reward_base_url,
            reward_api_key,
            wandb_run,
            epoch_idx,
            process_reward_weight=process_reward_weight,
            goal_reward_weight=goal_reward_weight,
            gamma=gamma,
            local_metrics_path=local_metrics_path,
            histories_path=histories_path,
        )

    if batch_idx in per_batch_rollouts:
        del per_batch_rollouts[batch_idx]
    impl = policy.module if hasattr(policy, "module") else policy
    if hasattr(impl, "clear_kv_cache"):
        impl.clear_kv_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

    return {
        "did_update": bool(should_update_all_ranks),
    }


def training_loop_goal_single(
    policy,
    which_agent_large="agent0",
    max_rounds: int = 100,
    kl_constraint: float = 0.02,
    entropy_coef: float = 0.0,
    ppo_clip: float = 0.2,
    process_reward_weight: float = 0.1,
    goal_reward_weight: float = 1.0,
    gamma: float = 1.0,
    max_new_tokens: int = 128,
    checkpoint_every: int = 10,
    save_path: str = "qwen_policy.json",
    backgrounds_path: Optional[str] = None,
    val_ratio: float = 0.15,
    seed: Optional[int] = None,
    wandb_run: Optional[object] = None,
    scene_batch_size: int = 1,
    gradient_batch_size: Optional[int] = None,
    num_groups: Optional[int] = None,
    epoch_idx=None,
    rank: int = 0,
    world_size: int = 1,
    run_timestamp: Optional[str] = None,
    local_metrics_path: Optional[str] = None,
    train_scenes: Optional[List[dict]] = None,
    start_batch_idx: int = 0,
    partner_base_url: str = "http://127.0.0.1:8000/v1",
    partner_model: str = "Qwen3.5-35B-A3B",
    partner_api_key: str = "EMPTY",
    reward_model: str = "Qwen3.5-35B-A3B",
    reward_base_url: str = "http://127.0.0.1:8000/v1",
    reward_api_key: str = "EMPTY",
    val_scenes: Optional[List[dict]] = None,
    test_every_batches: int = 1,
    test_max_scenes: int = 0,
    test_scene_batch_size: Optional[int] = None,
    test_num_groups: Optional[int] = None,
    judge_model: Optional[str] = None,
    judge_base_url: Optional[str] = None,
    judge_api_key: Optional[str] = None,
):
    if run_timestamp is not None:
        # Caller already controls the run-level directory layout.
        run_save_dir = os.path.abspath(save_path)
    else:
        active_timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_save_dir = os.path.abspath(f"{save_path}_{active_timestamp}")
    os.makedirs(run_save_dir, exist_ok=True)
    logger.info("[rank %d] Checkpoints for this run will be saved under: %s", _rank(), run_save_dir)
    epoch_seed = (
        int(seed) + int(epoch_idx)
        if seed is not None and epoch_idx is not None
        else seed
    )

    loader = DialogueDataLoader(
        backgrounds_path,
        batch_size=scene_batch_size,
        max_rounds=max_rounds,
        which_agent_large=which_agent_large,
        val_ratio=val_ratio,
        seed=epoch_seed,
        shuffle=True,
        rank=rank,
        world_size=world_size,
        train_scenes=train_scenes,
    )

    histories_path = os.path.join(run_save_dir, "histories.jsonl")
    saved_batches = set()
    per_batch_rollouts: Dict[int, dict] = {}
    groups_per_scene = (
        int(num_groups) if num_groups is not None and int(num_groups) > 0 else 1
    )
    if not local_metrics_path:
        local_metrics_path = os.path.join(
            run_save_dir, "training_metrics_summary.jsonl"
        )
    loader_iter = iter(loader)
    start_batch_idx = max(0, min(int(start_batch_idx), int(loader.num_batches)))
    if start_batch_idx > 0:
        logger.info(
            "[rank %d] Resuming within epoch from batch %d/%d",
            _rank(),
            start_batch_idx + 1,
            loader.num_batches,
        )
        states_to_skip = start_batch_idx * max_rounds
        for _ in range(states_to_skip):
            try:
                next(loader_iter)
            except StopIteration:
                break

    for batch_idx in range(start_batch_idx, loader.num_batches):
        batch_result = run_single_train_batch(
            batch_idx=batch_idx,
            loader=loader,
            loader_iter=loader_iter,
            policy=policy,
            per_batch_rollouts=per_batch_rollouts,
            saved_batches=saved_batches,
            histories_path=histories_path,
            which_agent_large=which_agent_large,
            groups_per_scene=groups_per_scene,
            max_rounds=max_rounds,
            max_new_tokens=max_new_tokens,
            gradient_batch_size=gradient_batch_size,
            kl_constraint=kl_constraint,
            entropy_coef=entropy_coef,
            ppo_clip=ppo_clip,
            process_reward_weight=process_reward_weight,
            goal_reward_weight=goal_reward_weight,
            gamma=gamma,
            reward_model=reward_model,
            reward_base_url=reward_base_url,
            reward_api_key=reward_api_key,
            wandb_run=wandb_run,
            epoch_idx=epoch_idx,
            local_metrics_path=local_metrics_path,
            partner_base_url=partner_base_url,
            partner_model=partner_model,
            partner_api_key=partner_api_key,
            world_size=world_size,
            rank=rank,
        )
        if (
            val_scenes
            and int(test_every_batches) > 0
            and ((batch_idx + 1) % int(test_every_batches) == 0)
        ):
            # Local import avoids a module-level cycle.
            from rl_testing_loop_goal_single import testing_loop_goal_single

            testing_loop_goal_single(
                policy=policy,
                val_scenes=val_scenes,
                which_agent_large=which_agent_large,
                max_rounds=max_rounds,
                max_new_tokens=max_new_tokens,
                scene_batch_size=(
                    int(test_scene_batch_size)
                    if test_scene_batch_size is not None
                    else int(scene_batch_size)
                ),
                num_groups=(
                    int(test_num_groups)
                    if test_num_groups is not None
                    else num_groups
                ),
                epoch_idx=epoch_idx,
                rank=rank,
                world_size=world_size,
                wandb_run=wandb_run,
                local_metrics_path=local_metrics_path,
                test_histories_path=os.path.join(
                    run_save_dir,
                    f"test_histories_batch{batch_idx + 1}.jsonl",
                ),
                test_max_scenes=test_max_scenes,
                partner_base_url=partner_base_url,
                partner_model=partner_model,
                partner_api_key=partner_api_key,
                judge_model=judge_model or reward_model,
                judge_base_url=judge_base_url or reward_base_url,
                judge_api_key=judge_api_key or reward_api_key,
            )
        if (
            batch_result.get("did_update")
            and checkpoint_every
            and checkpoint_every > 0
            and ((batch_idx + 1) % checkpoint_every == 0)
            and _rank() == 0
        ):
            try:
                impl = policy.module if hasattr(policy, "module") else policy
                checkpoint_path = os.path.join(run_save_dir, f"batch{batch_idx + 1}")
                impl.save(checkpoint_path)
                _save_train_state(run_save_dir, epoch_idx, batch_idx)
                logger.info(
                    "[rank %d] Saved checkpoint to %s (every %d batches)",
                    _rank(),
                    checkpoint_path,
                    checkpoint_every,
                )
            except Exception as e:
                logger.warning(
                    "[rank %d] Failed to save checkpoint after batch %d: %s",
                    _rank(),
                    batch_idx + 1,
                    e,
                )

    return policy
