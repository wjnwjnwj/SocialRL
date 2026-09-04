#!/usr/bin/env python3
from __future__ import annotations

import gc
import math
import os
import time
import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import wandb
import numpy as np

from rl_testing_loop_goal_single import testing_loop_goal_single
from rl_dataset import DialogueDataLoader
from utils import (
    _append_local_metrics,
    _ddp_aggregate_stats,
    _ddp_reduce_scalar,
    _extract_scene_id,
    append_histories_to_file,
    clone_history,
    collect_rewards,
    logger,
    run_dialogue_round,
    append_histories_with_rewards,
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
    process_reward_mean_per_group: List[float],
    process_reward_sum_per_group: List[float],
    process_step_count_per_group: List[int],
    total_output_round_count_per_group: List[int],
    small_output_round_count_per_group: List[int],
    old_value_mean_per_group: List[float],
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
        process_row = []
        process_sum_row = []
        process_steps_row = []
        total_output_rounds_row = []
        small_output_rounds_row = []
        old_value_row = []
        total_row = []
        goal_reason_row = []
        for group_idx, _group in enumerate(groups):
            global_idx = group_index_map.get((batch_idx, scene_idx, group_idx))
            if global_idx is None:
                goal_row.append("")
                process_row.append("")
                process_sum_row.append("")
                process_steps_row.append("")
                total_output_rounds_row.append("")
                small_output_rounds_row.append("")
                old_value_row.append("")
                total_row.append("")
                goal_reason_row.append("")
                continue
            goal_score = goal_scores_per_group[global_idx]
            goal_row.append(f"goal_reward={goal_score:.2f}")
            process_mean = (
                process_reward_mean_per_group[global_idx]
                if global_idx < len(process_reward_mean_per_group)
                else 0.0
            )
            process_row.append(f"process_reward_mean={process_mean:.4f}")
            process_sum = (
                process_reward_sum_per_group[global_idx]
                if global_idx < len(process_reward_sum_per_group)
                else 0.0
            )
            process_sum_row.append(f"process_reward_sum={process_sum:.4f}")
            process_steps = (
                process_step_count_per_group[global_idx]
                if global_idx < len(process_step_count_per_group)
                else 0
            )
            process_steps_row.append(f"process_step_count={int(process_steps)}")
            total_output_round_count = (
                total_output_round_count_per_group[global_idx]
                if global_idx < len(total_output_round_count_per_group)
                else 0
            )
            total_output_rounds_row.append(
                f"output_round_count={int(total_output_round_count)}"
            )
            small_output_round_count = (
                small_output_round_count_per_group[global_idx]
                if global_idx < len(small_output_round_count_per_group)
                else 0
            )
            small_output_rounds_row.append(
                f"small_output_round_count={int(small_output_round_count)}"
            )
            old_value_mean = (
                old_value_mean_per_group[global_idx]
                if global_idx < len(old_value_mean_per_group)
                else 0.0
            )
            old_value_row.append(f"old_value_mean={old_value_mean:.4f}")
            total_row.append(f"total_reward={goal_score + process_mean:.2f}")
            goal_reason = goal_reasons_per_group[global_idx]
            goal_reason_row.append(f"goal_reason={goal_reason}" if goal_reason else "")
        table.add_data(*goal_row)
        table.add_data(*process_row)
        table.add_data(*process_sum_row)
        table.add_data(*process_steps_row)
        table.add_data(*total_output_rounds_row)
        table.add_data(*small_output_rounds_row)
        table.add_data(*old_value_row)
        table.add_data(*total_row)
        table.add_data(*goal_reason_row)
        tables[f"dialogue_table_epoch{epoch_idx}_batch{batch_idx}_scene{scene_idx}"] = (
            table
        )
    return tables


def compute_advantages(
    all_groups: List[tuple],
    goal_scores_per_group: List[float],
    policy: object,
    critic: object,
    gamma: float,
    gae_lambda: float,
    goal_reward_weight: float,
    process_reward_weight: float,
    kl_beta: float = 0.0,
) -> tuple:
    """
    GAE + reward shaping. When `kl_beta > 0`, additionally compute the
    reference-policy logprob (= base model with PEFT adapter disabled) for
    every (prompt, response) pair so the PPO update can add a
    `β·KL(π_θ ‖ π_ref)` penalty. The reference forward is gradient-free and
    re-uses the already-loaded base weights, so this costs one extra forward
    per sample and zero extra GPU memory.
    """

    def _compute_gae(
        rewards: List[float], values: List[float], discount: float, lam: float
    ) -> tuple[List[float], List[float]]:
        t_len = len(rewards)
        if t_len == 0:
            return [], []
        adv = [0.0] * t_len
        last_gae = 0.0
        next_value = 0.0
        for t in reversed(range(t_len)):
            delta = float(rewards[t]) + float(discount) * float(next_value) - float(values[t])
            last_gae = float(delta) + float(discount) * float(lam) * float(last_gae)
            adv[t] = last_gae
            next_value = values[t]
        ret = [float(a + v) for a, v in zip(adv, values)]
        return adv, ret

    policy_impl = policy.module if hasattr(policy, "module") else policy
    critic_impl = critic.module if hasattr(critic, "module") else critic

    batch_input_prompts: List[str] = []
    batch_sequences: List[str] = []
    batch_old_logprobs: List[float] = []
    batch_ref_logprobs: List[float] = []
    batch_old_values: List[float] = []
    old_value_mean_per_group: List[float] = []
    batch_advantages: List[float] = []
    batch_returns: List[float] = []
    batch_raw_rewards: List[float] = []
    process_reward_mean_per_group: List[float] = []
    process_reward_sum_per_group: List[float] = []
    process_step_count_per_group: List[int] = []
    total_output_round_count_per_group: List[int] = []
    small_output_round_count_per_group: List[int] = []

    for group_idx, (_batch_idx, _b, _g, group, _scene, _agent) in enumerate(all_groups):
        prompts = list(group.get("prompts", []))
        responses = list(group.get("responses", []))
        all_replies = list(group.get("all_replies", []))
        horizon = min(len(prompts), len(responses))
        if horizon <= 0:
            continue
        prompts = prompts[:horizon]
        responses = responses[:horizon]
        terminal_reward = float(goal_scores_per_group[group_idx]) * float(goal_reward_weight)
        process_rewards = list(group.get("process_rewards", []))[:horizon]
        if len(process_rewards) != horizon:
            logger.warning(
                "[rank %d] process_rewards length mismatch: horizon=%d got=%d; padding/truncating.",
                _rank(),
                horizon,
                len(process_rewards),
            )
        if len(process_rewards) < horizon:
            process_rewards.extend([0.0] * (horizon - len(process_rewards)))
        rewards = [float(process_reward_weight) * float(r) for r in process_rewards]
        try:
            trajectory_normalize = bool(int(os.environ.get("TRAJECTORY_NORMALIZE", "0")))
        except Exception:
            trajectory_normalize = False
        if trajectory_normalize:
            try:
                expected_horizon_value = float(os.environ.get("EXPECTED_HORIZON", "5"))
            except Exception:
                expected_horizon_value = 5.0
            if expected_horizon_value <= 0:
                expected_horizon_value = 5.0
            rewards = [r / expected_horizon_value for r in rewards]
        rewards[-1] += terminal_reward
        process_reward_mean_per_group.append(
            float(np.mean(process_rewards)) if process_rewards else 0.0
        )
        process_reward_sum_per_group.append(
            float(np.sum(process_rewards)) if process_rewards else 0.0
        )
        process_step_count_per_group.append(int(len(process_rewards)))
        total_output_round_count_per_group.append(int(len(all_replies)))
        small_output_round_count_per_group.append(int(len(responses)))

        old_logprobs_t, _ = policy_impl.compute_response_logprob_and_entropy(
            prompts, responses, require_grad=False
        )
        old_logprobs_list = [
            float(v) for v in old_logprobs_t.detach().cpu().tolist()
        ]
        if kl_beta > 0.0:
            ref_logprobs_t, _ = policy_impl.compute_response_logprob_and_entropy(
                prompts,
                responses,
                require_grad=False,
                with_disabled_adapter=True,
            )
            ref_logprobs_list = [
                float(v) for v in ref_logprobs_t.detach().cpu().tolist()
            ]
        else:
            ref_logprobs_list = [0.0] * len(old_logprobs_list)
        old_values_t = critic_impl.evaluate_values(prompts, require_grad=False)
        old_values_list = [float(v) for v in old_values_t.detach().cpu().tolist()]
        old_value_mean_per_group.append(
            float(np.mean(old_values_list)) if old_values_list else 0.0
        )
        adv_t, ret_t = _compute_gae(
            rewards=rewards,
            values=old_values_list,
            discount=gamma,
            lam=gae_lambda,
        )

        batch_input_prompts.extend(prompts)
        batch_sequences.extend(responses)
        batch_old_logprobs.extend(old_logprobs_list)
        batch_ref_logprobs.extend(ref_logprobs_list)
        batch_old_values.extend(old_values_list)
        batch_advantages.extend(adv_t)
        batch_returns.extend(ret_t)
        batch_raw_rewards.extend(rewards)

        # Stash per-turn critic outputs / GAE products on the group so
        # ``append_histories_with_rewards`` can dump them next to process rewards.
        group["old_values"] = [float(v) for v in old_values_list]
        group["advantages"] = [float(a) for a in adv_t]
        group["returns"] = [float(r) for r in ret_t]
        group["shaped_rewards"] = [float(r) for r in rewards]

    return (
        batch_input_prompts,
        batch_sequences,
        batch_old_logprobs,
        batch_ref_logprobs,
        batch_old_values,
        old_value_mean_per_group,
        batch_advantages,
        batch_returns,
        batch_raw_rewards,
        process_reward_mean_per_group,
        process_reward_sum_per_group,
        process_step_count_per_group,
        total_output_round_count_per_group,
        small_output_round_count_per_group,
    )


def apply_policy_update(
    batch_input_prompts: List[str],
    batch_sequences: List[str],
    batch_old_logprobs: List[float],
    batch_ref_logprobs: List[float],
    batch_advantages: List[float],
    batch_returns: List[float],
    policy: object,
    critic: object,
    ppo_clip: float,
    ppo_epochs: int,
    ppo_minibatch_size: int,
    gradient_batch_size: Optional[int],
    vf_coef: float,
    ent_coef: float,
    normalize_advantage: bool,
    kl_beta: float = 0.0,
) -> Optional[object]:
    if not batch_sequences:
        logger.warning(
            "[rank %d] No small-model responses collected; skipped policy update.",
            _rank(),
        )
        return None

    policy_impl = policy.module if hasattr(policy, "module") else policy
    critic_impl = critic.module if hasattr(critic, "module") else critic

    policy_impl.model.train()
    critic_impl.model.train()

    policy_device = next(policy_impl.model.parameters()).device
    old_logprobs_t = torch.tensor(
        batch_old_logprobs, dtype=torch.float32, device=policy_device
    )
    use_kl_penalty = bool(kl_beta and kl_beta > 0.0)
    if use_kl_penalty:
        ref_logprobs_t = torch.tensor(
            batch_ref_logprobs, dtype=torch.float32, device=policy_device
        )
    else:
        ref_logprobs_t = None
    advantages_t = torch.tensor(batch_advantages, dtype=torch.float32, device=policy_device)
    returns_t = torch.tensor(batch_returns, dtype=torch.float32, device=policy_device)

    if normalize_advantage and advantages_t.numel() > 1:
        adv_std = advantages_t.std(unbiased=False)
        advantages_t = (advantages_t - advantages_t.mean()) / (adv_std + 1e-8)

    n = int(len(batch_sequences))
    if dist.is_available() and dist.is_initialized():
        sync_device = policy_device
        n_t = torch.tensor([n], device=sync_device, dtype=torch.int64)
        dist.all_reduce(n_t, op=dist.ReduceOp.MIN)
        n_sync = int(n_t.item())
        if n_sync != n:
            logger.warning(
                "[rank %d] PPO sample count mismatch across ranks: local=%d synced_min=%d; truncating for DDP consistency.",
                _rank(),
                n,
                n_sync,
            )
        n = n_sync
        if n <= 0:
            logger.warning(
                "[rank %d] Synced PPO sample count is zero; skip update this batch.",
                _rank(),
            )
            return None
        batch_input_prompts = batch_input_prompts[:n]
        batch_sequences = batch_sequences[:n]
        old_logprobs_t = old_logprobs_t[:n]
        if ref_logprobs_t is not None:
            ref_logprobs_t = ref_logprobs_t[:n]
        advantages_t = advantages_t[:n]
        returns_t = returns_t[:n]

    mb_size = max(1, int(ppo_minibatch_size))
    grad_chunk_size = (
        int(gradient_batch_size)
        if gradient_batch_size is not None and int(gradient_batch_size) > 0
        else None
    )
    epochs = max(1, int(ppo_epochs))
    total_minibatches = (n + mb_size - 1) // mb_size if n > 0 else 0

    policy_losses: List[float] = []
    value_losses: List[float] = []
    entropy_losses: List[float] = []
    approx_kls: List[float] = []
    kl_refs: List[float] = []
    clipfracs: List[float] = []
    actor_grad_norms: List[float] = []
    critic_grad_norms: List[float] = []

    for ppo_epoch_idx in range(epochs):
        logger.info(
            "[rank %d] PPO epoch %d/%d start: samples=%d minibatches=%d mb_size=%d grad_chunk=%s",
            _rank(),
            ppo_epoch_idx + 1,
            epochs,
            n,
            total_minibatches,
            mb_size,
            str(grad_chunk_size),
        )
        indices = list(range(n))
        random.shuffle(indices)
        for mb_idx_in_epoch, start in enumerate(range(0, n, mb_size)):
            mb_idx_list = indices[start : start + mb_size]
            mb_total = len(mb_idx_list)
            if mb_total <= 0:
                continue
            chunk_size = min(mb_total, grad_chunk_size) if grad_chunk_size else mb_total
            mb_no = mb_idx_in_epoch + 1

            policy_impl.optimizer.zero_grad()
            policy_loss_sum = 0.0
            entropy_sum = 0.0
            approx_kl_sum = 0.0
            kl_ref_sum = 0.0
            clipfrac_sum = 0.0
            for c_start in range(0, mb_total, chunk_size):
                c_indices = mb_idx_list[c_start : c_start + chunk_size]
                c_size = len(c_indices)
                c_idx = torch.tensor(c_indices, dtype=torch.long, device=policy_device)
                c_prompts = [batch_input_prompts[i] for i in c_indices]
                c_sequences = [batch_sequences[i] for i in c_indices]
                c_old_logprobs = old_logprobs_t[c_idx]
                c_advantages = advantages_t[c_idx]
                c_ref_logprobs = (
                    ref_logprobs_t[c_idx] if (use_kl_penalty and ref_logprobs_t is not None) else None
                )

                new_logprobs, entropies = policy_impl.compute_response_logprob_and_entropy(
                    c_prompts, c_sequences, require_grad=True
                )
                log_ratio = new_logprobs - c_old_logprobs
                ratio = torch.exp(log_ratio)
                surr1 = ratio * c_advantages
                surr2 = torch.clamp(ratio, 1.0 - ppo_clip, 1.0 + ppo_clip) * c_advantages
                policy_loss_chunk = -torch.min(surr1, surr2).mean()
                entropy_chunk = entropies.mean()
                if c_ref_logprobs is not None:
                    # k3 KL estimator (Schulman 2020): non-negative, unbiased.
                    log_ratio_ref = new_logprobs - c_ref_logprobs
                    kl_ref_chunk = (
                        torch.exp(log_ratio_ref) - log_ratio_ref - 1.0
                    ).mean()
                    actor_loss_chunk = (
                        policy_loss_chunk
                        - ent_coef * entropy_chunk
                        + kl_beta * kl_ref_chunk
                    )
                else:
                    kl_ref_chunk = torch.tensor(0.0, device=policy_device)
                    actor_loss_chunk = policy_loss_chunk - ent_coef * entropy_chunk
                scale = float(c_size) / float(mb_total)
                (actor_loss_chunk * scale).backward()

                with torch.no_grad():
                    approx_kl_chunk = (c_old_logprobs - new_logprobs).mean()
                    clipfrac_chunk = ((ratio - 1.0).abs() > ppo_clip).float().mean()
                policy_loss_sum += float(policy_loss_chunk.detach().cpu().item()) * scale
                entropy_sum += float(entropy_chunk.detach().cpu().item()) * scale
                approx_kl_sum += float(approx_kl_chunk.detach().cpu().item()) * scale
                kl_ref_sum += float(kl_ref_chunk.detach().cpu().item()) * scale
                clipfrac_sum += float(clipfrac_chunk.detach().cpu().item()) * scale

            logger.info(
                "[rank %d] PPO actor backward done: epoch=%d/%d mb=%d/%d",
                _rank(),
                ppo_epoch_idx + 1,
                epochs,
                mb_no,
                total_minibatches,
            )
            has_nan_grad = False
            for p in policy_impl.model.parameters():
                if p.grad is not None and torch.isnan(p.grad).any():
                    has_nan_grad = True
                    break
            if has_nan_grad:
                logger.warning(
                    "[rank %d] NaN detected in actor gradients, skipping this update",
                    _rank(),
                )
                policy_impl.optimizer.zero_grad()
                continue
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in policy_impl.model.parameters() if p.requires_grad], max_norm=1.0
            )
            if torch.isnan(actor_grad_norm):
                logger.warning(
                    "[rank %d] NaN in actor_grad_norm after clipping, skipping update",
                    _rank(),
                )
                policy_impl.optimizer.zero_grad()
                continue
            policy_impl.optimizer.step()
            logger.info(
                "[rank %d] PPO actor step done: epoch=%d/%d mb=%d/%d grad_norm=%.6f",
                _rank(),
                ppo_epoch_idx + 1,
                epochs,
                mb_no,
                total_minibatches,
                float(actor_grad_norm.detach().cpu().item()),
            )

            critic_impl.optimizer.zero_grad()
            value_loss_sum = 0.0
            logger.info(
                "[rank %d] PPO critic start: epoch=%d/%d mb=%d/%d mb_samples=%d chunk=%d",
                _rank(),
                ppo_epoch_idx + 1,
                epochs,
                mb_no,
                total_minibatches,
                mb_total,
                chunk_size,
            )
            for c_start in range(0, mb_total, chunk_size):
                c_indices = mb_idx_list[c_start : c_start + chunk_size]
                c_size = len(c_indices)
                c_idx = torch.tensor(c_indices, dtype=torch.long, device=policy_device)
                c_prompts = [batch_input_prompts[i] for i in c_indices]
                c_returns = returns_t[c_idx]

                value_pred = critic_impl.evaluate_values(c_prompts, require_grad=True)
                c_returns_critic = c_returns.to(value_pred.device)
                value_loss_chunk = torch.nn.functional.mse_loss(value_pred, c_returns_critic)
                scale = float(c_size) / float(mb_total)
                (vf_coef * value_loss_chunk * scale).backward()
                value_loss_sum += float(value_loss_chunk.detach().cpu().item()) * scale

            logger.info(
                "[rank %d] PPO critic backward done: epoch=%d/%d mb=%d/%d",
                _rank(),
                ppo_epoch_idx + 1,
                epochs,
                mb_no,
                total_minibatches,
            )
            has_nan_critic_grad = False
            for p in critic_impl.model.parameters():
                if p.grad is not None and torch.isnan(p.grad).any():
                    has_nan_critic_grad = True
                    break
            if has_nan_critic_grad:
                logger.warning(
                    "[rank %d] NaN detected in critic gradients, skipping this update",
                    _rank(),
                )
                critic_impl.optimizer.zero_grad()
                continue
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in critic_impl.model.parameters() if p.requires_grad], max_norm=1.0
            )
            if torch.isnan(critic_grad_norm):
                logger.warning(
                    "[rank %d] NaN in critic_grad_norm after clipping, skipping update",
                    _rank(),
                )
                critic_impl.optimizer.zero_grad()
                continue
            critic_impl.optimizer.step()
            logger.info(
                "[rank %d] PPO critic step done: epoch=%d/%d mb=%d/%d grad_norm=%.6f",
                _rank(),
                ppo_epoch_idx + 1,
                epochs,
                mb_no,
                total_minibatches,
                float(critic_grad_norm.detach().cpu().item()),
            )

            should_log_mb = (
                mb_idx_in_epoch == 0
                or (mb_idx_in_epoch + 1) == total_minibatches
                or ((mb_idx_in_epoch + 1) % 10 == 0)
            )
            if should_log_mb:
                logger.info(
                    "[rank %d] PPO epoch %d/%d minibatch %d/%d: mb_samples=%d chunk=%d policy_loss=%.6f value_loss=%.6f approx_kl=%.6e",
                    _rank(),
                    ppo_epoch_idx + 1,
                    epochs,
                    mb_no,
                    total_minibatches,
                    mb_total,
                    chunk_size,
                    policy_loss_sum,
                    value_loss_sum,
                    approx_kl_sum,
                )

            policy_losses.append(policy_loss_sum)
            value_losses.append(value_loss_sum)
            entropy_losses.append(entropy_sum)
            approx_kls.append(approx_kl_sum)
            kl_refs.append(kl_ref_sum)
            clipfracs.append(clipfrac_sum)
            actor_grad_norms.append(float(actor_grad_norm.detach().cpu().item()))
            critic_grad_norms.append(float(critic_grad_norm.detach().cpu().item()))

    policy_impl.model.eval()
    critic_impl.model.eval()

    def safe_mean(values: List[float]) -> float:
        """计算平均值，过滤掉 NaN"""
        valid = [v for v in values if not math.isnan(v)]
        return float(np.mean(valid)) if valid else 0.0

    return {
        "loss": safe_mean(policy_losses),
        "value_loss": safe_mean(value_losses),
        "entropy": safe_mean(entropy_losses),
        "approx_kl": safe_mean(approx_kls),
        "kl_ref": safe_mean(kl_refs),
        "clipfrac": safe_mean(clipfracs),
        "grad_norm": safe_mean(actor_grad_norms),
        "critic_grad_norm": safe_mean(critic_grad_norms),
    }


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
        float(metrics.get("approx_kl", 0.0)),
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
    old_values = stats.get("old_values", [])
    old_value_mean, old_value_std, old_value_min, old_value_max, _ = (
        _ddp_aggregate_stats(old_values)
    )
    process_sums = stats.get("process_reward_sums_per_group", [])
    process_sum_mean, process_sum_std, process_sum_min, process_sum_max, _ = (
        _ddp_aggregate_stats(process_sums)
    )
    process_steps = stats.get("process_step_counts_per_group", [])
    process_step_mean, process_step_std, process_step_min, process_step_max, _ = (
        _ddp_aggregate_stats(process_steps)
    )
    total_output_round_counts = stats.get("total_output_round_counts_per_group", [])
    (
        total_output_round_mean,
        total_output_round_std,
        total_output_round_min,
        total_output_round_max,
        _,
    ) = _ddp_aggregate_stats(total_output_round_counts)
    small_output_round_counts = stats.get("small_output_round_counts_per_group", [])
    (
        small_output_round_mean,
        small_output_round_std,
        small_output_round_min,
        small_output_round_max,
        _,
    ) = _ddp_aggregate_stats(small_output_round_counts)

    logger.info(
        "[rank %d] Dialogue PPO update (global groups=%d): loss=%.6f approx_kl=%.6e",
        _rank(),
        total_groups,
        loss_global,
        kl_global,
    )
    log_dict = {
        "round": max_round_idx,
        "scene_batch_size": total_scenes,
        "num_groups": groups_per_scene,
        "total_groups": total_groups,
        "loss": loss_global,
        "approx_kl": kl_global,
        "kl_ref": float(metrics.get("kl_ref", 0.0)),
        "value_loss": float(metrics.get("value_loss", 0.0)),
        "entropy": float(metrics.get("entropy", 0.0)),
        "clipfrac": float(metrics.get("clipfrac", 0.0)),
        "raw_reward_mean": raw_reward_mean,
        "raw_reward_std": raw_reward_std,
        "raw_reward_min": raw_reward_min,
        "raw_reward_max": raw_reward_max,
        "goal_reward_mean": goal_reward_mean,
        "goal_reward_std": goal_reward_std,
        "goal_reward_min": goal_reward_min,
        "goal_reward_max": goal_reward_max,
        "old_value_mean": old_value_mean,
        "old_value_std": old_value_std,
        "old_value_min": old_value_min,
        "old_value_max": old_value_max,
        "process_reward_sum_mean": process_sum_mean,
        "process_reward_sum_std": process_sum_std,
        "process_reward_sum_min": process_sum_min,
        "process_reward_sum_max": process_sum_max,
        "process_step_count_mean": process_step_mean,
        "process_step_count_std": process_step_std,
        "process_step_count_min": process_step_min,
        "process_step_count_max": process_step_max,
        "output_round_count_mean": total_output_round_mean,
        "output_round_count_std": total_output_round_std,
        "output_round_count_min": total_output_round_min,
        "output_round_count_max": total_output_round_max,
        "small_output_round_count_mean": small_output_round_mean,
        "small_output_round_count_std": small_output_round_std,
        "small_output_round_count_min": small_output_round_min,
        "small_output_round_count_max": small_output_round_max,
    }
    if "grad_norm" in metrics:
        log_dict["grad_norm"] = grad_norm_global
    if "critic_grad_norm" in metrics:
        log_dict["critic_grad_norm"] = float(metrics.get("critic_grad_norm", 0.0))
    return log_dict


def update_policy_global(
    completed: Dict[int, object],
    rollouts: Dict[int, dict],
    groups_per_scene: int,
    policy: object,
    critic: object,
    gamma: float,
    gae_lambda: float,
    ppo_clip: float,
    ppo_epochs: int,
    ppo_minibatch_size: int,
    gradient_batch_size: Optional[int],
    vf_coef: float,
    ent_coef: float,
    normalize_advantage: bool,
    goal_reward_weight: float,
    process_reward_weight: float,
    llm_model: str,
    llm_base_url: str,
    llm_api_key: str,
    wandb_run: object,
    epoch_idx: int,
    local_metrics_path: Optional[str] = None,
    histories_path: Optional[str] = None,
    kl_beta: float = 0.0,
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

    goal_scores_per_group, goal_reasons_per_group = collect_rewards(
        all_groups, llm_model, llm_base_url, llm_api_key
    )
    (
        batch_input_prompts,
        batch_sequences,
        batch_old_logprobs,
        batch_ref_logprobs,
        batch_old_values,
        old_value_mean_per_group,
        batch_advantages,
        batch_returns,
        batch_raw_rewards,
        process_reward_mean_per_group,
        process_reward_sum_per_group,
        process_step_count_per_group,
        total_output_round_count_per_group,
        small_output_round_count_per_group,
    ) = (
        compute_advantages(
            all_groups,
            goal_scores_per_group,
            policy,
            critic,
            gamma,
            gae_lambda,
            goal_reward_weight,
            process_reward_weight,
            kl_beta=kl_beta,
        )
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
                process_reward_mean_per_group,
                process_reward_sum_per_group,
                process_step_count_per_group,
                total_output_round_count_per_group,
                small_output_round_count_per_group,
                old_value_mean_per_group,
            )
        )
    if wandb_run is not None and table_payload:
        wandb_run.log(table_payload, commit=False)

    if histories_path:
        append_histories_with_rewards(
            histories_path, all_groups, goal_scores_per_group, goal_reasons_per_group
        )

    metrics = apply_policy_update(
        batch_input_prompts,
        batch_sequences,
        batch_old_logprobs,
        batch_ref_logprobs,
        batch_advantages,
        batch_returns,
        policy,
        critic,
        ppo_clip,
        ppo_epochs,
        ppo_minibatch_size,
        gradient_batch_size,
        vf_coef,
        ent_coef,
        normalize_advantage,
        kl_beta=kl_beta,
    )
    log_dict = log_policy_update(
        batch_raw_rewards,
        metrics,
        {
            "max_round_idx": max_round_idx,
            "total_scenes": total_scenes,
            "total_groups": len(all_groups),
            "goal_scores_per_group": goal_scores_per_group,
            "old_values": batch_old_values,
            "process_reward_sums_per_group": process_reward_sum_per_group,
            "process_step_counts_per_group": process_step_count_per_group,
            "total_output_round_counts_per_group": total_output_round_count_per_group,
            "small_output_round_counts_per_group": small_output_round_count_per_group,
            "batch_step": batch_step,
            "update_sample_count": len(batch_sequences),
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
                    "process_rewards": [],
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
    critic: object,
    gamma: float,
    gae_lambda: float,
    ppo_clip: float,
    ppo_epochs: int,
    ppo_minibatch_size: int,
    gradient_batch_size: Optional[int],
    vf_coef: float,
    ent_coef: float,
    normalize_advantage: bool,
    goal_reward_weight: float,
    process_reward_weight: float,
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
    clear_kv_cache_every_batch: bool = True,
    kl_beta: float = 0.0,
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
            policy,
            critic,
            gamma,
            gae_lambda,
            ppo_clip,
            ppo_epochs,
            ppo_minibatch_size,
            gradient_batch_size,
            vf_coef,
            ent_coef,
            normalize_advantage,
            goal_reward_weight,
            process_reward_weight,
            reward_model,
            reward_base_url,
            reward_api_key,
            wandb_run,
            epoch_idx,
            local_metrics_path=local_metrics_path,
            histories_path=histories_path,
            kl_beta=kl_beta,
        )

    if batch_idx in per_batch_rollouts:
        del per_batch_rollouts[batch_idx]
    impl = policy.module if hasattr(policy, "module") else policy
    if clear_kv_cache_every_batch and hasattr(impl, "clear_kv_cache"):
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
    critic,
    which_agent_large="agent0",
    max_rounds: int = 100,
    max_new_tokens: int = 128,
    checkpoint_every: int = 10,
    save_path: str = "qwen_policy.json",
    backgrounds_path: Optional[str] = None,
    val_ratio: float = 0.15,
    seed: Optional[int] = None,
    wandb_run: Optional[object] = None,
    scene_batch_size: int = 1,
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
    judge_model: Optional[str] = None,
    judge_base_url: Optional[str] = None,
    judge_api_key: Optional[str] = None,
    gamma: float = 0.95,
    gae_lambda: float = 0.95,
    ppo_clip: float = 0.2,
    ppo_epochs: int = 4,
    ppo_minibatch_size: int = 64,
    gradient_batch_size: Optional[int] = None,
    vf_coef: float = 0.5,
    ent_coef: float = 1e-3,
    normalize_advantage: bool = True,
    goal_reward_weight: float = 1.0,
    process_reward_weight: float = 0.05,
    clear_kv_cache_every_batch: bool = True,
    kl_beta: float = 0.0,
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
            critic=critic,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ppo_clip=ppo_clip,
            ppo_epochs=ppo_epochs,
            ppo_minibatch_size=ppo_minibatch_size,
            gradient_batch_size=gradient_batch_size,
            vf_coef=vf_coef,
            ent_coef=ent_coef,
            normalize_advantage=normalize_advantage,
            goal_reward_weight=goal_reward_weight,
            process_reward_weight=process_reward_weight,
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
            clear_kv_cache_every_batch=clear_kv_cache_every_batch,
            kl_beta=kl_beta,
        )
        if (
            val_scenes
            and int(test_every_batches) > 0
            and ((batch_idx + 1) % int(test_every_batches) == 0)
        ):
            # Local import avoids a module-level cycle.
            testing_loop_goal_single(
                policy=policy,
                val_scenes=val_scenes,
                which_agent_large=which_agent_large,
                max_rounds=max_rounds,
                max_new_tokens=max_new_tokens,
                scene_batch_size=scene_batch_size,
                num_groups=num_groups,
                epoch_idx=epoch_idx,
                rank=rank,
                world_size=world_size,
                wandb_run=wandb_run,
                local_metrics_path=local_metrics_path,
                test_histories_path=os.path.join(run_save_dir, f"test_histories_batch{batch_idx + 1}.jsonl"),
                test_max_scenes=test_max_scenes,
                partner_base_url=partner_base_url,
                partner_model=partner_model,
                partner_api_key=partner_api_key,
                judge_model=judge_model,
                judge_base_url=judge_base_url,
                judge_api_key=judge_api_key,
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
