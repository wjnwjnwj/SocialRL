#!/usr/bin/env python3
from __future__ import annotations

import time
from types import SimpleNamespace
import os
from typing import List, Optional

import torch.distributed as dist

from rl_dataset import build_batch_initial_states
from score_relation_delta import collect_test_relation_deltas
from utils import (
    _append_local_metrics,
    _ddp_aggregate_stats,
    _ddp_reduce_scalar,
    _extract_scene_id,
    clone_history,
    collect_test_rewards,
    logger,
    run_dialogue_round,
)


def _rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def testing_loop_goal_single(
    policy,
    val_scenes: Optional[List[dict]] = None,
    which_agent_large: str = "agent0",
    max_rounds: int = 10,
    max_new_tokens: int = 320,
    scene_batch_size: int = 1,
    num_groups: Optional[int] = None,
    epoch_idx: Optional[int] = None,
    rank: int = 0,
    world_size: int = 1,
    wandb_run: Optional[object] = None,
    local_metrics_path: Optional[str] = None,
    test_histories_path: Optional[str] = None,
    test_max_scenes: int = 0,
    partner_base_url: str = "http://127.0.0.1:8000/v1",
    partner_model: str = "Qwen3.5-35B-A3B",
    partner_api_key: str = "EMPTY", 
    judge_model: str = "Qwen3.5-35B-A3B",
    judge_base_url: str = "http://127.0.0.1:8000/v1",
    judge_api_key: str = "EMPTY",
) -> Optional[dict]:
    if not val_scenes:
        if rank == 0:
            logger.info("[rank %d] No test/validation scenes found, skip testing.", rank)
        return None

    scenes = list(val_scenes)
    if test_max_scenes and test_max_scenes > 0:
        scenes = scenes[: int(test_max_scenes)]
    if world_size > 1:
        scenes = scenes[rank::world_size]
    if not scenes:
        return None

    groups_per_scene = (
        int(num_groups) if num_groups is not None and int(num_groups) > 0 else 1
    )
    batch_size = max(1, int(scene_batch_size))
    all_groups: List[tuple] = []
    total_batches = (len(scenes) + batch_size - 1) // batch_size

    for batch_idx, start in enumerate(range(0, len(scenes), batch_size)):
        batch_scenes = scenes[start : start + batch_size]
        B = len(batch_scenes)
        agent0_list, agent1_list, history_messages_all = build_batch_initial_states(
            batch_scenes
        )
        small_agent_list = [
            agent1_list[i] if which_agent_large == "agent0" else agent0_list[i]
            for i in range(B)
        ]
        scene_groups = []
        for b in range(B):
            groups = []
            for _ in range(groups_per_scene):
                history_clone = {
                    agent0_list[b]: clone_history(history_messages_all[b][agent0_list[b]]),
                    agent1_list[b]: clone_history(history_messages_all[b][agent1_list[b]]),
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

        group_active_mask = [[True] * groups_per_scene for _ in range(B)]
        rollout = {"scene_groups": scene_groups}
        for round_idx in range(max_rounds):
            active_mask = [any(group_active_mask[i]) for i in range(B)]
            active_group_counts = [
                sum(1 for flag in masks if flag) for masks in group_active_mask
            ]
            scene_active_groups = ",".join(str(c) for c in active_group_counts)
            finished_scene_indices = [
                i for i, is_active in enumerate(active_mask) if not is_active
            ]
            logger.info(
                "[rank %d] Test batch %d/%d Round %d/%d (batch_size=%d, groups=%d, active_groups=%d, scene_active_groups=%s, finished_scenes=%s)",
                _rank(),
                batch_idx + 1,
                total_batches,
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
            if not any(active_mask):
                logger.info(
                    "[rank %d] Test batch %d/%d ended early at round %d: no active scenes left.",
                    _rank(),
                    batch_idx + 1,
                    total_batches,
                    round_idx + 1,
                )
                break
            is_large = (
                (which_agent_large == "agent0" and round_idx % 2 == 0)
                or (which_agent_large == "agent1" and round_idx % 2 == 1)
            )
            batch_state = SimpleNamespace(
                agent0_list=agent0_list,
                agent1_list=agent1_list,
                is_large=is_large,
                round_idx=round_idx,
                prompts=[None] * B,
            )
            _reply_texts, has_none_reply, stop_flags = run_dialogue_round(
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
                logger.warning(
                    "[rank %d] Test batch %d/%d round %d received None reply; stop this batch.",
                    _rank(),
                    batch_idx + 1,
                    total_batches,
                    round_idx + 1,
                )
                break
            for i in range(B):
                for g in range(groups_per_scene):
                    if group_active_mask[i][g] and stop_flags[i][g]:
                        group_active_mask[i][g] = False

        for b in range(B):
            for g in range(groups_per_scene):
                all_groups.append(
                    (
                        0,
                        b,
                        g,
                        scene_groups[b][g],
                        batch_scenes[b],
                        small_agent_list[b],
                    )
                )

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    ranker_model = judge_model
    ranker_base_url = judge_base_url
    ranker_api_key = judge_api_key
    goal_scores_per_group, _goal_reasons = collect_test_rewards(
        all_groups, ranker_model, ranker_base_url, ranker_api_key
    )
    relation_scores_per_group, relation_reasons_per_group = collect_test_relation_deltas(
        all_groups, ranker_model, ranker_base_url, ranker_api_key
    )
    try:
        from utils import append_histories_with_rewards
        if test_histories_path:
            hist_path = test_histories_path
        else:
            epoch_suffix = f"_epoch{int(epoch_idx) + 1}" if epoch_idx is not None else ""
            default_dir = (
                os.path.dirname(os.path.abspath(local_metrics_path))
                if local_metrics_path
                else os.getcwd()
            )
            hist_path = os.path.join(default_dir, f"test_histories{epoch_suffix}.jsonl")
        append_histories_with_rewards(
            hist_path,
            all_groups,
            goal_scores_per_group,
            _goal_reasons,
            relation_scores_per_group,
            relation_reasons_per_group,
        )
    except Exception:
        pass
    mean_v, std_v, min_v, max_v, cnt_v = _ddp_aggregate_stats(goal_scores_per_group)
    rel_mean, rel_std, rel_min, rel_max, rel_cnt = _ddp_aggregate_stats(
        relation_scores_per_group
    )
    scene_count_global = int(_ddp_reduce_scalar(float(len(scenes)), dist.ReduceOp.SUM))
    group_count_global = int(_ddp_reduce_scalar(float(len(all_groups)), dist.ReduceOp.SUM))

    result = {
        "test_epoch": int(epoch_idx) + 1 if epoch_idx is not None else -1,
        "test_scene_count": scene_count_global,
        "test_group_count": group_count_global,
        "test_goal_reward_mean": mean_v,
        "test_goal_reward_std": std_v,
        "test_goal_reward_min": min_v,
        "test_goal_reward_max": max_v,
        "test_goal_reward_count": int(cnt_v),
        "test_relation_delta_mean": rel_mean,
        "test_relation_delta_std": rel_std,
        "test_relation_delta_min": rel_min,
        "test_relation_delta_max": rel_max,
        "test_relation_delta_count": int(rel_cnt),
    }
    if wandb_run is not None:
        wandb_run.log(result)
    local_payload = dict(result)
    local_payload["epoch_idx"] = int(epoch_idx) if epoch_idx is not None else -1
    local_payload["batch_step"] = -1
    local_payload["logged_at"] = int(time.time())
    _append_local_metrics(local_metrics_path, local_payload)
    
    if rank == 0:
        logger.info(
            "[rank %d] Test epoch=%d scenes=%d groups=%d goal_mean=%.4f goal_std=%.4f "
            "relation_delta_mean=%.4f relation_delta_std=%.4f",
            _rank(),
            int(epoch_idx) + 1 if epoch_idx is not None else -1,
            scene_count_global,
            group_count_global,
            mean_v,
            std_v,
            rel_mean,
            rel_std,
        )
    return result
