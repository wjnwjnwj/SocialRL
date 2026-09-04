"""
Main evaluation loop for AgentSense.

Pipeline for each run item  (one run item = one policy character in one scene)
------------------------------------------------------------------------------
1. Build system prompts for all N characters.
2. Run multi-turn dialogue:
   - Speaker order: fixed round-robin over all N characters.
   - Max turns: N * 10  (avg 10 turns per character).
   - Each turn: if current speaker == policy_name → call policy model;
                else → call partner model with that character's system prompt.
   - Chat history is shared and updated after every turn.
   - Dialogue stops early if the current speaker's reply contains [LEAVE].
3. Judge scores the policy character's goal achievement (0-100).
4. Write one JSONL record per run item.
5. After all items, write a summary JSON.

Parallelism
-----------
Multiple run items can be evaluated concurrently (controlled by
--eval-workers / AGENTSENSE_EVAL_WORKERS).  Within each run item the
dialogue turns are sequential (each turn depends on the previous reply).

Resumption
----------
Already-scored run IDs are skipped so evaluation can resume after
interruption.
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

from agentsense_judge import score_goal_achievement, score_relation_delta
from agentsense_model import AgentModel
from agentsense_utils import (
    append_turn,
    build_agent_system_prompt,
    expand_all_scenes,
    has_stop_token,
    load_agentsense_data,
    load_done_run_ids,
    make_empty_history,
    make_run_id,
    strip_stop_token,
)

logger = logging.getLogger("agentsense_runner")


# ---------------------------------------------------------------------------
# Single run-item dialogue
# ---------------------------------------------------------------------------

def _run_dialogue(
    run_item: dict,
    policy_model: AgentModel,
    partner_model: AgentModel,
) -> tuple:
    """Execute one multi-turn dialogue for a single run item.

    Returns (history, system_prompts) where:
        history        – list of {"speaker", "content"} dicts
        system_prompts – dict mapping character name → system prompt string
    """
    characters = run_item["characters"]
    policy_name = run_item["policy_name"]
    speaker_order = run_item["speaker_order"]
    max_turns = run_item["max_turns"]

    # Pre-build system prompts for all characters (policy + all partners).
    system_prompts: Dict[str, str] = {}
    scenario = run_item["scenario"]
    for char in characters:
        system_prompts[char["name"]] = build_agent_system_prompt(
            agent_name=char["name"],
            characters=characters,
            scenario=scenario,
        )

    history = make_empty_history()
    n = len(speaker_order)
    partner_names = set(run_item["partner_names"])
    left_partners: set = set()

    from agentsense_utils import build_messages_for_agent

    for turn_idx in range(max_turns):
        current_name = speaker_order[turn_idx % n]

        # Skip partners who have already left.
        if current_name in left_partners:
            continue

        messages = build_messages_for_agent(
            system_prompt=system_prompts[current_name],
            history=history,
            focal_agent=current_name,
        )

        if current_name == policy_name:
            reply = policy_model.generate(messages)
        else:
            reply = partner_model.generate(messages)

        if reply is None:
            logger.warning(
                "run_id=%s turn=%d speaker=%s returned None; stopping dialogue.",
                make_run_id(run_item), turn_idx, current_name,
            )
            break

        clean_reply = strip_stop_token(reply)
        append_turn(history, current_name, clean_reply)

        if has_stop_token(reply):
            logger.debug(
                "run_id=%s: [LEAVE] from %s at turn %d.",
                make_run_id(run_item), current_name, turn_idx,
            )
            if current_name == policy_name:
                # Policy left → end immediately.
                break
            else:
                # Partner left → mark and check if all partners have left.
                left_partners.add(current_name)
                if left_partners >= partner_names:
                    logger.debug(
                        "run_id=%s: all partners left at turn %d; ending dialogue.",
                        make_run_id(run_item), turn_idx,
                    )
                    break

    return history, system_prompts


# ---------------------------------------------------------------------------
# Single run-item full evaluation
# ---------------------------------------------------------------------------

def _evaluate_run_item(
    run_item: dict,
    policy_model: AgentModel,
    partner_model: AgentModel,
    judge_model: str,
    judge_base_url: str,
    judge_api_key: str,
) -> dict:
    """Run dialogue + judge scoring for one run item."""
    run_id = make_run_id(run_item)
    t0 = time.time()

    history, system_prompts = _run_dialogue(run_item, policy_model, partner_model)

    judge_result = score_goal_achievement(
        run_item=run_item,
        history=history,
        model=judge_model,
        base_url=judge_base_url,
        api_key=judge_api_key,
    )

    relation_result = score_relation_delta(
        run_item=run_item,
        history=history,
        model=judge_model,
        base_url=judge_base_url,
        api_key=judge_api_key,
    )

    elapsed = round(time.time() - t0, 1)
    record = {
        "run_id": run_id,
        "sample_idx": run_item["sample_idx"],
        "policy_name": run_item["policy_name"],
        "num_characters": len(run_item["characters"]),
        "num_turns": len(history),
        "max_turns": run_item["max_turns"],
        "goal_score": judge_result["score"],
        "judge_analysis": judge_result["analysis"],
        "judge_key_evidence": judge_result["key_evidence"],
        "judge_raw_response": judge_result["raw_response"],
        "relation_delta_score": relation_result["score"],
        "relation_delta_reason": relation_result["reason"],
        "relation_delta_raw_response": relation_result["raw_response"],
        "character_system_prompts": system_prompts,
        "chat_history": history,
        "scored_at": int(time.time()),
        "elapsed_s": elapsed,
    }
    return record


# ---------------------------------------------------------------------------
# Summary computation
# ---------------------------------------------------------------------------

def _compute_summary(records: List[dict]) -> dict:
    """Aggregate scores across all run items."""
    goal_scores = [r["goal_score"] for r in records if isinstance(r.get("goal_score"), (int, float))]
    rel_scores = [r["relation_delta_score"] for r in records if isinstance(r.get("relation_delta_score"), (int, float))]

    by_n_chars_goal: Dict[int, list] = defaultdict(list)
    by_n_chars_rel: Dict[int, list] = defaultdict(list)
    by_sample: Dict[int, list] = defaultdict(list)

    for r in records:
        n = r.get("num_characters", 0)
        if isinstance(r.get("goal_score"), (int, float)):
            by_n_chars_goal[n].append(r["goal_score"])
        if isinstance(r.get("relation_delta_score"), (int, float)):
            by_n_chars_rel[n].append(r["relation_delta_score"])
        by_sample[r["sample_idx"]].append(r.get("goal_score", 0))

    def _stats(vals: list) -> dict:
        arr = np.array(vals, dtype=float)
        return {
            "count": int(arr.size),
            "mean": round(float(np.mean(arr)), 4) if arr.size else 0.0,
            "std": round(float(np.std(arr)), 4) if arr.size else 0.0,
            "min": round(float(np.min(arr)), 4) if arr.size else 0.0,
            "max": round(float(np.max(arr)), 4) if arr.size else 0.0,
        }

    n_chars_keys = sorted(set(list(by_n_chars_goal.keys()) + list(by_n_chars_rel.keys())))

    return {
        "overall_goal": _stats(goal_scores),
        "overall_relation_delta": _stats(rel_scores),
        # keep "overall" as an alias for goal scores for backward compatibility
        "overall": _stats(goal_scores),
        "by_num_characters": {
            str(n): {
                "goal": _stats(by_n_chars_goal.get(n, [])),
                "relation_delta": _stats(by_n_chars_rel.get(n, [])),
            }
            for n in n_chars_keys
        },
        "num_scenes": len(by_sample),
        "num_run_items": len(records),
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_agentsense_evaluation(
    data_path: str,
    output_dir: str,
    policy_model: AgentModel,
    partner_model: AgentModel,
    judge_model: str,
    judge_base_url: str,
    judge_api_key: str,
    max_scenes: int = 0,
    eval_workers: int = 4,
    run_timestamp: Optional[str] = None,
) -> dict:
    """Run the full AgentSense evaluation pipeline.

    Parameters
    ----------
    data_path : str
        Path to final_data.jsonl (or data_with_role.jsonl).
    output_dir : str
        Directory where results JSONL and summary JSON are written.
    policy_model / partner_model : AgentModel
        Initialized model clients.
    judge_model / judge_base_url / judge_api_key : str
        Judge model connection config.
    max_scenes : int
        Limit to first N scenes before expansion (0 = all).
    eval_workers : int
        Number of run items evaluated concurrently.
    run_timestamp : str, optional
        Timestamp string for output filenames (auto-generated if None).

    Returns
    -------
    dict  – summary statistics.
    """
    os.makedirs(output_dir, exist_ok=True)
    ts = run_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(output_dir, f"agentsense_results_{ts}.jsonl")
    summary_path = os.path.join(output_dir, f"agentsense_summary_{ts}.json")

    # Load and expand
    scenes = load_agentsense_data(data_path, max_items=max_scenes)
    logger.info("Loaded %d scenes from %s", len(scenes), data_path)

    run_items = expand_all_scenes(scenes)
    logger.info(
        "Expanded to %d run items (scenes=%d, avg characters=%.1f)",
        len(run_items), len(scenes),
        len(run_items) / len(scenes) if scenes else 0,
    )

    # Resume: skip already-scored run IDs
    done_ids = load_done_run_ids(results_path)
    if done_ids:
        before = len(run_items)
        run_items = [ri for ri in run_items if make_run_id(ri) not in done_ids]
        logger.info("Resuming: skipped %d already-scored items, %d remaining.",
                    before - len(run_items), len(run_items))

    all_records: List[dict] = []

    def _task(ri: dict) -> dict:
        return _evaluate_run_item(
            ri, policy_model, partner_model,
            judge_model, judge_base_url, judge_api_key,
        )

    workers = min(eval_workers, len(run_items)) if run_items else 1
    logger.info("Starting evaluation: %d run items, %d workers.", len(run_items), workers)

    with open(results_path, "a", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_ri = {executor.submit(_task, ri): ri for ri in run_items}
            pbar = tqdm(
                total=len(run_items),
                desc="Evaluating",
                unit="item",
                dynamic_ncols=True,
            )
            for future in as_completed(future_to_ri):
                ri = future_to_ri[future]
                try:
                    record = future.result()
                except Exception as exc:
                    logger.error("run_id=%s failed: %s", make_run_id(ri), exc)
                    record = {
                        "run_id": make_run_id(ri),
                        "sample_idx": ri["sample_idx"],
                        "policy_name": ri["policy_name"],
                        "num_characters": len(ri["characters"]),
                        "goal_score": None,
                        "error": str(exc),
                        "scored_at": int(time.time()),
                    }

                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                all_records.append(record)

                goal_str = (
                    f"{record['goal_score']:.1f}"
                    if isinstance(record.get("goal_score"), (int, float))
                    else "ERR"
                )
                rel_str = (
                    f"{record['relation_delta_score']:+.2f}"
                    if isinstance(record.get("relation_delta_score"), (int, float))
                    else "ERR"
                )
                pbar.set_postfix(
                    run_id=record["run_id"],
                    turns=record.get("num_turns", "?"),
                    goal=goal_str,
                    rel=rel_str,
                )
                pbar.update(1)
            pbar.close()

    # If resuming, reload all records for a complete summary
    if done_ids:
        reload: List[dict] = []
        with open(results_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    reload.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        all_records = reload

    valid_records = [r for r in all_records if isinstance(r.get("goal_score"), (int, float))]
    summary = _compute_summary(valid_records)
    summary["run_timestamp"] = ts
    summary["results_file"] = results_path

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("Evaluation complete. Results: %s", results_path)
    logger.info("Summary: %s", summary_path)
    _log_summary(summary)

    return summary


def _log_summary(summary: dict) -> None:
    goal = summary.get("overall_goal") or {}
    rel = summary.get("overall_relation_delta") or {}
    logger.info(
        "===== AgentSense Results (n=%d scenes / %d run items) =====\n"
        "  Goal score:     mean=%.2f  std=%.2f  min=%.2f  max=%.2f\n"
        "  Relation delta: mean=%.2f  std=%.2f  min=%.2f  max=%.2f",
        summary.get("num_scenes", 0),
        summary.get("num_run_items", 0),
        goal.get("mean", 0), goal.get("std", 0),
        goal.get("min", 0),  goal.get("max", 0),
        rel.get("mean", 0),  rel.get("std", 0),
        rel.get("min", 0),   rel.get("max", 0),
    )
    by_n = summary.get("by_num_characters") or {}
    for n_str, stats in sorted(by_n.items(), key=lambda x: int(x[0])):
        g = stats.get("goal") or {}
        r = stats.get("relation_delta") or {}
        logger.info(
            "  %s-char scenes: goal_mean=%.2f  rel_delta_mean=%.2f  n=%d",
            n_str, g.get("mean", 0), r.get("mean", 0), g.get("count", 0),
        )
