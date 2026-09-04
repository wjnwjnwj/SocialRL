#!/usr/bin/env python3
"""
Score relationship favorability change (Δrelation) for test dialogue histories.

Judge perspective: opponent (agent0) toward strategy model (agent1).
Output range: [-1, 1]. Aggregates are written into training_metrics_summary jsonl.
Per-dialogue scores are written into each test_histories*.jsonl record.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

from rl_reward import render_dialogue_for_lang_reward
from utils import _resolve_max_workers, call_api_model
from prompt import RELATION_DELTA_PROMPT

def _extract_block(text: str, start_label: str, end_labels: Tuple[str, ...]) -> str:
    start_idx = text.find(start_label)
    if start_idx < 0:
        return ""
    start_idx += len(start_label)
    end_idx = len(text)
    for label in end_labels:
        pos = text.find(label, start_idx)
        if pos >= 0:
            end_idx = min(end_idx, pos)
    return text[start_idx:end_idx].strip()


def parse_agent_system_context(system_content: str, agent_name: str) -> Dict[str, str]:
    scenario = _extract_block(
        system_content,
        "Scenario:",
        (
            f"{agent_name}'s social goal:",
            "\nYour Character Profile:",
            "\nNote:",
        ),
    )
    opponent_profile = _extract_block(
        system_content,
        "Your Character Profile:",
        ("\nYour Partner's Profile:", "\nNote:"),
    )
    strategy_profile = _extract_block(
        system_content,
        "Your Partner's Profile:",
        ("\nNote:",),
    )
    return {
        "scenario": scenario,
        "opponent_profile": opponent_profile,
        "strategy_profile": strategy_profile,
    }


def build_dialogue_text(record: dict) -> Tuple[str, str, str, str, str, str]:
    opponent = str(record.get("agent0") or "").strip()
    strategy = str(record.get("agent1") or "").strip()
    history = record.get("history") or {}
    opponent_msgs = history.get(opponent) or []
    if not opponent_msgs:
        raise ValueError(f"Missing history for opponent {opponent}")

    system_content = ""
    for msg in opponent_msgs:
        if (msg.get("role") or "").strip().lower() == "system":
            system_content = str(msg.get("content") or "")
            break

    ctx = parse_agent_system_context(system_content, opponent)
    dialogue = render_dialogue_for_lang_reward(
        opponent_msgs, other_speaker_name=strategy or "Partner"
    )
    return opponent, strategy, ctx["scenario"], ctx["opponent_profile"], ctx["strategy_profile"], dialogue


def build_record_from_group(item: tuple) -> dict:
    _batch_idx, _b, _g, group, _scene, _small_agent_name = item
    histories_all = group.get("histories_all") or {}
    agent_names = list(histories_all.keys())
    return {
        "agent0": agent_names[0] if len(agent_names) > 0 else "",
        "agent1": agent_names[1] if len(agent_names) > 1 else "",
        "history": histories_all,
    }


def parse_relation_response(resp: str) -> Tuple[float, str]:
    if not resp:
        return 0.0, ""

    cleaned = resp.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            reason = str(parsed.get("reason") or parsed.get("analysis") or "").strip()
            raw = parsed.get("relation_delta")
            if raw is None:
                raw = parsed.get("delta_relation")
            if raw is not None:
                score = float(raw)
                return max(-1.0, min(1.0, score)), reason
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    numbers = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
    for num in numbers:
        try:
            val = float(num)
            if -1.0 <= val <= 1.0:
                return val, cleaned[:300]
        except ValueError:
            continue
    return 0.0, cleaned[:300]


def evaluate_relation_delta(
    record: dict,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float = 0.0,
) -> Tuple[float, str]:
    opponent, strategy, scenario, opp_prof, strat_prof, dialogue = build_dialogue_text(record)
    prompt = RELATION_DELTA_PROMPT.format(
        opponent_name=opponent or "Opponent",
        strategy_name=strategy or "Strategy",
        scenario=scenario or "(unknown scenario)",
        opponent_profile=opp_prof or "(unknown)",
        strategy_profile=strat_prof or "(unknown)",
        history=dialogue or "(empty dialogue)",
    )
    resp = call_api_model(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=temperature,
        base_url=base_url,
        api_key=api_key,
    )
    return parse_relation_response(resp or "")


def aggregate_relation_delta_scores(scores: List[float]) -> Dict[str, float]:
    if not scores:
        return {
            "test_relation_delta_mean": 0.0,
            "test_relation_delta_std": 0.0,
            "test_relation_delta_min": 0.0,
            "test_relation_delta_max": 0.0,
            "test_relation_delta_count": 0,
        }
    return {
        "test_relation_delta_mean": float(statistics.mean(scores)),
        "test_relation_delta_std": float(statistics.pstdev(scores)) if len(scores) > 1 else 0.0,
        "test_relation_delta_min": float(min(scores)),
        "test_relation_delta_max": float(max(scores)),
        "test_relation_delta_count": int(len(scores)),
    }


def _score_relation_item_test(
    item: tuple,
    llm_model: str,
    llm_base_url: Optional[str],
    llm_api_key: Optional[str],
    temperature: float,
) -> tuple:
    group_idx, record = item
    try:
        score, reason = evaluate_relation_delta(
            record,
            model=llm_model,
            base_url=llm_base_url or "",
            api_key=llm_api_key or "EMPTY",
            temperature=temperature,
        )
    except Exception:
        score, reason = 0.0, ""
    return (group_idx, float(score), str(reason or ""))


def collect_test_relation_deltas(
    all_groups: List[tuple],
    llm_model: str,
    llm_base_url: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    temperature: float = 0.0,
) -> tuple:
    relation_scores: List[float] = [0.0 for _ in range(len(all_groups))]
    relation_reasons: List[str] = ["" for _ in range(len(all_groups))]

    if not all_groups:
        return relation_scores, relation_reasons

    if os.environ.get("SCORE_RELATION_DELTA", "1").strip().lower() in {"0", "false", "no"}:
        return relation_scores, relation_reasons

    tasks = []
    for group_idx, item in enumerate(all_groups):
        tasks.append((group_idx, build_record_from_group(item)))

    score_fn = partial(
        _score_relation_item_test,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        temperature=temperature,
    )
    with ThreadPoolExecutor(
        max_workers=_resolve_max_workers(
            env_name="RL_TEST_RELATION_MAX_WORKERS",
            task_count=len(tasks),
            default_workers=8,
        )
    ) as ex:
        results = list(ex.map(score_fn, tasks))

    for group_idx, score, reason in results:
        relation_scores[group_idx] = float(score)
        relation_reasons[group_idx] = str(reason or "")

    return relation_scores, relation_reasons


def discover_test_history_files(ckpt_dir: Path) -> List[Path]:
    ordered: List[Path] = []
    pretrain = ckpt_dir / "test_histories.jsonl"
    if pretrain.exists():
        ordered.append(pretrain)
    for epoch_dir in sorted(ckpt_dir.glob("checkpoints_epoch*")):
        for batch_file in sorted(epoch_dir.glob("test_histories_batch*.jsonl")):
            ordered.append(batch_file)
        epoch_hist = epoch_dir / "test_histories.jsonl"
        if epoch_hist.exists() and epoch_hist not in ordered:
            ordered.append(epoch_hist)
    for rank_file in sorted(ckpt_dir.glob("test_histories_rank*.jsonl")):
        if rank_file not in ordered:
            ordered.append(rank_file)
    return ordered


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def score_record_task(
    idx: int,
    record: dict,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float,
    force: bool,
) -> Tuple[int, float, str]:
    reward = record.setdefault("reward", {})
    if not force and reward.get("relation_delta") is not None:
        return idx, float(reward["relation_delta"]), str(reward.get("relation_reason") or "")

    score, reason = evaluate_relation_delta(
        record, model=model, base_url=base_url, api_key=api_key, temperature=temperature
    )
    reward["relation_delta"] = score
    reward["relation_reason"] = reason
    return idx, score, reason


def score_history_file(
    hist_path: Path,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float,
    max_workers: int,
    force: bool,
    limit: int,
) -> List[float]:
    rows = load_jsonl(hist_path)
    if limit > 0:
        rows = rows[:limit]

    pending = [
        (i, row)
        for i, row in enumerate(rows)
        if force or row.get("reward", {}).get("relation_delta") is None
    ]
    if pending:
        workers = max(1, min(max_workers, len(pending)))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(
                    score_record_task,
                    i,
                    row,
                    model,
                    base_url,
                    api_key,
                    temperature,
                    force,
                ): i
                for i, row in pending
            }
            done = 0
            for fut in as_completed(futures):
                idx, score, reason = fut.result()
                reward = rows[idx].setdefault("reward", {})
                reward["relation_delta"] = score
                reward["relation_reason"] = reason
                done += 1
                if done % 50 == 0 or done == len(pending):
                    print(f"  [{hist_path.name}] scored {done}/{len(pending)}")
        write_jsonl(hist_path, rows)
    else:
        print(f"  [{hist_path.name}] all records already scored, skip API calls")

    scores = []
    for row in rows:
        val = row.get("reward", {}).get("relation_delta")
        if val is not None:
            scores.append(float(val))
    return scores


def update_metrics_summary(
    summary_path: Path,
    test_files: List[Path],
    file_to_scores: Dict[str, List[float]],
) -> None:
    rows: List[dict] = []
    test_idx = 0
    with summary_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "test_epoch" in row and test_idx < len(test_files):
                hist_path = test_files[test_idx]
                scores = file_to_scores.get(str(hist_path), [])
                row.update(aggregate_relation_delta_scores(scores))
                row["test_histories_path"] = str(hist_path.relative_to(summary_path.parent))
                test_idx += 1
            rows.append(row)

    write_jsonl(summary_path, rows)


def check_vllm_service(base_url: str, api_key: str, model: str, timeout: float = 5.0) -> None:
    models_url = f"{base_url.rstrip('/')}/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        resp = requests.get(models_url, headers=headers, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise SystemExit(
            f"vLLM is not reachable at {base_url}\n"
            f"  ({exc})\n"
            "Start the server first, e.g.:\n"
            "  bash scripts-ppo-third-dynamic-reward/run_vllm_service.sh\n"
            "Then wait until /v1/models responds before running this script."
        ) from exc

    data = resp.json()
    model_ids = [m.get("id", "") for m in data.get("data", []) if isinstance(m, dict)]
    if model_ids and model not in model_ids:
        print(
            f"Warning: --model {model!r} not in served models {model_ids}. "
            "Use --model with a name from /v1/models."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Score Δrelation for test histories via vLLM judge.")
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "checkpoints",
        help="Checkpoint run directory containing test_histories*.jsonl",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000/v1",
        help="OpenAI-compatible vLLM base URL (see run_vllm_service.sh)",
    )
    parser.add_argument(
        "--model",
        default="Qwen3.5-35B-A3B",
        help="Served model name on vLLM",
    )
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="Max records per file (0=all)")
    parser.add_argument("--force", action="store_true", help="Re-score even if relation_delta exists")
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Optional subset of test history files (paths or names)",
    )
    parser.add_argument(
        "--update-metrics",
        action="store_true",
        help="Also update training_metrics_summary*.jsonl in ckpt-dir",
    )
    args = parser.parse_args()

    ckpt_dir = args.ckpt_dir.expanduser().resolve()
    all_files = discover_test_history_files(ckpt_dir)
    if args.files:
        wanted = {Path(p).name for p in args.files}
        test_files = [p for p in all_files if p.name in wanted or str(p) in args.files]
    else:
        test_files = all_files

    if not test_files:
        raise SystemExit(f"No test_histories*.jsonl found under {ckpt_dir}")

    print(f"Checkpoint dir: {ckpt_dir}")
    print(f"Test history files: {len(test_files)}")
    print(f"vLLM: {args.base_url} model={args.model}")
    check_vllm_service(args.base_url, args.api_key, args.model)

    file_to_scores: Dict[str, List[float]] = {}
    t0 = time.time()
    for i, hist_path in enumerate(test_files, 1):
        print(f"[{i}/{len(test_files)}] Scoring {hist_path}")
        scores = score_history_file(
            hist_path,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            temperature=args.temperature,
            max_workers=args.max_workers,
            force=args.force,
            limit=args.limit,
        )
        file_to_scores[str(hist_path)] = scores
        agg = aggregate_relation_delta_scores(scores)
        print(
            f"  mean={agg['test_relation_delta_mean']:.4f} "
            f"std={agg['test_relation_delta_std']:.4f} "
            f"count={agg['test_relation_delta_count']}"
        )

    if args.update_metrics:
        summary_files = sorted(ckpt_dir.glob("training_metrics_summary*.jsonl"))
        for summary_path in summary_files:
            print(f"Updating metrics summary: {summary_path}")
            update_metrics_summary(summary_path, test_files, file_to_scores)

    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
