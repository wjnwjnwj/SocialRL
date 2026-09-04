#!/usr/bin/env python3
"""Repeat AgentSense judging using the original prompt and judge functions.

Each result JSONL record already represents one policy character in one scene.
The policy character is the last speaker in the original round-robin order;
therefore this script rejudges every record, covering agent0 through agent4
scenes without changing the dialogue history.

The original ``score_goal_achievement`` and ``score_relation_delta`` functions
are called directly. The corresponding run item is reconstructed from the
original AgentSense source data by ``sample_idx`` and ``policy_name`` so the
goal list and all prompt formatting remain exactly the same as the original
evaluation.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(SCRIPT_DIR))

from agentsense_judge import score_goal_achievement, score_relation_delta  # noqa: E402
from agentsense_utils import expand_all_scenes, load_agentsense_data, make_run_id  # noqa: E402

RESULT_KEY = "repeated_evaluations"
DEFAULT_RESULT_DIR = SCRIPT_DIR.parent / "result"
DEFAULT_DATA = SCRIPT_DIR.parent / "SENSE" / "data" / "final_data.jsonl"


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def write_jsonl_atomic(path: Path, rows: List[dict]) -> None:
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def load_run_items(path: Path) -> Dict[Tuple[int, str], dict]:
    """Recreate the exact run items used by the original AgentSense runner."""
    scenes = load_agentsense_data(str(path))
    run_items = expand_all_scenes(scenes)
    result: Dict[Tuple[int, str], dict] = {}
    for item in run_items:
        key = (int(item["sample_idx"]), str(item["policy_name"]))
        if key in result:
            raise ValueError(f"Duplicate AgentSense run item: {key}")
        # The original runner puts the policy character last in each round.
        if item["speaker_order"][-1] != item["policy_name"]:
            raise ValueError(f"Policy is not the last speaker for run item {key}")
        result[key] = item
    return result


def validate_records(rows: List[dict], run_items: Dict[Tuple[int, str], dict], path: Path) -> None:
    missing: List[str] = []
    for index, row in enumerate(rows, 1):
        key = (int(row.get("sample_idx", -1)), str(row.get("policy_name") or ""))
        if key not in run_items:
            missing.append(f"{path.name}:{index} {key}")
            if len(missing) >= 10:
                break
    if missing:
        raise ValueError("Could not match result records to source run items: " + "; ".join(missing))


def get_existing_repeat(row: dict, repeat_index: int) -> dict | None:
    for result in (row.get(RESULT_KEY) or []):
        if isinstance(result, dict) and result.get("repeat_index") == repeat_index:
            return result
    return None


def is_complete(result: dict | None, skip_relation: bool) -> bool:
    if not isinstance(result, dict) or result.get("goal_score") not in (0, 1):
        return False
    if skip_relation:
        return True
    return isinstance(result.get("relation_delta_score"), (int, float))


def update_summary(row: dict) -> None:
    results = [r for r in row.get(RESULT_KEY, []) if isinstance(r, dict)]
    goals = [float(r["goal_score"]) for r in results if r.get("goal_score") is not None]
    relations = [
        float(r["relation_delta_score"])
        for r in results
        if r.get("relation_delta_score") is not None
    ]
    row["repeated_evaluation_summary"] = {
        "n": len(goals),
        "goal_score_mean": statistics.mean(goals) if goals else None,
        "goal_score_variance": statistics.variance(goals) if len(goals) > 1 else 0.0,
        "goal_score_std": statistics.stdev(goals) if len(goals) > 1 else 0.0,
        "relation_delta_mean": statistics.mean(relations) if relations else None,
        "relation_delta_variance": statistics.variance(relations) if len(relations) > 1 else 0.0,
        "relation_delta_std": statistics.stdev(relations) if len(relations) > 1 else 0.0,
    }


def judge_one(
    row: dict,
    run_item: dict,
    repeat_index: int,
    model: str,
    base_url: str,
    api_key: str,
    skip_relation: bool,
) -> dict:
    history = row.get("chat_history") or []
    if not history:
        raise ValueError("Empty chat_history")

    # These are the original AgentSense functions. They construct the exact
    # GOAL_JUDGE_PROMPT / RELATION_DELTA_PROMPT used by the first evaluation.
    goal = score_goal_achievement(run_item, history, model, base_url, api_key)
    result = {
        "repeat_index": repeat_index,
        "policy_name": run_item["policy_name"],
        "goal_score": int(goal.get("score", 0)),
        "judge_analysis": goal.get("analysis", ""),
        "judge_key_evidence": goal.get("key_evidence", ""),
        "judge_raw_response": goal.get("raw_response", ""),
        "judge_model": model,
        "judge_base_url": base_url,
    }
    if not skip_relation:
        relation = score_relation_delta(run_item, history, model, base_url, api_key)
        result.update(
            {
                "relation_delta_score": float(relation.get("score", 0.0)),
                "relation_delta_reason": relation.get("reason", ""),
                "relation_delta_raw_response": relation.get("raw_response", ""),
            }
        )
    return result


def append_repeat(row: dict, result: dict) -> None:
    results = row.setdefault(RESULT_KEY, [])
    for index, old in enumerate(results):
        if isinstance(old, dict) and old.get("repeat_index") == result["repeat_index"]:
            results[index] = result
            break
    else:
        results.append(result)
    results.sort(key=lambda item: int(item.get("repeat_index", 0)))
    update_summary(row)


def parse_endpoints(args: argparse.Namespace) -> List[str]:
    values = list(args.base_url or [])
    values.extend(value.strip() for value in args.base_urls.split(",") if value.strip())
    if not values:
        values = [os.environ.get("JUDGE_BASE_URL", "http://127.0.0.1:8000/v1")]
    return list(dict.fromkeys(values))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run five repeated AgentSense judgements using the original prompts."
    )
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--files", nargs="*", help="Result JSONL names; default is all *_vs_*.jsonl.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--start-repeat", type=int, default=1)
    parser.add_argument("--base-url", action="append", default=None)
    parser.add_argument("--base-urls", default=os.environ.get("JUDGE_BASE_URLS", ""))
    parser.add_argument("--model", default=os.environ.get("JUDGE_MODEL", "Qwen3.5-35B-A3B"))
    parser.add_argument("--api-key", default=os.environ.get("JUDGE_API_KEY", "EMPTY"))
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-relation", action="store_true")
    args = parser.parse_args()

    if args.repeats < 1 or args.start_repeat < 1 or args.max_workers < 1:
        parser.error("--repeats, --start-repeat, and --max-workers must be positive")
    result_dir = args.result_dir.expanduser().resolve()
    data_path = args.data_path.expanduser().resolve()
    endpoints = parse_endpoints(args)
    if not data_path.is_file():
        raise SystemExit(f"AgentSense source data not found: {data_path}")
    run_items = load_run_items(data_path)

    paths = sorted(result_dir.glob("*_vs_*.jsonl"))
    if args.files:
        wanted = {Path(value).name for value in args.files}
        paths = [path for path in paths if path.name in wanted]
    if not paths:
        raise SystemExit(f"No AgentSense result JSONL files found in {result_dir}")

    print(
        f"Judge={args.model}; endpoints={','.join(endpoints)}; "
        f"workers/endpoint={args.max_workers}; source={data_path}"
    )
    for path in paths:
        rows = load_jsonl(path)
        validate_records(rows, run_items, path)
        for repeat_index in range(args.start_repeat, args.start_repeat + args.repeats):
            pending = [
                index
                for index, row in enumerate(rows)
                if args.force
                or not is_complete(get_existing_repeat(row, repeat_index), args.skip_relation)
            ]
            if not pending:
                continue
            total_workers = min(len(pending), args.max_workers * len(endpoints))
            with ThreadPoolExecutor(max_workers=total_workers) as executor:
                futures = {
                    executor.submit(
                        judge_one,
                        rows[index],
                        run_items[(int(rows[index]["sample_idx"]), str(rows[index]["policy_name"]))],
                        repeat_index,
                        args.model,
                        endpoints[position % len(endpoints)],
                        args.api_key,
                        args.skip_relation,
                    ): index
                    for position, index in enumerate(pending)
                }
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"{path.name} r{repeat_index}"):
                    index = futures[future]
                    try:
                        append_repeat(rows[index], future.result())
                    except Exception as exc:
                        raise RuntimeError(f"Failed {path.name} row {index + 1}: {exc}") from exc
            write_jsonl_atomic(path, rows)
        print(f"Prepared {path.name}: records={len(rows)}")


if __name__ == "__main__":
    main()
