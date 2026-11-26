from __future__ import annotations

import json
import math
import re
from typing import Dict, List, Optional

from dynamic_reward_config import RUBRIC_DIMENSIONS
from prompt import (
    goal_achieve_prompt,
    goal_achieve_prompt_continuous,
    rubric_dynamic_generation_prompt,
    rubric_dynamic_scoring_prompt,
)

from utils import call_api_model


def standardize_scores(scores: List[float]) -> List[float]:
    """
    Standardize scores to have mean 0 and standard deviation 1.
    If all scores are the same, return zeros.
    """
    if not scores:
        return []

    # Convert to numpy for easy computation
    import numpy as np

    scores_array = np.array(scores, dtype=np.float64)

    # Handle edge cases
    if len(scores_array) == 1 or np.std(scores_array) == 0:
        return [0.0] * len(scores_array)

    # Standardize: (x - mean) / std
    standardized = (scores_array - np.mean(scores_array)) / (
        np.std(scores_array) + 0.01
    )
    return standardized.tolist()


def render_dialogue_for_lang_reward(
    history_messages: List[Dict[str, str]],
    other_speaker_name: Optional[str] = None,
) -> str:
    """
    Render dialogue history into plain speaker turns like:
    User: ...
    Benjamin: ...
    """
    if not history_messages:
        return ""

    system_text = ""
    for msg in history_messages:
        if msg.get("role") == "system":
            system_text = str(msg.get("content") or "")
            if system_text:
                break

    assistant_name = "Assistant"
    m = re.search(r"embodying\s+\*\*(.*?)\*\*", system_text, flags=re.IGNORECASE)
    if m and m.group(1).strip():
        assistant_name = m.group(1).strip()

    counterpart_name = (
        str(other_speaker_name).strip() if other_speaker_name else "User"
    ) or "User"

    lines: List[str] = []
    for msg in history_messages:
        role = (msg.get("role") or "").strip().lower()
        content = str(msg.get("content") or "").strip()
        if not content or role == "system":
            continue
        if role == "user":
            speaker = counterpart_name
        elif role == "assistant":
            speaker = assistant_name
        else:
            # Treat unexpected roles as the opposite side by default.
            speaker = counterpart_name
        lines.append(f"{speaker}: {content}")
    return "\n\n".join(lines)


def evaluate_social_goal_achievement_with_reason_binary(
    scene: dict,
    history_messages: List[Dict[str, str]],
    agent_name: str,
    ranker_model: str,
    temperature: float = 0.0,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> tuple:
    """
    Binary goal-achievement evaluation.
    Returns score in [0, 1] where 0=failure, 1=success.
    """
    agents_bg = scene.get("agents_background") or {}
    agent_goal = (scene.get("social_goals") or {}).get(agent_name, "")
    scenario = scene.get("scenario", "")

    counterpart_name = "User"
    if isinstance(agents_bg, dict):
        for name in agents_bg.keys():
            name_str = str(name).strip()
            if name_str and name_str != agent_name:
                counterpart_name = name_str
                break
    history_text = render_dialogue_for_lang_reward(
        history_messages or [], other_speaker_name=counterpart_name
    )
    if scenario:
        history_text = f"Scenario: {scenario}\n\n{history_text}".strip()

    user_prompt = goal_achieve_prompt.format(
        agent_name=agent_name,
        goal=agent_goal,
        history=history_text or "(No dialogue history)",
    )
    messages = [{"role": "user", "content": user_prompt}]
    resp = call_api_model(
        messages=messages,
        model=ranker_model,
        temperature=temperature,
        base_url=base_url,
        api_key=api_key,
    )
    if not resp:
        return 0.0, ""

    try:
        cleaned = resp.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            score_raw = parsed.get("goal_achievement_score")
            analysis = str(parsed.get("analysis") or parsed.get("reason") or "")
            evidence = str(parsed.get("key_evidence") or "")
            reason = analysis
            if evidence:
                reason = f"{analysis}\nEvidence: {evidence}" if analysis else evidence
            if score_raw is not None:
                score = float(score_raw)
                # Binary prompt expects 0/1, keep compatibility with accidental 0-100.
                if score > 1.0:
                    score = 1.0 if score >= 50.0 else 0.0
                score = 1.0 if score >= 0.5 else 0.0
                return score, reason
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    return 0.0, ""


def evaluate_social_goal_achievement_with_reason_continuous(
    scene: dict,
    history_messages: List[Dict[str, str]],
    agent_name: str,
    ranker_model: str,
    temperature: float = 0.0,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    max_workers: Optional[int] = None,
) -> tuple:
    """
    Continuous goal-achievement evaluation.
    Returns a score between 0-1 (float).
    """
    agents_bg = scene.get("agents_background") or {}
    agent_goal = (scene.get("social_goals") or {}).get(agent_name, "")
    scenario = scene.get("scenario", "")

    counterpart_name = "User"
    if isinstance(agents_bg, dict):
        for name in agents_bg.keys():
            name_str = str(name).strip()
            if name_str and name_str != agent_name:
                counterpart_name = name_str
                break
    history_text = render_dialogue_for_lang_reward(
        history_messages or [], other_speaker_name=counterpart_name
    )
    if scenario:
        history_text = f"Scenario: {scenario}\n\n{history_text}".strip()

    user_prompt = goal_achieve_prompt_continuous.format(
        agent_name=agent_name,
        goal=agent_goal,
        history=history_text or "(No dialogue history)",
    )

    messages = [
        {"role": "user", "content": user_prompt},
    ]
    resp = call_api_model(
        messages=messages,
        model=ranker_model,
        temperature=temperature,
        base_url=base_url,
        api_key=api_key,
    )
    if not resp:
        return 0.0, ""

    try:
        cleaned = resp.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            score_raw = parsed.get("goal_achievement_score")
            analysis = str(parsed.get("analysis") or parsed.get("reason") or "")
            evidence = str(parsed.get("key_evidence") or "")
            reason = analysis
            if evidence:
                reason = f"{analysis}\nEvidence: {evidence}" if analysis else evidence

            if score_raw is not None:
                score = float(score_raw)
                # Backward-compatible handling for accidental 0-100 outputs.
                if score > 1.0:
                    score = score / 100.0
                return max(0.0, min(1.0, score)), reason
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Fallback: parse the first number from free text.
    numbers = re.findall(r"\b\d+(?:\.\d+)?\b", resp)
    if numbers:
        try:
            score = float(numbers[0])
            if score > 1.0:
                score = score / 100.0
            return max(0.0, min(1.0, score)), ""
        except Exception:
            pass
    return 0.0, ""


def _render_goal_text(goal_value: object) -> str:
    if isinstance(goal_value, dict):
        parts: List[str] = []
        main_text = str(goal_value.get("main_text") or "").strip()
        if main_text:
            parts.append(main_text)
        tags = goal_value.get("tags") or {}
        if isinstance(tags, dict):
            for key, label in (
                ("extra_info", "Extra info"),
                ("clarification_hint", "Clarification hint"),
                ("strategy_hint", "Strategy hint"),
            ):
                items = tags.get(key) or []
                snippets: List[str] = []
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict):
                            content = str(item.get("content") or "").strip()
                        else:
                            content = str(item or "").strip()
                        if content:
                            snippets.append(content)
                if snippets:
                    parts.append(f"{label}: {' '.join(snippets)}")
        rendered = "\n".join(parts).strip()
        if rendered:
            return rendered
    text = str(goal_value or "").strip()
    return text or "(No goal provided)"


def _rubric_prompt_context(
    scene: dict,
    history_messages: List[Dict[str, str]],
    agent_name: str,
) -> Dict[str, str]:
    """Build context from history BEFORE the current reply for both stages."""
    agents_bg = scene.get("agents_background") or {}
    social_goals = scene.get("social_goals") or {}
    counterpart_name = "User"
    if isinstance(agents_bg, dict):
        for name in agents_bg.keys():
            name_str = str(name).strip()
            if name_str and name_str != agent_name:
                counterpart_name = name_str
                break

    history_text = render_dialogue_for_lang_reward(
        history_messages or [], other_speaker_name=counterpart_name
    )
    return {
        "scenario": str(scene.get("scenario") or "(No scenario provided)"),
        "agent_name": agent_name,
        "agent_background": str(agents_bg.get(agent_name) or "(No profile provided)"),
        "agent_goal": _render_goal_text(social_goals.get(agent_name, "")),
        "counterpart_name": counterpart_name,
        "counterpart_background": str(
            agents_bg.get(counterpart_name, "") or "(No profile provided)"
        ),
        "history": history_text or "(No dialogue history)",
    }


def _parse_rubric_response(resp: Optional[str]) -> dict:
    if not resp:
        return {}
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
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, ValueError, TypeError):
        return {}


def generate_rubric_dynamic(
    scene: dict,
    history_messages: List[Dict[str, str]],
    agent_name: str,
    ranker_model: str,
    temperature: float = 0.0,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    baseline_weights_text: str = "",
) -> tuple:
    """Generate criteria and weights without access to the current reply.

    Returns (rubric_items, llm_weights, weight_reasoning). Items contain only
    system-assigned item_id and criterion; invalid dimensions have no items.
    """
    user_prompt = rubric_dynamic_generation_prompt.format(
        **_rubric_prompt_context(scene, history_messages, agent_name),
        baseline_weights=str(baseline_weights_text or "(no baseline provided)"),
    )
    resp = call_api_model(
        messages=[{"role": "user", "content": user_prompt}],
        model=ranker_model,
        temperature=temperature,
        base_url=base_url,
        api_key=api_key,
    )
    parsed = _parse_rubric_response(resp)
    rubrics = {d: [] for d in RUBRIC_DIMENSIONS}
    raw_rubrics = parsed.get("rubrics")
    if isinstance(raw_rubrics, dict):
        for dim in RUBRIC_DIMENSIONS:
            items = raw_rubrics.get(dim)
            if not isinstance(items, list) or not 2 <= len(items) <= 4:
                continue
            criteria = [
                item.get("criterion") if isinstance(item, dict) else None
                for item in items
            ]
            if any(not isinstance(c, str) or not c.strip() for c in criteria):
                continue
            criteria = [c.strip() for c in criteria]
            if len(set(criteria)) != len(criteria):
                continue
            # Ignore any scores, reasons, or IDs leaked by the generator.
            rubrics[dim] = [
                {"item_id": f"{dim}_{idx}", "criterion": criterion}
                for idx, criterion in enumerate(criteria, start=1)
            ]

    weights: Dict[str, float] = {}
    raw_weights = parsed.get("weights")
    if isinstance(raw_weights, dict):
        for dim in RUBRIC_DIMENSIONS:
            try:
                value = float(raw_weights.get(dim))
            except (TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(value):
                weights[dim] = value
    return rubrics, weights or None, str(parsed.get("weight_reasoning") or "").strip()


def score_rubric_dynamic_with_reason(
    scene: dict,
    history_messages: List[Dict[str, str]],
    agent_name: str,
    current_response: str,
    rubric_items: Dict[str, list],
    ranker_model: str,
    temperature: float = 0.0,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> tuple:
    """Score fixed items returned by generate_rubric_dynamic in a fresh call.

    Returns (dim_scores, scored_rubrics, judge_reason). Criteria are copied from
    generation, never from the scorer. A dimension with incomplete/invalid
    judgements retains the neutral 0.5 fallback, with unscored items marked None.
    """
    fixed_items = {
        dim: [
            {"item_id": item["item_id"], "criterion": item["criterion"]}
            for item in rubric_items.get(dim, [])
        ]
        for dim in RUBRIC_DIMENSIONS
    }
    scores = {d: 0.5 for d in RUBRIC_DIMENSIONS}
    scored_rubrics = {
        dim: [dict(item, passed=None, reason="No valid judgement returned.") for item in items]
        for dim, items in fixed_items.items()
    }
    response_text = str(current_response or "").strip()
    if not response_text:
        return scores, scored_rubrics, "No actor reply available for rubric judgement."
    if not any(fixed_items.values()):
        return scores, scored_rubrics, "Rubric generation failed: no valid criteria returned."

    user_prompt = rubric_dynamic_scoring_prompt.format(
        **_rubric_prompt_context(scene, history_messages, agent_name),
        response=response_text,
        rubric_items=json.dumps(fixed_items, ensure_ascii=False),
    )
    resp = call_api_model(
        messages=[{"role": "user", "content": user_prompt}],
        model=ranker_model,
        temperature=temperature,
        base_url=base_url,
        api_key=api_key,
    )
    parsed = _parse_rubric_response(resp)
    judgements = parsed.get("judgements")
    if not isinstance(judgements, dict):
        return scores, scored_rubrics, "Rubric scoring failed: no valid judgements returned."

    invalid_dims = []
    for dim, items in scored_rubrics.items():
        raw_items = judgements.get(dim)
        by_id = {}
        duplicate_ids = set()
        if isinstance(raw_items, list):
            for item in raw_items:
                if not isinstance(item, dict) or not isinstance(item.get("item_id"), str):
                    continue
                item_id = item["item_id"]
                if item_id in by_id:
                    duplicate_ids.add(item_id)
                by_id[item_id] = item
        for item in items:
            item_id = item["item_id"]
            judgement = by_id.get(item_id)
            if judgement is None or item_id in duplicate_ids:
                continue
            try:
                passed = float(judgement.get("passed"))
            except (TypeError, ValueError, OverflowError):
                continue
            if passed not in (0.0, 1.0):
                continue
            item["passed"] = int(passed)
            item["reason"] = str(judgement.get("reason") or "").strip()
        if items and all(item["passed"] is not None for item in items):
            scores[dim] = sum(item["passed"] for item in items) / len(items)
        else:
            invalid_dims.append(dim)

    reason = str(parsed.get("judge_reason") or parsed.get("reason") or "").strip()
    if invalid_dims:
        fallback_reason = (
            "Neutral fallback for missing/invalid rubric judgements: "
            + ", ".join(invalid_dims)
        )
        reason = f"{reason}\n{fallback_reason}".strip()
    return scores, scored_rubrics, reason


def evaluate_rubric_dynamic_with_reason(
    scene: dict,
    history_messages: List[Dict[str, str]],
    agent_name: str,
    current_response: str,
    ranker_model: str,
    temperature: float = 0.0,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    baseline_weights_text: str = "",
) -> tuple:
    """Generate reply-independent rubrics, then score the reply in a second call.

    Returns (dim_scores, scored_rubrics, judge_reason, llm_weights, weight_reasoning).
    Both stages use the configured reward endpoint; weights come only from generation.
    """
    if not str(current_response or "").strip():
        return (
            {d: 0.5 for d in RUBRIC_DIMENSIONS},
            {d: [] for d in RUBRIC_DIMENSIONS},
            "No actor reply available for rubric judgement.",
            None,
            "",
        )
    rubrics, llm_weights, weight_reasoning = generate_rubric_dynamic(
        scene,
        history_messages,
        agent_name,
        ranker_model=ranker_model,
        temperature=temperature,
        base_url=base_url,
        api_key=api_key,
        baseline_weights_text=baseline_weights_text,
    )
    scores, scored_rubrics, reason = score_rubric_dynamic_with_reason(
        scene,
        history_messages,
        agent_name,
        current_response,
        rubrics,
        ranker_model=ranker_model,
        temperature=temperature,
        base_url=base_url,
        api_key=api_key,
    )
    return scores, scored_rubrics, reason, llm_weights, weight_reasoning
