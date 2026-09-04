from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

from prompt import (
    goal_achieve_prompt,
    goal_achieve_prompt_continuous,
    goal_directed_progress_prompt,
    multi_dim_judge_prompt,
    rubric_dynamic_judge_prompt,
    subgoal_progress_prompt,
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
    Returns score in [0, 100] where 0=failure, 100=success.
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
        return 0.5, ""

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
                score = 1.0 if score >= 0.5 else 0.0
                return (1.0 if score == 1.0 else 0.0), reason
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    return 0.5, ""


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
    Returns a score between 0-100 (float) instead of binary 0/100.
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
        return 0.5, ""

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
                score = float(score_raw) / 100.0
                return max(0.0, min(1.0, score)), reason
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Fallback: parse the first number from free text.
    numbers = re.findall(r"\b\d+(?:\.\d+)?\b", resp)
    if numbers:
        try:
            score = float(numbers[0]) / 100.0
            return max(0.0, min(1.0, score)), ""
        except Exception:
            pass
    return 0.5, ""


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


def evaluate_goal_directed_progress_with_reason(
    scene: dict,
    history_messages: List[Dict[str, str]],
    agent_name: str,
    current_response: str,
    ranker_model: str,
    temperature: float = 0.0,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> tuple:
    """Goal-directed per-turn progress reward.

    Unlike :func:`evaluate_subgoal_progress_with_reason` which scores partner
    approval, this scorer judges how much the actor's current reply advances
    the actor's own social goal from a neutral third-party perspective.
    Designed for scenarios where partner approval is mis-aligned with goal
    achievement (negotiation, boundary defense, etc.).
    Returns ``(progress_score_in_[0,1], reason)``.
    """
    agents_bg = scene.get("agents_background") or {}
    social_goals = scene.get("social_goals") or {}
    scenario = scene.get("scenario", "")
    counterpart_name = "User"
    if isinstance(agents_bg, dict):
        for name in agents_bg.keys():
            name_str = str(name).strip()
            if name_str and name_str != agent_name:
                counterpart_name = name_str
                break

    response_text = str(current_response or "").strip()
    if not response_text:
        return 0.0, "No actor reply available for goal-progress judgement."

    history_text = render_dialogue_for_lang_reward(
        history_messages or [], other_speaker_name=counterpart_name
    )
    user_prompt = goal_directed_progress_prompt.format(
        scenario=str(scenario or "(No scenario provided)"),
        agent_name=agent_name,
        agent_background=str(agents_bg.get(agent_name, "") or "(No profile provided)"),
        agent_goal=_render_goal_text(social_goals.get(agent_name, "")),
        counterpart_name=counterpart_name,
        counterpart_background=str(
            agents_bg.get(counterpart_name, "") or "(No profile provided)"
        ),
        history=history_text or "(No dialogue history)",
        response=response_text,
    )
    resp = call_api_model(
        messages=[{"role": "user", "content": user_prompt}],
        model=ranker_model,
        temperature=temperature,
        base_url=base_url,
        api_key=api_key,
    )
    if not resp:
        return 0.5, ""

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
            reason = str(parsed.get("reason") or parsed.get("analysis") or "").strip()
            raw_score = parsed.get("goal_progress_score")
            if raw_score is None:
                raw_score = parsed.get("progress_score")
            if raw_score is None:
                raw_score = parsed.get("score")
            if raw_score is not None:
                score = float(raw_score)
                return max(0.0, min(1.0, score)), reason
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    numbers = re.findall(r"\b(?:0(?:\.\d+)?|1(?:\.0+)?)\b", resp)
    if numbers:
        try:
            score = float(numbers[0])
            return max(0.0, min(1.0, score)), ""
        except Exception:
            pass
    return 0.5, ""


def evaluate_multi_dim_with_reason(
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
    """Per-turn multi-dimensional judgement with optional dynamic weights.

    Issues a single LLM call that returns 13 scores in [0,1] PLUS a candidate
    weight vector over the same 13 dimensions. The weights are advisory; the
    caller decides whether to blend them with the SCHEDULE baseline.

    ``baseline_weights_text`` is rendered into the prompt as a prior.

    Returns ``(dim_scores: Dict[str, float], reason: str,
    llm_weights: Optional[Dict[str, float]], weight_reasoning: str)``.
    On failure dims default to 0.5 and llm_weights is None.
    """
    DIMS = (
        "goal_progress",
        "assertiveness",
        "logical_coherence",
        "closure_appropriateness",
        "partner_approval",
        "rapport_building",
        "face_preservation",
        "self_disclosure",
        "emotional_intelligence",
        "role_consistency",
        "relevance",
        "novelty",
        "conciseness",
    )
    WEIGHT_DIMS = DIMS
    default = {d: 0.5 for d in DIMS}

    agents_bg = scene.get("agents_background") or {}
    social_goals = scene.get("social_goals") or {}
    scenario = scene.get("scenario", "")
    counterpart_name = "User"
    if isinstance(agents_bg, dict):
        for name in agents_bg.keys():
            name_str = str(name).strip()
            if name_str and name_str != agent_name:
                counterpart_name = name_str
                break

    response_text = str(current_response or "").strip()
    if not response_text:
        return default, "No actor reply available for multi-dim judgement.", None, ""

    history_text = render_dialogue_for_lang_reward(
        history_messages or [], other_speaker_name=counterpart_name
    )
    user_prompt = multi_dim_judge_prompt.format(
        scenario=str(scenario or "(No scenario provided)"),
        agent_name=agent_name,
        agent_background=str(agents_bg.get(agent_name, "") or "(No profile provided)"),
        agent_goal=_render_goal_text(social_goals.get(agent_name, "")),
        counterpart_name=counterpart_name,
        counterpart_background=str(
            agents_bg.get(counterpart_name, "") or "(No profile provided)"
        ),
        history=history_text or "(No dialogue history)",
        response=response_text,
        baseline_weights=str(baseline_weights_text or "(no baseline provided)"),
    )
    resp = call_api_model(
        messages=[{"role": "user", "content": user_prompt}],
        model=ranker_model,
        temperature=temperature,
        base_url=base_url,
        api_key=api_key,
    )
    if not resp:
        return default, "", None, ""

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
            weight_reasoning = str(parsed.get("weight_reasoning") or "").strip()
            scores = parsed.get("scores")
            out = dict(default)
            if isinstance(scores, dict):
                for d in DIMS:
                    v = scores.get(d)
                    if v is None:
                        continue
                    try:
                        out[d] = max(0.0, min(1.0, float(v)))
                    except (TypeError, ValueError):
                        continue
            llm_weights: Optional[Dict[str, float]] = None
            ws = parsed.get("weights")
            if isinstance(ws, dict):
                cand: Dict[str, float] = {}
                for d in WEIGHT_DIMS:
                    v = ws.get(d)
                    if v is None:
                        continue
                    try:
                        cand[d] = float(v)
                    except (TypeError, ValueError):
                        continue
                if cand:
                    llm_weights = cand
            return out, reason, llm_weights, weight_reasoning
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    return default, "", None, ""


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
    """Per-turn rubric-generated dynamic judgement.

    The judge generates 2-4 binary rubric items per high-level dimension and
    optional dynamic weights. The system converts each dimension's binary item
    pass rate into a score in [0,1].

    Returns ``(dim_scores, rubrics, judge_reason, llm_weights, weight_reasoning)``.
    """
    DIMS = (
        "goal_advancement",
        "strategic_positioning",
        "relational_attunement",
        "persona_consistency",
        "contextual_coherence",
        "turn_quality",
    )
    default_scores = {d: 0.5 for d in DIMS}
    default_rubrics = {d: [] for d in DIMS}

    agents_bg = scene.get("agents_background") or {}
    social_goals = scene.get("social_goals") or {}
    scenario = scene.get("scenario", "")
    counterpart_name = "User"
    if isinstance(agents_bg, dict):
        for name in agents_bg.keys():
            name_str = str(name).strip()
            if name_str and name_str != agent_name:
                counterpart_name = name_str
                break

    response_text = str(current_response or "").strip()
    if not response_text:
        return (
            default_scores,
            default_rubrics,
            "No actor reply available for rubric judgement.",
            None,
            "",
        )

    history_text = render_dialogue_for_lang_reward(
        history_messages or [], other_speaker_name=counterpart_name
    )
    user_prompt = rubric_dynamic_judge_prompt.format(
        scenario=str(scenario or "(No scenario provided)"),
        agent_name=agent_name,
        agent_background=str(agents_bg.get(agent_name, "") or "(No profile provided)"),
        agent_goal=_render_goal_text(social_goals.get(agent_name, "")),
        counterpart_name=counterpart_name,
        counterpart_background=str(
            agents_bg.get(counterpart_name, "") or "(No profile provided)"
        ),
        history=history_text or "(No dialogue history)",
        response=response_text,
        baseline_weights=str(baseline_weights_text or "(no baseline provided)"),
    )
    resp = call_api_model(
        messages=[{"role": "user", "content": user_prompt}],
        model=ranker_model,
        temperature=temperature,
        base_url=base_url,
        api_key=api_key,
    )
    if not resp:
        return default_scores, default_rubrics, "", None, ""

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
        if not isinstance(parsed, dict):
            return default_scores, default_rubrics, "", None, ""

        weight_reasoning = str(parsed.get("weight_reasoning") or "").strip()
        judge_reason = str(
            parsed.get("judge_reason") or parsed.get("reason") or parsed.get("analysis") or ""
        ).strip()

        out_rubrics: Dict[str, list] = {d: [] for d in DIMS}
        out_scores: Dict[str, float] = dict(default_scores)
        rubrics_raw = parsed.get("rubrics")
        if isinstance(rubrics_raw, dict):
            for dim in DIMS:
                items = rubrics_raw.get(dim)
                if not isinstance(items, list):
                    continue
                cleaned_items = []
                passed_values: List[float] = []
                for item in items[:4]:
                    if not isinstance(item, dict):
                        continue
                    criterion = str(item.get("criterion") or "").strip()
                    reason = str(item.get("reason") or "").strip()
                    raw_passed = item.get("passed")
                    if isinstance(raw_passed, bool):
                        passed = 1 if raw_passed else 0
                    else:
                        try:
                            passed = 1 if int(float(raw_passed)) >= 1 else 0
                        except (TypeError, ValueError):
                            continue
                    passed = 1 if passed else 0
                    cleaned_items.append(
                        {
                            "criterion": criterion,
                            "passed": int(passed),
                            "reason": reason,
                        }
                    )
                    passed_values.append(float(passed))
                if cleaned_items:
                    out_rubrics[dim] = cleaned_items
                    out_scores[dim] = float(sum(passed_values) / len(passed_values))

        llm_weights: Optional[Dict[str, float]] = None
        ws = parsed.get("weights")
        if isinstance(ws, dict):
            cand: Dict[str, float] = {}
            for d in DIMS:
                v = ws.get(d)
                if v is None:
                    continue
                try:
                    cand[d] = float(v)
                except (TypeError, ValueError):
                    continue
            if cand:
                llm_weights = cand
        return out_scores, out_rubrics, judge_reason, llm_weights, weight_reasoning
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    return default_scores, default_rubrics, "", None, ""


def evaluate_subgoal_progress_with_reason(
    scene: dict,
    history_messages: List[Dict[str, str]],
    agent_name: str,
    current_response: str,
    partner_response: str,
    ranker_model: str,
    temperature: float = 0.0,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> tuple:
    """
    Evaluate third-party approval of the small-model reply from the
    large-model opponent's perspective.
    Returns (approval_score_0_to_1, reason).
    """
    agents_bg = scene.get("agents_background") or {}
    social_goals = scene.get("social_goals") or {}
    scenario = scene.get("scenario", "")
    counterpart_name = "User"
    if isinstance(agents_bg, dict):
        for name in agents_bg.keys():
            name_str = str(name).strip()
            if name_str and name_str != agent_name:
                counterpart_name = name_str
                break

    partner_reply_text = str(partner_response or "").strip()
    if not partner_reply_text:
        return 0.0, "No large-model opponent reply available for approval judgement."

    history_text = render_dialogue_for_lang_reward(
        history_messages or [], other_speaker_name=counterpart_name
    )
    user_prompt = subgoal_progress_prompt.format(
        scenario=str(scenario or "(No scenario provided)"),
        agent_name=agent_name,
        agent_background=str(agents_bg.get(agent_name, "") or "(No profile provided)"),
        agent_goal=_render_goal_text(social_goals.get(agent_name, "")),
        counterpart_name=counterpart_name,
        counterpart_background=str(
            agents_bg.get(counterpart_name, "") or "(No profile provided)"
        ),
        counterpart_goal=_render_goal_text(social_goals.get(counterpart_name, "")),
        history=history_text or "(No dialogue history)",
        response=str(current_response or "").strip() or "(No current small-model reply)",
        partner_response=partner_reply_text,
    )
    resp = call_api_model(
        messages=[{"role": "user", "content": user_prompt}],
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
            reason = str(parsed.get("reason") or parsed.get("analysis") or "").strip()
            raw_score = parsed.get("approval_score")
            if raw_score is None:
                raw_score = parsed.get("score")
            if raw_score is not None:
                score = float(raw_score)
                return max(0.0, min(1.0, score)), reason
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Fallback: parse the first float in [0, 1] from free text.
    numbers = re.findall(r"\b(?:0(?:\.\d+)?|1(?:\.0+)?)\b", resp)
    if numbers:
        try:
            score = float(numbers[0])
            return max(0.0, min(1.0, score)), ""
        except Exception:
            pass
    return 0.0, ""
