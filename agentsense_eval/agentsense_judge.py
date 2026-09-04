"""
LM-as-Judge scoring for AgentSense evaluation.

Scores the policy agent's goal achievement on a continuous [0, 100] scale,
mirroring evaluate_social_goal_achievement_with_reason_continuous() from
scripts-ppo-third-dynamic-reward-rubric/rl_reward.py but fully independent.
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from typing import Optional

import requests

from agentsense_prompt import GOAL_JUDGE_PROMPT, RELATION_DELTA_PROMPT
from agentsense_utils import render_history_for_judge

logger = logging.getLogger("agentsense_judge")


# ---------------------------------------------------------------------------
# HTTP helper (independent copy)
# ---------------------------------------------------------------------------

def _call_api(
    messages: list,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float = 0.0,
) -> Optional[str]:
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": False},
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    max_retries = int(os.environ.get("AGENTSENSE_API_MAX_RETRIES", "6"))
    base_delay = float(os.environ.get("AGENTSENSE_API_RETRY_DELAY", "0.8"))
    last_exc: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            if resp.status_code == 200:
                data = resp.json()
                if "choices" in data:
                    return data["choices"][0]["message"]["content"].strip()
                last_exc = Exception(f"No 'choices' in response: {data}")
            else:
                last_exc = Exception(f"HTTP {resp.status_code}: {resp.text[:200]}")
        except requests.RequestException as exc:
            last_exc = exc
        if attempt < max_retries:
            sleep_s = base_delay * (2 ** attempt) + random.uniform(0, 0.25)
            logger.warning(
                "Judge API failed (attempt %d/%d): %s; retrying in %.2fs",
                attempt + 1, max_retries + 1, last_exc, sleep_s,
            )
            time.sleep(sleep_s)

    logger.error("Judge API failed after %d retries: %s", max_retries + 1, last_exc)
    return None


def _parse_json(raw: str) -> Optional[dict]:
    cleaned = raw.strip()
    for prefix in ("```json", "```"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    return None


# ---------------------------------------------------------------------------
# Partner info helper
# ---------------------------------------------------------------------------

def _build_partner_info(run_item: dict) -> str:
    """Build a short summary of non-policy characters for the judge prompt."""
    parts = []
    for c in run_item["characters"]:
        if c["name"] == run_item["policy_name"]:
            continue
        goals_text = "; ".join(g["goal"] for g in c["goals"])
        parts.append(f"[{c['name']}] Profile: {c['profile']}\nGoals: {goals_text}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Public scoring function
# ---------------------------------------------------------------------------

def score_goal_achievement(
    run_item: dict,
    history: list,
    model: str,
    base_url: str,
    api_key: str,
) -> dict:
    """Score the policy agent's goal achievement for one completed dialogue.

    Parameters
    ----------
    run_item : dict
        The run item produced by agentsense_utils.expand_scene_to_run_items().
    history : list
        Completed chat history (list of {"speaker": ..., "content": ...}).
    model / base_url / api_key : str
        Judge model connection config.

    Returns
    -------
    dict with keys:
        score          – int 0 or 1, default 0 on failure
        analysis       – str
        key_evidence   – str
        raw_response   – str (for debugging)
    """
    default = {"score": 0, "analysis": "", "key_evidence": "", "raw_response": ""}

    history_text = render_history_for_judge(history)
    if not history_text.strip():
        default["analysis"] = "Empty dialogue history."
        return default

    agent_goals_text = "\n".join(
        f"{i + 1}. {g}" for i, g in enumerate(run_item["policy_goals"])
    )

    prompt = GOAL_JUDGE_PROMPT.format(
        agent_name=run_item["policy_name"],
        agent_goals=agent_goals_text,
        history=history_text,
    )
    messages = [{"role": "user", "content": prompt}]
    raw = _call_api(messages, model=model, base_url=base_url, api_key=api_key)
    default["raw_response"] = raw or ""

    if not raw:
        return default

    parsed = _parse_json(raw)
    if parsed:
        result = dict(default)
        result["raw_response"] = raw
        result["analysis"] = str(parsed.get("analysis") or "").strip()
        result["key_evidence"] = str(parsed.get("key_evidence") or "").strip()
        raw_score = parsed.get("goal_achievement_score")
        if raw_score is not None:
            try:
                result["score"] = 1 if int(float(raw_score)) >= 1 else 0
            except (TypeError, ValueError):
                pass
        return result

    # Fallback: look for a bare 0 or 1 in the free text
    for token in re.findall(r"\b[01]\b", raw):
        default["raw_response"] = raw
        default["score"] = int(token)
        return default

    default["raw_response"] = raw
    return default


# ---------------------------------------------------------------------------
# Relation delta scoring  (Δrelation ∈ [-1, 1])
# Mirrors evaluate_relation_delta() from score_relation_delta.py.
# ---------------------------------------------------------------------------

def _build_opponent_strategy(run_item: dict) -> tuple:
    """Return (opponent_name, opponent_profile, strategy_name, strategy_profile).

    The policy character is the "strategy model"; the first partner is the
    "opponent" (the character whose favorability we measure).
    """
    policy_name = run_item["policy_name"]
    policy_profile = run_item["policy_profile"]
    char_map = {c["name"]: c for c in run_item["characters"]}

    # Pick the first non-policy character as the opponent.
    opponent_name = ""
    opponent_profile = ""
    for c in run_item["characters"]:
        if c["name"] != policy_name:
            opponent_name = c["name"]
            opponent_profile = c.get("profile", "")
            break

    return opponent_name, opponent_profile, policy_name, policy_profile


def score_relation_delta(
    run_item: dict,
    history: list,
    model: str,
    base_url: str,
    api_key: str,
) -> dict:
    """Score the relationship favorability change after one completed dialogue.

    Measures how much the first partner's (opponent's) favorability toward the
    policy character (strategy model) changed, on a scale of [-1, 1].

    Parameters
    ----------
    run_item : dict
        The run item produced by agentsense_utils.expand_scene_to_run_items().
    history : list
        Completed chat history (list of {"speaker": ..., "content": ...}).
    model / base_url / api_key : str
        Judge model connection config.

    Returns
    -------
    dict with keys:
        score        – float in [-1, 1], default 0.0 on failure
        reason       – str
        raw_response – str (for debugging)
    """
    default = {"score": 0.0, "reason": "", "raw_response": ""}

    history_text = render_history_for_judge(history)
    if not history_text.strip():
        default["reason"] = "Empty dialogue history."
        return default

    opponent_name, opponent_profile, strategy_name, strategy_profile = (
        _build_opponent_strategy(run_item)
    )
    if not opponent_name:
        default["reason"] = "No partner character found."
        return default

    prompt = RELATION_DELTA_PROMPT.format(
        opponent_name=opponent_name,
        strategy_name=strategy_name,
        scenario=run_item["scenario"],
        opponent_profile=opponent_profile or "(unknown)",
        strategy_profile=strategy_profile or "(unknown)",
        history=history_text,
    )
    messages = [{"role": "user", "content": prompt}]
    raw = _call_api(messages, model=model, base_url=base_url, api_key=api_key)
    default["raw_response"] = raw or ""

    if not raw:
        return default

    parsed = _parse_json(raw)
    if parsed:
        result = dict(default)
        result["raw_response"] = raw
        result["reason"] = str(parsed.get("reason") or "").strip()
        raw_score = parsed.get("relation_delta")
        if raw_score is None:
            raw_score = parsed.get("delta_relation")
        if raw_score is not None:
            try:
                result["score"] = max(-1.0, min(1.0, float(raw_score)))
            except (TypeError, ValueError):
                pass
        return result

    # Fallback: scan for a float in [-1, 1]
    for num in re.findall(r"-?\d+(?:\.\d+)?", raw):
        try:
            val = float(num)
            if -1.0 <= val <= 1.0:
                default["raw_response"] = raw
                default["score"] = val
                default["reason"] = raw[:300]
                return default
        except ValueError:
            continue

    default["raw_response"] = raw
    return default
