"""Dynamic multi-dimensional reward configuration.

Designed for the experiment that varies reward dimension weights by:
  - actor role family (expressive / assertive / acquisitive / cooperative)
  - dialogue phase (early / mid / late)

The 13 reward dimensions are all LLM-judged in one consolidated JSON judge
call per turn. ``soundness`` was split into ``logical_coherence`` +
``emotional_intelligence`` and ``relationship_quality`` into
``rapport_building`` + ``face_preservation`` for finer-grained credit; the
former rule-based ``novelty`` / ``conciseness`` are now LLM-judged too.

Combine with `TRAJECTORY_NORMALIZE` (per-turn reward /= expected_horizon)
to defuse the horizon-trap behavior PPO learns on per-turn process rewards.
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple


DIMENSIONS_LLM: Tuple[str, ...] = (
    # --- goal / task axis ---
    "goal_progress",
    "assertiveness",
    "logical_coherence",        # split from old "soundness": reasoning / argument quality
    "closure_appropriateness",
    # --- relationship / emotional axis ---
    "partner_approval",
    "rapport_building",         # split from old "relationship_quality": warmth / resonance / trust
    "face_preservation",        # split from old "relationship_quality": avoid offense / preserve face
    "self_disclosure",
    "emotional_intelligence",   # split from old "soundness": EQ in reading / responding to partner
    # --- persona / quality axis ---
    "role_consistency",         # 角色符合度: in-character fidelity to assigned profile
    "relevance",
    "novelty",                  # was rule-based; now LLM-judged
    "conciseness",              # was rule-based; now LLM-judged
)

# All dimensions are LLM-judged now (novelty / conciseness promoted from rules).
DIMENSIONS_RULE: Tuple[str, ...] = ()

ALL_DIMENSIONS: Tuple[str, ...] = DIMENSIONS_LLM + DIMENSIONS_RULE


ROLE_TO_FAMILY: Dict[str, str] = {
    # expressive: vulnerable / emotional disclosure dominant
    "support_seeker": "expressive",
    # assertive: stand-ground roles where caving directly destroys goal
    "lender": "assertive",
    "seller": "assertive",
    "persuader": "assertive",
    "boundary_setter": "assertive",
    "support_provider": "assertive",
    # acquisitive: goal = win partner concession; partner-approval correlates with goal
    "buyer": "acquisitive",
    "borrower": "acquisitive",
    "persuadee": "acquisitive",
    "boundary_challenger": "acquisitive",
    "verify_target": "acquisitive",
    "verifier": "assertive",
    # cooperative: mutual alignment, low-stakes coordination
    "co_planner": "cooperative",
    "planner": "cooperative",
    "safety_advocate": "cooperative",
    "safety_resister": "acquisitive",
    "other": "cooperative",
}


PHASE_NAMES: Tuple[str, ...] = ("early", "mid", "late")


def get_phase(turn_idx: int, expected_horizon: int = 5) -> str:
    """Map an absolute actor turn index to a discrete phase.

    expected_horizon is a soft estimate of typical horizon (5 for our setup).
    rel < 0.40 -> early, < 0.80 -> mid, otherwise late.
    Note: turn_idx is 0-based.
    """
    h = max(2, int(expected_horizon))
    rel = float(turn_idx) / float(h - 1)
    if rel < 0.40:
        return "early"
    if rel < 0.80:
        return "mid"
    return "late"


SCHEDULE: Dict[str, Dict[str, Dict[str, float]]] = {
    "expressive": {
        "early": {
            "self_disclosure": 0.40,
            "goal_progress": 0.20,
            "logical_coherence": 0.06,
            "emotional_intelligence": 0.04,
            "relevance": 0.10,
            "partner_approval": 0.05,
            "rapport_building": 0.03,
            "face_preservation": 0.02,
            "assertiveness": 0.05,
            "closure_appropriateness": 0.00,
            "novelty": 0.03,
            "conciseness": 0.02,
        },
        "mid": {
            "goal_progress": 0.30,
            "self_disclosure": 0.20,
            "logical_coherence": 0.09,
            "emotional_intelligence": 0.06,
            "rapport_building": 0.06,
            "face_preservation": 0.04,
            "assertiveness": 0.10,
            "partner_approval": 0.05,
            "relevance": 0.05,
            "closure_appropriateness": 0.00,
            "novelty": 0.03,
            "conciseness": 0.02,
        },
        "late": {
            "rapport_building": 0.18,
            "face_preservation": 0.12,
            "partner_approval": 0.20,
            "closure_appropriateness": 0.20,
            "goal_progress": 0.15,
            "logical_coherence": 0.03,
            "emotional_intelligence": 0.02,
            "self_disclosure": 0.05,
            "relevance": 0.03,
            "assertiveness": 0.00,
            "novelty": 0.01,
            "conciseness": 0.01,
        },
    },
    "assertive": {
        "early": {
            "goal_progress": 0.30,
            "assertiveness": 0.25,
            "logical_coherence": 0.09,
            "emotional_intelligence": 0.06,
            "self_disclosure": 0.10,
            "relevance": 0.10,
            "rapport_building": 0.03,
            "face_preservation": 0.02,
            "partner_approval": 0.00,
            "closure_appropriateness": 0.00,
            "novelty": 0.03,
            "conciseness": 0.02,
        },
        "mid": {
            "goal_progress": 0.35,
            "assertiveness": 0.25,
            "logical_coherence": 0.09,
            "emotional_intelligence": 0.06,
            "rapport_building": 0.06,
            "face_preservation": 0.04,
            "self_disclosure": 0.05,
            "relevance": 0.05,
            "partner_approval": 0.00,
            "closure_appropriateness": 0.00,
            "novelty": 0.03,
            "conciseness": 0.02,
        },
        "late": {
            "closure_appropriateness": 0.25,
            "goal_progress": 0.25,
            "rapport_building": 0.12,
            "face_preservation": 0.08,
            "logical_coherence": 0.06,
            "emotional_intelligence": 0.04,
            "partner_approval": 0.10,
            "assertiveness": 0.05,
            "self_disclosure": 0.02,
            "relevance": 0.01,
            "novelty": 0.01,
            "conciseness": 0.01,
        },
    },
    "acquisitive": {
        "early": {
            "self_disclosure": 0.25,
            "goal_progress": 0.25,
            "logical_coherence": 0.09,
            "emotional_intelligence": 0.06,
            "partner_approval": 0.15,
            "relevance": 0.10,
            "rapport_building": 0.03,
            "face_preservation": 0.02,
            "assertiveness": 0.05,
            "closure_appropriateness": 0.00,
            "novelty": 0.00,
            "conciseness": 0.00,
        },
        "mid": {
            "goal_progress": 0.35,
            "logical_coherence": 0.12,
            "emotional_intelligence": 0.08,
            "partner_approval": 0.15,
            "rapport_building": 0.09,
            "face_preservation": 0.06,
            "self_disclosure": 0.05,
            "assertiveness": 0.05,
            "relevance": 0.02,
            "closure_appropriateness": 0.00,
            "novelty": 0.02,
            "conciseness": 0.01,
        },
        "late": {
            "closure_appropriateness": 0.30,
            "partner_approval": 0.25,
            "rapport_building": 0.12,
            "face_preservation": 0.08,
            "goal_progress": 0.15,
            "logical_coherence": 0.03,
            "emotional_intelligence": 0.02,
            "assertiveness": 0.02,
            "self_disclosure": 0.01,
            "relevance": 0.01,
            "novelty": 0.00,
            "conciseness": 0.01,
        },
    },
    "cooperative": {
        "early": {
            "self_disclosure": 0.20,
            "goal_progress": 0.20,
            "logical_coherence": 0.09,
            "emotional_intelligence": 0.06,
            "relevance": 0.15,
            "partner_approval": 0.10,
            "rapport_building": 0.06,
            "face_preservation": 0.04,
            "assertiveness": 0.05,
            "closure_appropriateness": 0.00,
            "novelty": 0.03,
            "conciseness": 0.02,
        },
        "mid": {
            "goal_progress": 0.30,
            "logical_coherence": 0.12,
            "emotional_intelligence": 0.08,
            "rapport_building": 0.09,
            "face_preservation": 0.06,
            "partner_approval": 0.15,
            "self_disclosure": 0.05,
            "assertiveness": 0.05,
            "relevance": 0.05,
            "closure_appropriateness": 0.00,
            "novelty": 0.03,
            "conciseness": 0.02,
        },
        "late": {
            "closure_appropriateness": 0.30,
            "partner_approval": 0.20,
            "rapport_building": 0.12,
            "face_preservation": 0.08,
            "goal_progress": 0.15,
            "logical_coherence": 0.03,
            "emotional_intelligence": 0.02,
            "self_disclosure": 0.03,
            "assertiveness": 0.02,
            "relevance": 0.03,
            "novelty": 0.01,
            "conciseness": 0.01,
        },
    },
}


# Baseline weight for the persona-fidelity dimension, injected into every
# family x phase cell. Persona consistency is a steady "hygiene" signal: it
# matters throughout but should not dominate, so it tapers slightly by phase.
# (All aggregation paths renormalize by the sum of present weights, so adding
# this dimension does not require manually rebalancing the other weights.)
ROLE_CONSISTENCY_BASELINE: Dict[str, float] = {
    "early": 0.08,
    "mid": 0.07,
    "late": 0.05,
}

for _family_cells in SCHEDULE.values():
    for _phase_name, _cell in _family_cells.items():
        _cell.setdefault(
            "role_consistency", ROLE_CONSISTENCY_BASELINE.get(_phase_name, 0.05)
        )


def compute_weights(actor_role: str, turn_idx: int, expected_horizon: int = 5) -> Dict[str, float]:
    family = ROLE_TO_FAMILY.get((actor_role or "other").strip(), "cooperative")
    phase = get_phase(turn_idx, expected_horizon)
    return dict(SCHEDULE[family][phase])


def aggregate_weighted_score(
    dim_scores: Dict[str, float],
    actor_role: str,
    turn_idx: int,
    expected_horizon: int = 5,
) -> Tuple[float, Dict[str, float], str, str]:
    """Combine per-dimension scores into a single per-turn reward.

    Returns (weighted_score in [0,1], weights_used, family, phase).
    Weights are normalized to sum to 1.0 across whichever dimensions actually
    received a score (rule dims may be missing if no context).
    """
    family = ROLE_TO_FAMILY.get((actor_role or "other").strip(), "cooperative")
    phase = get_phase(turn_idx, expected_horizon)
    raw_w = SCHEDULE[family][phase]

    used: Dict[str, float] = {}
    total = 0.0
    sum_w = 0.0
    for dim, w in raw_w.items():
        if w <= 0:
            continue
        score = dim_scores.get(dim)
        if score is None:
            continue
        w_f = float(w)
        used[dim] = w_f
        total += w_f * float(score)
        sum_w += w_f

    if sum_w <= 0:
        return 0.5, used, family, phase
    return float(total / sum_w), used, family, phase


# ---------------------------------------------------------------------------
# Rubric-generated dynamic reward support
# ---------------------------------------------------------------------------

RUBRIC_DIMENSIONS: Tuple[str, ...] = (
    "goal_advancement",
    "strategic_positioning",
    "relational_attunement",
    "persona_consistency",
    "contextual_coherence",
    "turn_quality",
)


RUBRIC_SCHEDULE: Dict[str, Dict[str, Dict[str, float]]] = {
    "expressive": {
        "early": {
            "goal_advancement": 0.18,
            "strategic_positioning": 0.12,
            "relational_attunement": 0.22,
            "persona_consistency": 0.18,
            "contextual_coherence": 0.20,
            "turn_quality": 0.10,
        },
        "mid": {
            "goal_advancement": 0.26,
            "strategic_positioning": 0.16,
            "relational_attunement": 0.20,
            "persona_consistency": 0.14,
            "contextual_coherence": 0.14,
            "turn_quality": 0.10,
        },
        "late": {
            "goal_advancement": 0.20,
            "strategic_positioning": 0.10,
            "relational_attunement": 0.25,
            "persona_consistency": 0.12,
            "contextual_coherence": 0.10,
            "turn_quality": 0.23,
        },
    },
    "assertive": {
        "early": {
            "goal_advancement": 0.28,
            "strategic_positioning": 0.24,
            "relational_attunement": 0.12,
            "persona_consistency": 0.14,
            "contextual_coherence": 0.16,
            "turn_quality": 0.06,
        },
        "mid": {
            "goal_advancement": 0.32,
            "strategic_positioning": 0.24,
            "relational_attunement": 0.14,
            "persona_consistency": 0.12,
            "contextual_coherence": 0.10,
            "turn_quality": 0.08,
        },
        "late": {
            "goal_advancement": 0.26,
            "strategic_positioning": 0.18,
            "relational_attunement": 0.18,
            "persona_consistency": 0.10,
            "contextual_coherence": 0.08,
            "turn_quality": 0.20,
        },
    },
    "acquisitive": {
        "early": {
            "goal_advancement": 0.24,
            "strategic_positioning": 0.18,
            "relational_attunement": 0.20,
            "persona_consistency": 0.14,
            "contextual_coherence": 0.16,
            "turn_quality": 0.08,
        },
        "mid": {
            "goal_advancement": 0.32,
            "strategic_positioning": 0.20,
            "relational_attunement": 0.18,
            "persona_consistency": 0.10,
            "contextual_coherence": 0.12,
            "turn_quality": 0.08,
        },
        "late": {
            "goal_advancement": 0.24,
            "strategic_positioning": 0.14,
            "relational_attunement": 0.24,
            "persona_consistency": 0.08,
            "contextual_coherence": 0.08,
            "turn_quality": 0.22,
        },
    },
    "cooperative": {
        "early": {
            "goal_advancement": 0.20,
            "strategic_positioning": 0.14,
            "relational_attunement": 0.20,
            "persona_consistency": 0.16,
            "contextual_coherence": 0.20,
            "turn_quality": 0.10,
        },
        "mid": {
            "goal_advancement": 0.28,
            "strategic_positioning": 0.18,
            "relational_attunement": 0.20,
            "persona_consistency": 0.12,
            "contextual_coherence": 0.14,
            "turn_quality": 0.08,
        },
        "late": {
            "goal_advancement": 0.22,
            "strategic_positioning": 0.12,
            "relational_attunement": 0.24,
            "persona_consistency": 0.10,
            "contextual_coherence": 0.10,
            "turn_quality": 0.22,
        },
    },
}


def compute_rubric_weights(
    actor_role: str,
    turn_idx: int,
    expected_horizon: int = 5,
) -> Dict[str, float]:
    family = ROLE_TO_FAMILY.get((actor_role or "other").strip(), "cooperative")
    phase = get_phase(turn_idx, expected_horizon)
    return dict(RUBRIC_SCHEDULE[family][phase])


def render_rubric_baseline_for_prompt(
    actor_role: str,
    turn_idx: int,
    expected_horizon: int = 5,
) -> str:
    family = ROLE_TO_FAMILY.get((actor_role or "other").strip(), "cooperative")
    phase = get_phase(turn_idx, expected_horizon)
    base = RUBRIC_SCHEDULE[family][phase]
    total = sum(float(v) for v in base.values()) or 1.0
    rows = []
    for dim in RUBRIC_DIMENSIONS:
        w = float(base.get(dim, 0.0)) / total
        rows.append(f'    "{dim}": {w:.2f}')
    return (
        f"actor_role={actor_role}, family={family}, phase={phase} "
        f"(turn_idx={turn_idx} of expected_horizon={expected_horizon}).\n"
        f"Baseline rubric weight prior (sum=1.0):\n"
        + "{\n"
        + ",\n".join(rows)
        + "\n}"
    )


def safe_normalize_rubric_weights(
    weights: Dict[str, float],
    per_dim_clip: float = 0.5,
) -> Dict[str, float]:
    safe: Dict[str, float] = {}
    if not isinstance(weights, dict):
        return safe
    for dim in RUBRIC_DIMENSIONS:
        v = weights.get(dim)
        if v is None:
            continue
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if x < 0:
            x = 0.0
        if x > per_dim_clip:
            x = per_dim_clip
        if x > 0:
            safe[dim] = x
    total = sum(safe.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in safe.items()}


def mix_rubric_weight_dicts(
    llm_weights: Dict[str, float],
    baseline: Dict[str, float],
    mix_ratio: float = 0.5,
) -> Tuple[Dict[str, float], bool]:
    baseline = safe_normalize_rubric_weights(baseline, per_dim_clip=1.0)
    if not baseline:
        baseline = {dim: 1.0 / len(RUBRIC_DIMENSIONS) for dim in RUBRIC_DIMENSIONS}
    safe_llm = safe_normalize_rubric_weights(llm_weights)
    if not safe_llm or sum(safe_llm.values()) < 0.30:
        return baseline, False

    r = max(0.0, min(1.0, float(mix_ratio)))
    out: Dict[str, float] = {}
    for dim in RUBRIC_DIMENSIONS:
        b = float(baseline.get(dim, 0.0))
        l = float(safe_llm.get(dim, 0.0))
        out[dim] = r * l + (1.0 - r) * b
    total = sum(out.values())
    if total <= 0:
        return baseline, False
    out = {k: v / total for k, v in out.items()}

    cap = 0.5
    for _ in range(5):
        over = max(out.values()) - cap
        if over <= 1e-6:
            break
        max_key = max(out, key=out.get)
        out[max_key] = cap
        receivers = {k: v for k, v in out.items() if k != max_key and v > 0}
        rec_sum = sum(receivers.values())
        if rec_sum <= 0:
            return baseline, False
        for k, v in receivers.items():
            out[k] = v + over * (v / rec_sum)
    total = sum(out.values()) or 1.0
    return {k: v / total for k, v in out.items()}, True


def mix_rubric_with_baseline(
    llm_weights: Dict[str, float],
    actor_role: str,
    turn_idx: int,
    expected_horizon: int = 5,
    mix_ratio: float = 0.5,
) -> Tuple[Dict[str, float], bool]:
    baseline = compute_rubric_weights(actor_role, turn_idx, expected_horizon)
    return mix_rubric_weight_dicts(llm_weights, baseline, mix_ratio)


def aggregate_rubric_weighted_score(
    dim_scores: Dict[str, float],
    actor_role: str,
    turn_idx: int,
    expected_horizon: int = 5,
) -> Tuple[float, Dict[str, float], str, str]:
    family = ROLE_TO_FAMILY.get((actor_role or "other").strip(), "cooperative")
    phase = get_phase(turn_idx, expected_horizon)
    weights = safe_normalize_rubric_weights(RUBRIC_SCHEDULE[family][phase], per_dim_clip=1.0)

    used: Dict[str, float] = {}
    total = 0.0
    sum_w = 0.0
    for dim, w in weights.items():
        if w <= 0:
            continue
        score = dim_scores.get(dim)
        if score is None:
            continue
        used[dim] = float(w)
        total += float(w) * float(score)
        sum_w += float(w)
    if sum_w <= 0:
        return 0.5, used, family, phase
    return float(total / sum_w), used, family, phase


def _canonical_agent_name(name: object) -> str:
    text = str(name or "").strip().lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def _valid_actor_role(role: object) -> Optional[str]:
    if not isinstance(role, str):
        return None
    role = role.strip()
    if role in ROLE_TO_FAMILY:
        return role
    return None


def _lookup_role_mapping(mapping: object, agent_name: str) -> Optional[str]:
    if not isinstance(mapping, dict):
        return None

    role = _valid_actor_role(mapping.get(agent_name))
    if role:
        return role

    target = _canonical_agent_name(agent_name)
    if target:
        for key, value in mapping.items():
            if _canonical_agent_name(key) == target:
                role = _valid_actor_role(value)
                if role:
                    return role
    return None


def _goal_text_for_agent(scene: dict, agent_name: str) -> str:
    social_goals = scene.get("social_goals") or {}
    goal = None
    if isinstance(social_goals, dict):
        goal = social_goals.get(agent_name)
        if goal is None:
            target = _canonical_agent_name(agent_name)
            for key, value in social_goals.items():
                if _canonical_agent_name(key) == target:
                    goal = value
                    break
    if isinstance(goal, dict):
        parts = [str(goal.get("main_text") or "")]
        tags = goal.get("tags") or {}
        if isinstance(tags, dict):
            for items in tags.values():
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict):
                            parts.append(str(item.get("content") or ""))
                        else:
                            parts.append(str(item or ""))
        return "\n".join(part for part in parts if part).lower()
    return str(goal or "").lower()


def _infer_role_from_bucket_and_goal(scene: dict, agent_name: str) -> Optional[str]:
    bucket = str(scene.get("scene_bucket") or scene.get("bucket") or "").strip().lower()
    goal_text = _goal_text_for_agent(scene, agent_name)
    if not bucket or not goal_text:
        return None

    if bucket == "boundary_conflict":
        setter_terms = (
            "prevent", "delete", "stop", "set boundary", "privacy", "not comfortable",
            "do not want", "don't want", "refuse", "protect", "without permission",
        )
        challenger_terms = (
            "borrow", "keep", "use", "explain why you want", "convince", "persuade",
            "ask", "request", "regain", "forgiveness", "apolog",
        )
        setter_hits = sum(1 for term in setter_terms if term in goal_text)
        challenger_hits = sum(1 for term in challenger_terms if term in goal_text)
        return "boundary_setter" if setter_hits >= challenger_hits else "boundary_challenger"

    if bucket == "negotiation_transaction":
        seller_terms = ("sell", "price", "payment", "compensation", "fair value", "offer your")
        buyer_terms = ("buy", "purchase", "discount", "borrow", "obtain", "change", "lend", "loan")
        if any(term in goal_text for term in seller_terms):
            return "seller"
        if any(term in goal_text for term in buyer_terms):
            return "buyer"

    if bucket == "persuasion_influence":
        persuadee_terms = ("be supportive", "skeptical", "resistant", "manage", "consider", "decide whether")
        if any(term in goal_text for term in persuadee_terms):
            return "persuadee"
        return "persuader"

    if bucket == "support_empathy":
        provider_terms = ("support", "comfort", "help", "listen", "reassure", "encourage")
        seeker_terms = ("seek", "share", "express", "process", "struggling", "upset", "worried")
        provider_hits = sum(1 for term in provider_terms if term in goal_text)
        seeker_hits = sum(1 for term in seeker_terms if term in goal_text)
        return "support_provider" if provider_hits > seeker_hits else "support_seeker"

    if bucket == "coordination_planning":
        if any(term in goal_text for term in ("safety", "risk", "careful", "concern")):
            return "safety_advocate"
        return "co_planner"

    if bucket == "information_verification":
        if any(term in goal_text for term in ("verify", "find out", "ask", "learn", "confirm")):
            return "verify_target"
        return "other"

    return None


def resolve_actor_role(scene: dict, agent_name: str) -> str:
    """Resolve the actor role used by dynamic reward schedules.

    Preferred source is the per-scene ``role_side_by_agent`` annotation added
    to the bucketed data. The lookup is deliberately tolerant to copied scenes,
    role-swap wrappers, and minor agent-name formatting differences. When an
    older scene lacks per-agent roles, fall back to legacy fields and finally to
    a conservative scene-bucket/goal-text heuristic instead of collapsing every
    actor to ``other``.
    """
    if not isinstance(scene, dict):
        return "other"

    for field in (
        "role_side_by_agent",
        "actor_role_by_agent",
        "agent_role_by_agent",
        "role_by_agent",
    ):
        role = _lookup_role_mapping(scene.get(field), agent_name)
        if role:
            return role

    single_role = _valid_actor_role(scene.get("actor_role") or scene.get("role_side"))
    if single_role:
        return single_role

    inferred = _infer_role_from_bucket_and_goal(scene, agent_name)
    if inferred:
        return inferred

    return "other"


# ---------------------------------------------------------------------------
# LLM-judged dynamic weight support
# ---------------------------------------------------------------------------


def render_baseline_for_prompt(
    actor_role: str,
    turn_idx: int,
    expected_horizon: int = 5,
) -> str:
    """Format the baseline schedule weights as a JSON-ish string for the
    judge prompt. The LLM uses this as a prior and can adjust each weight.
    """
    family = ROLE_TO_FAMILY.get((actor_role or "other").strip(), "cooperative")
    phase = get_phase(turn_idx, expected_horizon)
    base = SCHEDULE[family][phase]
    # Renormalize to sum 1.0 for an accurate prior shown to the judge (the
    # role_consistency injection can push the raw sum slightly above 1.0).
    total = sum(float(v) for v in base.values()) or 1.0
    rows = []
    for dim in ALL_DIMENSIONS:
        w = float(base.get(dim, 0.0)) / total
        rows.append(f'    "{dim}": {w:.2f}')
    return (
        f"actor_role={actor_role}, family={family}, phase={phase} "
        f"(turn_idx={turn_idx} of expected_horizon={expected_horizon}).\n"
        f"Baseline weight prior (sum=1.0):\n"
        + "{\n"
        + ",\n".join(rows)
        + "\n}"
    )


def safe_normalize_weights(
    weights: Dict[str, float],
    per_dim_clip: float = 0.5,
) -> Dict[str, float]:
    """Project a candidate weight dict to a safe simplex.

    Steps:
      * keep only known dimensions, coerce to float >= 0
      * clip each weight to [0, per_dim_clip]
      * if total is non-positive, return empty dict (caller fallbacks)
      * else divide by total so the surviving weights sum to 1
    """
    safe: Dict[str, float] = {}
    if not isinstance(weights, dict):
        return safe
    for dim in ALL_DIMENSIONS:
        v = weights.get(dim)
        if v is None:
            continue
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if x < 0:
            x = 0.0
        if x > per_dim_clip:
            x = per_dim_clip
        if x > 0:
            safe[dim] = x
    total = sum(safe.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in safe.items()}


def mix_weight_dicts(
    llm_weights: Dict[str, float],
    baseline: Dict[str, float],
    mix_ratio: float = 0.5,
) -> Tuple[Dict[str, float], bool]:
    """Linearly blend safe LLM weights with an explicit ``baseline`` dict.

    Returns (final_weights, used_llm_flag). If LLM weights are degenerate
    (empty after safety projection or sum too small), the baseline alone is
    returned and used_llm_flag is False. The final vector is normalized to
    sum 1.0 with a per-dimension cap of 0.5 (overflow redistributed).
    """
    baseline = dict(baseline)
    safe_llm = safe_normalize_weights(llm_weights)
    if not safe_llm or sum(safe_llm.values()) < 0.30:
        return baseline, False

    r = max(0.0, min(1.0, float(mix_ratio)))
    out: Dict[str, float] = {}
    for dim in ALL_DIMENSIONS:
        b = float(baseline.get(dim, 0.0))
        l = float(safe_llm.get(dim, 0.0))
        out[dim] = r * l + (1.0 - r) * b
    total = sum(out.values())
    if total <= 0:
        return baseline, False
    out = {k: v / total for k, v in out.items()}
    # Final cap: no single dim may exceed 0.5. We redistribute overflow to
    # OTHER non-zero dims proportionally. If all other dims are zero (the
    # LLM dumped 100% on one axis), the cap cannot be achieved without
    # synthesizing structure, so we fall back to baseline.
    cap = 0.5
    for _ in range(5):
        over = max(out.values()) - cap
        if over <= 1e-6:
            break
        max_key = max(out, key=out.get)
        out[max_key] = cap
        receivers = {k: v for k, v in out.items() if k != max_key and v > 0}
        rec_sum = sum(receivers.values())
        if rec_sum <= 0:
            return baseline, False
        for k, v in receivers.items():
            out[k] = v + over * (v / rec_sum)
    return out, True


def mix_with_baseline(
    llm_weights: Dict[str, float],
    actor_role: str,
    turn_idx: int,
    expected_horizon: int = 5,
    mix_ratio: float = 0.5,
) -> Tuple[Dict[str, float], bool]:
    """Blend safe LLM weights with the per-turn SCHEDULE baseline."""
    family = ROLE_TO_FAMILY.get((actor_role or "other").strip(), "cooperative")
    phase = get_phase(turn_idx, expected_horizon)
    return mix_weight_dicts(llm_weights, dict(SCHEDULE[family][phase]), mix_ratio)


def aggregate_with_explicit_weights(
    dim_scores: Dict[str, float],
    weights: Dict[str, float],
) -> Tuple[float, Dict[str, float]]:
    """Like aggregate_weighted_score but uses an externally supplied weight
    dict (already normalized). Returns (weighted_score, used_weights)."""
    used: Dict[str, float] = {}
    total = 0.0
    sum_w = 0.0
    for dim, w in weights.items():
        if w <= 0:
            continue
        score = dim_scores.get(dim)
        if score is None:
            continue
        used[dim] = float(w)
        total += float(w) * float(score)
        sum_w += float(w)
    if sum_w <= 0:
        return 0.5, used
    return float(total / sum_w), used


__all__ = [
    "DIMENSIONS_LLM",
    "DIMENSIONS_RULE",
    "ALL_DIMENSIONS",
    "RUBRIC_DIMENSIONS",
    "ROLE_TO_FAMILY",
    "PHASE_NAMES",
    "SCHEDULE",
    "RUBRIC_SCHEDULE",
    "get_phase",
    "compute_weights",
    "compute_rubric_weights",
    "aggregate_weighted_score",
    "aggregate_rubric_weighted_score",
    "aggregate_with_explicit_weights",
    "resolve_actor_role",
    "render_baseline_for_prompt",
    "render_rubric_baseline_for_prompt",
    "safe_normalize_weights",
    "safe_normalize_rubric_weights",
    "mix_with_baseline",
    "mix_rubric_with_baseline",
    "mix_weight_dicts",
    "mix_rubric_weight_dicts",
]
