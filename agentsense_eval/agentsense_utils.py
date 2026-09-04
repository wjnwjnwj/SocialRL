"""
Data loading and scene expansion utilities for AgentSense evaluation.

Key concepts
------------
- A "scene" in AgentSense is one row of final_data.jsonl with N characters.
- We expand each scene into N "run items", one per character who will be
  played by the policy model.  The other N-1 characters are driven by the
  partner model.
- Each run item is a self-contained dict that carries everything needed for
  one dialogue run + judge evaluation, with no external references.
- Chat history is maintained as a flat list of dicts:
    {"speaker": name, "role_for_policy": "assistant"|"user", "content": text}
  The "role_for_policy" field is always relative to the policy character so
  that we can build an OpenAI-style messages list for any agent on the fly.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional, Set

from agentsense_prompt import AGENT_SYSTEM_PROMPT, STOP_TOKEN


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_agentsense_data(data_path: str, max_items: int = 0) -> List[dict]:
    """Load AgentSense JSONL and return a list of raw scene dicts."""
    scenes: List[dict] = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            scenes.append(json.loads(line))
    if max_items and max_items > 0:
        scenes = scenes[:max_items]
    return scenes


# ---------------------------------------------------------------------------
# Prompt-building helpers
# ---------------------------------------------------------------------------

def _build_scenario_text(scene: dict) -> str:
    """Combine background + description into a single scenario string."""
    bg = scene.get("background", "").strip()
    desc = scene.get("description", "").strip()
    if bg and desc:
        return f"{bg}\n{desc}"
    return bg or desc


def _build_goals_text(goals: List[dict]) -> str:
    """Format a character's goal list into a numbered string."""
    lines = []
    for i, g in enumerate(goals, 1):
        lines.append(f"{i}. {g['goal']}")
    return "\n".join(lines)


def _build_partner_background_text(characters: List[dict], exclude_name: str) -> str:
    """Build the partner_background block listing all characters except the focal one."""
    parts = []
    for c in characters:
        if c["name"] == exclude_name:
            continue
        parts.append(f"[{c['name']}]: {c['profile']}")
    return "\n\n".join(parts)


def _build_context_section(scenario: str, agent_name: str, goals_text: str) -> str:
    """Build the context_section injected into the system prompt."""
    lines = [
        f"Scenario: {scenario}",
        f"{agent_name}'s social goal(s):",
        goals_text,
    ]
    return "\n".join(lines)


def build_agent_system_prompt(
    agent_name: str,
    characters: List[dict],
    scenario: str,
) -> str:
    """Build the full system prompt for one agent in a scene."""
    char_map = {c["name"]: c for c in characters}
    char = char_map[agent_name]

    goals_text = _build_goals_text(char["goals"])
    context_section = _build_context_section(scenario, agent_name, goals_text)
    agent_background = char["profile"]
    partner_background = _build_partner_background_text(characters, exclude_name=agent_name)

    return AGENT_SYSTEM_PROMPT.format(
        agent_name=agent_name,
        context_section=context_section,
        agent_background=agent_background,
        partner_background=partner_background,
        stop_token=STOP_TOKEN,
    )


# ---------------------------------------------------------------------------
# Scene expansion: one scene → N run items
# ---------------------------------------------------------------------------

def expand_scene_to_run_items(scene: dict) -> List[dict]:
    """Expand a scene into N run items, one per policy character.

    Each run item contains:
        sample_idx      : original scene index
        policy_name     : name of the character played by the policy model
        characters      : full list of character dicts (all N)
        scenario        : combined background + description
        speaker_order   : list of names in fixed round-robin order
        max_turns       : N * 10
        policy_goals    : list of goal strings for the policy character
        policy_profile  : profile string for the policy character
        partner_names   : list of names driven by the partner model
    """
    characters = scene["characters"]
    scenario = _build_scenario_text(scene)
    n = len(characters)
    max_turns = n * 10

    run_items: List[dict] = []
    for char in characters:
        policy_name = char["name"]
        partner_names = [c["name"] for c in characters if c["name"] != policy_name]
        # Partners speak first; policy speaks last in each round.
        speaker_order = partner_names + [policy_name]
        run_item = {
            "sample_idx": scene["sample_idx"],
            "policy_name": policy_name,
            "characters": characters,
            "scenario": scenario,
            "speaker_order": speaker_order,
            "max_turns": max_turns,
            "policy_goals": [g["goal"] for g in char["goals"]],
            "policy_profile": char["profile"],
            "partner_names": partner_names,
        }
        run_items.append(run_item)
    return run_items


def expand_all_scenes(scenes: List[dict]) -> List[dict]:
    """Expand all scenes into run items.  Total count = sum of character counts."""
    run_items: List[dict] = []
    for scene in scenes:
        run_items.extend(expand_scene_to_run_items(scene))
    return run_items


# ---------------------------------------------------------------------------
# Chat history management
# ---------------------------------------------------------------------------

def make_empty_history() -> List[dict]:
    return []


def append_turn(
    history: List[dict],
    speaker: str,
    content: str,
) -> None:
    """Append one turn to the shared chat history (in-place)."""
    history.append({"speaker": speaker, "content": content})


def build_messages_for_agent(
    system_prompt: str,
    history: List[dict],
    focal_agent: str,
) -> List[dict]:
    """Convert shared chat history into an OpenAI messages list for one agent.

    Every utterance by the focal agent becomes role=assistant.
    Every utterance by anyone else becomes role=user, with the speaker name
    prepended: "SpeakerName: <text>".

    In multi-character scenes (N≥3) several non-focal speakers may speak
    consecutively before the focal agent replies.  Strict chat-template
    models (e.g. Gemma3) require perfectly alternating user/assistant turns,
    so consecutive user-role turns are merged into a single user message
    separated by newlines.

    The system prompt is prepended as role=system.
    """
    messages = [{"role": "system", "content": system_prompt}]
    for turn in history:
        speaker = turn["speaker"]
        content = turn["content"]
        if speaker == focal_agent:
            messages.append({"role": "assistant", "content": content})
        else:
            user_text = f"{speaker}: {content}"
            # Merge with the previous message if it is already a user turn,
            # preserving strict user/assistant alternation required by some
            # models (e.g. Gemma3).
            if messages and messages[-1]["role"] == "user":
                messages[-1]["content"] += f"\n{user_text}"
            else:
                messages.append({"role": "user", "content": user_text})

    # Some servers reject a messages list with no user turn.
    # Case 1: history is empty → first speaker in the scene, no prior context.
    # Case 2: last turn was assistant → focal agent spoke last, nudge it to continue.
    if len(messages) == 1:
        messages.append({
            "role": "user",
            "content": "The scene has just begun. Please start the conversation in character.",
        })
    elif messages[-1]["role"] != "user":
        messages.append({"role": "user", "content": "Please continue."})
    return messages


def render_history_for_judge(history: List[dict]) -> str:
    """Render chat history as a plain-text dialogue string for the judge."""
    lines = []
    for turn in history:
        lines.append(f"{turn['speaker']}: {turn['content']}")
    return "\n\n".join(lines)


def strip_stop_token(text: str) -> str:
    """Remove [LEAVE] from a reply so it isn't shown in the dialogue."""
    return text.replace(STOP_TOKEN, "").strip()


def has_stop_token(text: str) -> bool:
    return STOP_TOKEN in text


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def load_done_run_ids(results_path: str) -> Set[str]:
    """Return the set of run IDs already written to the results file."""
    done: Set[str] = set()
    if not os.path.exists(results_path):
        return done
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                rid = rec.get("run_id")
                if rid:
                    done.add(str(rid))
            except (json.JSONDecodeError, ValueError):
                pass
    return done


def make_run_id(run_item: dict) -> str:
    """Stable identifier for a run item: <sample_idx>_<policy_name>."""
    return f"{run_item['sample_idx']}_{run_item['policy_name']}"
