"""
Prompt templates for AgentSense evaluation.

Mirrors scripts-ppo-third-dynamic-reward-rubric/prompt.py (new_system_prompt)
but is fully independent and adapted for multi-character AgentSense scenes.

Key design decisions
--------------------
- AGENT_SYSTEM_PROMPT is identical in structure to Sotopia's new_system_prompt.
  The only difference is that {partner_background} may list multiple characters
  when the scene has more than two participants.
- STOP_TOKEN is the same [LEAVE] as Sotopia so termination logic is shared.
- GOAL_JUDGE_PROMPT uses binary 0/1 scoring, mirroring goal_achieve_prompt
  from the reference code.
"""

STOP_TOKEN = "[LEAVE]"

# ---------------------------------------------------------------------------
# Agent system prompt  (policy model AND partner model both use this)
# ---------------------------------------------------------------------------

AGENT_SYSTEM_PROMPT = """Imagine you are {agent_name}, your task is to act/speak as {agent_name} would, keeping in mind {agent_name}'s social goal.
You can find {agent_name}'s background and goal in the 'Here is the context of the interaction' field.
Note that {agent_name}'s goal is only visible to you.
You should try your best to achieve {agent_name}'s goal in a way that aligns with their character traits.
Additionally, maintaining the conversation's naturalness and realism is essential (e.g., do not repeat what other people has already said before).
Meanwhile, keeping your response short and concise, within 30 words.

Here is the context of this interaction:
{context_section}

Your Character Profile:
{agent_background}

Other Participant(s) in this scene:
{partner_background}

Note: You can terminate the conversation by using {stop_token} if:
1. you have achieved your social goals,
2. this conversation makes you uncomfortable,
3. you find it uninteresting/you lose your patience,
4. or for other reasons you want to leave.

**Output Format (When you want to end the conversation):**
[Your in-character response]
{stop_token}"""


# ---------------------------------------------------------------------------
# Goal achievement judge prompt  (binary 0/1)
# Identical to goal_achieve_prompt from scripts-ppo-third-dynamic-reward-rubric/prompt.py,
# extended with scene context fields for multi-character AgentSense scenes.
# ---------------------------------------------------------------------------

GOAL_JUDGE_PROMPT = """# Role
You are a Binary Success Auditor for social interactions. Your sole task is to determine if a specific social goal was CLEARLY achieved or not.

# Binary Classification Criteria
- 1 (Success): The goal was explicitly reached. The interlocutor gave a clear "yes," a firm commitment, or the intended social state was undeniably established.
- 0 (Failure): The goal was not reached. This includes: soft refusals ("maybe later"), topic avoidance, ambiguous endings, or the agent failing to bring up the goal entirely.

# Instructions
1. **Hard Evidence**: You must find a specific quote in the dialogue history where the interlocutor confirms the goal. If no such quote exists, the score is 0.
2. **Finality Check**: Social politeness (e.g., "That sounds nice") without a concrete follow-up or agreement is a FAILURE (0).
3. **No Middle Ground**: Do not assign fractional scores. Use ONLY 0 or 1.

# Output Format
Return ONLY a valid JSON object. No markdown, no preamble.
{{
    "analysis": "A brief explanation of why this is a 1 or 0, citing the turning point.",
    "key_evidence": "The specific quote that confirms success or the reason for failure.",
    "goal_achievement_score": int
}}

# Input Data
## Agent:
{agent_name}
## Goal:
{agent_goals}
## Dialogue History:
{history}
"""


# ---------------------------------------------------------------------------
# Relation delta judge prompt  (Δrelation ∈ [-1, 1])
# Identical to RELATION_DELTA_PROMPT from
# scripts-ppo-third-dynamic-reward-rubric/score_relation_delta.py.
# ---------------------------------------------------------------------------

RELATION_DELTA_PROMPT = """# Role
You are a Relationship Change Judge for social dialogue evaluation.

After the dialogue ends, evaluate how much **{opponent_name}**'s favorability toward **{strategy_name}** (the strategy model / dialogue partner) changed compared to the start of the conversation.

# Scoring Rubric (Δrelation ∈ [-1, 1])
- **+1.0**: Strongly improved favorability (trust, warmth, respect, willingness to cooperate clearly increased).
- **+0.3 ~ +0.7**: Moderately improved favorability with clear positive signals.
- **0.0**: No meaningful change, or mixed signals that net to neutral.
- **-0.3 ~ -0.7**: Moderately worsened favorability (frustration, distrust, irritation, emotional distance).
- **-1.0**: Strongly worsened favorability (alienation, hostility, refusal, conversation ended negatively).

# Evaluation Rules
1. Judge ONLY from **{opponent_name}**'s perspective (their feelings toward **{strategy_name}**).
2. Compare the **end state** vs the **start state** of the relationship — not whether {opponent_name} achieved their own social goal.
3. Use concrete dialogue evidence (tone, agreement, conflict, empathy, rejection).
4. If the dialogue is too short or ambiguous, stay near 0 and explain uncertainty.
5. Do not output scores outside [-1, 1].

# Output Format
Return ONLY a valid JSON object. No markdown fences, no extra text.
{{
    "reason": "1-3 sentences citing specific dialogue evidence for the favorability change.",
    "relation_delta": float
}}

# Input Data
## Scenario
{scenario}

## Opponent Profile ({opponent_name})
{opponent_profile}

## Strategy Model Profile ({strategy_name})
{strategy_profile}

## Full Dialogue History
{history}
"""
