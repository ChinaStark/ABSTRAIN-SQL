"""Boundary-aware reward for v3 Text-to-SQL RL.

Reward ladder when all three output sections are present:

    1.2  final SQL correct + draft SQL correct
    1.0  final SQL correct + draft SQL wrong
    0.6  ABSTAIN on a BEYOND-capability prompt
    0.4  final SQL wrong + draft SQL correct
    0.3  final SQL wrong + draft SQL wrong
    0.3  ABSTAIN on a WITHIN-capability prompt

Partial format rewards:
    0.2  two required sections
    0.1  one required section
    0.0  no required sections

The small 0.1 draft-correct-on-final-wrong signal prevents an all-wrong
GRPO group from becoming completely zero-variance.
"""
from __future__ import annotations

from dataclasses import dataclass

WITHIN = "within"
BEYOND = "beyond"


@dataclass(frozen=True)
class RewardValues:
    # Format reward: draft_sql / reflection / answer, 0.1 each.
    format_per_tag: float = 0.1
    format_full: float = 0.3

    # Task reward, added only after full format is present.
    sql_correct: float = 0.7
    draft_correct_bonus: float = 0.2
    sql_wrong: float = 0.0
    draft_correct_on_wrong: float = 0.1

    # 0.3 + 0.3 = 0.6 for a wise abstention.
    abstain_beyond: float = 0.3
    # 0.3 + 0.0 = 0.3 for an unnecessary abstention.
    abstain_within: float = 0.0


def classify_probe(exec_scores: list[float], correct_threshold: float = 1.0) -> str:
    """At least one correct probe -> WITHIN; all probes wrong -> BEYOND."""
    if not exec_scores:
        raise ValueError("exec_scores must not be empty")
    return WITHIN if any(s >= correct_threshold for s in exec_scores) else BEYOND


def format_score(
    has_draft: bool,
    has_reflection: bool,
    has_answer: bool,
    values: RewardValues = RewardValues(),
) -> float:
    """Return 0.1 for each required output section that is present."""
    return (
        int(has_draft) * values.format_per_tag
        + int(has_reflection) * values.format_per_tag
        + int(has_answer) * values.format_per_tag
    )


def trajectory_reward(
    *,
    final_exec: float,
    draft_exec: float,
    abstained: bool,
    boundary: str,
    reflection_present: bool,
    has_draft: bool,
    has_answer: bool,
    correct_threshold: float = 1.0,
    values: RewardValues = RewardValues(),
) -> float:
    """Compute the boundary-aware scalar trajectory reward."""
    fmt = format_score(has_draft, reflection_present, has_answer, values)

    # Task reward is gated by complete protocol format.
    if fmt < values.format_full:
        return fmt

    draft_correct = draft_exec >= correct_threshold
    final_correct = final_exec >= correct_threshold

    if final_correct:
        reward = values.format_full + values.sql_correct
        if draft_correct:
            reward += values.draft_correct_bonus
        return reward

    if abstained:
        if boundary == BEYOND:
            return values.format_full + values.abstain_beyond
        if boundary == WITHIN:
            return values.format_full + values.abstain_within
        raise ValueError(f"unknown boundary={boundary!r}")

    reward = values.format_full + values.sql_wrong
    if draft_correct:
        reward += values.draft_correct_on_wrong
    return reward
