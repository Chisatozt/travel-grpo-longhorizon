"""Per-task metric projection with no hidden-label artifact fields."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


RESULT_METRIC_KEYS = (
    "micro_avg",
    "micro_max",
    "number_of_1",
    "number_of_08",
    "terminal_reward",
    "gold_itinerary",
    "correct_itinerary",
    "user_aligned_success",
    "completion",
    "active_coverage",
    "passive_coverage",
    "efficiency",
    "policy_penalty",
    "actor_attempts",
    "environment_steps",
    "invalid_actions",
    "exact_repeats",
    "semantic_repeats",
)


def sanitize_reward(report: Mapping[str, Any]) -> dict[str, Any]:
    allowed = (
        "reward_version", "reward_valid", "terminal_reward", "raw_terminal_reward",
        "termination_reason", "grounded_quality", "quality_by_aspect", "completion_rate",
        "active_preference_coverage", "passive_preference_coverage", "efficiency",
        "policy_penalty", "gold_itinerary", "correct_itinerary", "fully_grounded",
        "user_aligned_success", "infrastructure_invalid", "infrastructure_errors",
        "invalid_actions", "exact_repeats", "semantic_repeats", "ambiguous_actions",
        "unsearched_answers", "wrong_answers",
    )
    return {key: report[key] for key in allowed if key in report}


def result_metrics(result: Mapping[str, Any]) -> dict[str, float]:
    if result.get("infrastructure_valid") is not True:
        return {}
    reward = result.get("reward")
    if not isinstance(reward, Mapping):
        return {}
    qualities = reward.get("quality_by_aspect", {})
    if not isinstance(qualities, Mapping):
        qualities = {}
    values = [float(value) for value in qualities.values()]
    return {
        "micro_avg": sum(values) / len(values) if values else 0.0,
        "micro_max": max(values) if values else 0.0,
        "number_of_1": float(sum(value == 1.0 for value in values)),
        "number_of_08": float(sum(value == 0.8 for value in values)),
        "terminal_reward": float(reward.get("terminal_reward", 0.0)),
        "gold_itinerary": float(reward.get("gold_itinerary") is True),
        "correct_itinerary": float(reward.get("correct_itinerary") is True),
        "user_aligned_success": float(reward.get("user_aligned_success") is True),
        "completion": float(reward.get("completion_rate", 0.0)),
        "active_coverage": float(reward.get("active_preference_coverage", 0.0)),
        "passive_coverage": float(reward.get("passive_preference_coverage", 0.0)),
        "efficiency": float(reward.get("efficiency", 0.0)),
        "policy_penalty": float(reward.get("policy_penalty", 0.0)),
        "actor_attempts": float(result.get("actor_attempts", 0)),
        "environment_steps": float(result.get("environment_steps", 0)),
        "invalid_actions": float(reward.get("invalid_actions", 0)),
        "exact_repeats": float(reward.get("exact_repeats", 0)),
        "semantic_repeats": float(reward.get("semantic_repeats", 0)),
    }
