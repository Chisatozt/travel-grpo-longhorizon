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
    "answer_submission_rate",
    "active_coverage",
    "passive_coverage",
    "efficiency",
    "policy_penalty",
    "actor_attempts",
    "environment_steps",
    "invalid_actions",
    "exact_repeats",
    "semantic_repeats",
    "answer_quality",
    "preference_coverage",
    "phase_transition_score",
    "guard_rejection_rate",
    "blocked_aspects",
)


# [项目注释] 功能：`sanitize_reward`：计算奖励、指标或聚合统计，供训练、评测或报告使用。
# [项目注释] 输入：`report`: Mapping[str, Any]。
# [项目注释] 输出：标注返回 `dict[str, Any]`；具体值由各分支决定。
def sanitize_reward(report: Mapping[str, Any]) -> dict[str, Any]:
    allowed = (
        "reward_version", "reward_valid", "terminal_reward", "raw_terminal_reward",
        "termination_reason", "grounded_quality", "quality_by_aspect", "completion_rate",
        "correct_answer_rate", "answer_submission_rate",
        "active_preference_coverage", "passive_preference_coverage", "search_coverage",
        "efficiency", "environment_steps", "actor_attempts", "effective_steps",
        "answer_quality", "preference_coverage", "phase_transition_score",
        "phase_transition_breakdown", "accepted_actor_attempts",
        "guard_rejections", "guard_rejection_rate", "blocked_aspects",
        "policy_penalty", "gold_itinerary", "correct_itinerary", "fully_grounded",
        "user_aligned_success", "infrastructure_invalid", "infrastructure_errors",
        "reward_degraded", "simulator_fallback_counts",
        "invalid_actions", "exact_repeats", "semantic_repeats", "ambiguous_actions",
        "unsearched_answers", "wrong_answers",
    )
    return {key: report[key] for key in allowed if key in report}


# [项目注释] 功能：`result_metrics`：计算奖励、指标或聚合统计，供训练、评测或报告使用。 主要协作调用：min, isinstance, float, values。
# [项目注释] 输入：`result`: Mapping[str, Any]。
# [项目注释] 输出：标注返回 `dict[str, float]`；具体值由各分支决定。
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
    default_answer_quality = sum(values) / len(values) if values else 0.0
    answer_quality_value = reward.get("answer_quality")
    answer_quality = (
        float(answer_quality_value)
        if isinstance(answer_quality_value, (int, float))
        and not isinstance(answer_quality_value, bool)
        else default_answer_quality
    )
    default_preference_coverage = min(
        1.0,
        float(reward.get("active_preference_coverage", 0.0))
        + float(reward.get("passive_preference_coverage", 0.0)),
    )
    preference_value = reward.get("preference_coverage")
    preference_coverage = (
        float(preference_value)
        if isinstance(preference_value, (int, float))
        and not isinstance(preference_value, bool)
        else default_preference_coverage
    )
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
        "answer_submission_rate": float(
            reward.get("answer_submission_rate", reward.get("completion_rate", 0.0))
            or 0.0
        ),
        "active_coverage": float(reward.get("active_preference_coverage", 0.0)),
        "passive_coverage": float(reward.get("passive_preference_coverage", 0.0)),
        "efficiency": float(reward.get("efficiency", 0.0)),
        "policy_penalty": float(reward.get("policy_penalty", 0.0)),
        "actor_attempts": float(result.get("actor_attempts", 0)),
        "environment_steps": float(result.get("environment_steps", 0)),
        "invalid_actions": float(reward.get("invalid_actions", 0)),
        "exact_repeats": float(reward.get("exact_repeats", 0)),
        "semantic_repeats": float(reward.get("semantic_repeats", 0)),
        "answer_quality": answer_quality,
        "preference_coverage": preference_coverage,
        "phase_transition_score": (
            float(reward["phase_transition_score"])
            if isinstance(reward.get("phase_transition_score"), (int, float))
            and not isinstance(reward.get("phase_transition_score"), bool)
            else 1.0
        ),
        "guard_rejection_rate": float(reward.get("guard_rejection_rate", 0.0)),
        "blocked_aspects": float(reward.get("blocked_aspects", 0.0)),
    }
