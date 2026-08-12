"""Deterministic terminal reward for UserBench travel trajectories.

The pinned simulator's step rewards are retained as diagnostics.  GRPO is
trained with the terminal reward defined here, which is computed only from
frozen task labels and an auditable episode evidence ledger.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence, Set as AbstractSet
from dataclasses import dataclass, field
from typing import Any

REWARD_VERSION = "userbench-travel-reward-v3-priority"
LEGACY_REWARD_VERSIONS = frozenset({"userbench-travel-reward-v2"})
SUPPORTED_REWARD_VERSIONS = frozenset({REWARD_VERSION, *LEGACY_REWARD_VERSIONS})
PRIORITY_REWARD_SCALE = 3.4
NEGATIVE_REWARD_TEMPERATURE = 1.5


class UserBenchRewardError(ValueError):
    """Raised when reward inputs violate the deterministic contract."""


@dataclass
class RawRewardTrace:
    """Preserve upstream step rewards for diagnostics only."""

    _values: list[float] = field(default_factory=list, repr=False)

    def append(self, value: float) -> None:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise UserBenchRewardError("UserBench reward must be numeric")
        normalized = float(value)
        if not math.isfinite(normalized):
            raise UserBenchRewardError("UserBench reward must be finite")
        self._values.append(normalized)

    @property
    def values(self) -> tuple[float, ...]:
        return tuple(self._values)

    @property
    def total(self) -> float:
        return float(sum(self._values))


@dataclass(frozen=True)
class TravelRewardTask:
    """Minimal frozen labels needed by the reward (never shown to the Actor)."""

    task_id: str
    aspects: tuple[str, ...]
    best_ids: Mapping[str, str]
    correct_ids: Mapping[str, frozenset[str]]
    preference_ids_by_aspect: Mapping[str, frozenset[str]]
    # Field names are used only by the local phase controller to avoid asking
    # irrelevant questions. Preference values and IDs remain hidden.
    preference_fields_by_aspect: Mapping[str, tuple[str, ...]] = field(
        default_factory=dict
    )

    @classmethod
    def from_upstream(cls, task: Mapping[str, Any]) -> "TravelRewardTask":
        task_id = task.get("id")
        dimensions = task.get("dimensions")
        preferences = task.get("preferences")
        if not isinstance(task_id, str) or not task_id:
            raise UserBenchRewardError("task.id must be a non-empty string")
        if (
            not isinstance(dimensions, Sequence)
            or isinstance(dimensions, (str, bytes))
            or not dimensions
        ):
            raise UserBenchRewardError("task.dimensions must be a non-empty sequence")
        aspects = tuple(str(value) for value in dimensions)
        if len(aspects) != len(set(aspects)):
            raise UserBenchRewardError("task.dimensions must be unique")
        if not isinstance(preferences, Mapping):
            raise UserBenchRewardError("task.preferences must be a mapping")

        best_ids: dict[str, str] = {}
        correct_ids: dict[str, frozenset[str]] = {}
        preference_ids: dict[str, frozenset[str]] = {}
        preference_fields: dict[str, tuple[str, ...]] = {}
        next_preference = 1
        for aspect in aspects:
            data = preferences.get(aspect)
            if not isinstance(data, Mapping):
                raise UserBenchRewardError(f"missing task data for aspect {aspect!r}")
            best = data.get("best_id")
            correct = data.get("correct_ids")
            raw_preferences = data.get("preferences", ())
            if not isinstance(best, str) or not best:
                raise UserBenchRewardError(f"missing best_id for aspect {aspect!r}")
            if not isinstance(correct, Sequence) or isinstance(correct, (str, bytes)):
                raise UserBenchRewardError(f"correct_ids for {aspect!r} must be a sequence")
            accepted = frozenset(str(value) for value in correct) | {best}
            if not isinstance(raw_preferences, Sequence) or isinstance(
                raw_preferences, (str, bytes)
            ):
                raise UserBenchRewardError(
                    f"preferences for {aspect!r} must be a sequence"
                )
            ids = frozenset(
                f"P{value}"
                for value in range(next_preference, next_preference + len(raw_preferences))
            )
            next_preference += len(raw_preferences)
            fields: list[str] = []
            for preference in raw_preferences:
                if (
                    isinstance(preference, Sequence)
                    and not isinstance(preference, (str, bytes))
                    and len(preference) >= 2
                    and str(preference[1]).strip()
                ):
                    value = str(preference[1]).strip()
                    if value not in fields:
                        fields.append(value)
            best_ids[aspect] = best
            correct_ids[aspect] = accepted
            preference_ids[aspect] = ids
            preference_fields[aspect] = tuple(fields)
        return cls(
            task_id,
            aspects,
            best_ids,
            correct_ids,
            preference_ids,
            preference_fields,
        )


@dataclass(frozen=True)
class UserBenchRewardSnapshot:
    """Hidden TravelEnv state sampled by the wrapper after a transition."""

    remaining_preference_ids: frozenset[str]
    active_elicited_count: int
    passive_elicited_count: int
    remaining_search_aspects: frozenset[str]
    choice_initials: frozenset[str]


def _ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator <= 0 else min(1.0, max(0.0, numerator / denominator))


def squash_terminal_reward(
    raw_reward: float,
    *,
    negative_temperature: float = NEGATIVE_REWARD_TEMPERATURE,
) -> float:
    """Keep positive rewards bounded while smoothly compressing negative ones."""

    try:
        normalized_raw = float(raw_reward)
    except (TypeError, ValueError) as exc:
        raise UserBenchRewardError("raw reward must be finite") from exc
    if not math.isfinite(normalized_raw):
        raise UserBenchRewardError("raw reward must be finite")
    try:
        normalized_temperature = float(negative_temperature)
    except (TypeError, ValueError) as exc:
        raise UserBenchRewardError(
            "negative reward temperature must be positive and finite"
        ) from exc
    if (
        not math.isfinite(normalized_temperature)
        or normalized_temperature <= 0.0
    ):
        raise UserBenchRewardError(
            "negative reward temperature must be positive and finite"
        )

    if normalized_raw >= 0.0:
        return min(1.0, normalized_raw)
    return -math.tanh((-normalized_raw) / normalized_temperature)


def scale_priority_reward(
    raw_reward: float,
    *,
    scale: float = PRIORITY_REWARD_SCALE,
) -> float:
    """Map the v3 score to ``[-1, 1]`` without negative-tail saturation."""

    try:
        normalized_raw = float(raw_reward)
        normalized_scale = float(scale)
    except (TypeError, ValueError) as exc:
        raise UserBenchRewardError("priority reward and scale must be finite") from exc
    if not math.isfinite(normalized_raw):
        raise UserBenchRewardError("priority reward must be finite")
    if not math.isfinite(normalized_scale) or normalized_scale <= 0.0:
        raise UserBenchRewardError("priority reward scale must be positive and finite")
    return max(-1.0, min(1.0, normalized_raw / normalized_scale))


def _count(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise UserBenchRewardError(f"{name} must be a non-negative integer")
    return value


def compute_travel_reward(
    *,
    task: TravelRewardTask,
    answers: Mapping[str, str],
    active_preference_ids: AbstractSet[str],
    passive_preference_ids: AbstractSet[str],
    searched_aspects: AbstractSet[str],
    steps: int,
    actor_attempts: int | None = None,
    max_steps: int = 20,
    invalid_actions: int = 0,
    exact_repeats: int = 0,
    semantic_repeats: int = 0,
    ambiguous_actions: int = 0,
    unsearched_answers: int = 0,
    wrong_answers: int = 0,
    parallel_tool_calls: bool = False,
    no_tool_output: bool = False,
    max_steps_reached: bool = False,
    guard_rejections: int = 0,
    blocked_aspects: int = 0,
    valid_search_required_transitions: int = 0,
    search_required_opportunities: int = 0,
    valid_candidate_answer_transitions: int = 0,
    candidate_answer_opportunities: int = 0,
    valid_retry_search_transitions: int = 0,
    retry_search_opportunities: int = 0,
    valid_aspect_switch_transitions: int = 0,
    aspect_switch_opportunities: int = 0,
    reward_valid: bool = True,
    termination_reason: str | None = None,
) -> dict[str, Any]:
    """Compute the completion-priority Travel Reward v3."""

    guard_rejections = _count(guard_rejections, "guard_rejections")
    blocked_aspects = _count(blocked_aspects, "blocked_aspects")
    transition_counts = {
        "search_required": (
            _count(valid_search_required_transitions, "valid_search_required_transitions"),
            _count(search_required_opportunities, "search_required_opportunities"),
        ),
        "candidate_answer": (
            _count(valid_candidate_answer_transitions, "valid_candidate_answer_transitions"),
            _count(candidate_answer_opportunities, "candidate_answer_opportunities"),
        ),
        "retry_search": (
            _count(valid_retry_search_transitions, "valid_retry_search_transitions"),
            _count(retry_search_opportunities, "retry_search_opportunities"),
        ),
        "aspect_switch": (
            _count(valid_aspect_switch_transitions, "valid_aspect_switch_transitions"),
            _count(aspect_switch_opportunities, "aspect_switch_opportunities"),
        ),
    }

    aspects = task.aspects
    all_preferences = set().union(
        *(set(task.preference_ids_by_aspect[aspect]) for aspect in aspects)
    )
    active = set(active_preference_ids) & all_preferences
    passive = (set(passive_preference_ids) & all_preferences) - active
    searched = set(searched_aspects) & set(aspects)

    quality_by_aspect: dict[str, float] = {}
    grounding_by_aspect: dict[str, float] = {}
    grounded_quality_by_aspect: dict[str, float] = {}
    active_coverage_by_aspect: dict[str, float] = {}
    best_by_aspect: dict[str, bool] = {}
    correct_by_aspect: dict[str, bool] = {}
    for aspect in aspects:
        selected = answers.get(aspect)
        best = selected == task.best_ids[aspect]
        correct = isinstance(selected, str) and selected in task.correct_ids[aspect]
        quality = 1.0 if best else 0.8 if correct else 0.0
        aspect_preferences = task.preference_ids_by_aspect[aspect]
        coverage = _ratio(len(active & set(aspect_preferences)), len(aspect_preferences))
        grounding = (0.25 + 0.75 * coverage) if aspect in searched else 0.0
        quality_by_aspect[aspect] = quality
        active_coverage_by_aspect[aspect] = coverage
        grounding_by_aspect[aspect] = grounding
        grounded_quality_by_aspect[aspect] = quality * grounding
        best_by_aspect[aspect] = best
        correct_by_aspect[aspect] = correct

    grounded_quality = sum(grounded_quality_by_aspect.values()) / len(aspects)
    active_coverage = _ratio(len(active), len(all_preferences))
    passive_coverage = _ratio(len(passive), len(all_preferences))
    search_coverage = _ratio(len(searched), len(aspects))
    answer_submission_rate = _ratio(
        sum(aspect in answers for aspect in aspects), len(aspects)
    )
    correct_answer_rate = _ratio(
        sum(correct_by_aspect.values()), len(aspects)
    )
    # Completion means a correct answer, not merely an answer-shaped tool call.
    completion_rate = correct_answer_rate

    environment_steps = max(0, steps)
    normalized_actor_attempts = (
        environment_steps
        if actor_attempts is None
        else max(0, actor_attempts)
    )
    counted_guard_rejections = min(guard_rejections, normalized_actor_attempts)
    accepted_actor_attempts = max(
        0, normalized_actor_attempts - counted_guard_rejections
    )
    effective_steps = max(environment_steps, accepted_actor_attempts) + 0.25 * (
        counted_guard_rejections
    )
    useful_turn_budget = len(all_preferences) + 2 * len(aspects)
    if effective_steps <= useful_turn_budget:
        efficiency = 1.0
    elif useful_turn_budget >= max_steps:
        efficiency = 0.0
    else:
        efficiency = max(
            0.0,
            1.0
            - (effective_steps - useful_turn_budget)
            / (max_steps - useful_turn_budget),
        )

    aspect_count = len(aspects)
    penalty_components = {
        "guard_rejection": 0.08 * _ratio(min(guard_rejections, 4), 4),
        "blocked_aspect": 0.08 * _ratio(min(blocked_aspects, aspect_count), aspect_count),
        "invalid_action": 0.03 * _ratio(min(max(0, invalid_actions), 4), 4),
        "parallel_tool_calls": 0.05 if parallel_tool_calls else 0.0,
        "exact_repeat": 0.02 * _ratio(min(max(0, exact_repeats), 4), 4),
        "semantic_repeat": 0.02 * _ratio(min(max(0, semantic_repeats), 4), 4),
        "ambiguous_action": 0.02 * _ratio(min(max(0, ambiguous_actions), aspect_count), aspect_count),
        "unsearched_answer": 0.03 * _ratio(min(max(0, unsearched_answers), aspect_count), aspect_count),
        "wrong_answer": 0.04 * _ratio(min(max(0, wrong_answers), aspect_count), aspect_count),
        "no_tool_output": 0.02 if no_tool_output else 0.0,
        "max_steps": 0.02 if max_steps_reached else 0.0,
    }
    policy_penalty = sum(penalty_components.values())
    transition_successes = sum(value[0] for value in transition_counts.values())
    transition_opportunities = sum(value[1] for value in transition_counts.values())
    phase_transition_score = (
        1.0
        if transition_opportunities == 0
        else _ratio(min(transition_successes, transition_opportunities), transition_opportunities)
    )
    phase_transition_breakdown = {
        name: {
            "successes": successes,
            "opportunities": opportunities,
            "rate": (
                1.0
                if opportunities == 0
                else _ratio(min(successes, opportunities), opportunities)
            ),
        }
        for name, (successes, opportunities) in transition_counts.items()
    }
    preference_coverage = _ratio(len(active | passive), len(all_preferences))
    answer_quality = _ratio(sum(quality_by_aspect.values()), len(aspects))
    raw_reward = (
        3.00 * completion_rate
        + 0.20 * preference_coverage
        + 0.08 * phase_transition_score
        + 0.06 * search_coverage
        + 0.04 * answer_quality
        + 0.02 * efficiency
        - policy_penalty
    )
    if not math.isfinite(raw_reward):
        raise UserBenchRewardError("raw reward must be finite")
    terminal_reward = scale_priority_reward(raw_reward) if reward_valid else 0.0
    if not math.isfinite(terminal_reward):
        raise UserBenchRewardError("terminal reward must be finite")

    all_answered = answer_submission_rate == 1.0
    gold_itinerary = all_answered and all(best_by_aspect.values())
    correct_itinerary = all_answered and all(correct_by_aspect.values())
    fully_grounded = all(
        grounding_by_aspect[aspect] == 1.0 for aspect in aspects
    )
    return {
        "reward_version": REWARD_VERSION,
        "reward_valid": bool(reward_valid),
        "terminal_reward": terminal_reward,
        "raw_terminal_reward": raw_reward,
        "termination_reason": termination_reason,
        "grounded_quality": grounded_quality,
        "answer_quality": answer_quality,
        "quality_by_aspect": quality_by_aspect,
        "grounding_by_aspect": grounding_by_aspect,
        "grounded_quality_by_aspect": grounded_quality_by_aspect,
        "active_coverage_by_aspect": active_coverage_by_aspect,
        "best_by_aspect": best_by_aspect,
        "correct_by_aspect": correct_by_aspect,
        "completion_rate": completion_rate,
        "correct_answer_rate": correct_answer_rate,
        "answer_submission_rate": answer_submission_rate,
        "active_preference_coverage": active_coverage,
        "passive_preference_coverage": passive_coverage,
        "preference_coverage": preference_coverage,
        "search_coverage": search_coverage,
        "phase_transition_score": phase_transition_score,
        "phase_transition_breakdown": phase_transition_breakdown,
        "efficiency": efficiency,
        "environment_steps": environment_steps,
        "actor_attempts": (
            None if actor_attempts is None else normalized_actor_attempts
        ),
        "effective_steps": effective_steps,
        "accepted_actor_attempts": accepted_actor_attempts,
        "guard_rejections": guard_rejections,
        "guard_rejection_rate": _ratio(guard_rejections, normalized_actor_attempts),
        "blocked_aspects": min(blocked_aspects, aspect_count),
        "policy_penalty": policy_penalty,
        "penalty_components": penalty_components,
        "searched_aspects": sorted(searched),
        "answers": dict(answers),
        "gold_itinerary": gold_itinerary,
        "correct_itinerary": correct_itinerary,
        "fully_grounded": fully_grounded,
        "user_aligned_success": bool(
            reward_valid and correct_itinerary and fully_grounded
        ),
        "infrastructure_invalid": not reward_valid,
    }
