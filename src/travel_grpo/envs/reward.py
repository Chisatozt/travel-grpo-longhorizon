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

REWARD_VERSION = "userbench-travel-reward-v2"
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
    reward_valid: bool = True,
    termination_reason: str | None = None,
) -> dict[str, Any]:
    """Compute Travel Reward v2 and a complete metric decomposition."""

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
    completion_rate = _ratio(sum(aspect in answers for aspect in aspects), len(aspects))

    environment_steps = max(0, steps)
    normalized_actor_attempts = (
        environment_steps
        if actor_attempts is None
        else max(0, actor_attempts)
    )
    effective_steps = max(environment_steps, normalized_actor_attempts)
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

    penalty_components = {
        # Parallel calls are one separate protocol event, not N malformed calls.
        "invalid_action": 0.0
        if parallel_tool_calls
        else max(0, invalid_actions) * 0.10,
        "parallel_tool_calls": 0.25 if parallel_tool_calls else 0.0,
        "exact_repeat": max(0, exact_repeats) * 0.05,
        "semantic_repeat": max(0, semantic_repeats) * 0.05,
        "ambiguous_action": max(0, ambiguous_actions) * 0.05,
        "unsearched_answer": max(0, unsearched_answers) * 0.10,
        "wrong_answer": max(0, wrong_answers) * 0.15,
        "no_tool_output": 0.15 if no_tool_output else 0.0,
        "max_steps": 0.15 if max_steps_reached else 0.0,
    }
    policy_penalty = sum(penalty_components.values())
    raw_reward = (
        0.65 * (2.0 * grounded_quality - 1.0)
        + 0.15 * active_coverage
        + 0.10 * search_coverage
        + 0.10 * efficiency
        - 0.10 * passive_coverage
        - 0.30 * (1.0 - completion_rate)
        - policy_penalty
    )
    if not math.isfinite(raw_reward):
        raise UserBenchRewardError("raw reward must be finite")
    terminal_reward = squash_terminal_reward(raw_reward) if reward_valid else 0.0
    if not math.isfinite(terminal_reward):
        raise UserBenchRewardError("terminal reward must be finite")

    all_answered = completion_rate == 1.0
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
        "quality_by_aspect": quality_by_aspect,
        "grounding_by_aspect": grounding_by_aspect,
        "grounded_quality_by_aspect": grounded_quality_by_aspect,
        "active_coverage_by_aspect": active_coverage_by_aspect,
        "best_by_aspect": best_by_aspect,
        "correct_by_aspect": correct_by_aspect,
        "completion_rate": completion_rate,
        "active_preference_coverage": active_coverage,
        "passive_preference_coverage": passive_coverage,
        "search_coverage": search_coverage,
        "efficiency": efficiency,
        "environment_steps": environment_steps,
        "actor_attempts": (
            None if actor_attempts is None else normalized_actor_attempts
        ),
        "effective_steps": effective_steps,
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
