"""Deterministic Travel Reward v2 contracts."""

import math

import pytest

from travel_grpo.envs.reward import (
    REWARD_VERSION,
    RawRewardTrace,
    TravelRewardTask,
    UserBenchRewardError,
    compute_travel_reward,
    squash_terminal_reward,
)


def _task() -> TravelRewardTask:
    return TravelRewardTask(
        task_id="task-1",
        aspects=("flight", "hotel"),
        best_ids={"flight": "F1", "hotel": "H1"},
        correct_ids={
            "flight": frozenset({"F1", "F2"}),
            "hotel": frozenset({"H1", "H2"}),
        },
        preference_ids_by_aspect={
            "flight": frozenset({"P1", "P2"}),
            "hotel": frozenset({"P3", "P4"}),
        },
    )


def _score(**overrides):
    values = {
        "task": _task(),
        "answers": {"flight": "F1", "hotel": "H1"},
        "active_preference_ids": {"P1", "P2", "P3", "P4"},
        "passive_preference_ids": set(),
        "searched_aspects": {"flight", "hotel"},
        "steps": 8,
    }
    values.update(overrides)
    return compute_travel_reward(**values)


def test_fully_grounded_gold_is_one_and_correct_alternative_is_point_seventy_four():
    gold = _score()
    alternative = _score(answers={"flight": "F2", "hotel": "H2"})
    assert gold["reward_version"] == REWARD_VERSION
    assert gold["terminal_reward"] == pytest.approx(1.0)
    assert gold["user_aligned_success"] is True
    assert alternative["terminal_reward"] == pytest.approx(0.74)


def test_ungrounded_guess_and_incomplete_trajectory_are_negative():
    guess = _score(
        active_preference_ids=set(),
        searched_aspects=set(),
        steps=2,
        unsearched_answers=2,
    )
    incomplete = _score(answers={}, active_preference_ids=set(), searched_aspects=set(), steps=0)
    assert guess["raw_terminal_reward"] < 0.0
    assert guess["terminal_reward"] < 0.0
    assert -1.0 < incomplete["terminal_reward"] < 0.0
    assert guess["terminal_reward"] > incomplete["terminal_reward"]


def test_policy_penalties_are_decomposed_without_a_total_or_component_cap():
    report = _score(
        invalid_actions=4,
        exact_repeats=4,
        semantic_repeats=4,
        ambiguous_actions=4,
        unsearched_answers=4,
        wrong_answers=4,
    )
    assert report["policy_penalty"] == pytest.approx(2.0)
    assert report["penalty_components"]["invalid_action"] == pytest.approx(0.40)


def test_error_counts_remain_monotonic_after_many_repeated_errors():
    r1 = _score(invalid_actions=1)["terminal_reward"]
    r4 = _score(invalid_actions=4)["terminal_reward"]
    r8 = _score(invalid_actions=8)["terminal_reward"]
    assert r1 > r4 > r8


def test_negative_terminal_rewards_are_smooth_and_distinct():
    values = tuple(squash_terminal_reward(value) for value in (-1.0, -1.1, -1.4))
    assert values[0] > values[1] > values[2]
    assert len(set(values)) == 3
    assert all(-1.0 < value < 0.0 for value in values)


def test_search_coverage_distinguishes_progress_and_actor_attempts_affect_efficiency():
    no_progress = _score(
        answers={}, active_preference_ids=set(), searched_aspects=set(), steps=0
    )
    searched = _score(
        answers={},
        active_preference_ids={"P1", "P2", "P3", "P4"},
        searched_aspects={"flight", "hotel"},
        steps=4,
    )
    efficient = _score(steps=8, actor_attempts=8)
    inefficient = _score(steps=8, actor_attempts=20)
    assert searched["search_coverage"] == pytest.approx(1.0)
    assert searched["terminal_reward"] > no_progress["terminal_reward"]
    assert efficient["effective_steps"] == 8
    assert inefficient["effective_steps"] == 20
    assert inefficient["efficiency"] < efficient["efficiency"]


def test_infrastructure_invalid_is_zero_not_a_negative_training_example():
    report = _score(reward_valid=False, termination_reason="infrastructure_error")
    assert report["terminal_reward"] == 0.0
    assert report["infrastructure_invalid"] is True


def test_reward_trace_remains_raw_diagnostic_only():
    trace = RawRewardTrace()
    for reward in (0.2, 0.2, 0.8, 1.0):
        trace.append(reward)
    assert trace.values == (0.2, 0.2, 0.8, 1.0)
    assert trace.total == pytest.approx(2.2)


@pytest.mark.parametrize("value", [True, math.inf, -math.inf, math.nan, "1"])
def test_invalid_raw_rewards_fail(value):
    with pytest.raises(UserBenchRewardError):
        RawRewardTrace().append(value)
