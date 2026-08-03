"""Deterministic Travel Reward v2 contracts."""

import math

import pytest

from travel_grpo.envs.reward import (
    REWARD_VERSION,
    RawRewardTrace,
    TravelRewardTask,
    UserBenchRewardError,
    compute_travel_reward,
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


def test_fully_grounded_gold_is_one_and_correct_alternative_is_point_seven():
    gold = _score()
    alternative = _score(answers={"flight": "F2", "hotel": "H2"})
    assert gold["reward_version"] == REWARD_VERSION
    assert gold["terminal_reward"] == pytest.approx(1.0)
    assert gold["user_aligned_success"] is True
    assert alternative["terminal_reward"] == pytest.approx(0.7)


def test_ungrounded_guess_and_incomplete_trajectory_are_negative():
    guess = _score(
        active_preference_ids=set(), searched_aspects=set(), steps=2
    )
    incomplete = _score(answers={}, active_preference_ids=set(), searched_aspects=set(), steps=0)
    assert guess["terminal_reward"] == pytest.approx(-0.65)
    assert incomplete["terminal_reward"] == -1.0


def test_policy_penalties_are_decomposed_and_capped():
    report = _score(
        invalid_actions=4,
        exact_repeats=4,
        semantic_repeats=4,
        ambiguous_actions=4,
        unsearched_answers=4,
        wrong_answers=4,
    )
    assert report["policy_penalty"] == 0.75
    assert report["penalty_components"]["invalid_action"] == 0.30


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
