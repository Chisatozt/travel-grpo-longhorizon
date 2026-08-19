"""Deterministic completion-priority Travel Reward v3 contracts."""

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


# [项目注释] 功能：`_task`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：TravelRewardTask, frozenset。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `TravelRewardTask`；具体值由各分支决定。
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


# [项目注释] 功能：`_score`：计算奖励、指标或聚合统计，供训练、评测或报告使用。 主要协作调用：update, compute_travel_reward, _task, set。
# [项目注释] 输入：**`overrides`。
# [项目注释] 输出：返回运行时计算结果；没有显式返回值的分支返回 `None`。
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


# [项目注释] 功能：`test_completion_dominates_and_gold_remains_one`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。 主要协作调用：_score,
# [项目注释]    approx。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_completion_dominates_and_gold_remains_one():
    gold = _score()
    alternative = _score(answers={"flight": "F2", "hotel": "H2"})
    assert gold["reward_version"] == REWARD_VERSION
    assert gold["terminal_reward"] == pytest.approx(1.0)
    assert gold["completion_rate"] == pytest.approx(1.0)
    assert gold["user_aligned_success"] is True
    assert alternative["completion_rate"] == pytest.approx(1.0)
    assert alternative["terminal_reward"] == pytest.approx(3.392 / 3.4)


# [项目注释] 功能：`test_completion_is_correct_answer_rate_not_submission_rate`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：_score, approx, set。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_completion_is_correct_answer_rate_not_submission_rate():
    wrong = _score(
        answers={"flight": "F99", "hotel": "H99"},
        active_preference_ids=set(),
        searched_aspects=set(),
        steps=2,
        unsearched_answers=2,
        wrong_answers=2,
    )
    partial = _score(
        answers={"flight": "F1", "hotel": "H99"},
        active_preference_ids=set(),
        searched_aspects=set(),
        steps=2,
        unsearched_answers=2,
        wrong_answers=1,
    )
    assert wrong["answer_submission_rate"] == pytest.approx(1.0)
    assert wrong["completion_rate"] == pytest.approx(0.0)
    assert wrong["correct_answer_rate"] == pytest.approx(0.0)
    assert partial["answer_submission_rate"] == pytest.approx(1.0)
    assert partial["completion_rate"] == pytest.approx(0.5)
    assert partial["terminal_reward"] > wrong["terminal_reward"]


# [项目注释] 功能：`test_policy_penalties_are_decomposed_and_bounded`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。 主要协作调用：_score,
# [项目注释]    approx。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_policy_penalties_are_decomposed_and_bounded():
    report = _score(
        invalid_actions=4,
        exact_repeats=4,
        semantic_repeats=4,
        ambiguous_actions=4,
        unsearched_answers=4,
        wrong_answers=4,
        guard_rejections=4,
        blocked_aspects=2,
    )
    assert report["policy_penalty"] == pytest.approx(0.32)
    assert report["penalty_components"]["invalid_action"] == pytest.approx(0.03)
    assert report["penalty_components"]["guard_rejection"] == pytest.approx(0.08)
    assert report["penalty_components"]["blocked_aspect"] == pytest.approx(0.08)


# [项目注释] 功能：`test_error_counts_decrease_reward_until_each_cap`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。 主要协作调用：_score,
# [项目注释]    approx。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_error_counts_decrease_reward_until_each_cap():
    r1 = _score(invalid_actions=1)["terminal_reward"]
    r4 = _score(invalid_actions=4)["terminal_reward"]
    r8 = _score(invalid_actions=8)["terminal_reward"]
    assert r1 > r4
    assert r4 == pytest.approx(r8)


# [项目注释] 功能：`test_passive_preference_coverage_is_positive_and_not_a_penalty`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：_score, set。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_passive_preference_coverage_is_positive_and_not_a_penalty():
    no_passive = _score(active_preference_ids=set(), passive_preference_ids=set())
    passive = _score(active_preference_ids=set(), passive_preference_ids={"P1"})
    assert passive["preference_coverage"] > no_passive["preference_coverage"]
    assert passive["terminal_reward"] > no_passive["terminal_reward"]


# [项目注释] 功能：`test_blocked_aspect_is_not_completion`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。 主要协作调用：_score, approx, set。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_blocked_aspect_is_not_completion():
    report = _score(answers={}, active_preference_ids=set(), searched_aspects=set(), blocked_aspects=1)
    assert report["completion_rate"] == pytest.approx(0.0)
    assert report["blocked_aspects"] == 1


# [项目注释] 功能：`test_public_phase_transition_score_is_vacuously_one_without_opportunities`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：_score, approx。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_public_phase_transition_score_is_vacuously_one_without_opportunities():
    report = _score()
    assert report["phase_transition_score"] == pytest.approx(1.0)
    assert report["phase_transition_breakdown"]["search_required"]["rate"] == pytest.approx(1.0)


# [项目注释] 功能：`test_public_phase_failures_are_small_relative_to_completion`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：_score, approx。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_public_phase_failures_are_small_relative_to_completion():
    report = _score(
        valid_search_required_transitions=0,
        search_required_opportunities=2,
        valid_candidate_answer_transitions=0,
        candidate_answer_opportunities=2,
        guard_rejections=4,
    )
    assert report["phase_transition_score"] == pytest.approx(0.0)
    assert report["completion_rate"] == pytest.approx(1.0)
    assert report["terminal_reward"] > 0.8


# [项目注释] 功能：`test_negative_terminal_rewards_are_smooth_and_distinct`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：tuple, all, len, squash_terminal_reward。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_negative_terminal_rewards_are_smooth_and_distinct():
    values = tuple(squash_terminal_reward(value) for value in (-1.0, -1.1, -1.4))
    assert values[0] > values[1] > values[2]
    assert len(set(values)) == 3
    assert all(-1.0 < value < 0.0 for value in values)


# [项目注释] 功能：`test_search_coverage_distinguishes_progress_and_actor_attempts_affect_efficiency`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：_score, approx, set。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
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


# [项目注释] 功能：`test_infrastructure_invalid_is_zero_not_a_negative_training_example`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：_score。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_infrastructure_invalid_is_zero_not_a_negative_training_example():
    report = _score(reward_valid=False, termination_reason="infrastructure_error")
    assert report["terminal_reward"] == 0.0
    assert report["infrastructure_invalid"] is True


# [项目注释] 功能：`test_reward_trace_remains_raw_diagnostic_only`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：RawRewardTrace, approx。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_reward_trace_remains_raw_diagnostic_only():
    trace = RawRewardTrace()
    for reward in (0.2, 0.2, 0.8, 1.0):
        trace.append(reward)
    assert trace.values == (0.2, 0.2, 0.8, 1.0)
    assert trace.total == pytest.approx(2.2)


@pytest.mark.parametrize("value", [True, math.inf, -math.inf, math.nan, "1"])
# [项目注释] 功能：`test_invalid_raw_rewards_fail`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。 主要协作调用：parametrize, raises,
# [项目注释]    RawRewardTrace。
# [项目注释] 输入：`value`。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_invalid_raw_rewards_fail(value):
    with pytest.raises(UserBenchRewardError):
        RawRewardTrace().append(value)
