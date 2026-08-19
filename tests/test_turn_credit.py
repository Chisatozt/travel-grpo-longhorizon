"""CPU contract tests for conservative-turn-credit-v2."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from travel_grpo.envs.userbench_context import UserBenchSessionState
from travel_grpo.envs.userbench_tools import UserBenchAction
from travel_grpo.training.grpo.turn_credit import (
    TURN_CREDIT_VERSION,
    AspectCausalTrace,
    TurnCreditConfig,
    TurnCreditError,
    TurnEvent,
    allocate_turn_evidence,
    assistant_turn_spans,
    build_aspect_causal_traces,
    build_turn_credit_trace,
    reshape_batch_advantages,
    reshape_turn_advantages,
)


# [项目注释] 功能：`event`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：TurnEvent。
# [项目注释] 输入：`index`: int；`aspect`: str；`choice`: str | None；**`kwargs`。
# [项目注释] 输出：标注返回 `TurnEvent`；具体值由各分支决定。
def event(
    index: int,
    *,
    aspect: str = "hotel",
    choice: str | None = None,
    **kwargs,
) -> TurnEvent:
    return TurnEvent(
        turn_index=index,
        public_aspect=aspect,
        phase_before="none",
        phase_after="none",
        choice=choice,
        **kwargs,
    )


# [项目注释] 功能：`completed_hotel_events`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：event。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `list[TurnEvent]`；具体值由各分支决定。
def completed_hotel_events() -> list[TurnEvent]:
    return [
        event(0, choice="action", field_resolved=True, reward_new_preference_count=1),
        event(1, choice="action", field_resolved=True, reward_new_preference_count=1),
        event(2, choice="action", no_progress_action=True, semantic_repeat=True),
        event(3, choice="search", normal_candidates_observed=True),
        event(
            4,
            choice="answer",
            answer_id_visible=True,
            reward_correct_answer=True,
        ),
    ]


# [项目注释] 功能：`test_completed_aspect_splits_preference_search_answer_budget`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：completed_hotel_events, build_aspect_causal_traces, allocate_turn_evidence, approx。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def test_completed_aspect_splits_preference_search_answer_budget() -> None:
    events = completed_hotel_events()
    traces = build_aspect_causal_traces(events, ["hotel"])
    assert traces == (
        AspectCausalTrace(
            "hotel",
            preference_turn_indices=(0, 1),
            successful_search_turn_index=3,
            accepted_answer_turn_index=4,
            correctly_completed=True,
        ),
    )
    evidence = allocate_turn_evidence(events, traces, reward_valid=True)
    assert evidence == pytest.approx((0.10, 0.10, -0.10, 0.45, 0.35))
    assert evidence[-1] / evidence[-2] == pytest.approx(0.35 / 0.45)


# [项目注释] 功能：`test_one_preference_turn_receives_the_whole_preference_budget`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：completed_hotel_events, build_aspect_causal_traces, allocate_turn_evidence, approx。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def test_one_preference_turn_receives_the_whole_preference_budget() -> None:
    events = completed_hotel_events()
    events[0].field_resolved = False
    traces = build_aspect_causal_traces(events, ["hotel"])
    evidence = allocate_turn_evidence(events, traces, reward_valid=True)
    assert evidence[0] == 0.0
    assert evidence[1] == pytest.approx(0.20)
    assert evidence[3:] == pytest.approx((0.45, 0.35))


# [项目注释] 功能：`test_successful_aspect_does_not_add_partial_progress_twice`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：completed_hotel_events, build_turn_credit_trace, sum, approx。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def test_successful_aspect_does_not_add_partial_progress_twice() -> None:
    events = completed_hotel_events()
    trace = build_turn_credit_trace(events, ["hotel"], reward_valid=True)
    assert sum(value for value in trace.evidence if value > 0) == pytest.approx(1.0)


# [项目注释] 功能：`test_incomplete_aspect_uses_partial_progress_and_wrong_answer_penalty`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：build_turn_credit_trace, event, approx。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def test_incomplete_aspect_uses_partial_progress_and_wrong_answer_penalty() -> None:
    events = [
        event(0, choice="action", field_resolved=True),
        event(1, choice="action", field_resolved=True),
        event(2, choice="action", field_resolved=True),
        event(3, choice="search", normal_candidates_observed=True),
        event(
            4,
            choice="answer",
            answer_id_visible=True,
            reward_wrong_answer=True,
        ),
    ]
    trace = build_turn_credit_trace(events, ["hotel"], reward_valid=True)
    assert trace.evidence == pytest.approx((0.05, 0.05, 0.0, 0.20, -0.10))


# [项目注释] 功能：`test_blocked_aspect_never_gets_completion_evidence`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：build_turn_credit_trace, event。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def test_blocked_aspect_never_gets_completion_evidence() -> None:
    events = [
        event(0, choice="search", first_fallback=True),
        event(1, choice="search", second_fallback=True),
    ]
    trace = build_turn_credit_trace(
        events, ["hotel"], blocked_aspects=["hotel"], reward_valid=True
    )
    assert trace.aspects[0].blocked is True
    assert trace.aspects[0].correctly_completed is False
    assert trace.evidence == (0.0, 0.0)


# [项目注释] 功能：`test_specific_violation_does_not_stack_semantic_and_guard_penalties`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：build_turn_credit_trace, event。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def test_specific_violation_does_not_stack_semantic_and_guard_penalties() -> None:
    events = [
        event(
            0,
            choice="search",
            exact_query_repeat=True,
            semantic_repeat=True,
            guard_rejected=True,
        )
    ]
    trace = build_turn_credit_trace(events, ["hotel"], reward_valid=True)
    assert trace.evidence == (-0.20,)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"wrong_aspect": True, "guard_rejected": True}, -0.20),
        ({"invisible_answer_id": True, "guard_rejected": True}, -0.20),
        ({"guard_rejected": True}, -0.15),
        ({"malformed_tool_call": True}, -0.15),
        ({"no_tool_output": True}, -0.15),
    ],
)
# [项目注释] 功能：`test_error_root_cause_weights`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。 主要协作调用：parametrize,
# [项目注释]    build_turn_credit_trace, event, approx。
# [项目注释] 输入：`kwargs`；`expected`。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def test_error_root_cause_weights(kwargs, expected) -> None:
    events = [event(0, **kwargs)]
    trace = build_turn_credit_trace(events, ["hotel"], reward_valid=True)
    assert trace.evidence == pytest.approx((expected,))


# [项目注释] 功能：`test_infrastructure_failure_and_invalid_reward_zero_credit`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：completed_hotel_events, event, build_turn_credit_trace, len。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def test_infrastructure_failure_and_invalid_reward_zero_credit() -> None:
    failed = [event(0, choice="search", normal_candidates_observed=True, infrastructure_failure=True)]
    assert build_turn_credit_trace(failed, ["hotel"], reward_valid=True).evidence == (0.0,)
    successful = completed_hotel_events()
    assert build_turn_credit_trace(successful, ["hotel"], reward_valid=False).evidence == (
        0.0,
    ) * len(successful)


# [项目注释] 功能：`test_multi_aspect_chains_never_cross`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：build_turn_credit_trace, event, approx。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def test_multi_aspect_chains_never_cross() -> None:
    events = [
        event(0, aspect="flight", choice="action", field_resolved=True),
        event(1, aspect="flight", choice="search", normal_candidates_observed=True),
        event(2, aspect="flight", choice="answer", answer_id_visible=True, reward_correct_answer=True),
        event(3, aspect="hotel", choice="search", normal_candidates_observed=True),
        event(4, aspect="hotel", choice="answer", answer_id_visible=True, reward_correct_answer=True),
    ]
    trace = build_turn_credit_trace(events, ["flight", "hotel"], reward_valid=True)
    assert trace.evidence == pytest.approx((0.20, 0.45, 0.35, 0.45, 0.35))
    assert trace.aspects[0].accepted_answer_turn_index == 2
    assert trace.aspects[1].accepted_answer_turn_index == 4


# [项目注释] 功能：`test_assistant_turn_spans_are_separated_by_tool_tokens`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：assistant_turn_spans。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def test_assistant_turn_spans_are_separated_by_tool_tokens() -> None:
    assert assistant_turn_spans([1, 1, 0, 0, 1, 0, 1, 1]) == (
        (0, 2),
        (4, 5),
        (6, 8),
    )


# [项目注释] 功能：`test_bounded_reshaping_preserves_sign_and_lambda_zero_parity`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：reshape_turn_advantages, all, approx, TurnCreditConfig。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def test_bounded_reshaping_preserves_sign_and_lambda_zero_parity() -> None:
    evidence = [0.10, -0.10, 0.35, 0.45]
    positive = reshape_turn_advantages(0.4, evidence)
    negative = reshape_turn_advantages(-0.4, evidence)
    assert all(0.36 - 1e-9 <= value <= 0.44 + 1e-9 for value in positive)
    assert all(-0.44 - 1e-9 <= value <= -0.36 + 1e-9 for value in negative)
    assert positive[-1] > positive[0] > positive[1]
    assert negative[-1] == pytest.approx(negative[0])
    assert negative[0] > negative[1]
    parity = reshape_turn_advantages(
        0.4, evidence, config=TurnCreditConfig(mix_lambda=0.0)
    )
    assert parity == pytest.approx((0.4,) * len(evidence))


# [项目注释] 功能：`test_token_weighted_reshaping_exactly_conserves_sequence_advantage`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：reshape_turn_advantages, all, weighted_mean, approx。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def test_token_weighted_reshaping_exactly_conserves_sequence_advantage() -> None:
    evidence = [0.20, 0.45, 0.35, -0.20]
    lengths = [2, 7, 3, 5]
    positive = reshape_turn_advantages(0.4, evidence, turn_token_lengths=lengths)
    negative = reshape_turn_advantages(-0.3, evidence, turn_token_lengths=lengths)

    # [项目注释] 功能：`weighted_mean`：计算奖励、指标或聚合统计，供训练、评测或报告使用。 主要协作调用：sum, zip。
    # [项目注释] 输入：`values`。
    # [项目注释] 输出：返回运行时计算结果；没有显式返回值的分支返回 `None`。
    def weighted_mean(values):
        return sum(n * value for n, value in zip(lengths, values, strict=True)) / sum(lengths)

    assert weighted_mean(positive) == pytest.approx(0.4)
    assert weighted_mean(negative) == pytest.approx(-0.3)
    assert all(value > 0.0 for value in positive)
    assert all(value < 0.0 for value in negative)
    assert positive[1] > positive[2] > positive[0]
    assert negative[3] < negative[0]


# [项目注释] 功能：`test_repeated_same_root_cause_is_blamed_once_per_aspect`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：build_turn_credit_trace, event, approx。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def test_repeated_same_root_cause_is_blamed_once_per_aspect() -> None:
    events = [
        event(0, choice="action", no_progress_action=True),
        event(1, choice="action", no_progress_action=True),
        event(2, aspect="flight", choice="action", no_progress_action=True),
    ]
    trace = build_turn_credit_trace(events, ["hotel", "flight"], reward_valid=True)
    assert trace.evidence == pytest.approx((-0.10, 0.0, -0.10))


# [项目注释] 功能：`test_zero_sequence_advantage_stays_zero`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：reshape_turn_advantages。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def test_zero_sequence_advantage_stays_zero() -> None:
    assert reshape_turn_advantages(0.0, [0.45, -0.20], turn_token_lengths=[4, 2]) == (0.0, 0.0)


# [项目注释] 功能：`test_batch_reshaping_uses_turn_records_and_keeps_tool_tokens_zero`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：importorskip, SimpleNamespace, reshape_batch_advantages, bool。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def test_batch_reshaping_uses_turn_records_and_keeps_tool_tokens_zero() -> None:
    torch = pytest.importorskip("torch")
    batch = {
        "response_mask": torch.tensor([[1, 1, 0, 1, 0, 1, 1]], dtype=torch.float32),
        "advantages": torch.tensor([[0.4, 0.4, 0.0, 0.4, 0.0, 0.4, 0.4]]),
        "returns": torch.tensor([[0.4, 0.4, 0.0, 0.4, 0.0, 0.4, 0.4]]),
    }
    data = SimpleNamespace(
        batch=batch,
        non_tensor_batch={
            "turn_credit": [
                {
                    "version": TURN_CREDIT_VERSION,
                    "mode": "train",
                    "reward_valid": True,
                    "turn_count": 3,
                    "evidence": [0.10, -0.10, 0.45],
                }
            ]
        },
    )
    result = reshape_batch_advantages(
        data,
        {
            "travel_turn_credit": {
                "mode": "train",
                "mix_lambda": 0.5,
                "multiplier_band": 0.2,
            }
        },
    )
    assert result.batch["advantages"][0, 2].item() == 0.0
    assert result.batch["advantages"][0, 4].item() == 0.0
    assert result.batch["advantages"][0, 5].item() > result.batch["advantages"][0, 3].item()
    active = result.batch["response_mask"][0].bool()
    assert result.batch["advantages"][0][active].mean().item() == pytest.approx(0.4)
    assert result.batch["returns"].tolist() == result.batch["advantages"].tolist()


# [项目注释] 功能：`test_batch_turn_span_mismatch_fails_closed`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。 主要协作调用：importorskip,
# [项目注释]    SimpleNamespace, raises, reshape_batch_advantages。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def test_batch_turn_span_mismatch_fails_closed() -> None:
    torch = pytest.importorskip("torch")
    data = SimpleNamespace(
        batch={
            "response_mask": torch.tensor([[1, 0, 1]], dtype=torch.float32),
            "advantages": torch.tensor([[0.2, 0.0, 0.2]]),
            "returns": torch.tensor([[0.2, 0.0, 0.2]]),
        },
        non_tensor_batch={
            "turn_credit": [
                {"version": TURN_CREDIT_VERSION, "evidence": [0.1]}
            ]
        },
    )
    with pytest.raises(TurnCreditError, match="turn/span mismatch"):
        reshape_batch_advantages(
            data, {"travel_turn_credit": {"mode": "train"}}
        )


# [项目注释] 功能：`test_extra_field_contains_no_hidden_ids_or_values`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：build_turn_credit_trace, repr, completed_hotel_events, to_extra_field。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def test_extra_field_contains_no_hidden_ids_or_values() -> None:
    trace = build_turn_credit_trace(
        completed_hotel_events(), ["hotel"], reward_valid=True
    )
    serialized = repr(trace.to_extra_field(mode="shadow"))
    for forbidden in (
        "remaining_preference_ids",
        "correct_ids",
        "best_ids",
        "reward_snapshot",
        "reward delta",
        "hidden preference",
        "H1",
    ):
        assert forbidden not in serialized




# [项目注释] 功能：`test_session_ledger_records_public_guard_root_cause_without_environment`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：UserBenchSessionState, begin_actor_turn, from_parameters, reject_actor_turn。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def test_session_ledger_records_public_guard_root_cause_without_environment() -> None:
    session = UserBenchSessionState(
        "request-credit",
        "task-credit",
        object(),
        public_initial_message="I need a hotel in Boston.",
        turn_credit_mode="shadow",
    )
    session.begin_actor_turn()
    action = UserBenchAction.from_parameters(
        {
            "thought": "retry",
            "choice": "search",
            "content": "Search for a hotel in Boston",
        }
    )
    session.reject_actor_turn(
        reason="search query was already attempted for this public aspect",
        action=action,
        category="public_phase_guard",
    )
    trace = session.finalize_turn_credit({"reward_valid": True})

    assert trace.evidence == (-0.20,)
    assert trace.events[0].exact_query_repeat is True
    assert trace.events[0].public_aspect == "hotel"
    assert session.turn_events[0].reward_new_preference_count == 0


# [项目注释] 功能：`test_session_ledger_off_mode_has_no_runtime_recording`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：UserBenchSessionState, begin_actor_turn, finalize_turn_credit, object。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def test_session_ledger_off_mode_has_no_runtime_recording() -> None:
    session = UserBenchSessionState(
        "request-off",
        "task-off",
        object(),
        public_initial_message="I need a hotel in Boston.",
    )
    session.begin_actor_turn()
    trace = session.finalize_turn_credit({"reward_valid": True})
    assert trace.events == ()
    assert trace.evidence == ()



# [项目注释] 功能：`test_rejected_action_uses_guard_penalty_not_accepted_no_progress_penalty`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：UserBenchSessionState, begin_actor_turn, from_parameters, reject_actor_turn。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def test_rejected_action_uses_guard_penalty_not_accepted_no_progress_penalty() -> None:
    session = UserBenchSessionState(
        "request-guard",
        "task-guard",
        object(),
        public_initial_message="I need a hotel in Boston.",
        turn_credit_mode="shadow",
    )
    session.begin_actor_turn()
    action = UserBenchAction.from_parameters(
        {
            "thought": "ask again",
            "choice": "action",
            "content": "Do you have a hotel price preference?",
        }
    )
    session.reject_actor_turn(
        reason="SEARCH_REQUIRED accepts choice=search only",
        action=action,
        category="public_phase_guard",
    )
    trace = session.finalize_turn_credit({"reward_valid": True})
    assert trace.evidence == (-0.15,)
    assert trace.events[0].guard_rejected is True
    assert trace.events[0].no_progress_action is False
