"""CPU tests for the finite public recovery phase guard."""

import asyncio

from travel_grpo.envs.observation import UserBenchObservation, UserBenchStepResult
from travel_grpo.envs.public_control import (
    PublicAspectStatus,
    PublicControlEvent,
    RecoveryMode,
    advance_public_aspect,
    is_substantive_query_change,
    new_public_control_state,
    reduce_public_control_state,
    reduce_public_feedback,
    validate_public_action,
)
from travel_grpo.envs.userbench_context import (
    UserBenchSessionState,
    clear_current_session,
    set_current_session,
)
from travel_grpo.envs.userbench_tools import UserBenchAction
from travel_grpo.training.grpo.adapter.tools import execute_userbench_action


# [项目注释] 功能：`_action`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：from_parameters。
# [项目注释] 输入：`choice`: str；`content`: str。
# [项目注释] 输出：标注返回 `UserBenchAction`；具体值由各分支决定。
def _action(choice: str, content: str) -> UserBenchAction:
    return UserBenchAction.from_parameters(
        {"thought": "public test", "choice": choice, "content": content}
    )


# [项目注释] 功能：`test_normal_eliciting_turn_preserves_progress_and_then_requires_answer`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：new_public_control_state, reduce_public_feedback, _action。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_normal_eliciting_turn_preserves_progress_and_then_requires_answer():
    state = new_public_control_state("I need a hotel.", no_progress_threshold=2)
    state = reduce_public_feedback(
        state,
        _action("action", "hotel name"),
        "Please tell me the hotel preference you want.",
    )
    assert state.phase is RecoveryMode.ELICITING
    assert state.consecutive_no_progress == 0

    state = reduce_public_feedback(
        state,
        _action("search", "hotel Madrid"),
        "Candidates: H1, H2",
    )
    assert state.phase is RecoveryMode.ANSWER_REQUIRED
    assert state.current is not None
    assert state.current.visible_option_ids == {"H1", "H2"}


# [项目注释] 功能：`test_synonymous_no_preference_repeats_force_search_required`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：new_public_control_state, reduce_public_feedback, validate_public_action, _action。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_synonymous_no_preference_repeats_force_search_required():
    state = new_public_control_state("I need a hotel.", no_progress_threshold=2)
    repeated_feedback = "I still need a hotel preference."
    for content in ("hotel name", "please provide the hotel name", "hotel name"):
        state = reduce_public_feedback(
            state,
            _action("action", content),
            repeated_feedback,
        )
    assert state.consecutive_no_progress == 2
    assert state.phase is RecoveryMode.SEARCH_REQUIRED
    assert validate_public_action(state, _action("action", "ask again")) == (
        "SEARCH_REQUIRED accepts choice=search only"
    )


# [项目注释] 功能：`test_threshold_forces_search_and_rejects_action`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：new_public_control_state, _action, reduce_public_control_state, PublicControlEvent。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_threshold_forces_search_and_rejects_action():
    state = new_public_control_state("I need a hotel.", no_progress_threshold=2)
    action = _action("action", "hotel name")
    state = reduce_public_control_state(state, PublicControlEvent(action=action))
    state = reduce_public_control_state(state, PublicControlEvent(action=action))

    assert state.phase is RecoveryMode.SEARCH_REQUIRED
    assert validate_public_action(state, action) == (
        "SEARCH_REQUIRED accepts choice=search only"
    )
    assert validate_public_action(state, _action("search", "hotel Madrid")) is None


# [项目注释] 功能：`test_normal_candidates_force_one_visible_answer_only`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：new_public_control_state, reduce_public_feedback, _action, validate_public_action。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_normal_candidates_force_one_visible_answer_only():
    state = new_public_control_state("I need a hotel.")
    state = reduce_public_feedback(
        state, _action("search", "hotel Madrid"), "Candidates: H1, H2"
    )
    assert state.phase is RecoveryMode.ANSWER_REQUIRED
    assert validate_public_action(state, _action("search", "hotel Madrid")) == (
        "ANSWER_REQUIRED accepts choice=answer only"
    )
    assert validate_public_action(state, _action("answer", "H1,H2")) == (
        "answer must contain exactly one official option ID"
    )
    assert validate_public_action(state, _action("answer", "F1")) == (
        "answer ID is not visible in the current candidate list"
    )
    assert validate_public_action(state, _action("answer", "H2")) is None


# [项目注释] 功能：`test_fallback_retry_is_one_substantive_rewrite_then_blocks`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：new_public_control_state, _action, reduce_public_feedback, validate_public_action。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_fallback_retry_is_one_substantive_rewrite_then_blocks():
    state = new_public_control_state("I need a hotel.")
    first = _action("search", "hotel Madrid")
    state = reduce_public_feedback(
        state, first, "Currently the searching backend is experiencing some issues."
    )
    assert state.phase is RecoveryMode.SEARCH_RETRY_REQUIRED
    assert validate_public_action(state, _action("search", "HOTEL, MADRID")) == (
        "search query was already attempted for this public aspect"
    )
    assert validate_public_action(state, _action("search", "Madrid hotel")) == (
        "retry query must materially change the previous public query"
    )
    revised = _action("search", "hotel Madrid near airport")
    assert validate_public_action(state, revised) is None

    state = reduce_public_feedback(
        state,
        revised,
        "Currently the searching backend is experiencing some issues again.",
    )
    assert state.current is not None
    assert state.current.status is PublicAspectStatus.BLOCKED
    assert state.phase is RecoveryMode.SWITCH_ASPECT_REQUIRED
    assert state.blocked_count == 1
    assert state.answered_count == 0
    assert validate_public_action(state, _action("search", "hotel another query")) == (
        "public aspect 'hotel' is terminal"
    )


# [项目注释] 功能：`test_first_fallback_rewritten_query_can_recover_to_answer_required`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：new_public_control_state, reduce_public_feedback, _action, validate_public_action。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_first_fallback_rewritten_query_can_recover_to_answer_required():
    state = new_public_control_state("I need a hotel.")
    state = reduce_public_feedback(
        state,
        _action("search", "hotel Madrid"),
        "The searching backend is experiencing some issues.",
    )
    revised = _action("search", "hotel Madrid near airport")
    assert validate_public_action(state, revised) is None
    state = reduce_public_feedback(state, revised, "Candidates: H1, H2")
    assert state.phase is RecoveryMode.ANSWER_REQUIRED
    assert state.current is not None
    assert state.current.search_fallbacks == 1
    assert state.current.search_attempts == 2
    assert state.current.visible_option_ids == {"H1", "H2"}


# [项目注释] 功能：`test_blocked_aspect_rejects_old_search_but_accepts_next_aspect_after_advance`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：new_public_control_state, reduce_public_feedback, advance_public_aspect, _action。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_blocked_aspect_rejects_old_search_but_accepts_next_aspect_after_advance():
    state = new_public_control_state("I need a hotel and a flight.")
    state = reduce_public_feedback(
        state,
        _action("search", "hotel Madrid"),
        "The searching backend is experiencing some issues.",
    )
    state = reduce_public_feedback(
        state,
        _action("search", "hotel Madrid near airport"),
        "The searching backend is experiencing some issues again.",
    )
    assert state.current is not None
    assert state.current.status is PublicAspectStatus.BLOCKED
    assert state.answered_aspects == ()
    assert state.blocked_aspects == ("hotel",)
    assert validate_public_action(state, _action("search", "hotel another query")) == (
        "public aspect 'hotel' is terminal"
    )

    state = advance_public_aspect(state)
    assert state.current_aspect == "flight"
    assert validate_public_action(state, _action("search", "flight Madrid")) is None
    state = reduce_public_feedback(
        state,
        _action("search", "flight Madrid"),
        "Candidates: F1",
    )
    assert state.phase is RecoveryMode.ANSWER_REQUIRED
    state = reduce_public_feedback(state, _action("answer", "F1"), "accepted")
    assert state.answered_aspects == ("flight",)
    assert state.blocked_aspects == ("hotel",)


# [项目注释] 功能：`test_query_normalization_ignores_order_but_accepts_new_public_token`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：is_substantive_query_change。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_query_normalization_ignores_order_but_accepts_new_public_token():
    assert not is_substantive_query_change("Hotel, Madrid", "madrid hotel")
    assert not is_substantive_query_change("Hotel Madrid", " hotel   madrid ")
    assert is_substantive_query_change("Hotel Madrid", "Hotel Madrid airport")


# [项目注释] 功能：`test_answered_and_blocked_aspects_advance_and_terminate_separately`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：new_public_control_state, reduce_public_feedback, advance_public_aspect, _action。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_answered_and_blocked_aspects_advance_and_terminate_separately():
    state = new_public_control_state("I need a hotel and a flight.")
    state = reduce_public_feedback(state, _action("search", "hotel"), "H1")
    state = reduce_public_feedback(state, _action("answer", "H1"), "accepted")
    assert state.answered_aspects == ("hotel",)
    assert state.blocked_aspects == ()
    assert state.phase is RecoveryMode.SWITCH_ASPECT_REQUIRED

    state = advance_public_aspect(state)
    assert state.current_aspect == "flight"
    assert state.episode_done is False
    state = reduce_public_feedback(
        state, _action("search", "flight"), "Currently the searching backend is experiencing some issues."
    )
    state = reduce_public_feedback(
        state,
        _action("search", "flight revised query"),
        "Currently the searching backend is experiencing some issues again.",
    )
    assert state.blocked_aspects == ("flight",)
    assert state.answered_aspects == ("hotel",)
    assert state.all_aspects_terminal is True
    assert state.episode_done is True

    state = advance_public_aspect(state)
    assert state.episode_done is True
    assert state.current_aspect is None
    assert state.answered_count == 1
    assert state.blocked_count == 1


# [项目注释] 类型：`_PublicGuardWrapper` 封装相关状态、协议或数据结构。类属性和方法共同维护其不变量。
class _PublicGuardWrapper:
    # [项目注释] 功能：`__init__`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。
    # [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
    # [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
    def __init__(self):
        self.calls = 0
        self.closed = False

    # [项目注释] 功能：`astep`：异步地实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：UserBenchObservation, UserBenchStepResult。
    # [项目注释] 输入：`action`。
    # [项目注释] 输出：返回运行时计算结果；没有显式返回值的分支返回 `None`。
    async def astep(self, action):
        self.calls += 1
        observation = UserBenchObservation("Candidates: H1, H2", 1, False, 0.0, {})
        return UserBenchStepResult("public-task", observation, 0.0, False, False, {})

    # [项目注释] 功能：`reward_snapshot`：计算奖励、指标或聚合统计，供训练、评测或报告使用。
    # [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
    # [项目注释] 输出：返回运行时计算结果；没有显式返回值的分支返回 `None`。
    def reward_snapshot(self):
        return None

    # [项目注释] 功能：`close`：清理运行时资源或恢复边界状态，保证后续调用不会继承脏状态。
    # [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
    # [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
    def close(self):
        self.closed = True


# [项目注释] 功能：`test_session_render_actor_feedback_is_unified_and_idempotent`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：_PublicGuardWrapper, UserBenchSessionState, render_actor_feedback, startswith。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_session_render_actor_feedback_is_unified_and_idempotent():
    wrapper = _PublicGuardWrapper()
    session = UserBenchSessionState(
        "public-request",
        "public-task",
        wrapper,
        public_initial_message="I need a hotel.",
    )
    rendered = session.render_actor_feedback("Candidates: H1")
    assert rendered.startswith("Candidates: H1\n\nPublic control |")
    assert rendered.count("Public control |") == 1
    assert "Current control state: ELICITING" in rendered
    assert session.render_actor_feedback(rendered) == rendered
    assert session.append_recovery_instruction("Candidates: H1") == rendered


# [项目注释] 功能：`test_tool_guard_rejects_answer_required_search_before_environment`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：run, _PublicGuardWrapper, UserBenchSessionState, set_current_session。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_tool_guard_rejects_answer_required_search_before_environment():
    # [项目注释] 功能：`scenario`：异步地实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：_PublicGuardWrapper,
    # [项目注释]    UserBenchSessionState, set_current_session, clear_current_session。
    # [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
    # [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
    async def scenario():
        wrapper = _PublicGuardWrapper()
        session = UserBenchSessionState(
            "public-request",
            "public-task",
            wrapper,
            public_initial_message="I need a hotel.",
        )
        set_current_session(session)
        try:
            first = await execute_userbench_action(
                {"thought": "search", "choice": "search", "content": "hotel"}
            )
            assert wrapper.calls == 1
            assert "Candidates" in first.text
            second = await execute_userbench_action(
                {"thought": "repeat", "choice": "search", "content": "hotel"}
            )
            assert second.metadata["environment_executed"] is False
            assert wrapper.calls == 1
            assert "Current control state: ANSWER_REQUIRED" in second.text
            assert "Allowed next tool calls: answer (one visible option ID)" in second.text
        finally:
            clear_current_session()
            assert wrapper.closed

    asyncio.run(scenario())
