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


def _action(choice: str, content: str) -> UserBenchAction:
    return UserBenchAction.from_parameters(
        {"thought": "public test", "choice": choice, "content": content}
    )


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


def test_query_normalization_ignores_order_but_accepts_new_public_token():
    assert not is_substantive_query_change("Hotel, Madrid", "madrid hotel")
    assert not is_substantive_query_change("Hotel Madrid", " hotel   madrid ")
    assert is_substantive_query_change("Hotel Madrid", "Hotel Madrid airport")


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


class _PublicGuardWrapper:
    def __init__(self):
        self.calls = 0
        self.closed = False

    async def astep(self, action):
        self.calls += 1
        observation = UserBenchObservation("Candidates: H1, H2", 1, False, 0.0, {})
        return UserBenchStepResult("public-task", observation, 0.0, False, False, {})

    def reward_snapshot(self):
        return None

    def close(self):
        self.closed = True


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


def test_tool_guard_rejects_answer_required_search_before_environment():
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
