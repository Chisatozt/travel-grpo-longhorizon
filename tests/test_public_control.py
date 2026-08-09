"""CPU tests for the public-only control state boundary."""

import inspect

import pytest

from travel_grpo.envs.public_control import (
    PublicAspectStatus,
    PublicControlEvent,
    PublicObservationKind,
    RecoveryMode,
    advance_public_aspect,
    classify_public_observation,
    extract_public_aspects,
    new_public_control_state,
    public_action_signature,
    public_search_signature,
    public_semantic_signature,
    reduce_public_control_state,
    reduce_public_feedback,
    render_actor_control_info,
)
from travel_grpo.envs.userbench_tools import ActionChoice, UserBenchAction


def _action(choice: str, content: str, thought: str = "public thought") -> UserBenchAction:
    return UserBenchAction.from_parameters(
        {"thought": thought, "choice": choice, "content": content}
    )


def test_public_aspects_only_use_explicit_initial_message_mentions():
    assert extract_public_aspects("Please arrange a hotel, then a flight and a rental car.") == (
        "hotel",
        "flight",
        "rental_car",
    )
    assert extract_public_aspects("I need a trip, but no hotel was mentioned.") == (
        "hotel",
    )
    assert extract_public_aspects("No travel composition was named.") == ()


def test_observation_classifier_uses_text_not_hidden_diagnostics():
    normal = classify_public_observation(
        "Normal results: H1 and H2", choice=ActionChoice.SEARCH
    )
    assert normal.kind is PublicObservationKind.SEARCH_NORMAL
    assert normal.visible_option_ids == {"H1", "H2"}

    fallback = classify_public_observation(
        "The searching backend is experiencing some issues; H1 may be stale.",
        choice="search",
    )
    assert fallback.kind is PublicObservationKind.SEARCH_FALLBACK
    assert fallback.is_fallback
    assert not fallback.is_normal_search

    empty = classify_public_observation("No candidates were returned.", choice="search")
    assert empty.kind is PublicObservationKind.SEARCH_EMPTY
    assert "diagnostics" not in inspect.signature(classify_public_observation).parameters


def test_public_signatures_do_not_use_actor_thought_or_hidden_aspects():
    first = _action("action", "hotel name", thought="one")
    second = _action("action", "hotel name", thought="a different thought")
    assert public_action_signature(first) == public_action_signature(second)
    assert public_search_signature(first) is None
    search = _action("search", "hotel in Madrid")
    assert public_search_signature(search) == "search:hotel in madrid"
    assert public_semantic_signature(first, ("hotel",)) == ("hotel", "name")
    assert public_semantic_signature(_action("action", "flight company"), ("hotel",)) is None


def test_normal_search_records_only_visible_ids_for_public_aspects_and_requires_answer():
    state = new_public_control_state("I need a hotel and a flight.")
    state = reduce_public_feedback(
        state,
        _action("search", "hotel in Madrid"),
        "Candidates: H1, H2, F9.",
    )
    hotel = state.current
    assert hotel is not None
    assert hotel.visible_option_ids == {"H1", "H2"}
    assert hotel.normal_search_seen is True
    assert state.recovery_mode is RecoveryMode.ANSWER_REQUIRED
    assert state.aspects[1].visible_option_ids == frozenset()


def test_visible_wrong_answer_is_publicly_answered_without_correctness_lookup():
    state = new_public_control_state("I need a hotel.")
    state = reduce_public_feedback(
        state,
        _action("search", "hotel in Madrid"),
        "Candidates: H1, H2.",
    )
    state = reduce_public_feedback(state, _action("answer", "H2"), "accepted")
    assert state.current is not None
    assert state.current.status is PublicAspectStatus.ANSWERED
    assert state.current.submitted_answer == "H2"
    assert state.submitted_answer_ids == ("H2",)
    assert state.recovery_mode is RecoveryMode.SWITCH_ASPECT_REQUIRED


def test_unseen_answer_is_recorded_as_actor_input_but_not_marked_answered():
    state = new_public_control_state("I need a hotel.")
    state = reduce_public_feedback(state, _action("answer", "H99"), "not accepted")
    assert state.current is not None
    assert state.current.status is PublicAspectStatus.OPEN
    assert state.submitted_answer_ids == ("H99",)


def test_first_and_second_public_search_fallbacks_are_separate_recovery_signals():
    state = new_public_control_state("I need a hotel.")
    state = reduce_public_feedback(
        state,
        _action("search", "hotel in Madrid"),
        "The searching backend is experiencing some issues.",
    )
    assert state.current is not None
    assert state.current.search_attempts == 1
    assert state.current.search_fallbacks == 1
    assert state.current.status is PublicAspectStatus.OPEN
    assert state.recovery_mode is RecoveryMode.SEARCH_RETRY_REQUIRED

    state = reduce_public_feedback(
        state,
        _action("search", "hotel Madrid with revised query"),
        "The searching backend is experiencing some issues again.",
    )
    assert state.current is not None
    assert state.current.status is PublicAspectStatus.BLOCKED
    assert state.current.search_attempts == 2
    assert state.current.search_fallbacks == 2
    assert state.recovery_mode is RecoveryMode.BLOCKED


def test_no_public_aspect_does_not_invent_a_search_recovery_target():
    state = new_public_control_state("Please help with my trip.", no_progress_threshold=1)
    state = reduce_public_control_state(
        state,
        PublicControlEvent(action=_action("action", "ask for more details")),
    )
    assert state.current_aspect is None
    assert state.recovery_mode is RecoveryMode.NONE


def test_no_progress_threshold_is_public_and_does_not_need_reward_snapshot():
    state = new_public_control_state("I need a hotel.", no_progress_threshold=2)
    action = _action("action", "hotel name")
    state = reduce_public_control_state(state, PublicControlEvent(action=action))
    assert state.consecutive_no_progress == 1
    assert state.recovery_mode is RecoveryMode.NONE
    state = reduce_public_control_state(state, PublicControlEvent(action=action))
    assert state.consecutive_no_progress == 2
    assert state.recovery_mode is RecoveryMode.SEARCH_REQUIRED


def test_public_aspect_advance_follows_initial_message_order_and_terminates():
    state = new_public_control_state("I need a hotel and a flight.")
    state = reduce_public_feedback(state, _action("search", "hotel"), "H1")
    state = reduce_public_feedback(state, _action("answer", "H1"), "accepted")
    state = advance_public_aspect(state)
    assert state.current_aspect == "flight"
    assert state.episode_done is False

    state = reduce_public_feedback(state, _action("search", "flight"), "F1")
    state = reduce_public_feedback(state, _action("answer", "F1"), "accepted")
    state = advance_public_aspect(state)
    assert state.current_aspect is None
    assert state.episode_done is True


def test_renderer_contains_public_evidence_only():
    state = new_public_control_state("I need a hotel.")
    state = reduce_public_feedback(state, _action("search", "hotel"), "H1 and H2")
    rendered = render_actor_control_info(state)
    assert "H1" in rendered and "H2" in rendered
    for secret in (
        "remaining_preference_ids",
        "reward_snapshot",
        "correct_ids",
        "best_ids",
        "ground_truth",
    ):
        assert secret not in rendered


def test_public_reducer_api_has_no_hidden_reward_inputs():
    parameters = set(inspect.signature(reduce_public_control_state).parameters)
    assert parameters == {"state", "event"}
    event_parameters = set(inspect.signature(PublicControlEvent).parameters)
    assert event_parameters == {"action", "observation"}
