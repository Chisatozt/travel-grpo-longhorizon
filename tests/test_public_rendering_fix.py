from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from travel_grpo.envs.public_control import (
    PublicAspectStatus,
    RecoveryMode,
    advance_public_aspect,
    mark_public_preference_complete,
    new_public_control_state,
    reduce_public_feedback,
    render_actor_control_info,
    validate_public_action,
)
from travel_grpo.envs.userbench_tools import UserBenchAction
from travel_grpo.training.recovery_sft import public_state_from_payload

ROOT = Path(__file__).resolve().parents[1]


def _action(choice: str, content: str) -> UserBenchAction:
    return UserBenchAction.from_parameters({"thought": "test", "choice": choice, "content": content})


def test_public_completion_hint_renders_search_required_and_rejects_action() -> None:
    state = mark_public_preference_complete(new_public_control_state("I need a hotel."))
    assert state.phase is RecoveryMode.SEARCH_REQUIRED
    assert validate_public_action(state, _action("action", "ask hotel preference")) == (
        "SEARCH_REQUIRED accepts choice=search only"
    )
    rendered = render_actor_control_info(state)
    assert "Current control state: SEARCH_REQUIRED" in rendered
    assert "Allowed next tool calls: search" in rendered
    assert "Do not call action" in rendered
    for forbidden in ("remaining_preference_ids", "correct_ids", "best_ids", "reward_snapshot"):
        assert forbidden not in rendered


def test_advance_preserves_explicit_answered_switch_note() -> None:
    state = new_public_control_state("I need a hotel and a flight.")
    state = reduce_public_feedback(state, _action("search", "hotel Paris"), "Candidates: H1, H2")
    state = reduce_public_feedback(state, _action("answer", "H1"), "accepted")
    state = advance_public_aspect(state)
    assert state.current_aspect == "flight"
    assert state.last_transition_aspect == "hotel"
    assert state.last_transition_status is PublicAspectStatus.ANSWERED
    rendered = render_actor_control_info(state)
    assert "Transition: SWITCH_ASPECT_REQUIRED" in rendered
    assert "Previous public aspect: hotel is ANSWERED" in rendered
    assert "Continue only with current public aspect: flight" in rendered


def test_advance_preserves_explicit_blocked_switch_note() -> None:
    state = new_public_control_state("I need a hotel and a flight.")
    state = reduce_public_feedback(state, _action("search", "hotel Paris"), "The search backend is experiencing some issues.")
    state = reduce_public_feedback(state, _action("search", "hotel Paris airport"), "The search backend is experiencing some issues again.")
    state = advance_public_aspect(state)
    assert state.blocked_aspects == ("hotel",)
    assert state.current_aspect == "flight"
    rendered = render_actor_control_info(state)
    assert "Transition: SWITCH_ASPECT_REQUIRED" in rendered
    assert "Previous public aspect: hotel is BLOCKED" in rendered
    assert "do not call that aspect again" in rendered


def test_boundary_phase_hint_round_trips_without_hidden_fields() -> None:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "I need a hotel in Paris."},
    ]
    payload = {
        "current_aspect": "hotel",
        "recovery_mode": "ELICITING",
        "fallback_count": 0,
        "visible_option_ids": [],
        "answered_aspects": [],
        "blocked_aspects": [],
        "search_attempts": 0,
        "normal_search_seen": False,
        "consecutive_no_progress": 0,
    }
    state = public_state_from_payload(
        payload, messages, phase_hint="preference_complete_to_search"
    )
    assert state.phase is RecoveryMode.SEARCH_REQUIRED
    assert state.current is not None and state.current.preferences_complete
    assert "remaining_preference_ids" not in json.dumps(state, default=str)


def test_inference_gate_renders_corrected_public_phases() -> None:
    spec = importlib.util.spec_from_file_location(
        "inference_gate", ROOT / "scripts/eval/inference_gate.py"
    )
    assert spec is not None and spec.loader is not None
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    records = gate.load_boundary_records()
    preference_records = gate.choose_probe_records(
        records, boundary_type="preference_complete_to_search", count=24
    )
    assert all(
        gate._public_messages(record, "B")[1]["recovery_mode"] == "SEARCH_REQUIRED"
        for record in preference_records
    )
    second_records = gate.choose_probe_records(
        records, boundary_type="second_fallback", count=24
    )
    for record in second_records[:1]:
        messages, payload = gate._public_messages(record, "B")
        assert "Transition: SWITCH_ASPECT_REQUIRED" in messages[-1]["content"]
        assert payload["last_transition_status"] == "BLOCKED"

def test_switch_note_clears_on_next_public_event() -> None:
    state = new_public_control_state("I need a hotel and a flight.")
    state = reduce_public_feedback(state, _action("search", "hotel Paris"), "Candidates: H1, H2")
    state = reduce_public_feedback(state, _action("answer", "H1"), "accepted")
    state = advance_public_aspect(state)
    assert "Transition: SWITCH_ASPECT_REQUIRED" in render_actor_control_info(state)
    state = reduce_public_feedback(state, _action("action", "flight airline"), "Please state a flight preference.")
    assert state.last_transition_aspect is None
    assert "Transition: SWITCH_ASPECT_REQUIRED" not in render_actor_control_info(state)

