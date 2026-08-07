"""Pinned provenance, simulator isolation, and ContextVar trajectory tests."""

import asyncio

import pytest

from travel_grpo.envs.observation import UserBenchObservation, UserBenchStepResult
from travel_grpo.envs.reward import TravelRewardTask, UserBenchRewardSnapshot
from travel_grpo.envs.userbench_context import (
    PINNED_USERBENCH_COMMIT,
    UserBenchSessionState,
    clear_current_session,
    require_current_session,
    set_current_session,
    validate_embedded_userbench,
)
from travel_grpo.envs.userbench_tools import UserBenchAction
from travel_grpo.envs.userbench_interaction import (
    SimulatorBoundaryError,
    SimulatorRole,
    UserSimulatorRuntime,
    _reset_user_simulator_binding_for_tests,
    bind_user_simulator_process,
)


class CloseOnlyWrapper:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def _stall_state(*, threshold=2, aspects=("hotel",)):
    preferences = {
        aspect: frozenset({f"P{index}"})
        for index, aspect in enumerate(aspects, start=1)
    }
    task = TravelRewardTask(
        task_id="stall-task",
        aspects=tuple(aspects),
        best_ids={aspect: f"{aspect[0].upper()}1" for aspect in aspects},
        correct_ids={
            aspect: frozenset({f"{aspect[0].upper()}1", f"{aspect[0].upper()}2"})
            for aspect in aspects
        },
        preference_ids_by_aspect=preferences,
    )
    before = UserBenchRewardSnapshot(
        frozenset().union(*preferences.values()),
        0,
        0,
        frozenset(aspects),
        frozenset(),
    )
    state = UserBenchSessionState(
        "stall-request",
        "stall-task",
        CloseOnlyWrapper(),
        reward_task=task,
        reward_snapshot=before,
        stall_recovery_enabled=True,
        stall_no_progress_threshold=threshold,
    )
    return state, before


def _step(state, snapshot, *, action=None, feedback="no change", step=1):
    state.record_step(
        UserBenchStepResult(
            state.task_id,
            UserBenchObservation(feedback, step, False, 0.0, {}),
            0.0,
            False,
            False,
            {},
        ),
        action,
        snapshot,
    )


def test_embedded_source_commit_and_license_are_pinned():
    source = validate_embedded_userbench()
    assert source.upstream_commit == PINNED_USERBENCH_COMMIT
    assert source.license == "Apache-2.0"
    assert source.license_file.name == "LICENSE.txt"


def test_simulator_repr_hides_api_key_and_process_rejects_mixing():
    _reset_user_simulator_binding_for_tests()
    grpo = UserSimulatorRuntime(
        SimulatorRole.GRPO, "deepseek-v4-flash", "http://grpo/v1", "top-secret"
    )
    evaluation = UserSimulatorRuntime(
        SimulatorRole.EVAL, "eval-model", "http://eval/v1", "other-secret"
    )
    environment = {}
    assert "top-secret" not in repr(grpo)
    bind_user_simulator_process(grpo, environ=environment)
    bind_user_simulator_process(grpo, environ=environment)
    assert environment["OPENAI_BASE_URL"] == "http://grpo/v1"
    with pytest.raises(SimulatorBoundaryError, match="already bound"):
        bind_user_simulator_process(evaluation, environ=environment)
    _reset_user_simulator_binding_for_tests()


@pytest.mark.parametrize(
    "role,prefix",
    [
        (SimulatorRole.COLLECTION, "COLLECTION_USER_SIM"),
        (SimulatorRole.GRPO, "GRPO_USER_SIM"),
    ],
)
def test_deepseek_simulator_roles_load_from_separate_variables(role, prefix):
    runtime = UserSimulatorRuntime.from_environment(
        role,
        {
            f"{prefix}_MODEL": "deepseek-v4-flash",
            f"{prefix}_BASE_URL": "https://provider.example/v1",
            f"{prefix}_API_KEY": "secret",
        },
    )
    assert runtime.role is role
    assert runtime.model == "deepseek-v4-flash"
    assert runtime.timeout == 60.0
    assert "secret" not in repr(runtime)


def test_grpo_simulator_rejects_a_different_model():
    with pytest.raises(SimulatorBoundaryError, match="must be"):
        UserSimulatorRuntime.from_environment(
            SimulatorRole.GRPO,
            {
                "GRPO_USER_SIM_MODEL": "some-other-model",
                "GRPO_USER_SIM_BASE_URL": "https://provider.example/v1",
                "GRPO_USER_SIM_API_KEY": "secret",
            },
        )


def test_concurrent_contexts_keep_task_and_raw_rewards_isolated():
    async def trajectory(task_id, values):
        wrapper = CloseOnlyWrapper()
        state = UserBenchSessionState(task_id, task_id, wrapper)
        set_current_session(state)
        try:
            for index, value in enumerate(values, start=1):
                await asyncio.sleep(0)
                current = require_current_session()
                current.record_step(
                    UserBenchStepResult(
                        task_id=task_id,
                        observation=UserBenchObservation("ok", index, False, value, {}),
                        reward=value,
                        terminated=index == len(values),
                        truncated=False,
                        diagnostics={},
                    )
                )
            return require_current_session().metrics()
        finally:
            clear_current_session()
            assert wrapper.closed

    async def scenario():
        return await asyncio.gather(
            trajectory("task-a", [0.2, 1.0]),
            trajectory("task-b", [0.8]),
        )

    first, second = asyncio.run(scenario())
    assert first["task_id"] == "task-a"
    assert first["step_rewards"] == [0.2, 1.0]
    assert first["cumulative_reward"] == pytest.approx(1.2)
    assert second["task_id"] == "task-b"
    assert second["cumulative_reward"] == 0.8


def test_complete_fallback_is_degraded_but_still_reward_valid():
    task = TravelRewardTask(
        task_id="task-fallback",
        aspects=("flight",),
        best_ids={"flight": "F1"},
        correct_ids={"flight": frozenset({"F1"})},
        preference_ids_by_aspect={"flight": frozenset({"P1"})},
    )
    before = UserBenchRewardSnapshot(
        frozenset({"P1"}), 0, 0, frozenset({"flight"}), frozenset()
    )
    after = UserBenchRewardSnapshot(
        frozenset(), 1, 0, frozenset(), frozenset()
    )
    state = UserBenchSessionState(
        "request-fallback",
        "task-fallback",
        CloseOnlyWrapper(),
        reward_task=task,
        reward_snapshot=before,
    )
    state.record_step(
        UserBenchStepResult(
            "task-fallback",
            UserBenchObservation("fallback response", 1, False, 0.2, {}),
            0.2,
            False,
            False,
            {"userbench_search_fallbacks": 1},
        ),
        snapshot=after,
    )
    report = state.reward_report()
    assert report["reward_valid"] is True
    assert report["reward_degraded"] is True
    assert report["simulator_fallback_counts"] == {
        "userbench_search_fallbacks": 1
    }
    assert report["infrastructure_errors"] == []


def test_fallback_with_missing_snapshot_is_hard_invalid():
    task = TravelRewardTask(
        task_id="task-invalid",
        aspects=("flight",),
        best_ids={"flight": "F1"},
        correct_ids={"flight": frozenset({"F1"})},
        preference_ids_by_aspect={"flight": frozenset({"P1"})},
    )
    before = UserBenchRewardSnapshot(
        frozenset({"P1"}), 0, 0, frozenset({"flight"}), frozenset()
    )
    state = UserBenchSessionState(
        "request-invalid",
        "task-invalid",
        CloseOnlyWrapper(),
        reward_task=task,
        reward_snapshot=before,
    )
    state.record_step(
        UserBenchStepResult(
            "task-invalid",
            UserBenchObservation("missing evidence", 1, False, 0.0, {}),
            0.0,
            False,
            False,
            {"userbench_search_fallbacks": 1},
        ),
        snapshot=None,
    )
    report = state.reward_report()
    assert report["reward_valid"] is False
    assert report["terminal_reward"] == 0.0
    assert report["reward_degraded"] is False
    assert "missing_reward_evidence" in report["infrastructure_errors"]


def test_progress_from_preference_search_and_answer_resets_streak():
    preference_state, before = _stall_state()
    preference_after = UserBenchRewardSnapshot(
        frozenset(), 1, 0, before.remaining_search_aspects, frozenset()
    )
    _step(
        preference_state,
        preference_after,
        action=UserBenchAction.from_parameters(
            {"thought": "ask", "choice": "action", "content": "hotel name"}
        ),
    )
    assert preference_state.consecutive_no_progress == 0

    search_state, search_before = _stall_state()
    search_after = UserBenchRewardSnapshot(
        search_before.remaining_preference_ids,
        0,
        0,
        frozenset(),
        frozenset(),
    )
    _step(
        search_state,
        search_after,
        action=UserBenchAction.from_parameters(
            {"thought": "search", "choice": "search", "content": "hotel"}
        ),
        feedback="Search results: H1 and H2",
    )
    assert search_state.consecutive_no_progress == 0
    assert search_state.visible_answer_options == {"H1", "H2"}

    answer_state, answer_before = _stall_state()
    answer_after = UserBenchRewardSnapshot(
        answer_before.remaining_preference_ids,
        0,
        0,
        answer_before.remaining_search_aspects,
        frozenset({"H"}),
    )
    _step(
        answer_state,
        answer_after,
        action=UserBenchAction.from_parameters(
            {"thought": "answer", "choice": "answer", "content": "H1"}
        ),
    )
    assert answer_state.answers == {"hotel": "H1"}
    assert answer_state.consecutive_no_progress == 0


def test_no_progress_and_invalid_protocol_events_increment_streak():
    state, before = _stall_state(threshold=3)
    for step in range(1, 3):
        _step(
            state,
            before,
            action=UserBenchAction.from_parameters(
                {"thought": "repeat", "choice": "search", "content": "hotel"}
            ),
            step=step,
        )
    state.record_non_progress("malformed_tool_call")
    assert state.consecutive_no_progress == 3
    assert state.max_consecutive_no_progress == 3
    assert state.terminated is True
    assert state.truncated is False
    assert state.termination_reason == "stalled_no_progress"


def test_stall_without_visible_answer_evidence_hard_cuts_valid_reward():
    state, before = _stall_state(threshold=2)
    _step(state, before, step=1)
    _step(state, before, step=2)
    assert state.terminated is True
    assert state.truncated is False
    assert state.termination_reason == "stalled_no_progress"
    report = state.reward_report()
    assert report["reward_valid"] is True
    assert report["terminal_reward"] != 0.0
    assert report["penalty_components"]["max_steps"] == 0.0


def test_stall_with_visible_options_enters_one_recovery_pending_state():
    state, before = _stall_state(threshold=2)
    searched = UserBenchRewardSnapshot(
        before.remaining_preference_ids,
        0,
        0,
        frozenset(),
        frozenset(),
    )
    _step(
        state,
        searched,
        action=UserBenchAction.from_parameters(
            {"thought": "search", "choice": "search", "content": "hotel"}
        ),
        feedback="H1 H2 H3",
    )
    _step(state, searched, step=2)
    _step(state, searched, step=3)
    assert state.stall_recovery_triggered is True
    assert state.answer_only_pending is True
    assert state.terminated is False
    assert state.visible_answer_options == {"H1", "H2", "H3"}


def test_infrastructure_invalid_does_not_become_stall():
    state, _ = _stall_state(threshold=1)
    _step(state, None)
    assert state.infrastructure_errors == ["missing_reward_evidence"]
    assert state.termination_reason is None
    assert state.stall_recovery_triggered is False


def test_feature_off_keeps_no_progress_control_flow_unchanged():
    state, before = _stall_state(threshold=1)
    state.configure_stall_recovery(enabled=False, threshold=1)
    _step(state, before)
    state.record_non_progress("invalid_tool_call")
    assert state.done is False
    assert state.consecutive_no_progress == 0
    assert state.stall_recovery_triggered is False


def test_second_stall_after_recovery_use_hard_cuts_without_retry():
    state, before = _stall_state(threshold=2)
    state.stall_recovery_used = True
    _step(state, before, step=1)
    _step(state, before, step=2)
    assert state.termination_reason == "stalled_no_progress"
    assert state.stall_hard_truncated is True
