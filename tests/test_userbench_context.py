"""Pinned provenance, simulator isolation, and ContextVar trajectory tests."""

import asyncio

import pytest

from travel_grpo.envs.observation import UserBenchObservation, UserBenchStepResult
from travel_grpo.envs.userbench_context import (
    PINNED_USERBENCH_COMMIT,
    UserBenchSessionState,
    clear_current_session,
    require_current_session,
    set_current_session,
    validate_embedded_userbench,
)
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


def test_embedded_source_commit_and_license_are_pinned():
    source = validate_embedded_userbench()
    assert source.upstream_commit == PINNED_USERBENCH_COMMIT
    assert source.license == "Apache-2.0"
    assert source.license_file.name == "LICENSE.txt"


def test_simulator_repr_hides_api_key_and_process_rejects_mixing():
    _reset_user_simulator_binding_for_tests()
    train = UserSimulatorRuntime(
        SimulatorRole.TRAIN, "train-model", "http://train/v1", "top-secret"
    )
    evaluation = UserSimulatorRuntime(
        SimulatorRole.EVAL, "eval-model", "http://eval/v1", "other-secret"
    )
    environment = {}
    assert "top-secret" not in repr(train)
    bind_user_simulator_process(train, environ=environment)
    bind_user_simulator_process(train, environ=environment)
    assert environment["OPENAI_BASE_URL"] == "http://train/v1"
    with pytest.raises(SimulatorBoundaryError, match="already bound"):
        bind_user_simulator_process(evaluation, environ=environment)
    _reset_user_simulator_binding_for_tests()


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
