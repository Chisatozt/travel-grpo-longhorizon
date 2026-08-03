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
