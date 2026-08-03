"""Offline wrapper tests using an injected TravelEnv-compatible fake."""

import asyncio

import pytest

from travel_grpo.envs.userbench_interaction import (
    SimulatorRole,
    UserSimulatorRuntime,
    _reset_user_simulator_binding_for_tests,
)
from travel_grpo.envs.userbench_wrapper import (
    UserBenchEnvironmentConfig,
    UserBenchLifecycleError,
    UserBenchWrapper,
)


class FakeTravelEnv:
    def __init__(self, task_id, rewards=(0.2, 1.0)):
        self.task_id = task_id
        self.rewards = rewards
        self.actions = []
        self.closed = False

    def _observation(self, index, reward=0.0, complete=False):
        return {
            "feedback": f"feedback-{index}",
            "step_count": index,
            "episode_complete": complete,
            "last_reward": reward,
            "preference_list": ["secret"],
            "ground_truth": ["hidden"],
            "remaining_best_options": ["hidden"],
        }

    def reset(self, *, seed=None, options=None):
        assert seed == 42
        return self._observation(0), {"task_id": self.task_id}

    def step(self, action_input):
        self.actions.append(action_input)
        index = len(self.actions)
        reward = self.rewards[index - 1]
        done = index == len(self.rewards)
        return (
            self._observation(index, reward, done),
            reward,
            done,
            False,
            {
                "task_id": self.task_id,
                "ground_truth": ["still-hidden"],
            },
        )

    async def step_async(self, action_input):
        await asyncio.sleep(0)
        return self.step(action_input)

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def reset_process_binding():
    _reset_user_simulator_binding_for_tests()
    yield
    _reset_user_simulator_binding_for_tests()


def runtime():
    return UserSimulatorRuntime(
        role=SimulatorRole.TRAIN,
        model="fake-user",
        base_url="http://127.0.0.1:9999/v1",
        api_key="secret",
    )


def build_wrapper(fake):
    return UserBenchWrapper(
        "task-1",
        runtime(),
        environment_factory=lambda *_: fake,
    )


def test_reset_and_sync_steps_project_actor_safe_feedback_only():
    fake = FakeTravelEnv("task-1")
    wrapper = build_wrapper(fake)
    observation = wrapper.reset()
    assert observation.feedback == "feedback-0"
    assert observation.to_tool_text() == "feedback-0"
    assert "preference_list" in observation.diagnostics
    assert "secret" not in observation.to_tool_text()

    first = wrapper.step({"thought": "look", "choice": "search", "content": "Paris"})
    second = wrapper.step({"thought": "done", "choice": "answer", "content": "hotel-1"})
    assert first.reward == 0.2 and not first.done
    assert second.reward == 1.0 and second.done
    assert second.diagnostics["ground_truth"] == ["still-hidden"]
    assert fake.actions == ["[search] Paris", "[answer] hotel-1"]
    with pytest.raises(UserBenchLifecycleError, match="completed"):
        wrapper.step({"thought": "again", "choice": "search", "content": "x"})
    wrapper.close()
    wrapper.close()
    assert fake.closed


def test_async_step_uses_the_async_environment_path():
    async def scenario():
        fake = FakeTravelEnv("task-1", rewards=(0.8,))
        wrapper = build_wrapper(fake)
        wrapper.reset()
        result = await wrapper.astep(
            {"thought": "finish", "choice": "answer", "content": "option"}
        )
        assert result.done
        assert result.observation.to_tool_text() == "feedback-1"
        wrapper.close()

    asyncio.run(scenario())


def test_step_before_reset_and_step_after_close_fail_loudly():
    wrapper = build_wrapper(FakeTravelEnv("task-1"))
    with pytest.raises(UserBenchLifecycleError, match="reset"):
        wrapper.step({"thought": "x", "choice": "search", "content": "x"})
    wrapper.close()
    with pytest.raises(UserBenchLifecycleError, match="closed"):
        wrapper.reset()


@pytest.mark.parametrize(
    "override",
    [
        {"seed": 7},
        {"max_steps": 21},
        {"one_choice_per_aspect": False},
        {"search_correct_reward": 1.0},
        {"normalize_rewards": True},
    ],
)
def test_environment_contract_rejects_unpinned_modes(override):
    with pytest.raises(ValueError):
        UserBenchEnvironmentConfig(**override)
