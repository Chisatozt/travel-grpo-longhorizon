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


# [项目注释] 类型：`FakeTravelEnv` 封装相关状态、协议或数据结构。类属性和方法共同维护其不变量。
class FakeTravelEnv:
    # [项目注释] 功能：`__init__`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。
    # [项目注释] 输入：`task_id`；`rewards`。
    # [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
    def __init__(self, task_id, rewards=(0.2, 1.0)):
        self.task_id = task_id
        self.rewards = rewards
        self.actions = []
        self.closed = False

    # [项目注释] 功能：`_observation`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。
    # [项目注释] 输入：`index`；`reward`；`complete`。
    # [项目注释] 输出：返回运行时计算结果；没有显式返回值的分支返回 `None`。
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

    # [项目注释] 功能：`reset`：清理运行时资源或恢复边界状态，保证后续调用不会继承脏状态。 主要协作调用：_observation。
    # [项目注释] 输入：`seed`；`options`。
    # [项目注释] 输出：返回运行时计算结果；没有显式返回值的分支返回 `None`。
    def reset(self, *, seed=None, options=None):
        assert seed == 42
        return self._observation(0), {"task_id": self.task_id}

    # [项目注释] 功能：`step`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：len, _observation。
    # [项目注释] 输入：`action_input`。
    # [项目注释] 输出：返回运行时计算结果；没有显式返回值的分支返回 `None`。
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

    # [项目注释] 功能：`step_async`：异步地实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：step, sleep。
    # [项目注释] 输入：`action_input`。
    # [项目注释] 输出：返回运行时计算结果；没有显式返回值的分支返回 `None`。
    async def step_async(self, action_input):
        await asyncio.sleep(0)
        return self.step(action_input)

    # [项目注释] 功能：`close`：清理运行时资源或恢复边界状态，保证后续调用不会继承脏状态。
    # [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
    # [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
# [项目注释] 功能：`reset_process_binding`：清理运行时资源或恢复边界状态，保证后续调用不会继承脏状态。 主要协作调用：fixture,
# [项目注释]    _reset_user_simulator_binding_for_tests。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：生成器/迭代器，逐步产出中间值。
def reset_process_binding():
    _reset_user_simulator_binding_for_tests()
    yield
    _reset_user_simulator_binding_for_tests()


# [项目注释] 功能：`runtime`：编排一个训练、采集、评测或 replay 流程，并汇总其结果。 主要协作调用：UserSimulatorRuntime。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：返回运行时计算结果；没有显式返回值的分支返回 `None`。
def runtime():
    return UserSimulatorRuntime(
        role=SimulatorRole.GRPO,
        model="deepseek-v4-flash",
        base_url="http://127.0.0.1:9999/v1",
        api_key="secret",
    )


# [项目注释] 功能：`build_wrapper`：根据输入配置和中间状态构建或生成新的项目产物。 主要协作调用：UserBenchWrapper, runtime。
# [项目注释] 输入：`fake`；`config`。
# [项目注释] 输出：返回运行时计算结果；没有显式返回值的分支返回 `None`。
def build_wrapper(fake, config=None):
    return UserBenchWrapper(
        "task-1",
        runtime(),
        config,
        environment_factory=lambda *_: fake,
    )


# [项目注释] 功能：`test_reset_and_sync_steps_project_actor_safe_feedback_only`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：FakeTravelEnv, build_wrapper, reset, step。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
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


# [项目注释] 功能：`test_async_step_uses_the_async_environment_path`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。 主要协作调用：run,
# [项目注释]    FakeTravelEnv, build_wrapper, reset。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_async_step_uses_the_async_environment_path():
    # [项目注释] 功能：`scenario`：异步地实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：FakeTravelEnv, build_wrapper, reset,
    # [项目注释]    close。
    # [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
    # [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
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


# [项目注释] 功能：`test_async_reset_uses_non_blocking_wrapper_entrypoint`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。 主要协作调用：run,
# [项目注释]    FakeTravelEnv, build_wrapper, close。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_async_reset_uses_non_blocking_wrapper_entrypoint():
    # [项目注释] 功能：`scenario`：异步地实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：FakeTravelEnv, build_wrapper, close,
    # [项目注释]    areset。
    # [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
    # [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
    async def scenario():
        fake = FakeTravelEnv("task-1", rewards=(0.8,))
        wrapper = build_wrapper(fake)
        observation = await wrapper.areset()
        assert observation.to_tool_text() == "feedback-0"
        wrapper.close()

    asyncio.run(scenario())


# [项目注释] 功能：`test_async_step_reconciles_pinned_one_choice_termination`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：run, AsyncOneChoiceEnv, build_wrapper, reset。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_async_step_reconciles_pinned_one_choice_termination():
    # [项目注释] 类型：`AsyncOneChoiceEnv` 封装相关状态、协议或数据结构。类属性和方法共同维护其不变量。
    class AsyncOneChoiceEnv(FakeTravelEnv):
        current_task = {"dimensions": ["hotel", "restaurant"]}
        state_list = {"choice_initials": {"H", "R"}}

        # [项目注释] 功能：`step_async`：异步地实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：_observation, sleep。
        # [项目注释] 输入：`action_input`。
        # [项目注释] 输出：返回运行时计算结果；没有显式返回值的分支返回 `None`。
        async def step_async(self, action_input):
            await asyncio.sleep(0)
            observation = self._observation(1, 1.0, complete=False)
            return observation, 1.0, False, False, {"task_id": self.task_id}

    # [项目注释] 功能：`scenario`：异步地实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：AsyncOneChoiceEnv, build_wrapper,
    # [项目注释]    reset, close。
    # [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
    # [项目注释] 输出：返回运行时计算结果；没有显式返回值的分支返回 `None`。
    async def scenario():
        fake = AsyncOneChoiceEnv("task-1", rewards=(1.0,))
        wrapper = build_wrapper(fake)
        wrapper.reset()
        result = await wrapper.astep(
            {"thought": "finish", "choice": "answer", "content": "option"}
        )
        wrapper.close()
        return result

    result = asyncio.run(scenario())
    assert result.terminated is True
    assert result.diagnostics["wrapper_async_termination_reconciled"] == 1


# [项目注释] 功能：`test_collection_mode_captures_upstream_fallback_without_raw_stdout`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：run, PrintingTravelEnv, build_wrapper, reset。
# [项目注释] 输入：`capsys`。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_collection_mode_captures_upstream_fallback_without_raw_stdout(capsys):
    # [项目注释] 类型：`PrintingTravelEnv` 封装相关状态、协议或数据结构。类属性和方法共同维护其不变量。
    class PrintingTravelEnv(FakeTravelEnv):
        # [项目注释] 功能：`step_async`：异步地实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：print, step_async, super。
        # [项目注释] 输入：`action_input`。
        # [项目注释] 输出：返回运行时计算结果；没有显式返回值的分支返回 `None`。
        async def step_async(self, action_input):
            print(
                "[TravelGym - Async Judging Conversation] Invalid judgment: None; "
                "By default turn to normal conversation"
            )
            return await super().step_async(action_input)

    # [项目注释] 功能：`scenario`：异步地实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：PrintingTravelEnv, build_wrapper,
    # [项目注释]    reset, close。
    # [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
    # [项目注释] 输出：返回运行时计算结果；没有显式返回值的分支返回 `None`。
    async def scenario():
        fake = PrintingTravelEnv("task-1", rewards=(0.0,))
        wrapper = build_wrapper(
            fake, UserBenchEnvironmentConfig(capture_upstream_diagnostics=True)
        )
        wrapper.reset()
        result = await wrapper.astep(
            {"thought": "ask", "choice": "action", "content": "Kitchen?"}
        )
        wrapper.close()
        return result

    result = asyncio.run(scenario())
    assert result.diagnostics["userbench_judgment_fallbacks"] == 1
    assert "Invalid judgment" not in capsys.readouterr().out


# [项目注释] 功能：`test_step_before_reset_and_step_after_close_fail_loudly`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：build_wrapper, close, FakeTravelEnv, raises。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
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
# [项目注释] 功能：`test_environment_contract_rejects_unpinned_modes`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：parametrize, raises, UserBenchEnvironmentConfig。
# [项目注释] 输入：`override`。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_environment_contract_rejects_unpinned_modes(override):
    with pytest.raises(ValueError):
        UserBenchEnvironmentConfig(**override)
