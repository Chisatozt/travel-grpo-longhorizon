# [项目注释] 模块：测试模块，负责验证 test_evaluation_rollout 的行为契约。
# [项目注释] 该文件的公共边界、输入输出和调用关系由下方实现及架构文档共同定义。

from __future__ import annotations

import asyncio
import json

from travel_grpo.envs.observation import UserBenchObservation, UserBenchStepResult
from travel_grpo.envs.userbench_interaction import (
    DEEPSEEK_V4_FLASH_MODEL,
    SimulatorRole,
    UserSimulatorRuntime,
)
from travel_grpo.envs.userbench_tools import UserBenchAction
from travel_grpo.evaluation.rollout import rollout_task
from travel_grpo.models.openai_compatible import TeacherToolCall


# [项目注释] 类型：`_FakeWrapper` 封装相关状态、协议或数据结构。类属性和方法共同维护其不变量。
class _FakeWrapper:
    instances: list["_FakeWrapper"] = []

    # [项目注释] 功能：`__init__`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。
    # [项目注释] 输入：`task_id`；`simulator`；`config`；`source_root`。
    # [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
    def __init__(self, task_id, simulator, config, *, source_root=None):
        self.task_id = task_id
        self.calls = 0
        self.actions = []
        self.closed = False
        self.__class__.instances.append(self)

    # [项目注释] 功能：`areset`：异步地清理运行时资源或恢复边界状态，保证后续调用不会继承脏状态。 主要协作调用：UserBenchObservation。
    # [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
    # [项目注释] 输出：返回运行时计算结果；没有显式返回值的分支返回 `None`。
    async def areset(self):
        return UserBenchObservation(
            "Initial public travel request.", 0, False, 0.0, {}
        )

    # [项目注释] 功能：`reward_task`：计算奖励、指标或聚合统计，供训练、评测或报告使用。
    # [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
    # [项目注释] 输出：返回运行时计算结果；没有显式返回值的分支返回 `None`。
    def reward_task(self):
        return None

    # [项目注释] 功能：`reward_snapshot`：计算奖励、指标或聚合统计，供训练、评测或报告使用。
    # [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
    # [项目注释] 输出：返回运行时计算结果；没有显式返回值的分支返回 `None`。
    def reward_snapshot(self):
        return None

    # [项目注释] 功能：`astep`：异步地实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：UserBenchStepResult, UserBenchObservation。
    # [项目注释] 输入：`action`。
    # [项目注释] 输出：返回运行时计算结果；没有显式返回值的分支返回 `None`。
    async def astep(self, action):
        self.calls += 1
        self.actions.append(action)
        if action.choice.value == "search":
            feedback = "Here are all the options: H1, H2"
        else:
            feedback = "Please provide the hotel preference."
        return UserBenchStepResult(
            self.task_id,
            UserBenchObservation(feedback, self.calls, False, 0.0, {}),
            0.0,
            False,
            False,
            {},
        )

    # [项目注释] 功能：`close`：清理运行时资源或恢复边界状态，保证后续调用不会继承脏状态。
    # [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
    # [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
    def close(self):
        self.closed = True


# [项目注释] 类型：`_FakeActor` 封装相关状态、协议或数据结构。类属性和方法共同维护其不变量。
class _FakeActor:
    # [项目注释] 功能：`__init__`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。
    # [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
    # [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
    def __init__(self):
        self.messages = []
        self.calls = 0
        self._actions = [
            {"thought": "ask", "choice": "action", "content": "hotel preference"},
            {"thought": "search", "choice": "search", "content": "hotel Paris"},
            {"thought": "wrong phase", "choice": "action", "content": "hotel rating"},
            {"thought": "answer", "choice": "answer", "content": "H1"},
        ]

    # [项目注释] 功能：`generate_action`：异步地根据输入配置和中间状态构建或生成新的项目产物。 主要协作调用：TeacherToolCall, min, dict,
    # [项目注释]    from_parameters。
    # [项目注释] 输入：`messages`。
    # [项目注释] 输出：返回运行时计算结果；没有显式返回值的分支返回 `None`。
    async def generate_action(self, messages):
        self.messages.append([dict(message) for message in messages])
        parameters = self._actions[min(self.calls, len(self._actions) - 1)]
        self.calls += 1
        return TeacherToolCall(
            call_id=f"fake-{self.calls}",
            action=UserBenchAction.from_parameters(parameters),
        )


SIMULATOR = UserSimulatorRuntime(
    role=SimulatorRole.EVAL,
    model=DEEPSEEK_V4_FLASH_MODEL,
    base_url="http://eval.invalid/v1",
    api_key="test-key",
)
TASK = {
    "task_id": "task-1",
    "composition": "22",
    "prompt": [
        {"role": "system", "content": "You are a travel agent."},
        {"role": "user", "content": "I need a hotel in Paris."},
    ],
}


# [项目注释] 功能：`test_guard_rejects_before_simulator_and_renders_public_feedback`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：_FakeActor, clear, run, dumps。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_guard_rejects_before_simulator_and_renders_public_feedback():
    actor = _FakeActor()
    _FakeWrapper.instances.clear()

    result = asyncio.run(
        rollout_task(
            TASK,
            actor=actor,
            simulator=SIMULATOR,
            wrapper_factory=_FakeWrapper,
            public_control_enabled=True,
        )
    )

    wrapper = _FakeWrapper.instances[-1]
    assert wrapper.calls == 3  # action + search + answer; rejected action never reached UserBench
    assert result["phase_guard_version"] == "public-control-v1"
    assert result["guard_rejections"] == 1
    assert result["guard_rejection_reasons"] == {
        "ANSWER_REQUIRED accepts choice=answer only": 1
    }
    transcript = json.dumps(result["visible_transcript"], ensure_ascii=False)
    assert "Current control state: ANSWER_REQUIRED" in transcript
    assert "Allowed next tool calls: answer (one visible option ID)" in transcript
    for forbidden in (
        "remaining_preference_ids",
        "correct_ids",
        "best_ids",
        "reward_snapshot",
        "reward delta",
    ):
        assert forbidden not in transcript
    assert wrapper.closed is True


# [项目注释] 功能：`test_raw_rollout_remains_explicitly_available_for_ablation`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：_FakeActor, clear, run, rollout_task。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_raw_rollout_remains_explicitly_available_for_ablation():
    actor = _FakeActor()
    _FakeWrapper.instances.clear()
    result = asyncio.run(
        rollout_task(
            TASK,
            actor=actor,
            simulator=SIMULATOR,
            wrapper_factory=_FakeWrapper,
            public_control_enabled=False,
        )
    )
    assert "phase_guard_version" not in result
    assert _FakeWrapper.instances[-1].calls == 20
