"""Opt-in smoke test for the editable pinned UserBench installation."""

import asyncio
import json
import os
from pathlib import Path

import pytest

from travel_grpo.envs.userbench_interaction import (
    SimulatorRole,
    UserSimulatorRuntime,
    _reset_user_simulator_binding_for_tests,
)
from travel_grpo.envs.userbench_wrapper import UserBenchWrapper

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(
    os.getenv("TRAVEL_GRPO_REAL_USERBENCH_SMOKE") != "1",
    reason="set TRAVEL_GRPO_REAL_USERBENCH_SMOKE=1 after editable UserBench install",
)
# [项目注释] 功能：`test_real_userbench_without_network`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。 主要协作调用：skipif, importorskip,
# [项目注释]    loads, next。
# [项目注释] 输入：`monkeypatch`。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_real_userbench_without_network(monkeypatch):
    travel_env = pytest.importorskip("travelgym.env.travel_env")
    task_path = ROOT / "environments/UserBench/travelgym/data/travelgym_data_22.json"
    tasks = json.loads(task_path.read_text(encoding="utf-8"))
    task_id = next(iter(tasks))

    # [项目注释] 功能：`fake_async_evaluator`：异步地实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。
    # [项目注释] 输入：*`args`；**`kwargs`。
    # [项目注释] 输出：返回运行时计算结果；没有显式返回值的分支返回 `None`。
    async def fake_async_evaluator(*args, **kwargs):
        return "offline evaluator feedback", [], 0.2

    monkeypatch.setattr(travel_env, "async_evaluate_action", fake_async_evaluator)
    _reset_user_simulator_binding_for_tests()
    runtime = UserSimulatorRuntime(
        SimulatorRole.GRPO,
        "deepseek-v4-flash",
        "http://127.0.0.1:9/v1",
        "offline-key",
    )
    wrapper = UserBenchWrapper(task_id, runtime)
    try:
        wrapper.reset()
        result = asyncio.run(
            wrapper.astep(
                {"thought": "offline smoke", "choice": "action", "content": "hello"}
            )
        )
        assert result.observation.feedback == "offline evaluator feedback"
        assert result.reward == 0.2
    finally:
        wrapper.close()
        _reset_user_simulator_binding_for_tests()
