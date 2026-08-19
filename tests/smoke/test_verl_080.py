"""Opt-in import contract for the pinned veRL 0.8 runtime."""

import os
import pytest
from travel_grpo.training.grpo.compat import require_verl_080

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.getenv("TRAVEL_GRPO_VERL_SMOKE") != "1",
    reason="set TRAVEL_GRPO_VERL_SMOKE=1 in the Linux veRL 0.8 environment",
)
# [项目注释] 功能：`test_external_verl_version_and_adapter_subclasses`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。 主要协作调用：skipif,
# [项目注释]    issubclass, require_verl_080, getenv。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_external_verl_version_and_adapter_subclasses():
    assert require_verl_080() == "0.8.0"
    from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
    from verl.tools.base_tool import BaseTool
    from travel_grpo.training.grpo.adapter.agent_loop import UserBenchAgentLoop
    from travel_grpo.training.grpo.adapter.tools import UserBenchTool

    assert issubclass(UserBenchTool, BaseTool)
    assert issubclass(UserBenchAgentLoop, ToolAgentLoop)
