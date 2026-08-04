"""Opt-in import contract for the pinned veRL 0.8 runtime."""

import os
import pytest
from travel_grpo.training.grpo.compat import require_verl_080

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.getenv("TRAVEL_GRPO_VERL_SMOKE") != "1",
    reason="set TRAVEL_GRPO_VERL_SMOKE=1 in the Linux veRL 0.8 environment",
)
def test_external_verl_version_and_adapter_subclasses():
    assert require_verl_080() == "0.8.0"
    from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
    from verl.tools.base_tool import BaseTool
    from travel_grpo.training.grpo.adapter.agent_loop import UserBenchAgentLoop
    from travel_grpo.training.grpo.adapter.tools import UserBenchTool

    assert issubclass(UserBenchTool, BaseTool)
    assert issubclass(UserBenchAgentLoop, ToolAgentLoop)
