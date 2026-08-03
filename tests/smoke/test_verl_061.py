"""Opt-in import contract for an external veRL 0.6.1 editable checkout."""

import os

import pytest

from travel_grpo.training.grpo.compat import require_verl_061

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.getenv("TRAVEL_GRPO_VERL_SMOKE") != "1",
    reason="set TRAVEL_GRPO_VERL_SMOKE=1 after editable veRL 0.6.1 install",
)
def test_external_verl_version_and_adapter_subclasses():
    assert require_verl_061() == "0.6.1"

    from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
    from verl.interactions.base import BaseInteraction
    from verl.tools.base_tool import BaseTool

    from travel_grpo.training.grpo.adapter.agent_loop import UserBenchAgentLoop
    from travel_grpo.training.grpo.adapter.session import UserBenchInteraction
    from travel_grpo.training.grpo.adapter.tools import UserBenchTool

    assert issubclass(UserBenchTool, BaseTool)
    assert issubclass(UserBenchInteraction, BaseInteraction)
    assert issubclass(UserBenchAgentLoop, ToolAgentLoop)
