"""veRL ToolAgentLoop that stops immediately when TravelGym finishes."""

from __future__ import annotations

import json
from typing import Any

from travel_grpo.envs.userbench_context import (
    clear_current_session,
    get_current_session,
)
from travel_grpo.training.grpo.compat import require_verl_061

try:  # Importable without the optional training stack.
    from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop
    from verl.tools.schemas import ToolResponse
except ImportError:  # pragma: no cover - expected for core-only installs.
    AgentState = None  # type: ignore[assignment]
    ToolAgentLoop = object  # type: ignore[assignment,misc]
    ToolResponse = None  # type: ignore[assignment]


def session_requests_termination() -> bool:
    """Provider-neutral state decision used by offline tests."""

    session = get_current_session()
    return session is not None and session.done


def select_post_tool_state(default_state: Any, terminated_state: Any) -> Any:
    """Select termination in the tool-processing turn when TravelGym is done."""

    return terminated_state if session_requests_termination() else default_state


class UserBenchAgentLoop(ToolAgentLoop):  # type: ignore[misc]
    """Close each trajectory reliably and prevent a post-terminal actor turn."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        require_verl_061()
        super().__init__(*args, **kwargs)

    async def _handle_processing_tools_state(self, state: Any) -> Any:
        next_state = await super()._handle_processing_tools_state(state)
        return select_post_tool_state(next_state, AgentState.TERMINATED)

    async def _call_tool(
        self, tool_call: Any, tools_kwargs: dict[str, Any]
    ) -> tuple[Any, float, dict[str, Any]]:
        """Keep actor-format failures stable while letting runtime failures escape."""

        try:
            parameters = json.loads(tool_call.arguments)
        except (json.JSONDecodeError, TypeError) as exc:
            return (
                ToolResponse(text=f"Error: invalid tool arguments: {exc}"),
                0.0,
                {"validation_error": str(exc)},
            )
        if tool_call.name not in self.tools:
            return (
                ToolResponse(text=f"Error: unknown tool {tool_call.name!r}"),
                0.0,
                {"validation_error": "unknown tool"},
            )

        tool = self.tools[tool_call.name]
        kwargs = tools_kwargs.get(tool_call.name, {})
        if not isinstance(kwargs, dict):
            raise TypeError(f"tools_kwargs for {tool_call.name!r} must be a mapping")
        instance_id = None
        try:
            instance_id, _ = await tool.create(
                create_kwargs=kwargs.get("create_kwargs", {})
            )
            response, reward, metadata = await tool.execute(instance_id, parameters)
        finally:
            if instance_id is not None:
                await tool.release(instance_id)

        response_text = response.text
        if response_text and len(response_text) > self.max_tool_response_length:
            length = self.max_tool_response_length
            if self.tool_response_truncate_side == "left":
                response_text = response_text[:length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                response_text = "(truncated)..." + response_text[-length:]
            else:
                half = length // 2
                response_text = (
                    response_text[:half] + "...(truncated)..." + response_text[-half:]
                )
        return ToolResponse(text=response_text), reward, metadata

    async def run(self, *args: Any, **kwargs: Any) -> Any:
        try:
            output = await super().run(*args, **kwargs)
            session = get_current_session()
            if session is None:
                raise RuntimeError(
                    "UserBench AgentLoop completed without an active session"
                )
            if session.num_tool_calls == 0 and session.protocol_error is None:
                session.protocol_error = (
                    "rollout ended without interact_with_env output"
                )

            interaction = self.interaction_map.get("userbench")
            if interaction is None:
                raise RuntimeError(
                    "UserBench interaction is missing from the AgentLoop"
                )
            output.reward_score = await interaction.calculate_score()
            metrics = session.metrics()
            extra_fields = getattr(output, "extra_fields", None)
            if extra_fields is None:
                output.extra_fields = {}
                extra_fields = output.extra_fields
            extra_fields["userbench"] = metrics
            await interaction.finalize_interaction()
            return output
        finally:
            clear_current_session(close=True)
