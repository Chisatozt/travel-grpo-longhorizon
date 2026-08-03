"""Expose the official UserBench action as one veRL native tool."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from travel_grpo.envs.userbench_context import require_current_session
from travel_grpo.envs.userbench_tools import (
    TOOL_NAME,
    UserBenchAction,
    UserBenchActionError,
    get_interact_with_env_schema,
)
from travel_grpo.training.grpo.adapter.session import ENVIRONMENT_NAME
from travel_grpo.training.grpo.compat import require_verl_061

try:  # Keep core imports light; actual construction requires veRL 0.6.1.
    from verl.tools.base_tool import BaseTool
    from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
except ImportError:  # pragma: no cover - expected outside the optional runtime.
    BaseTool = object  # type: ignore[assignment,misc]
    OpenAIFunctionToolSchema = None  # type: ignore[assignment]
    ToolResponse = None  # type: ignore[assignment]


@dataclass(frozen=True)
class UserBenchToolExecution:
    """Provider-neutral tool result used by both veRL and offline tests."""

    text: str
    reward: float
    metadata: dict[str, Any]


async def execute_userbench_action(
    parameters: Mapping[str, Any],
) -> UserBenchToolExecution:
    """Validate and execute one action in the session held by the current context."""

    session = require_current_session()
    if session.done:
        raise RuntimeError(
            "cannot execute a tool call after the UserBench episode ended"
        )
    try:
        action = UserBenchAction.from_parameters(parameters)
    except UserBenchActionError as exc:
        return UserBenchToolExecution(
            text=f"Error: invalid {TOOL_NAME} call: {exc}",
            reward=0.0,
            metadata={"validation_error": str(exc), "task_id": session.task_id},
        )

    result = await session.wrapper.astep(action)
    session.record_step(result)
    return UserBenchToolExecution(
        text=result.observation.to_tool_text(),
        reward=0.0,
        metadata={
            "task_id": session.task_id,
            "raw_reward": result.reward,
            "cumulative_reward": session.rewards.total,
            "step_count": result.observation.step_count,
            "terminated": result.terminated,
            "truncated": result.truncated,
        },
    )


class UserBenchTool(BaseTool):  # type: ignore[misc]
    """veRL BaseTool adapter; step rewards are metadata, never tool rewards."""

    def __init__(
        self, config: dict[str, Any], tool_schema: OpenAIFunctionToolSchema
    ) -> None:
        require_verl_061()
        super().__init__(config, tool_schema)

    async def create(
        self, instance_id: str | None = None, **kwargs: Any
    ) -> tuple[str, Any]:
        session = require_current_session()
        create_kwargs = kwargs.get("create_kwargs", {})
        if not isinstance(create_kwargs, Mapping):
            raise TypeError("create_kwargs must be a mapping")
        task_id = create_kwargs.get("id")
        if task_id is not None and task_id != session.task_id:
            raise ValueError(
                f"tool task ID {task_id!r} does not match session {session.task_id!r}"
            )
        env_name = create_kwargs.get("env_name")
        if env_name is not None and env_name != ENVIRONMENT_NAME:
            raise ValueError(f"tool env_name must be {ENVIRONMENT_NAME!r}")
        resolved_id = instance_id or session.request_id
        return resolved_id, ToolResponse(text="")

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs: Any
    ) -> tuple[Any, float, dict[str, Any]]:
        session = require_current_session()
        if instance_id != session.request_id:
            raise ValueError(
                f"tool instance ID {instance_id!r} does not match {session.request_id!r}"
            )
        result = await execute_userbench_action(parameters)
        return ToolResponse(text=result.text), result.reward, result.metadata

    async def calc_reward(self, instance_id: str, **kwargs: Any) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs: Any) -> None:
        return None

    @staticmethod
    def schema_dict() -> dict[str, Any]:
        return get_interact_with_env_schema()
