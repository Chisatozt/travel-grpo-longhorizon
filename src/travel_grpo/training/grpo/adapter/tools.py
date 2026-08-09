"""Expose the official UserBench action as one veRL native tool."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from travel_grpo.envs.userbench_context import require_current_session
from travel_grpo.envs.userbench_wrapper import UserBenchEnvironmentError
from travel_grpo.envs.userbench_tools import (
    TOOL_NAME,
    UserBenchAction,
    UserBenchActionError,
    get_interact_with_env_schema,
)
from travel_grpo.training.grpo.adapter.session import ENVIRONMENT_NAME
from travel_grpo.training.grpo.compat import require_verl_080

try:  # Keep core imports light; actual construction requires veRL 0.8.0.
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


def _rejected_tool_execution(
    session: Any,
    message: str,
    *,
    reason: str,
) -> UserBenchToolExecution:
    """Return an actor-visible error without touching the environment."""

    session.invalid_actions += 1
    session.record_non_progress(reason)
    return UserBenchToolExecution(
        text=session.render_actor_feedback(f"Error: {message}"),
        reward=0.0,
        metadata={
            "validation_error": message,
            "task_id": session.task_id,
            "environment_executed": False,
        },
    )


async def execute_userbench_action(
    parameters: Mapping[str, Any],
) -> UserBenchToolExecution:
    """Validate and execute one action in the session held by the current context."""

    session = require_current_session()
    # A terminal answered/blocked aspect is advanced before the next call;
    # all guard decisions remain public-only and happen before UserBench.
    session.prepare_public_action()
    if session.done:
        # Public sessions return a recoverable actor-visible rejection instead
        # of raising after the public ledger has reached a terminal state.
        # Legacy sessions retain the strict lifecycle exception.
        if getattr(session, "public_control_state", None) is not None:
            return _rejected_tool_execution(
                session,
                "the rollout is terminal; no further tool call is allowed",
                reason="public_control_complete",
            )
        raise RuntimeError(
            "cannot execute a tool call after the UserBench episode ended"
        )
    try:
        action = UserBenchAction.from_parameters(parameters)
    except UserBenchActionError as exc:
        return _rejected_tool_execution(
            session,
            f"invalid {TOOL_NAME} call: {exc}",
            reason="invalid_tool_call",
        )

    public_error = session.validate_public_action(action)
    if public_error is not None:
        return _rejected_tool_execution(
            session,
            public_error,
            reason="public_phase_guard",
        )

    recovery_error = session.validate_answer_only_action(action)
    if recovery_error is not None:
        return _rejected_tool_execution(
            session,
            recovery_error,
            reason="answer_only_violation",
        )

    try:
        result = await session.wrapper.astep(action)
    except Exception as exc:
        module = exc.__class__.__module__.split(".", 1)[0]
        infrastructure = isinstance(
            exc, (UserBenchEnvironmentError, TimeoutError, ConnectionError, OSError)
        ) or module in {"openai", "httpx", "httpcore"}
        if not infrastructure:
            raise
        reason = f"simulator_{exc.__class__.__name__}"
        session.infrastructure_errors.append(reason)
        session.protocol_error = "simulator_infrastructure_failure"
        session.termination_reason = "simulator_infrastructure_failure"
        session.terminated = True
        return UserBenchToolExecution(
            text=session.render_actor_feedback(
                "Error: UserBench simulator infrastructure failure; trajectory invalid."
            ),
            reward=0.0,
            metadata={
                "task_id": session.task_id,
                "infrastructure_invalid": True,
                "error_type": exc.__class__.__name__,
                "terminated": True,
                "truncated": False,
            },
        )
    try:
        snapshot = session.wrapper.reward_snapshot()
    except AttributeError:
        snapshot = None
    except Exception:
        # A transition without a readable post-step snapshot cannot be scored
        # from the evidence ledger.  Keep the rollout observable to veRL, but
        # classify it as hard-invalid rather than letting the worker crash.
        session.infrastructure_errors.append("reward_snapshot_unavailable")
        snapshot = None
    session.record_step(result, action, snapshot)
    report = session.reward_report()
    feedback = session.render_actor_feedback(result.observation.to_tool_text())
    return UserBenchToolExecution(
        text=feedback,
        reward=0.0,
        metadata={
            "task_id": session.task_id,
            "raw_reward": result.reward,
            "raw_cumulative_reward": session.rewards.total,
            "terminal_reward_preview": report["terminal_reward"],
            "reward_valid": report["reward_valid"],
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
        require_verl_080()
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
