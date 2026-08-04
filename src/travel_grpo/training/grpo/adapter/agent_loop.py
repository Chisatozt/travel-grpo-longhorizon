"""veRL 0.8 ToolAgentLoop with one direct UserBench session per rollout."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from travel_grpo.envs.userbench_context import (
    clear_current_session,
    get_current_session,
)
from travel_grpo.training.grpo.adapter.session import (
    UserBenchRolloutRuntime,
    task_id_from_run_kwargs,
)
from travel_grpo.training.grpo.compat import require_verl_080
from travel_grpo.envs.userbench_tools import (
    TOOL_NAME,
    UserBenchAction,
    UserBenchActionError,
)

try:  # Lightweight installs keep the integration importable.
    from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop
    from verl.tools.schemas import ToolResponse
except ImportError:  # pragma: no cover
    AgentState = None  # type: ignore[assignment]
    ToolAgentLoop = object  # type: ignore[assignment,misc]
    ToolResponse = None  # type: ignore[assignment]


def session_requests_termination() -> bool:
    session = get_current_session()
    return session is not None and session.done


def select_post_tool_state(default_state: Any, terminated_state: Any) -> Any:
    return terminated_state if session_requests_termination() else default_state


def reject_parallel_tool_calls(state: Any) -> bool:
    tool_calls = getattr(state, "tool_calls", None)
    if not isinstance(tool_calls, (list, tuple)) or len(tool_calls) <= 1:
        return False
    session = get_current_session()
    if session is None:
        raise RuntimeError("parallel tool calls were emitted without a UserBench session")
    session.invalid_actions += 1
    session.parallel_tool_calls = True
    session.protocol_error = "parallel_tool_calls"
    session.termination_reason = "parallel_tool_calls"
    session.terminated = True
    return True


def finalize_actor_stop(session: Any) -> None:
    """Classify a rollout that returned without an environment terminal step."""

    if session.num_tool_calls == 0 and session.protocol_error is None:
        session.invalid_actions += 1
        session.protocol_error = "no_tool_output"
        session.termination_reason = "no_tool_output"
    elif not session.done and session.termination_reason is None:
        session.termination_reason = "actor_stopped"


class UserBenchAgentLoop(ToolAgentLoop):  # type: ignore[misc]
    """Own the environment lifecycle and assign Reward v2 exactly once."""

    def __init__(
        self,
        *args: Any,
        environment_config_path: str | Path = "configs/interaction_config/userbench.yaml",
        simulator_config_path: str | Path = "configs/interaction_config/simulator_train.yaml",
        max_steps: int = 20,
        **kwargs: Any,
    ) -> None:
        require_verl_080()
        if int(max_steps) != 20:
            raise ValueError("the UserBench rollout contract requires max_steps=20")
        super().__init__(*args, **kwargs)
        self.userbench_runtime = UserBenchRolloutRuntime.from_config_files(
            environment_config_path, simulator_config_path
        )

    async def _handle_processing_tools_state(self, state: Any) -> Any:
        if reject_parallel_tool_calls(state):
            return AgentState.TERMINATED
        next_state = await super()._handle_processing_tools_state(state)
        return select_post_tool_state(next_state, AgentState.TERMINATED)

    async def _handle_generating_state(
        self, state: Any, sampling_params: Any, ignore_termination: bool = False
    ) -> Any:
        session = get_current_session()
        if session is None:
            raise RuntimeError("Actor generation has no active UserBench session")
        session.actor_attempts += 1
        return await super()._handle_generating_state(
            state, sampling_params, ignore_termination=ignore_termination
        )

    async def _call_tool(
        self, tool_call: Any, tools_kwargs: dict[str, Any], agent_data: Any
    ) -> tuple[Any, float, dict[str, Any]]:
        """Return stable protocol errors without invoking the environment."""

        session = get_current_session()
        if session is None:
            raise RuntimeError("tool dispatch has no active UserBench session")
        name = getattr(tool_call, "name", None)
        if name != TOOL_NAME:
            session.invalid_actions += 1
            return (
                ToolResponse(text=f"Error: unsupported tool {name!r}; use {TOOL_NAME}."),
                0.0,
                {"validation_error": "unknown_tool"},
            )
        try:
            raw = getattr(tool_call, "arguments", None)
            parameters = json.loads(raw) if isinstance(raw, str) else None
            if not isinstance(parameters, Mapping):
                raise UserBenchActionError("arguments must be one JSON object")
            UserBenchAction.from_parameters(parameters)
        except (json.JSONDecodeError, UserBenchActionError) as exc:
            session.invalid_actions += 1
            return (
                ToolResponse(text=f"Error: invalid {TOOL_NAME} call: {exc}"),
                0.0,
                {"validation_error": str(exc)},
            )
        return await super()._call_tool(tool_call, tools_kwargs, agent_data)

    async def run(self, sampling_params: Any, **kwargs: Any) -> Any:
        task_id = task_id_from_run_kwargs(kwargs)
        session = await self.userbench_runtime.astart_session(task_id)
        try:
            output = await super().run(sampling_params, **kwargs)
            finalize_actor_stop(session)
            reward = session.reward_report()
            output.reward_score = float(reward["terminal_reward"])
            extra_fields = getattr(output, "extra_fields", None)
            if extra_fields is None:
                output.extra_fields = {}
                extra_fields = output.extra_fields
            extra_fields["userbench"] = {
                **session.metrics(),
                "reward": reward,
                "infrastructure_invalid": not bool(reward.get("reward_valid")),
            }
            # veRL's validation logger flattens this mapping into the generation
            # JSONL.  It contains metrics only, never hidden IDs or snapshots.
            extra_fields["reward_extra_info"] = {
                "task_id": session.task_id,
                "terminal_reward": float(reward["terminal_reward"]),
                "reward_valid": bool(reward.get("reward_valid")),
                "correct_itinerary": bool(reward.get("correct_itinerary")),
                "gold_itinerary": bool(reward.get("gold_itinerary")),
                "user_aligned_success": bool(reward.get("user_aligned_success")),
                "completion_rate": float(reward.get("completion_rate", 0.0)),
                "active_preference_coverage": float(
                    reward.get("active_preference_coverage", 0.0)
                ),
                "passive_preference_coverage": float(
                    reward.get("passive_preference_coverage", 0.0)
                ),
                "efficiency": float(reward.get("efficiency", 0.0)),
                "policy_penalty": float(reward.get("policy_penalty", 0.0)),
                "quality_by_aspect": dict(reward.get("quality_by_aspect", {})),
                "actor_attempts": session.actor_attempts,
                "environment_steps": session.num_tool_calls,
                "invalid_actions": session.invalid_actions,
                "exact_repeats": session.exact_repeats,
                "semantic_repeats": session.semantic_repeats,
                "termination_reason": session.termination_reason,
            }
            return output
        finally:
            clear_current_session(close=True)
