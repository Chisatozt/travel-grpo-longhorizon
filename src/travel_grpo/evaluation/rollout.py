"""One deterministic Actor/UserBench rollout for frozen evaluation."""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from travel_grpo.envs.userbench_context import UserBenchSessionState
from travel_grpo.envs.userbench_interaction import SimulatorRole, UserSimulatorRuntime
from travel_grpo.envs.userbench_tools import UserBenchAction
from travel_grpo.envs.userbench_wrapper import UserBenchEnvironmentConfig, UserBenchWrapper
from travel_grpo.evaluation.metrics import sanitize_reward
from travel_grpo.models.openai_compatible import TeacherApiError, TeacherProtocolError


async def rollout_task(
    task: Mapping[str, Any],
    *,
    actor: Any,
    simulator: UserSimulatorRuntime,
    source_root: str | Path | None = None,
    wrapper_factory: Any = UserBenchWrapper,
) -> dict[str, Any]:
    if simulator.role is not SimulatorRole.EVAL:
        raise ValueError("frozen evaluation requires the eval simulator role")
    task_id = str(task["task_id"])
    prompt = task.get("prompt")
    if not isinstance(prompt, Sequence) or isinstance(prompt, (str, bytes)):
        raise ValueError("evaluation task prompt must be a message sequence")
    messages = [dict(value) for value in prompt]
    wrapper = wrapper_factory(task_id, simulator, UserBenchEnvironmentConfig(), source_root=source_root)
    session: UserBenchSessionState | None = None
    attempts = 0
    try:
        async_reset = getattr(wrapper, "areset", None)
        if callable(async_reset):
            await async_reset()
        else:
            import asyncio

            await asyncio.to_thread(wrapper.reset)
        session = UserBenchSessionState(
            request_id=f"eval-{uuid.uuid4().hex}", task_id=task_id, wrapper=wrapper,
            reward_task=wrapper.reward_task(), reward_snapshot=wrapper.reward_snapshot(),
        )
        for _ in range(20):
            attempts += 1
            session.actor_attempts = attempts
            try:
                call = await actor.generate_action(messages)
            except TeacherProtocolError as exc:
                session.invalid_actions += 1
                message = str(exc)
                if "exactly one interact_with_env call" in message:
                    session.parallel_tool_calls = True
                    session.protocol_error = "parallel_tool_calls"
                    session.termination_reason = "parallel_tool_calls"
                    session.terminated = True
                    break
                session.protocol_error = "invalid_actor_tool_call"
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Error: emit exactly one valid interact_with_env tool call "
                            "with only thought, choice, and content."
                        ),
                    }
                )
                continue
            except TeacherApiError as exc:
                session.infrastructure_errors.append(f"actor_{exc.__class__.__name__}")
                session.termination_reason = "actor_infrastructure_failure"
                break
            action = UserBenchAction.from_parameters(call.parameters)
            messages.append(call.to_assistant_message())
            try:
                result = await wrapper.astep(action)
                snapshot = wrapper.reward_snapshot()
            except Exception as exc:
                session.infrastructure_errors.append(f"simulator_{exc.__class__.__name__}")
                session.termination_reason = "simulator_infrastructure_failure"
                break
            session.record_step(result, action, snapshot)
            messages.append({"role": "tool", "tool_call_id": call.call_id, "name": "interact_with_env", "content": result.observation.feedback})
            if session.done:
                break
        if not session.done and session.termination_reason is None:
            session.termination_reason = "actor_turn_limit"
        reward = session.reward_report()
        return {
            "schema_version": "travel-evaluation-task-v1",
            "task_id": task_id,
            "composition": str(task["composition"]),
            "infrastructure_valid": reward.get("reward_valid") is True,
            "actor_attempts": attempts,
            "environment_steps": session.num_tool_calls,
            "termination_reason": session.termination_reason,
            "reward": sanitize_reward(reward),
            "visible_transcript": messages,
        }
    finally:
        wrapper.close()
