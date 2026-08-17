"""One deterministic Actor/UserBench rollout for frozen evaluation."""

from __future__ import annotations

import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from travel_grpo.protocols.actor_messages import normalize_actor_messages
from travel_grpo.envs.userbench_context import UserBenchSessionState
from travel_grpo.envs.userbench_interaction import SimulatorRole, UserSimulatorRuntime
from travel_grpo.envs.userbench_tools import UserBenchAction, UserBenchActionError
from travel_grpo.envs.userbench_wrapper import UserBenchEnvironmentConfig, UserBenchWrapper
from travel_grpo.evaluation.metrics import sanitize_reward
from travel_grpo.models.openai_compatible import TeacherApiError, TeacherProtocolError
from travel_grpo.prompts.actor_policy import (
    ACTOR_RUNTIME_POLICY_VERSION,
    ensure_actor_runtime_policy,
)


PUBLIC_CONTROL_PHASE_GUARD_VERSION = "public-control-v1"


def _initial_user_content(messages: Sequence[Mapping[str, Any]]) -> str:
    """Return the first actor-visible user message for public state setup."""

    for message in messages:
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            return str(message["content"])
    raise ValueError("evaluation prompt has no public user message")


async def rollout_task(
    task: Mapping[str, Any],
    *,
    actor: Any,
    simulator: UserSimulatorRuntime,
    source_root: str | Path | None = None,
    wrapper_factory: Any = UserBenchWrapper,
    apply_actor_policy: bool = True,
    public_control_enabled: bool = False,
) -> dict[str, Any]:
    """Run one rollout, optionally through the public production guard."""

    if public_control_enabled:
        return await _guarded_rollout_task(
            task,
            actor=actor,
            simulator=simulator,
            source_root=source_root,
            wrapper_factory=wrapper_factory,
            apply_actor_policy=apply_actor_policy,
        )
    if simulator.role is not SimulatorRole.EVAL:
        raise ValueError("frozen evaluation requires the eval simulator role")
    task_id = str(task["task_id"])
    prompt = task.get("prompt")
    if not isinstance(prompt, Sequence) or isinstance(prompt, (str, bytes)):
        raise ValueError("evaluation task prompt must be a message sequence")
    messages = (
        ensure_actor_runtime_policy(prompt)
        if apply_actor_policy
        else [dict(value) for value in prompt]
    )
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
            "actor_policy_version": (
                ACTOR_RUNTIME_POLICY_VERSION if apply_actor_policy else "none"
            ),
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


async def _guarded_rollout_task(
    task: Mapping[str, Any],
    *,
    actor: Any,
    simulator: UserSimulatorRuntime,
    source_root: str | Path | None = None,
    wrapper_factory: Any = UserBenchWrapper,
    apply_actor_policy: bool = True,
) -> dict[str, Any]:
    """Run one production-style rollout with public guard and feedback.

    The action is validated before ``wrapper.astep``. Rejected calls get a
    recoverable actor-visible error and never consume a simulator transition.
    Only public initial text, actor actions, and simulator feedback enter the
    control ledger; reward snapshots remain offline scoring inputs.
    """

    if simulator.role is not SimulatorRole.EVAL:
        raise ValueError("frozen evaluation requires the eval simulator role")
    task_id = str(task["task_id"])
    prompt = task.get("prompt")
    if not isinstance(prompt, Sequence) or isinstance(prompt, (str, bytes)):
        raise ValueError("evaluation task prompt must be a message sequence")
    normalized_prompt = normalize_actor_messages(prompt)
    initial_user_message = _initial_user_content(normalized_prompt)
    messages = (
        ensure_actor_runtime_policy(normalized_prompt)
        if apply_actor_policy
        else [dict(value) for value in normalized_prompt]
    )
    wrapper = wrapper_factory(
        task_id,
        simulator,
        UserBenchEnvironmentConfig(),
        source_root=source_root,
    )
    session: UserBenchSessionState | None = None
    guard_rejections = 0
    guard_rejection_reasons: Counter[str] = Counter()
    try:
        async_reset = getattr(wrapper, "areset", None)
        if callable(async_reset):
            await async_reset()
        else:
            import asyncio

            await asyncio.to_thread(wrapper.reset)
        session = UserBenchSessionState(
            request_id=f"eval-{uuid.uuid4().hex}",
            task_id=task_id,
            wrapper=wrapper,
            reward_task=wrapper.reward_task(),
            reward_snapshot=wrapper.reward_snapshot(),
            public_initial_message=initial_user_message,
        )
        for _ in range(20):
            if session.done:
                break

            # Advance answered/blocked aspects before generation and expose
            # the public transition note exactly once.
            before_phase = (
                session.public_control_state.phase
                if session.public_control_state is not None
                else None
            )
            session.prepare_public_action()
            after_phase = (
                session.public_control_state.phase
                if session.public_control_state is not None
                else None
            )
            if (
                before_phase != after_phase
                and session.public_control_state is not None
            ):
                messages.append(
                    {"role": "user", "content": session.render_actor_feedback("")}
                )

            session.actor_attempts += 1
            try:
                call = await actor.generate_action(messages)
                action = UserBenchAction.from_parameters(call.parameters)
            except TeacherApiError as exc:
                session.infrastructure_errors.append(
                    f"actor_{exc.__class__.__name__}"
                )
                session.termination_reason = "actor_infrastructure_failure"
                break
            except (TeacherProtocolError, UserBenchActionError, ValueError) as exc:
                session.invalid_actions += 1
                session.protocol_error = "invalid_actor_tool_call"
                messages.append(
                    {
                        "role": "user",
                        "content": session.render_actor_feedback(
                            "Error: emit exactly one valid interact_with_env "
                            f"tool call ({exc.__class__.__name__})."
                        ),
                    }
                )
                continue

            messages.append(call.to_assistant_message())
            reason = session.validate_public_action(action)
            if reason is not None:
                guard_rejections += 1
                guard_rejection_reasons[reason] += 1
                session.invalid_actions += 1
                session.record_public_guard_rejection(reason)
                session.record_public_non_progress(reason)
                messages.append(
                    {
                        "role": "user",
                        "content": session.render_actor_feedback(
                            f"Error: public control rejected this call: {reason}"
                        ),
                    }
                )
                continue

            try:
                result = await wrapper.astep(action)
                snapshot = wrapper.reward_snapshot()
            except Exception as exc:
                session.infrastructure_errors.append(
                    f"simulator_{exc.__class__.__name__}"
                )
                session.termination_reason = "simulator_infrastructure_failure"
                break

            session.record_step(result, action, snapshot)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.call_id,
                    "name": "interact_with_env",
                    "content": session.render_actor_feedback(
                        result.observation.feedback
                    ),
                }
            )
            if session.done:
                break

        if not session.done and session.termination_reason is None:
            session.termination_reason = "actor_turn_limit"
        reward = session.reward_report()
        return {
            "schema_version": "travel-evaluation-task-v1",
            "actor_policy_version": (
                ACTOR_RUNTIME_POLICY_VERSION if apply_actor_policy else "none"
            ),
            "phase_guard_version": PUBLIC_CONTROL_PHASE_GUARD_VERSION,
            "task_id": task_id,
            "composition": str(task["composition"]),
            "infrastructure_valid": reward.get("reward_valid") is True,
            "guard_rejections": guard_rejections,
            "guard_rejection_reasons": dict(guard_rejection_reasons),
            "actor_attempts": session.actor_attempts,
            "environment_steps": session.num_tool_calls,
            "termination_reason": session.termination_reason,
            "reward": sanitize_reward(reward),
            "visible_transcript": messages,
        }
    finally:
        wrapper.close()


__all__ = ["PUBLIC_CONTROL_PHASE_GUARD_VERSION", "rollout_task"]
