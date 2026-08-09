"""veRL 0.8 ToolAgentLoop with one direct UserBench session per rollout."""

from __future__ import annotations

import copy
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
from travel_grpo.training.grpo.preflight import is_validation_sampling
from travel_grpo.prompts.actor_policy import (
    ACTOR_RUNTIME_POLICY_VERSION,
    ensure_actor_runtime_policy,
)
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


def _parse_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean")


def _parse_threshold(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("stall_no_progress_threshold must be an integer >= 1")
    if isinstance(value, int):
        threshold = value
    elif isinstance(value, str) and value.strip().isdigit():
        threshold = int(value.strip())
    else:
        raise ValueError("stall_no_progress_threshold must be an integer >= 1")
    if threshold < 1:
        raise ValueError("stall_no_progress_threshold must be an integer >= 1")
    return threshold


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
    session.terminated = True
    if session.answer_only_pending:
        session.hard_stop_stalled()
    else:
        session.termination_reason = "parallel_tool_calls"
    return True


def finalize_actor_stop(session: Any) -> None:
    """Classify a rollout that returned without an environment terminal step."""

    if (
        not session.done
        and not getattr(session, "infrastructure_errors", ())
        and (
            getattr(session, "answer_only_pending", False)
            or getattr(session, "answer_only_generation_started", False)
        )
    ):
        session.hard_stop_stalled()
        return
    if session.num_tool_calls == 0 and session.protocol_error is None:
        session.invalid_actions += 1
        session.protocol_error = "no_tool_output"
        session.termination_reason = "no_tool_output"
    elif not session.done and session.termination_reason is None:
        session.termination_reason = "actor_stopped"


_DISABLED_ACTOR_POLICY_VERSION = "disabled"


def _parse_actor_policy_version(value: Any, *, enabled: bool) -> str:
    if value is None:
        version = ACTOR_RUNTIME_POLICY_VERSION
    else:
        version = str(value).strip()
    if not version:
        raise ValueError("actor_policy_version must be a non-empty string")
    if enabled and version != ACTOR_RUNTIME_POLICY_VERSION:
        raise ValueError(
            "actor_policy_version must match the shared production policy "
            f"{ACTOR_RUNTIME_POLICY_VERSION!r}; got {version!r}"
        )
    return version


def prepare_actor_prompt(
    raw_prompt: Any,
    *,
    actor_policy_enabled: bool | str = True,
    actor_policy_version: str | None = ACTOR_RUNTIME_POLICY_VERSION,
) -> list[dict[str, Any]]:
    """Copy a veRL raw prompt and optionally add one Actor policy block.

    The copy happens before any validation or mutation.  This function is
    intentionally independent of veRL so it can be tested on CPU and reused
    by train/validation parity checks.
    """

    enabled = _parse_bool(actor_policy_enabled, name="actor_policy_enabled")
    copied = copy.deepcopy(raw_prompt)
    if isinstance(copied, (str, bytes)):
        raise ValueError("raw_prompt must be a message sequence")
    if not isinstance(copied, list):
        try:
            copied = list(copied)
        except TypeError as exc:
            raise ValueError("raw_prompt must be a message sequence") from exc
    if not enabled:
        return copied
    version = _parse_actor_policy_version(actor_policy_version, enabled=True)
    if version != ACTOR_RUNTIME_POLICY_VERSION:
        # Keep the check explicit even though _parse_actor_policy_version also
        # validates it; this documents that only the shared block is injected.
        raise ValueError(
            "cannot inject an unknown Actor policy version: "
            f"{version!r}"
        )
    return ensure_actor_runtime_policy(copied)


def actor_policy_metadata(
    *, actor_policy_enabled: bool, actor_policy_version: str
) -> dict[str, Any]:
    """Return only non-sensitive policy provenance for rollout logging."""

    return {
        "actor_policy_enabled": bool(actor_policy_enabled),
        "actor_policy_version": (
            actor_policy_version if actor_policy_enabled else _DISABLED_ACTOR_POLICY_VERSION
        ),
    }


class UserBenchAgentLoop(ToolAgentLoop):  # type: ignore[misc]
    """Own the environment lifecycle and assign Reward v2 exactly once."""

    def __init__(
        self,
        *args: Any,
        environment_config_path: str | Path = "configs/interaction_config/userbench.yaml",
        simulator_config_path: str | Path = "configs/interaction_config/simulator_train.yaml",
        max_steps: int = 20,
        stall_recovery_enabled: bool | str = False,
        stall_no_progress_threshold: int | str = 4,
        actor_policy_enabled: bool | str = True,
        actor_policy_version: str | None = ACTOR_RUNTIME_POLICY_VERSION,
        **kwargs: Any,
    ) -> None:
        require_verl_080()
        if int(max_steps) != 20:
            raise ValueError("the UserBench rollout contract requires max_steps=20")
        self.stall_recovery_enabled = _parse_bool(
            stall_recovery_enabled, name="stall_recovery_enabled"
        )
        self.stall_no_progress_threshold = _parse_threshold(
            stall_no_progress_threshold
        )
        self.actor_policy_enabled = _parse_bool(
            actor_policy_enabled, name="actor_policy_enabled"
        )
        self.actor_policy_version = _parse_actor_policy_version(
            actor_policy_version, enabled=self.actor_policy_enabled
        )
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
        configure_stall_recovery = getattr(
            session, "configure_stall_recovery", None
        )
        if callable(configure_stall_recovery):
            configure_stall_recovery(
                enabled=(
                    self.stall_recovery_enabled
                    and not is_validation_sampling(sampling_params)
                ),
                threshold=self.stall_no_progress_threshold,
            )
        session.actor_attempts += 1
        if session.answer_only_pending:
            session.begin_answer_only_generation()
            if session.done:
                return AgentState.TERMINATED
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
            session.record_non_progress("unknown_tool")
            return (
                ToolResponse(
                    text=session.render_actor_feedback(
                        f"Error: unsupported tool {name!r}; use {TOOL_NAME}."
                    )
                ),
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
            session.record_non_progress("malformed_tool_call")
            return (
                ToolResponse(
                    text=session.render_actor_feedback(
                        f"Error: invalid {TOOL_NAME} call: {exc}"
                    )
                ),
                0.0,
                {"validation_error": str(exc)},
            )
        return await super()._call_tool(tool_call, tools_kwargs, agent_data)

    async def run(self, sampling_params: Any, **kwargs: Any) -> Any:
        # veRL reuses the input mapping when launching multiple rollouts.
        # Prepare a private prompt before the parent loop sees it so policy
        # injection cannot leak across samples.
        parent_kwargs = dict(kwargs)
        if "raw_prompt" in kwargs:
            parent_kwargs["raw_prompt"] = prepare_actor_prompt(
                kwargs["raw_prompt"],
                actor_policy_enabled=self.actor_policy_enabled,
                actor_policy_version=self.actor_policy_version,
            )
        task_id = task_id_from_run_kwargs(kwargs)
        session = await self.userbench_runtime.astart_session(task_id)
        try:
            configure_stall_recovery = getattr(
                session, "configure_stall_recovery", None
            )
            if callable(configure_stall_recovery):
                configure_stall_recovery(
                    enabled=(
                        self.stall_recovery_enabled
                        and not is_validation_sampling(sampling_params)
                    ),
                    threshold=self.stall_no_progress_threshold,
                )
            output = await super().run(sampling_params, **parent_kwargs)
            if getattr(output, "extra_fields", None) is None:
                output.extra_fields = {}
            output.extra_fields.update(
                actor_policy_metadata(
                    actor_policy_enabled=self.actor_policy_enabled,
                    actor_policy_version=self.actor_policy_version,
                )
            )
            finalize_actor_stop(session)
            reward = session.reward_report()
            output.reward_score = float(reward["terminal_reward"])
            extra_fields = getattr(output, "extra_fields", None)
            if extra_fields is None:
                output.extra_fields = {}
                extra_fields = output.extra_fields
            policy_metadata = actor_policy_metadata(
                actor_policy_enabled=self.actor_policy_enabled,
                actor_policy_version=self.actor_policy_version,
            )
            extra_fields["userbench"] = {
                **session.metrics(),
                **policy_metadata,
                "reward": reward,
                "infrastructure_invalid": not bool(reward.get("reward_valid")),
            }
            # veRL's validation logger flattens this mapping into the generation
            # JSONL.  It contains metrics only, never hidden IDs or snapshots.
            quality_by_aspect = dict(reward.get("quality_by_aspect", {}))
            fallback_counts = dict(reward.get("simulator_fallback_counts", {}))
            public_state = getattr(session, "public_control_state", None)
            public_phase = getattr(getattr(public_state, "phase", None), "value", None)
            extra_fields["reward_extra_info"] = {
                **policy_metadata,
                "public_control_phase": public_phase,
                "public_control_episode_done": bool(
                    getattr(public_state, "episode_done", False)
                ),
                "public_answered_aspect_count": int(
                    getattr(public_state, "answered_count", 0)
                ),
                "public_blocked_aspect_count": int(
                    getattr(public_state, "blocked_count", 0)
                ),
                "task_id": session.task_id,
                "terminal_reward": float(reward["terminal_reward"]),
                "reward_valid": bool(reward.get("reward_valid")),
                "correct_itinerary": bool(reward.get("correct_itinerary")),
                "gold_itinerary": bool(reward.get("gold_itinerary")),
                "user_aligned_success": bool(reward.get("user_aligned_success")),
                "completion_rate": float(reward.get("completion_rate", 0.0)),
                "search_coverage": float(reward.get("search_coverage", 0.0)),
                "active_preference_coverage": float(
                    reward.get("active_preference_coverage", 0.0)
                ),
                "passive_preference_coverage": float(
                    reward.get("passive_preference_coverage", 0.0)
                ),
                "efficiency": float(reward.get("efficiency", 0.0)),
                "policy_penalty": float(reward.get("policy_penalty", 0.0)),
                "quality_flight": float(quality_by_aspect.get("flight", 0.0)),
                "quality_hotel": float(quality_by_aspect.get("hotel", 0.0)),
                "quality_restaurant": float(quality_by_aspect.get("restaurant", 0.0)),
                "quality_apartment": float(quality_by_aspect.get("apartment", 0.0)),
                "quality_rental_car": float(quality_by_aspect.get("rental_car", 0.0)),
                "actor_attempts": session.actor_attempts,
                "environment_steps": session.num_tool_calls,
                "effective_steps": int(reward.get("effective_steps", session.num_tool_calls)),
                "invalid_actions": session.invalid_actions,
                "exact_repeats": session.exact_repeats,
                "semantic_repeats": session.semantic_repeats,
                "reward_degraded": bool(reward.get("reward_degraded", False)),
                "userbench_judgment_fallbacks": int(
                    fallback_counts.get("userbench_judgment_fallbacks", 0)
                ),
                "userbench_response_fallbacks": int(
                    fallback_counts.get("userbench_response_fallbacks", 0)
                ),
                "userbench_search_fallbacks": int(
                    fallback_counts.get("userbench_search_fallbacks", 0)
                ),
                "termination_reason": session.termination_reason,
                "stall_recovery_enabled": session.stall_recovery_enabled,
                "stall_recovery_triggered": session.stall_recovery_triggered,
                "stall_recovery_used": session.stall_recovery_used,
                "stall_hard_truncated": session.stall_hard_truncated,
                "consecutive_no_progress": session.consecutive_no_progress,
                "max_consecutive_no_progress": session.max_consecutive_no_progress,
                "answer_only_generation_started": session.answer_only_generation_started,
                "visible_answer_option_count": len(session.visible_answer_options),
            }
            return output
        finally:
            clear_current_session(close=True)
