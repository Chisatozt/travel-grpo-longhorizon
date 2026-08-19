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
from travel_grpo.trajectory.turn_credit import (
    TurnCreditConfig,
    validate_turn_credit_mode,
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


# [项目注释] 功能：`session_requests_termination`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：get_current_session。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `bool`；具体值由各分支决定。
def session_requests_termination() -> bool:
    session = get_current_session()
    return session is not None and session.done


# [项目注释] 功能：`_parse_bool`：读取并解析外部数据，将其转换为项目内部可消费的结构。 主要协作调用：isinstance, ValueError, casefold, strip。
# [项目注释] 输入：`value`: Any；`name`: str。
# [项目注释] 输出：标注返回 `bool`；具体值由各分支决定。
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


# [项目注释] 功能：`_parse_threshold`：读取并解析外部数据，将其转换为项目内部可消费的结构。 主要协作调用：isinstance, ValueError, isdigit, int。
# [项目注释] 输入：`value`: Any。
# [项目注释] 输出：标注返回 `int`；具体值由各分支决定。
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


# [项目注释] 功能：`select_post_tool_state`：按固定约束拆分、采样或选择输入集合，保持确定性和边界条件。 主要协作调用：session_requests_termination。
# [项目注释] 输入：`default_state`: Any；`terminated_state`: Any。
# [项目注释] 输出：标注返回 `Any`；具体值由各分支决定。
def select_post_tool_state(default_state: Any, terminated_state: Any) -> Any:
    return terminated_state if session_requests_termination() else default_state


# [项目注释] 功能：`reject_parallel_tool_calls`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：getattr,
# [项目注释]    get_current_session, callable, RuntimeError。
# [项目注释] 输入：`state`: Any。
# [项目注释] 输出：标注返回 `bool`；具体值由各分支决定。
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
    finalize_turn = getattr(session, "finalize_pending_actor_turn", None)
    if callable(finalize_turn):
        finalize_turn(reason="parallel_tool_calls")
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
        finalize_turn = getattr(session, "finalize_pending_actor_turn", None)
        if callable(finalize_turn):
            finalize_turn(reason=session.termination_reason or "stalled_no_progress")
        return
    if session.num_tool_calls == 0 and session.protocol_error is None:
        session.invalid_actions += 1
        session.protocol_error = "no_tool_output"
        session.termination_reason = "no_tool_output"
    elif not session.done and session.termination_reason is None:
        session.termination_reason = "actor_stopped"
    finalize_turn = getattr(session, "finalize_pending_actor_turn", None)
    if callable(finalize_turn):
        finalize_turn(reason=session.termination_reason or "no_tool_output")


_DISABLED_ACTOR_POLICY_VERSION = "disabled"


# [项目注释] 功能：`_parse_actor_policy_version`：读取并解析外部数据，将其转换为项目内部可消费的结构。 主要协作调用：strip, ValueError, str。
# [项目注释] 输入：`value`: Any；`enabled`: bool。
# [项目注释] 输出：标注返回 `str`；具体值由各分支决定。
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
    """Own the environment lifecycle and assign the v3 reward exactly once."""

    # [项目注释] 功能：`__init__`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：require_verl_080, _parse_bool,
    # [项目注释]    _parse_threshold, _parse_actor_policy_version。
    # [项目注释] 输入：`environment_config_path`: str | Path；`simulator_config_path`: str | Path；`max_steps`:
    # [项目注释]    int；`stall_recovery_enabled`: bool | str；`stall_no_progress_threshold`: int |
    # [项目注释]    str；`actor_policy_enabled`: bool | str；`actor_policy_version`: str |
    # [项目注释]    None；`turn_credit_mode`: str；`turn_credit_config`: Mapping[str, Any] |
    # [项目注释]    None；`turn_credit_config_json`: str | None；*`args`；**`kwargs`。
    # [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
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
        turn_credit_mode: str = "off",
        turn_credit_config: Mapping[str, Any] | None = None,
        turn_credit_config_json: str | None = None,
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
        self.turn_credit_mode = validate_turn_credit_mode(turn_credit_mode)
        if turn_credit_config_json:
            if turn_credit_config is not None:
                raise ValueError(
                    "turn_credit_config and turn_credit_config_json are mutually exclusive"
                )
            decoded_turn_credit = json.loads(turn_credit_config_json)
            if not isinstance(decoded_turn_credit, Mapping):
                raise ValueError("turn_credit_config_json must decode to one mapping")
            turn_credit_config = decoded_turn_credit
        self.turn_credit_config = TurnCreditConfig.from_mapping(turn_credit_config)
        super().__init__(*args, **kwargs)
        self.userbench_runtime = UserBenchRolloutRuntime.from_config_files(
            environment_config_path, simulator_config_path
        )

    # [项目注释] 功能：`_handle_processing_tools_state`：异步地实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。
    # [项目注释]    主要协作调用：reject_parallel_tool_calls, select_post_tool_state, _handle_processing_tools_state,
    # [项目注释]    super。
    # [项目注释] 输入：`state`: Any。
    # [项目注释] 输出：标注返回 `Any`；具体值由各分支决定。
    async def _handle_processing_tools_state(self, state: Any) -> Any:
        if reject_parallel_tool_calls(state):
            return AgentState.TERMINATED
        next_state = await super()._handle_processing_tools_state(state)
        return select_post_tool_state(next_state, AgentState.TERMINATED)

    # [项目注释] 功能：`_handle_generating_state`：异步地实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：get_current_session,
    # [项目注释]    getattr, callable, RuntimeError。
    # [项目注释] 输入：`state`: Any；`sampling_params`: Any；`ignore_termination`: bool。
    # [项目注释] 输出：标注返回 `Any`；具体值由各分支决定。
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
        begin_turn = getattr(session, "begin_actor_turn", None)
        if callable(begin_turn):
            begin_turn()
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
            reject_turn = getattr(session, "reject_actor_turn", None)
            if callable(reject_turn):
                reject_turn(reason=f"unsupported tool {name!r}", category="unknown_tool")
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
            reject_turn = getattr(session, "reject_actor_turn", None)
            if callable(reject_turn):
                reject_turn(reason=str(exc), category="malformed_tool_call")
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

    # [项目注释] 功能：`run`：异步地编排一个训练、采集、评测或 replay 流程，并汇总其结果。 主要协作调用：dict, task_id_from_run_kwargs,
    # [项目注释]    prepare_actor_prompt, astart_session。
    # [项目注释] 输入：`sampling_params`: Any；**`kwargs`。
    # [项目注释] 输出：标注返回 `Any`；具体值由各分支决定。
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
            turn_credit_mode = getattr(self, "turn_credit_mode", "off")
            configure_turn_credit = getattr(session, "configure_turn_credit", None)
            if callable(configure_turn_credit):
                configure_turn_credit(
                    mode=turn_credit_mode,
                    config=getattr(self, "turn_credit_config", TurnCreditConfig()),
                )
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
            finalize_turn_credit = getattr(session, "finalize_turn_credit", None)
            turn_trace = (
                finalize_turn_credit(reward)
                if turn_credit_mode != "off" and callable(finalize_turn_credit)
                else None
            )
            extra_fields = getattr(output, "extra_fields", None)
            if extra_fields is None:
                output.extra_fields = {}
                extra_fields = output.extra_fields
            policy_metadata = actor_policy_metadata(
                actor_policy_enabled=self.actor_policy_enabled,
                actor_policy_version=self.actor_policy_version,
            )
            if turn_trace is not None:
                extra_fields["turn_credit"] = turn_trace.to_extra_field(
                    mode=turn_credit_mode
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
                **(
                    turn_trace.metrics(mode=turn_credit_mode)
                    if turn_trace is not None else {}
                ),
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
                "correct_answer_rate": float(
                    reward.get("correct_answer_rate", reward.get("completion_rate", 0.0))
                ),
                "answer_submission_rate": float(
                    reward.get("answer_submission_rate", 0.0)
                ),
                "answer_quality": float(reward.get("answer_quality", 0.0)),
                "preference_coverage": float(reward.get("preference_coverage", 0.0)),
                "phase_transition_score": float(
                    reward.get("phase_transition_score", 0.0)
                ),
                "guard_rejections": int(getattr(session, "guard_rejections", 0)),
                "guard_rejection_rate": float(
                    reward.get("guard_rejection_rate", 0.0)
                ),
                "blocked_aspects": int(reward.get("blocked_aspects", 0)),
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
