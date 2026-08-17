"""Teacher trajectory collection against an isolated UserBench simulator API."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from travel_grpo.envs.userbench_interaction import (
    DEEPSEEK_V4_FLASH_MODEL,
    SimulatorRole,
    UserSimulatorRuntime,
)
from travel_grpo.envs.userbench_wrapper import (
    UserBenchEnvironmentConfig,
    UserBenchWrapper,
)
from travel_grpo.envs.userbench_context import UserBenchSessionState
from travel_grpo.envs.reward import REWARD_VERSION, SUPPORTED_REWARD_VERSIONS
from travel_grpo.models.openai_compatible import (
    TeacherClientProtocol,
    TeacherRequestConstraint,
)
from travel_grpo.prompts.actor_policy import (
    ACTOR_RUNTIME_POLICY,
    ACTOR_RUNTIME_POLICY_VERSION,
    TEACHER_GENERATION_INSTRUCTION,
    ensure_teacher_generation_messages,
    strip_teacher_generation_instruction,
)
from travel_grpo.envs.userbench_tools import (
    normalized_action_signature,
    semantic_action_signature,
)
from travel_grpo.training.teacher_policy import (
    POLICY_VERSION,
    AttemptStrategy,
    TeacherPhase,
    TeacherPolicyState,
    canonical_content_for,
)

from travel_grpo.training.sft.contracts import (
    COLLECTION_DIAGNOSTIC_SCHEMA_VERSION,
    TRAJECTORY_SCHEMA_VERSION,
    TeacherAttemptDiagnostic,
    TeacherTaskOutcome,
    TeacherTrajectory,
)
from travel_grpo.training.sft.errors import (
    TeacherAttemptAbort,
    TeacherCollectionError,
    TeacherGenerationError,
)
from travel_grpo.training.sft.planning import (
    assert_disjoint_from_evaluation,
    build_stratified_task_plan,
    load_teacher_task_pool,
    select_stratified_task_wave,
)


SIMULATOR_FALLBACK_TEXT = (
    "I'm sorry, I'm not sure how to respond to your latest utterance right now. "
    "Please try again."
)
# Compatibility export: old scripts imported this name for the Actor-facing
# suffix. It now resolves to the production runtime policy; Teacher-only
# generation controls are exposed separately and are never archived in Actor
# messages.
TEACHER_ACTOR_POLICY = ACTOR_RUNTIME_POLICY


def validate_teacher_collection_config(path: str | Path) -> Mapping[str, Any]:
    """Validate the model-role contract recorded in the collection YAML."""

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - installed by the API extra.
        raise TeacherCollectionError(
            "teacher collection configuration requires PyYAML; run "
            "`pip install -e .[api]`"
        ) from exc
    config_path = Path(path)
    try:
        document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TeacherCollectionError(
            f"cannot read teacher collection config: {config_path}"
        ) from exc
    if not isinstance(document, Mapping):
        raise TeacherCollectionError("teacher collection config must be a mapping")
    teacher = document.get("teacher")
    simulator = document.get("simulator")
    collection = document.get("collection")
    if not all(
        isinstance(value, Mapping) for value in (teacher, simulator, collection)
    ):
        raise TeacherCollectionError(
            "teacher collection config requires teacher, simulator, and collection mappings"
        )
    if str(teacher.get("model", "")).casefold() != DEEPSEEK_V4_FLASH_MODEL:
        raise TeacherCollectionError(
            "configured teacher model must be deepseek-v4-flash"
        )
    if str(simulator.get("model", "")).casefold() != DEEPSEEK_V4_FLASH_MODEL:
        raise TeacherCollectionError(
            "configured collection simulator model must be deepseek-v4-flash"
        )
    if collection.get("max_steps") != 20:
        raise TeacherCollectionError("teacher collection max_steps must be 20")
    if collection.get("strict_filter") is not True:
        raise TeacherCollectionError("teacher collection strict_filter must be true")
    if collection.get("capture_upstream_diagnostics") is not True:
        raise TeacherCollectionError(
            "teacher collection capture_upstream_diagnostics must be true"
        )
    if collection.get("reward_version") != REWARD_VERSION:
        raise TeacherCollectionError(
            "teacher collection reward_version must be " + REWARD_VERSION
        )
    if collection.get("minimum_terminal_reward") != 0.7:
        raise TeacherCollectionError(
            "teacher collection minimum_terminal_reward must be 0.7"
        )
    if collection.get("require_zero_policy_penalty") is not True:
        raise TeacherCollectionError(
            "teacher collection must require zero policy penalty"
        )
    if collection.get("policy_version") != POLICY_VERSION:
        raise TeacherCollectionError(
            f"teacher collection policy_version must be {POLICY_VERSION}"
        )
    configured_actor_policy_version = collection.get("actor_policy_version")
    if configured_actor_policy_version not in (None, ACTOR_RUNTIME_POLICY_VERSION):
        raise TeacherCollectionError(
            "teacher collection actor_policy_version must be "
            f"{ACTOR_RUNTIME_POLICY_VERSION}"
        )
    if collection.get("fail_fast_on_strict_violation") is not True:
        raise TeacherCollectionError(
            "teacher collection must fail fast on strict violations"
        )
    if collection.get("checkpoint_each_task") is not True:
        raise TeacherCollectionError(
            "teacher collection must checkpoint every completed task"
        )
    if collection.get("resume_safe") is not True:
        raise TeacherCollectionError("teacher collection resume_safe must be true")
    silver = collection.get("silver")
    if not isinstance(silver, Mapping) or silver.get("enabled") is not True:
        raise TeacherCollectionError("silver collection tier must be enabled")
    if silver.get("max_judgment_fallbacks") != 1:
        raise TeacherCollectionError("silver allows exactly one judgment fallback")
    if silver.get("max_vague_repairs") != 1:
        raise TeacherCollectionError("silver allows exactly one vague repair")
    if silver.get("max_search_repairs") != 1:
        raise TeacherCollectionError("silver allows exactly one search repair")
    if silver.get("max_elicitation_repairs_per_field") != 1:
        raise TeacherCollectionError(
            "silver allows exactly one elicitation repair per field"
        )
    if silver.get("minimum_raw_terminal_reward") != 0.7:
        raise TeacherCollectionError(
            "silver minimum_raw_terminal_reward must be 0.7"
        )
    return document


def write_stratified_selection_manifest(
    path: str | Path, document: Mapping[str, Any]
) -> Path:
    """Atomically persist a credential-free adaptive sampling manifest."""

    destination = Path(path)
    _atomic_json_record(document, destination)
    return destination


WrapperFactory = Callable[..., UserBenchWrapper]


def task_dimensions(task_id: str) -> tuple[str, ...]:
    """Extract UserBench travel aspects from its compound task ID."""

    dimensions = tuple(
        part.split(":", 1)[0] for part in task_id.split("|") if ":" in part
    )
    allowed = {"flight", "hotel", "apartment", "rental_car", "restaurant"}
    if not dimensions or any(value not in allowed for value in dimensions):
        raise TeacherCollectionError(
            f"cannot derive travel dimensions from task {task_id!r}"
        )
    if len(set(dimensions)) != len(dimensions):
        raise TeacherCollectionError(f"task {task_id!r} contains duplicate dimensions")
    return dimensions


def _prepare_teacher_messages(
    prompt: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build the Teacher request while keeping Teacher-only text request-local."""

    try:
        return ensure_teacher_generation_messages(prompt)
    except ValueError as exc:
        raise TeacherCollectionError(str(exc)) from exc


def _simulator_fallback(result: Any) -> bool:
    if result.observation.to_tool_text().strip() == SIMULATOR_FALLBACK_TEXT:
        return True
    diagnostics = result.diagnostics
    for key in ("simulator_fallback", "response_fallback", "generation_fallback"):
        if diagnostics.get(key) is True:
            return True
    return False


def _simulator_diagnostic_count(result: Any, name: str) -> int:
    value = result.diagnostics.get(name, 0)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _partial_trajectory_record(
    *,
    task_id: str,
    trajectory_attempt: int,
    messages: Sequence[Mapping[str, Any]],
    rewards: Sequence[float],
    dimensions: Sequence[str],
    answered_aspects: set[str],
    committed_actions: Sequence[Mapping[str, Any]],
    generation_diagnostics: Sequence[Mapping[str, Any]],
    simulator_fallbacks: int,
    simulator_judgment_fallbacks: int,
    simulator_search_fallbacks: int,
    terminated: bool,
    truncated: bool,
    teacher_request_count: int,
    teacher_usage: Mapping[str, int],
) -> dict[str, Any]:
    """Build a credential-free rejected-attempt snapshot for diagnosis."""

    return {
        "task_id": task_id,
        "trajectory_attempt": trajectory_attempt,
        "failure_environment_turn": len(rewards) + 1,
        "environment_steps_completed": len(rewards),
        "messages": strip_teacher_generation_instruction(messages),
        "actor_policy_version": ACTOR_RUNTIME_POLICY_VERSION,
        "step_rewards": list(rewards),
        "cumulative_reward": float(sum(rewards)),
        "expected_aspects": list(dimensions),
        "answered_aspects": [
            aspect for aspect in dimensions if aspect in answered_aspects
        ],
        "committed_actions": [dict(action) for action in committed_actions],
        "generation_diagnostics": [dict(value) for value in generation_diagnostics],
        "simulator_fallbacks": simulator_fallbacks,
        "simulator_judgment_fallbacks": simulator_judgment_fallbacks,
        "simulator_search_fallbacks": simulator_search_fallbacks,
        "terminated": terminated,
        "truncated": truncated,
        "teacher_request_count": teacher_request_count,
        "teacher_usage": dict(teacher_usage),
    }


def _message_contract_errors(messages: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    errors: list[str] = []
    if len(messages) < 4 or [message.get("role") for message in messages[:2]] != [
        "system",
        "user",
    ]:
        return ("invalid_message_prefix",)
    remainder = messages[2:]
    if len(remainder) % 2:
        errors.append("unpaired_assistant_tool_messages")
    for index in range(0, len(remainder) - 1, 2):
        assistant, tool = remainder[index : index + 2]
        calls = (
            assistant.get("tool_calls")
            if assistant.get("role") == "assistant"
            else None
        )
        if not isinstance(calls, list) or len(calls) != 1:
            errors.append("assistant_must_contain_one_tool_call")
            continue
        call_id = calls[0].get("id") if isinstance(calls[0], Mapping) else None
        if tool.get("role") != "tool" or tool.get("tool_call_id") != call_id:
            errors.append("tool_call_id_mismatch")
    return tuple(sorted(set(errors)))


def _feedback_policy_errors(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    """Map deterministic UserBench rejection feedback to stable SFT reasons."""

    errors: set[str] = set()
    for message in messages:
        if message.get("role") != "tool":
            continue
        content = str(message.get("content") or "")
        if "too vague and general" in content:
            errors.add("vague_action_feedback")
        if "already recommended an option" in content:
            errors.add("duplicate_recommendation_feedback")
        if "Invalid option ID format" in content:
            errors.add("invalid_option_id_feedback")
    return tuple(sorted(errors))


def trajectory_rejection_reasons(trajectory: TeacherTrajectory) -> tuple[str, ...]:
    """Apply the strict SFT admission gate to a completed collection attempt."""

    reasons: list[str] = []
    if not trajectory.terminated:
        reasons.append("not_terminated")
    if trajectory.truncated:
        reasons.append("truncated")
    if set(trajectory.answered_aspects) != set(trajectory.expected_aspects):
        reasons.append("incomplete_aspect_answers")
    if trajectory.simulator_fallbacks:
        reasons.append("simulator_fallback")
    if trajectory.simulator_judgment_fallbacks:
        reasons.append("simulator_judgment_fallback")
    if trajectory.simulator_search_fallbacks:
        reasons.append("simulator_search_fallback")
    reasons.extend(_message_contract_errors(trajectory.messages))
    reasons.extend(_feedback_policy_errors(trajectory.messages))
    reward = trajectory.reward_breakdown
    if not isinstance(reward, Mapping):
        reasons.append("missing_reward_evidence")
    else:
        if reward.get("reward_version") not in SUPPORTED_REWARD_VERSIONS:
            reasons.append("wrong_reward_version")
        if reward.get("reward_valid") is not True:
            reasons.append("reward_invalid")
        if reward.get("completion_rate") != 1.0:
            reasons.append("incomplete_reward_completion")
        if reward.get("correct_itinerary") is not True:
            reasons.append("incorrect_itinerary")
        policy_penalty = reward.get("policy_penalty")
        if not isinstance(policy_penalty, (int, float)) or isinstance(
            policy_penalty, bool
        ) or float(policy_penalty) != 0.0:
            reasons.append("policy_penalty")
        terminal_reward = reward.get("terminal_reward")
        if not isinstance(terminal_reward, (int, float)) or isinstance(
            terminal_reward, bool
        ) or not math.isfinite(float(terminal_reward)) or float(terminal_reward) < 0.7:
            reasons.append("terminal_reward_below_threshold")
        for field in (
            "invalid_actions",
            "exact_repeats",
            "semantic_repeats",
            "ambiguous_actions",
            "unsearched_answers",
            "wrong_answers",
        ):
            value = reward.get(field, 0)
            if not isinstance(value, int) or isinstance(value, bool) or value != 0:
                reasons.append(field)
        if reward.get("infrastructure_errors"):
            reasons.append("infrastructure_error")
    return tuple(sorted(set(reasons)))


_SILVER_ALLOWED_REASONS = frozenset(
    {
        "vague_action_feedback",
        "simulator_judgment_fallback",
        "reward_invalid",
        "terminal_reward_below_threshold",
        "infrastructure_error",
    }
)


def _silver_markers(trajectory: TeacherTrajectory) -> set[str]:
    markers = {
        str(value.get("reason"))
        for value in trajectory.generation_diagnostics
        if isinstance(value, Mapping) and value.get("reason")
    }
    if any(
        message.get("role") == "assistant" and message.get("loss_mask") is True
        for message in trajectory.messages
        if isinstance(message, Mapping)
    ):
        markers.add("loss_masked_repair")
    return markers


def quality_tier_for_trajectory(
    trajectory: TeacherTrajectory,
    strict_reasons: Sequence[str] | None = None,
) -> str | None:
    """Return ``gold``/``silver`` or ``None`` for a rejected trajectory.

    Silver is deliberately narrower than a generic relaxed gate.  It permits
    only one explicitly recorded recoverable repair or one valid simulator
    judgment fallback; correctness, termination, protocol integrity and
    visible-answer constraints remain hard requirements.
    """

    reasons = set(strict_reasons or trajectory_rejection_reasons(trajectory))
    markers = _silver_markers(trajectory)
    repair_markers = {
        "search_repair_allowed",
        "vague_action_repair_allowed",
        "judgment_fallback_allowed",
        "loss_masked_repair",
    }
    if not reasons:
        return "silver" if markers & repair_markers else "gold"
    if not markers & repair_markers:
        return None
    if reasons - _SILVER_ALLOWED_REASONS:
        return None
    if trajectory.simulator_fallbacks or trajectory.simulator_search_fallbacks:
        return None
    if trajectory.simulator_judgment_fallbacks:
        if trajectory.simulator_judgment_fallbacks != 1:
            return None
        # Older collectors persisted the consumed fallback turn with
        # ``loss_mask=True`` but did not persist the newer diagnostic marker.
        # ``quality_tier_for_trajectory`` already requires a repair marker;
        # the durable loss mask is sufficient evidence for those legacy Silver
        # records, and the fallback count is still capped at exactly one.
    reward = trajectory.reward_breakdown
    if not isinstance(reward, Mapping):
        return None
    if reward.get("completion_rate") != 1.0:
        return None
    if reward.get("correct_itinerary") is not True:
        return None
    if reward.get("policy_penalty") != 0.0:
        return None
    for field_name in (
        "invalid_actions",
        "exact_repeats",
        "semantic_repeats",
        "ambiguous_actions",
        "unsearched_answers",
        "wrong_answers",
    ):
        if reward.get(field_name, 0) != 0:
            return None
    raw_terminal = reward.get("raw_terminal_reward")
    if (
        not isinstance(raw_terminal, (int, float))
        or isinstance(raw_terminal, bool)
        or not math.isfinite(float(raw_terminal))
        or float(raw_terminal) < 0.7
    ):
        return None
    return "silver"


async def collect_teacher_trajectory(
    task: Mapping[str, Any],
    *,
    teacher: TeacherClientProtocol,
    simulator: UserSimulatorRuntime,
    wrapper_factory: WrapperFactory = UserBenchWrapper,
    source_root: str | Path | None = None,
    trajectory_attempt: int = 1,
) -> TeacherTrajectory:
    """Collect one state-machine-controlled tool trajectory."""

    if teacher.runtime.model.casefold() != DEEPSEEK_V4_FLASH_MODEL:
        raise TeacherCollectionError("teacher must use deepseek-v4-flash")
    if simulator.role is not SimulatorRole.COLLECTION:
        raise TeacherCollectionError("teacher collection requires collection simulator role")
    if simulator.model.casefold() != DEEPSEEK_V4_FLASH_MODEL:
        raise TeacherCollectionError("collection simulator must use deepseek-v4-flash")

    task_id = str(task.get("task_id") or "")
    prompt = task.get("prompt")
    if not task_id or not isinstance(prompt, list):
        raise TeacherCollectionError("teacher task is missing task_id or prompt")
    messages = _prepare_teacher_messages(prompt)
    dimensions = task_dimensions(task_id)
    strategy = AttemptStrategy.for_attempt(trajectory_attempt)
    wrapper = wrapper_factory(
        task_id,
        simulator,
        UserBenchEnvironmentConfig(capture_upstream_diagnostics=True),
        source_root=source_root,
    )
    rewards: list[float] = []
    terminated = False
    truncated = False
    answered_aspects: set[str] = set()
    exact_signatures: set[str] = set()
    semantic_signatures: set[tuple[str, str]] = set()
    generation_diagnostics: list[dict[str, Any]] = []
    committed_actions: list[dict[str, Any]] = []
    simulator_fallbacks = 0
    simulator_judgment_fallbacks = 0
    simulator_search_fallbacks = 0
    search_repairs: dict[str, int] = {}
    elicitation_repair_fields: set[str] = set()
    vague_repairs: dict[str, int] = {}
    reward_state: UserBenchSessionState | None = None
    reward_breakdown: dict[str, Any] | None = None
    teacher_request_count = 0
    teacher_usage: Counter[str] = Counter()
    try:
        wrapper.reset()
        reward_task = wrapper.reward_task()
        reward_state = UserBenchSessionState(
            request_id=f"teacher-{task_id}-{trajectory_attempt}",
            task_id=task_id,
            wrapper=wrapper,
            reward_task=reward_task,
            reward_snapshot=wrapper.reward_snapshot(),
        )
        policy = TeacherPolicyState(reward_task, strategy)
        for turn in range(wrapper.config.max_steps):
            try:
                plan = policy.next_plan(reward_state, messages)
            except RuntimeError as exc:
                code = str(exc)
                raise TeacherAttemptAbort(code, f"task {task_id!r}: {code}") from exc

            tool_call = None
            for generation_attempt in range(1, teacher.runtime.action_retries + 2):
                instruction = plan.instruction(generation_attempt)
                elicitation_repair_key = f"{plan.aspect}.{plan.field}"
                is_elicitation_repair = (
                    plan.phase is TeacherPhase.ELICIT
                    and elicitation_repair_key in elicitation_repair_fields
                )
                if is_elicitation_repair:
                    instruction += (
                        " The previous focused preference question was not recorded. "
                        "Use the materially different repair wording supplied by the "
                        "content constraint; ask about the same single field only."
                    )
                if plan.phase is TeacherPhase.SEARCH and search_repairs.get(plan.aspect, 0):
                    instruction += (
                        " The previous search for this aspect was not recorded. Emit a "
                        "materially different corrected query, not a copy of the previous "
                        "query; restate all public arguments and preferences."
                    )
                request_messages = [
                    *messages,
                    {"role": "user", "content": instruction},
                ]
                strongest_retry = (
                    strategy is AttemptStrategy.CANONICAL
                    or generation_attempt > teacher.runtime.action_retries
                    or (
                        strategy is AttemptStrategy.STRICT
                        and generation_attempt >= teacher.runtime.action_retries
                    )
                )
                if is_elicitation_repair:
                    assert plan.elicitation_repair_content is not None
                    allowed_contents = (plan.elicitation_repair_content,)
                else:
                    allowed_contents = (
                        canonical_content_for(plan)
                        if strongest_retry or plan.phase is TeacherPhase.ANSWER
                        else ()
                    )
                constraint = TeacherRequestConstraint(plan.choice, allowed_contents)
                candidate = await teacher.generate_action(
                    request_messages,
                    constraint=constraint,
                )
                teacher_request_count += candidate.protocol_attempts
                teacher_usage.update(candidate.usage)
                for protocol_rejection in candidate.protocol_rejections:
                    usage = protocol_rejection.get("usage")
                    if isinstance(usage, Mapping):
                        teacher_usage.update(
                            {
                                str(name): int(value)
                                for name, value in usage.items()
                                if isinstance(value, int) and not isinstance(value, bool)
                            }
                        )
                for rejection in candidate.protocol_rejections:
                    generation_diagnostics.append(
                        {
                            "environment_turn": turn + 1,
                            "generation_attempt": generation_attempt,
                            "reason": "teacher_protocol_retry",
                            "phase": plan.phase.value,
                            "aspect": plan.aspect,
                            "field": plan.field,
                            "attempt_strategy": strategy.value,
                            "detail": dict(rejection),
                        }
                    )
                exact = normalized_action_signature(candidate.action)
                semantic = semantic_action_signature(candidate.action, dimensions)
                rejection_reason = plan.validate(candidate.action)
                if rejection_reason is None and exact in exact_signatures:
                    rejection_reason = "duplicate_action"
                if (
                    rejection_reason is None
                    and semantic is not None
                    and semantic in semantic_signatures
                ):
                    rejection_reason = "semantic_duplicate_action"
                if rejection_reason is None:
                    tool_call = candidate
                    break
                content = candidate.action.content
                generation_diagnostics.append(
                    {
                        "environment_turn": turn + 1,
                        "generation_attempt": generation_attempt,
                        "reason": rejection_reason,
                        "phase": plan.phase.value,
                        "aspect": plan.aspect,
                        "field": plan.field,
                        "choice": candidate.action.choice.value,
                        "content": content[:500],
                        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                        "thought_length": len(candidate.action.thought),
                        "attempt_strategy": strategy.value,
                        "canonical_constraint": bool(allowed_contents),
                    }
                )
                if generation_attempt > teacher.runtime.action_retries:
                    raise TeacherGenerationError(
                        f"task {task_id!r} exhausted action retries: {rejection_reason}",
                        generation_diagnostics,
                        reason_code=f"teacher_action_exhausted.{rejection_reason}",
                    )
            assert tool_call is not None

            before_snapshot = reward_state.reward_snapshot
            before_wrong = reward_state.wrong_answers
            before_unsearched = reward_state.unsearched_answers
            result = await wrapper.astep(tool_call.action)
            after_snapshot = wrapper.reward_snapshot()
            active_delta = (
                0
                if before_snapshot is None
                else max(
                    0,
                    after_snapshot.active_elicited_count
                    - before_snapshot.active_elicited_count,
                )
            )
            passive_delta = (
                0
                if before_snapshot is None
                else max(
                    0,
                    after_snapshot.passive_elicited_count
                    - before_snapshot.passive_elicited_count,
                )
            )
            elicitation_not_recorded = (
                plan.phase is TeacherPhase.ELICIT and active_delta <= 0
            )
            reward_state.record_step(
                result,
                tool_call.action,
                after_snapshot,
                count_action_repetition=not elicitation_not_recorded,
            )
            exact_signatures.add(normalized_action_signature(tool_call.action))
            semantic = semantic_action_signature(tool_call.action, dimensions)
            if semantic is not None:
                semantic_signatures.add(semantic)
            answered_aspects = set(reward_state.answers)

            response_fallback = _simulator_fallback(result)
            judgment_fallbacks = _simulator_diagnostic_count(
                result, "userbench_judgment_fallbacks"
            )
            search_fallbacks = _simulator_diagnostic_count(
                result, "userbench_search_fallbacks"
            )
            if response_fallback:
                simulator_fallbacks += 1
            simulator_judgment_fallbacks += judgment_fallbacks
            simulator_search_fallbacks += search_fallbacks
            if plan.phase is not TeacherPhase.ELICIT or active_delta > 0:
                policy.record_committed(plan)
            committed_actions.append(
                {
                    "environment_turn": turn + 1,
                    "phase": plan.phase.value,
                    "aspect": plan.aspect,
                    "field": plan.field,
                    "attempt_strategy": strategy.value,
                    "choice": tool_call.action.choice.value,
                    "content": tool_call.action.content,
                    "thought_length": len(tool_call.action.thought),
                    "normalized_signature": normalized_action_signature(tool_call.action),
                    "semantic_aspect": semantic[0] if semantic else None,
                    "semantic_field": semantic[1] if semantic else None,
                    "active_preference_delta": active_delta,
                    "passive_preference_delta": passive_delta,
                    "teacher_latency_seconds": tool_call.latency_seconds,
                    "teacher_usage": dict(tool_call.usage),
                    "reward": result.reward,
                    "loss_mask": False,
                    "terminated": result.terminated,
                    "truncated": result.truncated,
                }
            )
            assistant_message_index = len(messages)
            messages.append(tool_call.to_assistant_message())
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.call_id,
                    "name": "interact_with_env",
                    "content": result.observation.to_tool_text(),
                }
            )
            rewards.append(result.reward)
            terminated = result.terminated
            truncated = result.truncated

            feedback = result.observation.to_tool_text()
            fail_reason = None
            if response_fallback:
                fail_reason = "simulator.response_fallback"
            elif judgment_fallbacks:
                fail_reason = "simulator.judgment_fallback"
            elif search_fallbacks:
                fail_reason = "simulator.search_fallback"
            elif "too vague and general" in feedback:
                fail_reason = "environment.vague_action_feedback"
            elif "already recommended an option" in feedback:
                fail_reason = "environment.duplicate_recommendation"
            elif "Invalid option ID format" in feedback:
                fail_reason = "environment.invalid_option_id"
            elif reward_state.wrong_answers > before_wrong:
                fail_reason = "environment.wrong_answer"
            elif reward_state.unsearched_answers > before_unsearched:
                fail_reason = "environment.unsearched_answer"
            elif elicitation_not_recorded:
                fail_reason = "environment.elicitation_not_recorded"
            elif (
                plan.phase is TeacherPhase.SEARCH
                and plan.aspect not in reward_state.searched_aspects
            ):
                fail_reason = "environment.search_not_recorded"
            elif (
                plan.phase is TeacherPhase.ANSWER
                and plan.aspect not in reward_state.answers
            ):
                fail_reason = "environment.answer_not_recorded"
            if (
                fail_reason == "simulator.judgment_fallback"
                and not response_fallback
                and simulator_judgment_fallbacks == 1
                and feedback.strip()
            ):
                # Keep the turn for auditability, but admit at most one valid
                # judgment fallback as silver. The reward ledger remains
                # infrastructure-invalid, so this can never become gold.
                messages[assistant_message_index]["loss_mask"] = True
                committed_actions[-1]["loss_mask"] = True
                generation_diagnostics.append(
                    {
                        "environment_turn": turn + 1,
                        "generation_attempt": 0,
                        "reason": "judgment_fallback_allowed",
                        "phase": plan.phase.value,
                        "aspect": plan.aspect,
                        "field": plan.field,
                        "attempt_strategy": strategy.value,
                    }
                )
                if plan.phase is TeacherPhase.ELICIT and active_delta <= 0:
                    repair_key = f"{plan.aspect}.{plan.field}"
                    elicitation_repair_fields.add(repair_key)
                    exact_signatures.discard(
                        normalized_action_signature(tool_call.action)
                    )
                    if semantic is not None:
                        semantic_signatures.discard(semantic)
                fail_reason = None
            elif fail_reason == "environment.vague_action_feedback":
                repair_key = f"{plan.aspect}.{plan.field or plan.phase.value}"
                if vague_repairs.get(repair_key, 0) < 1:
                    vague_repairs[repair_key] = vague_repairs.get(repair_key, 0) + 1
                    if plan.phase is TeacherPhase.ELICIT and plan.field is not None:
                        policy.asked_fields[plan.aspect].discard(plan.field)
                        if active_delta <= 0:
                            elicitation_repair_fields.add(repair_key)
                    exact_signatures.discard(
                        normalized_action_signature(tool_call.action)
                    )
                    if semantic is not None:
                        semantic_signatures.discard(semantic)
                    messages[assistant_message_index]["loss_mask"] = True
                    committed_actions[-1]["loss_mask"] = True
                    generation_diagnostics.append(
                        {
                            "environment_turn": turn + 1,
                            "generation_attempt": 0,
                            "reason": "vague_action_repair_allowed",
                            "phase": plan.phase.value,
                            "aspect": plan.aspect,
                            "field": plan.field,
                            "attempt_strategy": strategy.value,
                        }
                    )
                    fail_reason = None
            if fail_reason is not None:
                elicitation_repair_key = f"{plan.aspect}.{plan.field}"
                if (
                    fail_reason == "environment.elicitation_not_recorded"
                    and elicitation_repair_key not in elicitation_repair_fields
                ):
                    elicitation_repair_fields.add(elicitation_repair_key)
                    # The environment consumed this turn but did not credit an
                    # active preference. Preserve it for audit/context, exclude it
                    # from SFT loss, and retry the same uncommitted field once.
                    if semantic is not None:
                        semantic_signatures.discard(semantic)
                    exact_signatures.discard(
                        normalized_action_signature(tool_call.action)
                    )
                    messages[assistant_message_index]["loss_mask"] = True
                    committed_actions[-1]["loss_mask"] = True
                    generation_diagnostics.append(
                        {
                            "environment_turn": turn + 1,
                            "generation_attempt": 0,
                            "reason": "elicitation_repair_allowed",
                            "phase": plan.phase.value,
                            "aspect": plan.aspect,
                            "field": plan.field,
                            "attempt_strategy": strategy.value,
                        }
                    )
                    continue
                if (
                    fail_reason == "environment.search_not_recorded"
                    and search_repairs.get(plan.aspect, 0) < 1
                ):
                    search_repairs[plan.aspect] = search_repairs.get(plan.aspect, 0) + 1
                    # Keep the consumed turn in context, but do not train the
                    # actor to reproduce a search UserBench did not record.
                    messages[assistant_message_index]["loss_mask"] = True
                    committed_actions[-1]["loss_mask"] = True
                    generation_diagnostics.append(
                        {
                            "environment_turn": turn + 1,
                            "generation_attempt": 0,
                            "reason": "search_repair_allowed",
                            "phase": plan.phase.value,
                            "aspect": plan.aspect,
                            "field": plan.field,
                            "attempt_strategy": strategy.value,
                        }
                    )
                    continue
                raise TeacherAttemptAbort(
                    fail_reason,
                    f"task {task_id!r} cannot pass strict admission: {fail_reason}",
                )
            if result.done:
                break
        if not (terminated or truncated):
            raise TeacherAttemptAbort(
                "environment.max_steps_without_terminal",
                f"task {task_id!r} exceeded 20 calls without a terminal state",
            )
        reward_breakdown = reward_state.reward_report()
    except Exception as exc:
        exc.partial_trajectory = _partial_trajectory_record(
            task_id=task_id,
            trajectory_attempt=trajectory_attempt,
            messages=messages,
            rewards=rewards,
            dimensions=dimensions,
            answered_aspects=answered_aspects,
            committed_actions=committed_actions,
            generation_diagnostics=generation_diagnostics,
            simulator_fallbacks=simulator_fallbacks,
            simulator_judgment_fallbacks=simulator_judgment_fallbacks,
            simulator_search_fallbacks=simulator_search_fallbacks,
            terminated=terminated,
            truncated=truncated,
            teacher_request_count=teacher_request_count,
            teacher_usage=teacher_usage,
        )
        raise
    finally:
        wrapper.close()

    return TeacherTrajectory(
        task_id=task_id,
        composition=str(task["composition"]),
        difficulty=str(task["difficulty"]),
        source_split=str(task["source_split"]),
        teacher_model=teacher.runtime.model,
        simulator_model=simulator.model,
        # The Teacher request contains an extra generation-only system block;
        # archive only the Actor-visible runtime policy for future SFT.
        messages=tuple(strip_teacher_generation_instruction(messages)),
        step_rewards=tuple(rewards),
        terminated=terminated,
        truncated=truncated,
        expected_aspects=dimensions,
        answered_aspects=tuple(value for value in dimensions if value in answered_aspects),
        simulator_fallbacks=simulator_fallbacks,
        simulator_judgment_fallbacks=simulator_judgment_fallbacks,
        simulator_search_fallbacks=simulator_search_fallbacks,
        generation_diagnostics=tuple(generation_diagnostics),
        trajectory_attempt=trajectory_attempt,
        reward_breakdown=reward_breakdown,
        policy_version=POLICY_VERSION,
        actor_policy_version=ACTOR_RUNTIME_POLICY_VERSION,
        attempt_strategy=strategy.value,
        teacher_request_count=teacher_request_count,
        teacher_usage=dict(teacher_usage),
    )


async def collect_teacher_trajectories(
    tasks: Sequence[Mapping[str, Any]],
    *,
    teacher: TeacherClientProtocol,
    simulator: UserSimulatorRuntime,
    concurrency: int = 1,
    wrapper_factory: WrapperFactory = UserBenchWrapper,
    source_root: str | Path | None = None,
) -> tuple[TeacherTrajectory, ...]:
    """Collect a task pool concurrently while preserving input order."""

    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    semaphore = asyncio.Semaphore(concurrency)

    async def collect(task: Mapping[str, Any]) -> TeacherTrajectory:
        async with semaphore:
            return await collect_teacher_trajectory(
                task,
                teacher=teacher,
                simulator=simulator,
                wrapper_factory=wrapper_factory,
                source_root=source_root,
            )

    return tuple(await asyncio.gather(*(collect(task) for task in tasks)))


async def collect_teacher_task_with_retries(
    task: Mapping[str, Any],
    *,
    teacher: TeacherClientProtocol,
    simulator: UserSimulatorRuntime,
    max_attempts: int = 3,
    wrapper_factory: WrapperFactory = UserBenchWrapper,
    source_root: str | Path | None = None,
) -> TeacherTaskOutcome:
    """Retry a whole trajectory and admit only attempts passing the strict gate."""

    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    task_id = str(task.get("task_id") or "")
    attempts: list[TeacherAttemptDiagnostic] = []
    for attempt in range(1, max_attempts + 1):
        try:
            trajectory = await collect_teacher_trajectory(
                task,
                teacher=teacher,
                simulator=simulator,
                wrapper_factory=wrapper_factory,
                source_root=source_root,
                trajectory_attempt=attempt,
            )
            reasons = trajectory_rejection_reasons(trajectory)
            quality_tier = quality_tier_for_trajectory(trajectory, reasons)
            accepted = quality_tier is not None
            if accepted and trajectory.quality_tier != quality_tier:
                trajectory = replace(trajectory, quality_tier=quality_tier)
            attempts.append(
                TeacherAttemptDiagnostic(
                    task_id=task_id,
                    attempt=attempt,
                    accepted=accepted,
                    rejection_reasons=reasons,
                    generation_diagnostics=trajectory.generation_diagnostics,
                    partial_trajectory=(
                        None
                        if accepted
                        else {
                            **trajectory.to_record(),
                            "generation_diagnostics": list(
                                trajectory.generation_diagnostics
                            ),
                        }
                    ),
                    attempt_strategy=trajectory.attempt_strategy,
                    quality_tier=quality_tier or "rejected",
                )
            )
            if accepted:
                return TeacherTaskOutcome(
                    task_id, trajectory, tuple(attempts), quality_tier
                )
        except Exception as exc:  # collection failures belong in diagnostics, not SFT
            diagnostics = getattr(exc, "diagnostics", None)
            if diagnostics is None:
                diagnostics = [
                    {"reason": "teacher_protocol_retry", "detail": dict(value)}
                    for value in getattr(exc, "rejections", ())
                ]
            attempts.append(
                TeacherAttemptDiagnostic(
                    task_id=task_id,
                    attempt=attempt,
                    accepted=False,
                    rejection_reasons=(
                        getattr(
                            exc,
                            "reason_code",
                            f"collection_error.{exc.__class__.__name__}",
                        ),
                    ),
                    generation_diagnostics=tuple(diagnostics),
                    error_type=exc.__class__.__name__,
                    error_message=str(exc),
                    partial_trajectory=getattr(exc, "partial_trajectory", None),
                    attempt_strategy=AttemptStrategy.for_attempt(attempt).value,
                    quality_tier="rejected",
                )
            )
    return TeacherTaskOutcome(task_id, None, tuple(attempts), "rejected")


async def collect_teacher_outcomes(
    tasks: Sequence[Mapping[str, Any]],
    *,
    teacher: TeacherClientProtocol,
    simulator: UserSimulatorRuntime,
    concurrency: int = 1,
    max_attempts: int = 3,
    wrapper_factory: WrapperFactory = UserBenchWrapper,
    source_root: str | Path | None = None,
    on_outcome: Callable[[TeacherTaskOutcome], Any] | None = None,
) -> tuple[TeacherTaskOutcome, ...]:
    """Collect strict task outcomes concurrently while preserving task order."""

    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    semaphore = asyncio.Semaphore(concurrency)

    async def collect(task: Mapping[str, Any]) -> TeacherTaskOutcome:
        async with semaphore:
            outcome = await collect_teacher_task_with_retries(
                task,
                teacher=teacher,
                simulator=simulator,
                max_attempts=max_attempts,
                wrapper_factory=wrapper_factory,
                source_root=source_root,
            )
            if on_outcome is not None:
                result = on_outcome(outcome)
                if inspect.isawaitable(result):
                    await result
            return outcome

    return tuple(await asyncio.gather(*(collect(task) for task in tasks)))


def _atomic_json_record(record: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                dict(record),
                handle,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def initialize_teacher_run(
    run_dir: str | Path,
    task_ids: Sequence[str],
    *,
    resume: bool = False,
) -> Path:
    """Create or validate a resume-safe per-task checkpoint directory."""

    root = Path(run_dir)
    manifest_path = root / "manifest.json"
    expected = {
        "schema_version": "userbench-teacher-run-v1",
        "policy_version": POLICY_VERSION,
        "actor_policy_version": ACTOR_RUNTIME_POLICY_VERSION,
        "task_ids": list(task_ids),
    }
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(
                f"teacher run already exists; pass --resume: {root}"
            )
        try:
            current = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TeacherCollectionError("teacher run manifest is unreadable") from exc
        legacy_expected = {
            key: value for key, value in expected.items() if key != "actor_policy_version"
        }
        if current not in (expected, legacy_expected):
            raise TeacherCollectionError(
                "teacher run manifest does not match policy or ordered task IDs"
            )
    else:
        if resume:
            raise FileNotFoundError(f"teacher run does not exist: {root}")
        _atomic_json_record(expected, manifest_path)
    (root / "tasks").mkdir(parents=True, exist_ok=True)
    return root


def teacher_outcome_checkpoint_path(run_dir: str | Path, task_id: str) -> Path:
    digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()
    return Path(run_dir) / "tasks" / f"{digest}.json"


def write_teacher_outcome_checkpoint(
    outcome: TeacherTaskOutcome, run_dir: str | Path
) -> Path:
    destination = teacher_outcome_checkpoint_path(run_dir, outcome.task_id)
    _atomic_json_record(outcome.to_checkpoint_record(), destination)
    return destination


def load_teacher_outcome_checkpoints(
    run_dir: str | Path, task_ids: Sequence[str]
) -> dict[str, TeacherTaskOutcome]:
    outcomes: dict[str, TeacherTaskOutcome] = {}
    for task_id in task_ids:
        path = teacher_outcome_checkpoint_path(run_dir, task_id)
        if not path.exists():
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TeacherCollectionError(f"invalid task checkpoint: {path}") from exc
        outcome = TeacherTaskOutcome.from_checkpoint_record(record)
        if outcome.task_id != task_id:
            raise TeacherCollectionError("task checkpoint ID does not match its filename")
        outcomes[task_id] = outcome
    return outcomes


def summarize_teacher_outcomes(
    outcomes: Sequence[TeacherTaskOutcome],
) -> dict[str, Any]:
    """Return a credential-free run summary suitable for progress review."""

    rejection_reasons: Counter[str] = Counter()
    generation_reasons: Counter[str] = Counter()
    environment_steps = 0
    teacher_requests = 0
    teacher_usage: Counter[str] = Counter()
    for outcome in outcomes:
        for attempt in outcome.attempts:
            rejection_reasons.update(attempt.rejection_reasons)
            diagnostics = attempt.generation_diagnostics
            generation_reasons.update(
                str(value.get("reason", "unknown")) for value in diagnostics
            )
            partial = attempt.partial_trajectory or {}
            if attempt.accepted and outcome.trajectory is not None:
                steps = len(outcome.trajectory.step_rewards)
                recorded_requests = outcome.trajectory.teacher_request_count
                recorded_usage = outcome.trajectory.teacher_usage or {}
            else:
                steps = int(
                    partial.get(
                        "environment_steps_completed", partial.get("num_steps", 0)
                    )
                )
                recorded_requests = int(partial.get("teacher_request_count", 0))
                recorded_usage = partial.get("teacher_usage", {})
            environment_steps += steps
            action_rejections = sum(
                value.get("reason") != "teacher_protocol_retry"
                for value in diagnostics
            )
            protocol_retries = sum(
                value.get("reason") == "teacher_protocol_retry"
                for value in diagnostics
            )
            teacher_requests += recorded_requests or (
                steps + action_rejections + protocol_retries
            )
            if isinstance(recorded_usage, Mapping):
                teacher_usage.update(
                    {
                        str(name): int(value)
                        for name, value in recorded_usage.items()
                        if isinstance(value, int) and not isinstance(value, bool)
                    }
                )
    accepted = sum(value.accepted for value in outcomes)
    gold = sum(value.quality_tier == "gold" for value in outcomes)
    silver = sum(value.quality_tier == "silver" for value in outcomes)
    return {
        "tasks": len(outcomes),
        "accepted": accepted,
        "gold": gold,
        "silver": silver,
        "rejected": len(outcomes) - accepted,
        "acceptance_rate": 0.0 if not outcomes else accepted / len(outcomes),
        "attempts": sum(len(value.attempts) for value in outcomes),
        "environment_steps": environment_steps,
        "estimated_teacher_requests": teacher_requests,
        "teacher_usage": dict(sorted(teacher_usage.items())),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "generation_reasons": dict(sorted(generation_reasons.items())),
    }


def write_teacher_trajectories(
    trajectories: Sequence[TeacherTrajectory],
    output_path: str | Path,
    *,
    force: bool = False,
) -> Path:
    """Atomically write deterministic JSONL without credentials or endpoints."""

    destination = Path(output_path)
    if destination.exists() and not force:
        raise FileExistsError(f"trajectory output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for trajectory in trajectories:
                handle.write(
                    json.dumps(
                        trajectory.to_record(),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _write_jsonl_records(
    records: Sequence[Mapping[str, Any]], output_path: str | Path, *, force: bool
) -> Path:
    destination = Path(output_path)
    if destination.exists() and not force:
        raise FileExistsError(f"collection artifact already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(
                    json.dumps(
                        dict(record),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def write_teacher_collection_artifacts(
    outcomes: Sequence[TeacherTaskOutcome],
    *,
    accepted_path: str | Path,
    rejected_path: str | Path,
    diagnostics_path: str | Path,
    silver_path: str | Path | None = None,
    force: bool = False,

) -> tuple[Path, Path, Path, Path]:
    """Write gold, silver, rejected tasks, and diagnostics separately."""

    if silver_path is None:
        accepted = Path(accepted_path)
        stem = accepted.stem
        if stem.endswith(".accepted"):
            stem = stem[: -len(".accepted")]
        silver_path = accepted.with_name(f"{stem}.silver.jsonl")

    destinations = tuple(
        Path(value)
        for value in (accepted_path, silver_path, rejected_path, diagnostics_path)
    )
    if len(set(destinations)) != 4:
        raise ValueError("gold, silver, rejected, and diagnostics paths must be distinct")
    existing = [value for value in destinations if value.exists()]
    if existing and not force:
        raise FileExistsError(f"collection artifact already exists: {existing[0]}")
    accepted = [
        outcome.trajectory
        for outcome in outcomes
        if outcome.quality_tier == "gold" and outcome.trajectory is not None
    ]
    silver = [
        outcome.trajectory
        for outcome in outcomes
        if outcome.quality_tier == "silver" and outcome.trajectory is not None
    ]
    rejected = [
        outcome.rejected_record() for outcome in outcomes if not outcome.accepted
    ]
    diagnostics = [
        diagnostic.to_record()
        for outcome in outcomes
        for diagnostic in outcome.attempts
    ]
    write_teacher_trajectories(accepted, destinations[0], force=force)
    write_teacher_trajectories(silver, destinations[1], force=force)
    _write_jsonl_records(rejected, destinations[2], force=force)
    _write_jsonl_records(diagnostics, destinations[3], force=force)
    return destinations
