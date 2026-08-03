"""Teacher trajectory collection against an isolated UserBench simulator API."""

from __future__ import annotations

import asyncio
import copy
import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
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
from travel_grpo.models.openai_compatible import TeacherClientProtocol
from travel_grpo.envs.userbench_tools import (
    ActionChoice,
    FIELD_QUERY_HINTS,
    aspect_from_option_id,
    normalized_action_signature,
    semantic_action_signature,
)

TRAJECTORY_SCHEMA_VERSION = "userbench-teacher-trajectory-v3"
COLLECTION_DIAGNOSTIC_SCHEMA_VERSION = "userbench-teacher-diagnostic-v2"
SIMULATOR_FALLBACK_TEXT = (
    "I'm sorry, I'm not sure how to respond to your latest utterance right now. "
    "Please try again."
)
TEACHER_ACTOR_POLICY = """Teacher policy for strict UserBench trajectories:
- Emit exactly one interact_with_env call per turn. Keep thought to one short operational sentence of at most 200 characters.
- Search one travel aspect at a time. Ask exactly one concrete preference field per action; never ask vague "other preferences" questions or bundle fields.
- Field order: flight company/path/time/amenities/service; hotel or apartment name/room/amenities/service/rating; rental car brand/model/seats/insurance/service; restaurant cuisine/tags/rating/expectation.
- Never repeat an exact action or an already asked aspect/field. Ask no more than four focused questions per aspect.
- Reserve one turn per unanswered aspect plus two recovery turns. When only that reserve remains, answer an aspect instead of asking another question.
- Submit exactly one option ID for each answer and complete every requested aspect within 20 environment calls."""


class TeacherCollectionError(RuntimeError):
    """Raised when a task pool or collected trajectory violates the contract."""


class TeacherGenerationError(TeacherCollectionError):
    """Raised after request-local action correction attempts are exhausted."""

    def __init__(self, message: str, diagnostics: Sequence[Mapping[str, Any]]) -> None:
        super().__init__(message)
        self.diagnostics = tuple(dict(value) for value in diagnostics)


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
    return document


@dataclass(frozen=True)
class TeacherTrajectory:
    task_id: str
    composition: str
    difficulty: str
    source_split: str
    teacher_model: str
    simulator_model: str
    messages: tuple[dict[str, Any], ...]
    step_rewards: tuple[float, ...]
    terminated: bool
    truncated: bool
    expected_aspects: tuple[str, ...] = ()
    answered_aspects: tuple[str, ...] = ()
    simulator_fallbacks: int = 0
    simulator_judgment_fallbacks: int = 0
    simulator_search_fallbacks: int = 0
    generation_diagnostics: tuple[dict[str, Any], ...] = ()
    trajectory_attempt: int = 1

    @property
    def total_reward(self) -> float:
        return float(sum(self.step_rewards))

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": TRAJECTORY_SCHEMA_VERSION,
            "task_id": self.task_id,
            "composition": self.composition,
            "difficulty": self.difficulty,
            "source_split": self.source_split,
            "teacher_model": self.teacher_model,
            "simulator_model": self.simulator_model,
            "messages": list(self.messages),
            "step_rewards": list(self.step_rewards),
            "total_reward": self.total_reward,
            "num_steps": len(self.step_rewards),
            "terminated": self.terminated,
            "truncated": self.truncated,
            "expected_aspects": list(self.expected_aspects),
            "answered_aspects": list(self.answered_aspects),
            "simulator_fallbacks": self.simulator_fallbacks,
            "simulator_judgment_fallbacks": self.simulator_judgment_fallbacks,
            "simulator_search_fallbacks": self.simulator_search_fallbacks,
            "trajectory_attempt": self.trajectory_attempt,
        }


@dataclass(frozen=True)
class TeacherAttemptDiagnostic:
    task_id: str
    attempt: int
    accepted: bool
    rejection_reasons: tuple[str, ...]
    generation_diagnostics: tuple[dict[str, Any], ...] = ()
    error_type: str | None = None
    error_message: str | None = None
    partial_trajectory: Mapping[str, Any] | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": COLLECTION_DIAGNOSTIC_SCHEMA_VERSION,
            "task_id": self.task_id,
            "attempt": self.attempt,
            "accepted": self.accepted,
            "rejection_reasons": list(self.rejection_reasons),
            "generation_diagnostics": list(self.generation_diagnostics),
            "error_type": self.error_type,
            "error_message": self.error_message,
            "partial_trajectory": (
                None
                if self.partial_trajectory is None
                else dict(self.partial_trajectory)
            ),
        }


@dataclass(frozen=True)
class TeacherTaskOutcome:
    task_id: str
    trajectory: TeacherTrajectory | None
    attempts: tuple[TeacherAttemptDiagnostic, ...]

    @property
    def accepted(self) -> bool:
        return self.trajectory is not None

    def rejected_record(self) -> dict[str, Any]:
        final = self.attempts[-1]
        return {
            "schema_version": COLLECTION_DIAGNOSTIC_SCHEMA_VERSION,
            "task_id": self.task_id,
            "attempts": len(self.attempts),
            "rejection_reasons": list(final.rejection_reasons),
            "error_type": final.error_type,
            "error_message": final.error_message,
        }


def load_teacher_task_pool(
    path: str | Path, *, expected_source_split: str = "train"
) -> tuple[dict[str, Any], ...]:
    """Load the five-field SFT task contract produced by the split pipeline."""

    source = Path(path)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise TeacherCollectionError(
            f"cannot read teacher task pool: {source}"
        ) from exc
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TeacherCollectionError(
                f"invalid JSON in teacher task pool at line {index}"
            ) from exc
        if not isinstance(record, dict):
            raise TeacherCollectionError(f"teacher task line {index} must be an object")
        task_id = record.get("task_id")
        prompt = record.get("prompt")
        if not isinstance(task_id, str) or not task_id:
            raise TeacherCollectionError(f"teacher task line {index} has no task_id")
        if task_id in seen:
            raise TeacherCollectionError(f"duplicate teacher task ID {task_id!r}")
        if not isinstance(prompt, list) or [
            message.get("role") for message in prompt if isinstance(message, dict)
        ] != ["system", "user"]:
            raise TeacherCollectionError(
                f"teacher task {task_id!r} prompt must contain system,user roles"
            )
        for key in ("composition", "difficulty", "source_split"):
            if not isinstance(record.get(key), str) or not record[key]:
                raise TeacherCollectionError(
                    f"teacher task {task_id!r} is missing {key}"
                )
        if record["source_split"] != expected_source_split:
            raise TeacherCollectionError(
                f"task {task_id!r} must originate from official {expected_source_split}"
            )
        seen.add(task_id)
        records.append(record)
    if not records:
        raise TeacherCollectionError("teacher task pool is empty")
    return tuple(records)


def assert_disjoint_from_evaluation(
    tasks: Sequence[Mapping[str, Any]], evaluation_path: str | Path
) -> None:
    evaluation = load_teacher_task_pool(evaluation_path, expected_source_split="test")
    task_ids = {str(task["task_id"]) for task in tasks}
    overlap = task_ids & {str(task["task_id"]) for task in evaluation}
    if overlap:
        example = min(overlap)
        raise TeacherCollectionError(
            f"teacher task pool overlaps frozen evaluation task {example!r}"
        )


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
    messages = copy.deepcopy(list(prompt))
    if not messages or messages[0].get("role") != "system":
        raise TeacherCollectionError("teacher prompt must begin with a system message")
    system_content = messages[0].get("content")
    if not isinstance(system_content, str):
        raise TeacherCollectionError("teacher system message must contain text")
    messages[0]["content"] = f"{system_content}\n\n{TEACHER_ACTOR_POLICY}"
    return messages


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


def _duplicate_correction(
    *,
    action: Any,
    semantic: tuple[str, str] | None,
    dimensions: Sequence[str],
    completed_semantic_fields: set[tuple[str, str]],
    answered_aspects: set[str],
    generation_attempt: int,
    force_answer: bool,
) -> str:
    completed = (
        ", ".join(
            f"{aspect}/{field}" for aspect, field in sorted(completed_semantic_fields)
        )
        or "none"
    )
    available = [
        f"{aspect}/{field}"
        for aspect in dimensions
        if aspect not in answered_aspects
        for field in FIELD_QUERY_HINTS[aspect]
        if (aspect, field) not in completed_semantic_fields
    ]
    available_text = ", ".join(available) or "no unasked preference fields"
    if semantic is not None:
        duplicate = f"{semantic[0]}/{semantic[1]}"
    else:
        duplicate = f"exact {action.choice.value} choice/content"
    answer_rule = (
        " You are in the reserved answer phase: submit one option ID for an unanswered aspect."
        if force_answer
        else ""
    )
    return (
        f"Duplicate-action correction {generation_attempt}: `{duplicate}` was already "
        f"committed. Completed preference fields: {completed}. Available unasked fields: "
        f"{available_text}. Do not ask `{duplicate}` again. Choose one available field, "
        "switch aspect, search if needed, or answer when grounded. Emit exactly one "
        f"interact_with_env call.{answer_rule}"
    )


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
) -> dict[str, Any]:
    """Build a credential-free rejected-attempt snapshot for diagnosis."""

    return {
        "task_id": task_id,
        "trajectory_attempt": trajectory_attempt,
        "failure_environment_turn": len(rewards) + 1,
        "environment_steps_completed": len(rewards),
        "messages": copy.deepcopy(list(messages)),
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
    return tuple(sorted(set(reasons)))


async def collect_teacher_trajectory(
    task: Mapping[str, Any],
    *,
    teacher: TeacherClientProtocol,
    simulator: UserSimulatorRuntime,
    wrapper_factory: WrapperFactory = UserBenchWrapper,
    source_root: str | Path | None = None,
    trajectory_attempt: int = 1,
) -> TeacherTrajectory:
    """Collect one tool-only trajectory without sharing actor/simulator clients."""

    if teacher.runtime.model.casefold() != DEEPSEEK_V4_FLASH_MODEL:
        raise TeacherCollectionError("teacher must use deepseek-v4-flash")
    if simulator.role is not SimulatorRole.COLLECTION:
        raise TeacherCollectionError(
            "teacher collection requires collection simulator role"
        )
    if simulator.model.casefold() != DEEPSEEK_V4_FLASH_MODEL:
        raise TeacherCollectionError("collection simulator must use deepseek-v4-flash")

    task_id = str(task.get("task_id") or "")
    prompt = task.get("prompt")
    if not task_id or not isinstance(prompt, list):
        raise TeacherCollectionError("teacher task is missing task_id or prompt")
    messages = _prepare_teacher_messages(prompt)
    dimensions = task_dimensions(task_id)
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
    try:
        wrapper.reset()
        for turn in range(wrapper.config.max_steps):
            remaining_turns = wrapper.config.max_steps - turn
            unanswered = set(dimensions) - answered_aspects
            force_answer = bool(unanswered) and remaining_turns <= len(unanswered) + 2
            request_messages: Sequence[Mapping[str, Any]] = messages
            tool_call = None
            for generation_attempt in range(1, teacher.runtime.action_retries + 2):
                candidate = await teacher.generate_action(
                    request_messages, force_answer=force_answer
                )
                for rejection in candidate.protocol_rejections:
                    generation_diagnostics.append(
                        {
                            "environment_turn": turn + 1,
                            "generation_attempt": generation_attempt,
                            "reason": "teacher_protocol_retry",
                            "detail": dict(rejection),
                        }
                    )
                exact = normalized_action_signature(candidate.action)
                semantic = semantic_action_signature(candidate.action, dimensions)
                rejection_reason = None
                if exact in exact_signatures:
                    rejection_reason = "duplicate_action"
                elif semantic is not None and semantic in semantic_signatures:
                    rejection_reason = "semantic_duplicate_action"
                if rejection_reason is None:
                    tool_call = candidate
                    break
                diagnostic = {
                    "environment_turn": turn + 1,
                    "generation_attempt": generation_attempt,
                    "reason": rejection_reason,
                    "choice": candidate.action.choice.value,
                    "semantic_aspect": semantic[0] if semantic else None,
                    "semantic_field": semantic[1] if semantic else None,
                    "force_answer": force_answer,
                }
                generation_diagnostics.append(diagnostic)
                if generation_attempt > teacher.runtime.action_retries:
                    raise TeacherGenerationError(
                        f"task {task_id!r} exhausted duplicate-action retries",
                        generation_diagnostics,
                    )
                correction = {
                    "role": "user",
                    "content": _duplicate_correction(
                        action=candidate.action,
                        semantic=semantic,
                        dimensions=dimensions,
                        completed_semantic_fields=semantic_signatures,
                        answered_aspects=answered_aspects,
                        generation_attempt=generation_attempt,
                        force_answer=force_answer,
                    ),
                }
                request_messages = [*messages, correction]
            assert tool_call is not None
            result = await wrapper.astep(tool_call.action)
            exact_signatures.add(normalized_action_signature(tool_call.action))
            semantic = semantic_action_signature(tool_call.action, dimensions)
            if semantic is not None:
                semantic_signatures.add(semantic)
            if tool_call.action.choice is ActionChoice.ANSWER:
                aspect = aspect_from_option_id(tool_call.action.content)
                if aspect in dimensions:
                    answered_aspects.add(aspect)
            if _simulator_fallback(result):
                simulator_fallbacks += 1
            simulator_judgment_fallbacks += _simulator_diagnostic_count(
                result, "userbench_judgment_fallbacks"
            )
            simulator_search_fallbacks += _simulator_diagnostic_count(
                result, "userbench_search_fallbacks"
            )
            committed_actions.append(
                {
                    "environment_turn": turn + 1,
                    "choice": tool_call.action.choice.value,
                    "content": tool_call.action.content,
                    "thought_length": len(tool_call.action.thought),
                    "normalized_signature": normalized_action_signature(
                        tool_call.action
                    ),
                    "semantic_aspect": semantic[0] if semantic else None,
                    "semantic_field": semantic[1] if semantic else None,
                    "reward": result.reward,
                    "terminated": result.terminated,
                    "truncated": result.truncated,
                }
            )
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
            if result.done:
                break
        if not (terminated or truncated):
            raise TeacherCollectionError(
                f"task {task_id!r} exceeded 20 calls without an environment terminal state"
            )
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
        messages=tuple(messages),
        step_rewards=tuple(rewards),
        terminated=terminated,
        truncated=truncated,
        expected_aspects=dimensions,
        answered_aspects=tuple(
            value for value in dimensions if value in answered_aspects
        ),
        simulator_fallbacks=simulator_fallbacks,
        simulator_judgment_fallbacks=simulator_judgment_fallbacks,
        simulator_search_fallbacks=simulator_search_fallbacks,
        generation_diagnostics=tuple(generation_diagnostics),
        trajectory_attempt=trajectory_attempt,
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
            accepted = not reasons
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
                )
            )
            if accepted:
                return TeacherTaskOutcome(task_id, trajectory, tuple(attempts))
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
                    rejection_reasons=("collection_error",),
                    generation_diagnostics=tuple(diagnostics),
                    error_type=exc.__class__.__name__,
                    error_message=str(exc),
                    partial_trajectory=getattr(exc, "partial_trajectory", None),
                )
            )
    return TeacherTaskOutcome(task_id, None, tuple(attempts))


async def collect_teacher_outcomes(
    tasks: Sequence[Mapping[str, Any]],
    *,
    teacher: TeacherClientProtocol,
    simulator: UserSimulatorRuntime,
    concurrency: int = 1,
    max_attempts: int = 3,
    wrapper_factory: WrapperFactory = UserBenchWrapper,
    source_root: str | Path | None = None,
) -> tuple[TeacherTaskOutcome, ...]:
    """Collect strict task outcomes concurrently while preserving task order."""

    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    semaphore = asyncio.Semaphore(concurrency)

    async def collect(task: Mapping[str, Any]) -> TeacherTaskOutcome:
        async with semaphore:
            return await collect_teacher_task_with_retries(
                task,
                teacher=teacher,
                simulator=simulator,
                max_attempts=max_attempts,
                wrapper_factory=wrapper_factory,
                source_root=source_root,
            )

    return tuple(await asyncio.gather(*(collect(task) for task in tasks)))


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
    force: bool = False,
) -> tuple[Path, Path, Path]:
    """Write accepted SFT data, rejected tasks, and attempt diagnostics separately."""

    destinations = tuple(
        Path(value) for value in (accepted_path, rejected_path, diagnostics_path)
    )
    if len(set(destinations)) != 3:
        raise ValueError("accepted, rejected, and diagnostics paths must be distinct")
    existing = [value for value in destinations if value.exists()]
    if existing and not force:
        raise FileExistsError(f"collection artifact already exists: {existing[0]}")
    accepted = [
        outcome.trajectory for outcome in outcomes if outcome.trajectory is not None
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
    _write_jsonl_records(rejected, destinations[1], force=force)
    _write_jsonl_records(diagnostics, destinations[2], force=force)
    return destinations
