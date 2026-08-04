"""Teacher trajectory collection against an isolated UserBench simulator API."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
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
from travel_grpo.envs.reward import REWARD_VERSION
from travel_grpo.models.openai_compatible import (
    TeacherClientProtocol,
    TeacherRequestConstraint,
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

TRAJECTORY_SCHEMA_VERSION = "userbench-teacher-trajectory-v4"
COLLECTION_DIAGNOSTIC_SCHEMA_VERSION = "userbench-teacher-diagnostic-v4"
SIMULATOR_FALLBACK_TEXT = (
    "I'm sorry, I'm not sure how to respond to your latest utterance right now. "
    "Please try again."
)
TEACHER_ACTOR_POLICY = """Teacher policy for strict UserBench trajectories:
- Emit exactly one interact_with_env call per turn. Keep thought to one short operational sentence of at most 200 characters.
- Follow the controller's current phase, aspect, and preference field exactly.
- Ask one concrete preference field per action; never ask vague "other preferences" questions or bundle fields.
- Search each travel aspect at most once, after its preferences are complete.
- Answer immediately after search with exactly one visible option ID for the current aspect.
- Never repeat an exact action, semantic preference field, search aspect, or answered aspect."""


class TeacherCollectionError(RuntimeError):
    """Raised when a task pool or collected trajectory violates the contract."""


class TeacherGenerationError(TeacherCollectionError):
    """Raised after request-local action correction attempts are exhausted."""

    def __init__(
        self,
        message: str,
        diagnostics: Sequence[Mapping[str, Any]],
        *,
        reason_code: str = "teacher_action_exhausted",
    ) -> None:
        super().__init__(message)
        self.diagnostics = tuple(dict(value) for value in diagnostics)
        self.reason_code = reason_code


class TeacherAttemptAbort(TeacherCollectionError):
    """Stops an attempt as soon as strict admission becomes impossible."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


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
            "teacher collection reward_version must be userbench-travel-reward-v2"
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
    if silver.get("minimum_raw_terminal_reward") != 0.7:
        raise TeacherCollectionError(
            "silver minimum_raw_terminal_reward must be 0.7"
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
    reward_breakdown: Mapping[str, Any] | None = None
    policy_version: str = POLICY_VERSION
    attempt_strategy: str = AttemptStrategy.NATURAL.value
    teacher_request_count: int = 0
    teacher_usage: Mapping[str, int] | None = None
    quality_tier: str = "gold"

    @property
    def total_reward(self) -> float:
        return float(sum(self.step_rewards))

    def to_record(self) -> dict[str, Any]:
        reward = dict(self.reward_breakdown or {})
        teacher_usage = dict(self.teacher_usage or {})
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
            "policy_version": self.policy_version,
            "attempt_strategy": self.attempt_strategy,
            "teacher_request_count": self.teacher_request_count,
            "teacher_usage": teacher_usage,
            "quality_tier": self.quality_tier,
            "reward_version": reward.get("reward_version"),
            "reward_valid": reward.get("reward_valid"),
            "terminal_reward": reward.get("terminal_reward"),
            "reward_breakdown": reward or None,
            "completion_rate": reward.get("completion_rate"),
            "correct_itinerary": reward.get("correct_itinerary"),
            "gold_itinerary": reward.get("gold_itinerary"),
            "fully_grounded": reward.get("fully_grounded"),
            "active_preference_coverage": reward.get(
                "active_preference_coverage"
            ),
            "passive_preference_coverage": reward.get(
                "passive_preference_coverage"
            ),
            "policy_penalty": reward.get("policy_penalty"),
            "invalid_actions": reward.get("invalid_actions", 0),
            "exact_repeats": reward.get("exact_repeats", 0),
            "semantic_repeats": reward.get("semantic_repeats", 0),
            "ambiguous_actions": reward.get("ambiguous_actions", 0),
            "unsearched_answers": reward.get("unsearched_answers", 0),
            "wrong_answers": reward.get("wrong_answers", 0),
            "infrastructure_errors": reward.get("infrastructure_errors", []),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "TeacherTrajectory":
        if record.get("schema_version") != TRAJECTORY_SCHEMA_VERSION:
            raise TeacherCollectionError("checkpoint contains an unsupported trajectory")
        return cls(
            task_id=str(record["task_id"]),
            composition=str(record["composition"]),
            difficulty=str(record["difficulty"]),
            source_split=str(record["source_split"]),
            teacher_model=str(record["teacher_model"]),
            simulator_model=str(record["simulator_model"]),
            messages=tuple(copy.deepcopy(record["messages"])),
            step_rewards=tuple(float(value) for value in record["step_rewards"]),
            terminated=record.get("terminated") is True,
            truncated=record.get("truncated") is True,
            expected_aspects=tuple(str(value) for value in record["expected_aspects"]),
            answered_aspects=tuple(str(value) for value in record["answered_aspects"]),
            simulator_fallbacks=int(record.get("simulator_fallbacks", 0)),
            simulator_judgment_fallbacks=int(
                record.get("simulator_judgment_fallbacks", 0)
            ),
            simulator_search_fallbacks=int(record.get("simulator_search_fallbacks", 0)),
            generation_diagnostics=tuple(
                copy.deepcopy(record.get("generation_diagnostics", ()))
            ),
            trajectory_attempt=int(record.get("trajectory_attempt", 1)),
            reward_breakdown=copy.deepcopy(record.get("reward_breakdown")),
            policy_version=str(record.get("policy_version", "")),
            attempt_strategy=str(record.get("attempt_strategy", "")),
            teacher_request_count=int(record.get("teacher_request_count", 0)),
            teacher_usage=copy.deepcopy(record.get("teacher_usage", {})),
            quality_tier=str(record.get("quality_tier", "gold")),
        )


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
    policy_version: str = POLICY_VERSION
    attempt_strategy: str = AttemptStrategy.NATURAL.value
    quality_tier: str = "rejected"

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": COLLECTION_DIAGNOSTIC_SCHEMA_VERSION,
            "task_id": self.task_id,
            "attempt": self.attempt,
            "accepted": self.accepted,
            "policy_version": self.policy_version,
            "attempt_strategy": self.attempt_strategy,
            "quality_tier": self.quality_tier,
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

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "TeacherAttemptDiagnostic":
        if record.get("schema_version") != COLLECTION_DIAGNOSTIC_SCHEMA_VERSION:
            raise TeacherCollectionError("checkpoint contains unsupported diagnostics")
        return cls(
            task_id=str(record["task_id"]),
            attempt=int(record["attempt"]),
            accepted=record.get("accepted") is True,
            rejection_reasons=tuple(str(value) for value in record["rejection_reasons"]),
            generation_diagnostics=tuple(
                copy.deepcopy(record.get("generation_diagnostics", ()))
            ),
            error_type=(
                None if record.get("error_type") is None else str(record["error_type"])
            ),
            error_message=(
                None
                if record.get("error_message") is None
                else str(record["error_message"])
            ),
            partial_trajectory=copy.deepcopy(record.get("partial_trajectory")),
            policy_version=str(record.get("policy_version", "")),
            attempt_strategy=str(record.get("attempt_strategy", "")),
            quality_tier=str(record.get("quality_tier", "rejected")),
        )


@dataclass(frozen=True)
class TeacherTaskOutcome:
    task_id: str
    trajectory: TeacherTrajectory | None
    attempts: tuple[TeacherAttemptDiagnostic, ...]
    quality_tier: str = "rejected"

    @property
    def accepted(self) -> bool:
        return self.trajectory is not None

    @property
    def gold(self) -> bool:
        return self.quality_tier == "gold"

    def rejected_record(self) -> dict[str, Any]:
        final = self.attempts[-1]
        return {
            "schema_version": COLLECTION_DIAGNOSTIC_SCHEMA_VERSION,
            "task_id": self.task_id,
            "attempts": len(self.attempts),
            "rejection_reasons": list(final.rejection_reasons),
            "error_type": final.error_type,
            "error_message": final.error_message,
            "policy_version": final.policy_version,
            "attempt_strategy": final.attempt_strategy,
            "quality_tier": self.quality_tier,
        }

    def to_checkpoint_record(self) -> dict[str, Any]:
        return {
            "schema_version": "userbench-teacher-task-checkpoint-v1",
            "policy_version": POLICY_VERSION,
            "task_id": self.task_id,
            "trajectory": None if self.trajectory is None else self.trajectory.to_record(),
            "attempts": [value.to_record() for value in self.attempts],
            "quality_tier": self.quality_tier,
        }

    @classmethod
    def from_checkpoint_record(cls, record: Mapping[str, Any]) -> "TeacherTaskOutcome":
        if record.get("schema_version") != "userbench-teacher-task-checkpoint-v1":
            raise TeacherCollectionError("unsupported teacher task checkpoint")
        if record.get("policy_version") != POLICY_VERSION:
            raise TeacherCollectionError("teacher task checkpoint policy mismatch")
        trajectory_record = record.get("trajectory")
        return cls(
            task_id=str(record["task_id"]),
            trajectory=(
                None
                if trajectory_record is None
                else TeacherTrajectory.from_record(trajectory_record)
            ),
            attempts=tuple(
                TeacherAttemptDiagnostic.from_record(value)
                for value in record.get("attempts", ())
            ),
            quality_tier=str(
                record.get(
                    "quality_tier",
                    "rejected" if trajectory_record is None else "gold",
                )
            ),
        )


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


def _largest_remainder_quota(
    counts: Mapping[str, int], target: int
) -> dict[str, int]:
    """Allocate an integer target proportionally, deterministically."""

    if target <= 0:
        raise TeacherCollectionError("stratified target must be positive")
    total = sum(int(value) for value in counts.values())
    if total <= 0:
        raise TeacherCollectionError("cannot stratify an empty task pool")
    if target > total:
        raise TeacherCollectionError(
            f"stratified target {target} exceeds task pool size {total}"
        )
    quotas = {
        str(key): target * int(value) // total for key, value in counts.items()
    }
    remainder = target - sum(quotas.values())
    ranked = sorted(
        (str(key) for key in counts),
        key=lambda key: (
            -(target * int(counts[key]) % total),
            key,
        ),
    )
    for key in ranked[:remainder]:
        quotas[key] += 1
    return quotas


def build_stratified_task_plan(
    tasks: Sequence[Mapping[str, Any]],
    *,
    target: int,
    field: str = "composition",
    seed: str = "sft-stratified-v1",
) -> tuple[tuple[dict[str, Any], ...], dict[str, int]]:
    """Build a reproducible, composition-stratified candidate order.

    The returned order is grouped by ``field`` and deterministically shuffled
    within each group.  The quota is the exact largest-remainder allocation
    for ``target`` tasks; rejected tasks remain in the candidate order so a
    later wave can refill the same stratum without changing other strata.
    """

    if not isinstance(field, str) or not field:
        raise TeacherCollectionError("stratification field must be non-empty")
    if not isinstance(seed, str) or not seed:
        raise TeacherCollectionError("stratification seed must be non-empty")
    groups: dict[str, list[dict[str, Any]]] = {}
    for task in tasks:
        value = task.get(field)
        if not isinstance(value, str) or not value:
            raise TeacherCollectionError(
                f"task {task.get('task_id', '<unknown>')!r} is missing stratification field {field!r}"
            )
        task_id = task.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise TeacherCollectionError("stratified task is missing task_id")
        groups.setdefault(value, []).append(dict(task))
    counts = {key: len(value) for key, value in groups.items()}
    quotas = _largest_remainder_quota(counts, target)
    ordered: list[dict[str, Any]] = []
    for value in sorted(groups):
        ordered.extend(
            sorted(
                groups[value],
                key=lambda task: (
                    hashlib.sha256(
                        f"{seed}\0{task['task_id']}".encode("utf-8")
                    ).hexdigest(),
                    str(task["task_id"]),
                ),
            )
        )
    return tuple(ordered), quotas


def select_stratified_task_wave(
    tasks: Sequence[Mapping[str, Any]],
    *,
    quotas: Mapping[str, int],
    attempted_task_ids: set[str],
    accepted_task_ids: set[str],
    field: str = "composition",
    wave_size: int = 32,
) -> tuple[dict[str, Any], ...]:
    """Select the next wave, proportional to each stratum's remaining deficit.

    Accepted means Gold or Silver.  Rejected and infrastructure-invalid tasks
    are only marked attempted, so the next call naturally refills that same
    stratum from its remaining candidates.
    """

    if wave_size <= 0:
        raise TeacherCollectionError("stratified wave size must be positive")
    groups: dict[str, list[dict[str, Any]]] = {}
    accepted_by_stratum: Counter[str] = Counter()
    for task in tasks:
        task_id = str(task["task_id"])
        value = str(task[field])
        if task_id not in attempted_task_ids:
            groups.setdefault(value, []).append(dict(task))
        if task_id in accepted_task_ids:
            accepted_by_stratum[value] += 1
    deficits = {
        value: max(int(quotas.get(value, 0)) - accepted_by_stratum[value], 0)
        for value in quotas
    }
    capacities = {
        value: min(deficits[value], len(groups.get(value, ())))
        for value in deficits
        if deficits[value] > 0 and groups.get(value)
    }
    if not capacities:
        return ()
    total = min(wave_size, sum(capacities.values()))
    allocations = {value: 0 for value in capacities}
    # D'Hondt allocation gives an integer proportional split and handles a
    # stratum whose remaining candidate pool is smaller than its quota.
    for _ in range(total):
        available = [
            value for value in sorted(capacities) if allocations[value] < capacities[value]
        ]
        if not available:
            break
        value = max(
            available,
            key=lambda item: (
                capacities[item] / (allocations[item] + 1),
                capacities[item],
                item,
            ),
        )
        allocations[value] += 1
    selected: list[dict[str, Any]] = []
    for value in sorted(allocations):
        selected.extend(groups[value][: allocations[value]])
    return tuple(selected)


def write_stratified_selection_manifest(
    path: str | Path, document: Mapping[str, Any]
) -> Path:
    """Atomically persist a credential-free adaptive sampling manifest."""

    destination = Path(path)
    _atomic_json_record(document, destination)
    return destination


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
        if reward.get("reward_version") != REWARD_VERSION:
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
        if "judgment_fallback_allowed" not in markers:
            return None
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
            reward_state.record_step(result, tool_call.action, after_snapshot)
            policy.record_committed(plan)
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
                fail_reason = None
            elif fail_reason == "environment.vague_action_feedback":
                repair_key = f"{plan.aspect}.{plan.field or plan.phase.value}"
                if vague_repairs.get(repair_key, 0) < 1:
                    vague_repairs[repair_key] = vague_repairs.get(repair_key, 0) + 1
                    if plan.phase is TeacherPhase.ELICIT and plan.field is not None:
                        policy.asked_fields[plan.aspect].discard(plan.field)
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
        messages=tuple(messages),
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
        if current != expected:
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
