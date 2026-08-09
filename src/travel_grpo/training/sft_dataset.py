"""Strict UserBench trajectory validation and action-only SFT rendering."""

from __future__ import annotations

import copy
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from travel_grpo.envs.reward import REWARD_VERSION
from travel_grpo.envs.userbench_tools import (
    TOOL_NAME,
    UserBenchAction,
    UserBenchActionError,
    get_interact_with_env_schema,
)
from travel_grpo.training.sft_collection import (
    TRAJECTORY_SCHEMA_VERSION,
    TeacherTrajectory,
    quality_tier_for_trajectory,
    trajectory_rejection_reasons as collection_rejection_reasons,
)

IGNORE_INDEX = -100
MIN_TERMINAL_REWARD = 0.7
PREFIX_SCHEMA_VERSION = "userbench-teacher-prefix-v1"
RECOVERY_SFT_SCHEMA_VERSION = "recovery-sft-v1"
SFT_RECORD_FORMATS = frozenset({"trajectory", "prefix", "recovery"})
_PREFIX_FINAL_ANSWER_FAILURE_PREFIXES = (
    "environment.wrong_answer",
    "environment.answer_not_recorded",
    "environment.answer_not_matching_public_requirement",
)


class SFTDatasetError(ValueError):
    """Raised when a trajectory cannot safely become an SFT example."""


class SFTTrajectoryTooLongError(SFTDatasetError):
    """Raised so callers can reject one whole trajectory without truncation."""


class ChatTemplateTokenizer(Protocol):
    pad_token_id: int | None
    eos_token_id: int | None
    padding_side: str

    def apply_chat_template(self, conversation: Sequence[Mapping[str, Any]], **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class TrajectoryRejection:
    line_number: int
    task_id: str | None
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class TrajectoryAudit:
    source: Path
    records: tuple[dict[str, Any], ...]
    rejections: tuple[TrajectoryRejection, ...]

    def summary(self) -> dict[str, Any]:
        reasons = Counter(reason for item in self.rejections for reason in item.reasons)
        compositions = Counter(str(record["composition"]) for record in self.records)
        return {
            "source": str(self.source),
            "total_trajectories": len(self.records) + len(self.rejections),
            "accepted_trajectories": len(self.records),
            "rejected_trajectories": len(self.rejections),
            "composition_distribution": dict(sorted(compositions.items())),
            "rejection_reasons": dict(sorted(reasons.items())),
            "rejections": [
                {
                    "line_number": item.line_number,
                    "task_id": item.task_id,
                    "reasons": list(item.reasons),
                }
                for item in self.rejections
            ],
        }


@dataclass(frozen=True)
class ActionOnlyExample:
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    labels: tuple[int, ...]
    task_id: str
    trajectory_id: str
    assistant_turn_index: int
    composition: str

    @property
    def sequence_length(self) -> int:
        return len(self.input_ids)

    @property
    def label_tokens(self) -> int:
        return sum(value != IGNORE_INDEX for value in self.labels)

    def to_trainer_dict(self) -> dict[str, Any]:
        return {
            "input_ids": list(self.input_ids),
            "attention_mask": list(self.attention_mask),
            "labels": list(self.labels),
            "task_id": self.task_id,
            "trajectory_id": self.trajectory_id,
            "assistant_turn_index": self.assistant_turn_index,
        }


def load_tool_schema(path: str | Path) -> dict[str, Any]:
    """Load the sole veRL tool and prove it matches the Python contract."""

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - covered by project extras.
        raise SFTDatasetError("PyYAML is required to validate the tool schema") from exc
    source = Path(path)
    try:
        document = yaml.safe_load(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SFTDatasetError(f"cannot read tool schema: {source}") from exc
    tools = document.get("tools") if isinstance(document, Mapping) else None
    if not isinstance(tools, list) or len(tools) != 1:
        raise SFTDatasetError("tool configuration must define exactly one tool")
    entry = tools[0]
    schema = entry.get("tool_schema") if isinstance(entry, Mapping) else None
    expected = get_interact_with_env_schema()
    if schema != expected:
        raise SFTDatasetError(
            "tool YAML schema does not match the Python interact_with_env contract"
        )
    return expected


def _message_reasons(messages: Any) -> tuple[str, ...]:
    reasons: set[str] = set()
    if not isinstance(messages, list) or len(messages) < 4:
        return ("invalid_messages",)
    if [message.get("role") for message in messages[:2] if isinstance(message, Mapping)] != [
        "system",
        "user",
    ]:
        reasons.add("invalid_message_prefix")
    remainder = messages[2:]
    if len(remainder) % 2:
        reasons.add("unpaired_assistant_tool_messages")
    seen_call_ids: set[str] = set()
    for offset in range(0, len(remainder), 2):
        if offset + 1 >= len(remainder):
            break
        assistant, tool = remainder[offset], remainder[offset + 1]
        if not isinstance(assistant, Mapping) or assistant.get("role") != "assistant":
            reasons.add("invalid_assistant_message")
            continue
        if "loss_mask" in assistant and not isinstance(assistant["loss_mask"], bool):
            reasons.add("invalid_loss_mask")
        if assistant.get("content") not in (None, ""):
            reasons.add("assistant_content_must_be_empty")
        calls = assistant.get("tool_calls")
        if not isinstance(calls, list) or len(calls) != 1:
            reasons.add("assistant_must_contain_one_tool_call")
            continue
        call = calls[0]
        function = call.get("function") if isinstance(call, Mapping) else None
        call_id = call.get("id") if isinstance(call, Mapping) else None
        if not isinstance(call, Mapping) or call.get("type") != "function":
            reasons.add("wrong_tool_call_type")
        if not isinstance(call_id, str) or not call_id or call_id in seen_call_ids:
            reasons.add("invalid_or_duplicate_tool_call_id")
        else:
            seen_call_ids.add(call_id)
        if not isinstance(function, Mapping) or function.get("name") != TOOL_NAME:
            reasons.add("wrong_function_name")
        else:
            arguments = function.get("arguments")
            try:
                parameters = json.loads(arguments) if isinstance(arguments, str) else None
            except json.JSONDecodeError:
                parameters = None
            if not isinstance(parameters, Mapping):
                reasons.add("invalid_tool_arguments_json")
            else:
                try:
                    UserBenchAction.from_parameters(parameters)
                except UserBenchActionError:
                    reasons.add("invalid_tool_arguments")
        if not isinstance(tool, Mapping) or tool.get("role") != "tool":
            reasons.add("invalid_tool_message")
        elif tool.get("tool_call_id") != call_id:
            reasons.add("tool_call_id_mismatch")
        elif tool.get("name") != TOOL_NAME:
            reasons.add("wrong_tool_message_name")
        elif not isinstance(tool.get("content"), str):
            reasons.add("invalid_tool_message_content")

        content = str(tool.get("content") or "") if isinstance(tool, Mapping) else ""
        if "too vague and general" in content:
            reasons.add("vague_action_feedback")
        if "already recommended an option" in content:
            reasons.add("duplicate_recommendation_feedback")
        if "Invalid option ID format" in content:
            reasons.add("invalid_option_id_feedback")
    return tuple(sorted(reasons))


def trajectory_rejection_reasons(record: Any) -> tuple[str, ...]:
    """Return every deterministic reason a record is not trainable."""

    if not isinstance(record, Mapping):
        return ("record_not_mapping",)
    reasons: set[str] = set()
    if record.get("schema_version") != TRAJECTORY_SCHEMA_VERSION:
        reasons.add("legacy_or_unknown_schema")
    for field in (
        "task_id",
        "composition",
        "difficulty",
        "source_split",
        "teacher_model",
        "simulator_model",
    ):
        if not isinstance(record.get(field), str) or not str(record[field]).strip():
            reasons.add(f"missing_{field}")
    reasons.update(_message_reasons(record.get("messages")))
    if record.get("terminated") is not True:
        reasons.add("not_terminated")
    if record.get("truncated") is not False:
        reasons.add("truncated")

    reward = record.get("reward_breakdown")
    if not isinstance(reward, Mapping):
        reasons.add("missing_reward_evidence")
    else:
        if record.get("reward_version") != REWARD_VERSION or reward.get("reward_version") != REWARD_VERSION:
            reasons.add("wrong_reward_version")
        if record.get("reward_valid") is not True or reward.get("reward_valid") is not True:
            reasons.add("reward_invalid")
        completion_rate = record.get("completion_rate")
        if (
            not isinstance(completion_rate, (int, float))
            or isinstance(completion_rate, bool)
            or not math.isfinite(float(completion_rate))
            or float(completion_rate) != 1.0
        ):
            reasons.add("incomplete_reward_completion")
        if record.get("correct_itinerary") is not True:
            reasons.add("incorrect_itinerary")
        policy_penalty = record.get("policy_penalty")
        if (
            not isinstance(policy_penalty, (int, float))
            or isinstance(policy_penalty, bool)
            or not math.isfinite(float(policy_penalty))
            or float(policy_penalty) != 0.0
        ):
            reasons.add("policy_penalty")
        terminal = record.get("terminal_reward")
        if not isinstance(terminal, (int, float)) or isinstance(terminal, bool) or not math.isfinite(float(terminal)):
            reasons.add("invalid_terminal_reward")
        elif float(terminal) < MIN_TERMINAL_REWARD:
            reasons.add("terminal_reward_below_threshold")
        for field in (
            "invalid_actions",
            "exact_repeats",
            "semantic_repeats",
            "ambiguous_actions",
            "unsearched_answers",
            "wrong_answers",
        ):
            value = record.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value != 0:
                reasons.add(field)
            nested = reward.get(field)
            if (
                not isinstance(nested, int)
                or isinstance(nested, bool)
                or nested != value
            ):
                reasons.add(f"reward_mismatch_{field}")
        for field in (
            "reward_version",
            "reward_valid",
            "terminal_reward",
            "completion_rate",
            "correct_itinerary",
            "gold_itinerary",
            "fully_grounded",
            "active_preference_coverage",
            "passive_preference_coverage",
            "policy_penalty",
            "infrastructure_errors",
        ):
            if reward.get(field) != record.get(field):
                reasons.add(f"reward_mismatch_{field}")
        for field in ("gold_itinerary", "fully_grounded"):
            if not isinstance(record.get(field), bool):
                reasons.add(f"invalid_{field}")
        for field in ("active_preference_coverage", "passive_preference_coverage"):
            value = record.get(field)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                reasons.add(f"invalid_{field}")
        infrastructure_errors = record.get("infrastructure_errors")
        if not isinstance(infrastructure_errors, list):
            reasons.add("invalid_infrastructure_errors")
        elif infrastructure_errors:
            reasons.add("infrastructure_error")
    if record.get("source_split") != "train":
        reasons.add("not_official_train")
    expected = record.get("expected_aspects")
    answered = record.get("answered_aspects")
    if not isinstance(expected, list) or not isinstance(answered, list) or set(expected) != set(answered):
        reasons.add("incomplete_aspect_answers")
    return tuple(sorted(reasons))


_SILVER_RELAXABLE_RECORD_REASONS = frozenset(
    {
        "vague_action_feedback",
        "reward_invalid",
        "terminal_reward_below_threshold",
        "infrastructure_error",
    }
)


def sft_admission_reasons(
    record: Any,
    *,
    accepted_quality_tiers: Sequence[str] = ("gold",),
) -> tuple[str, ...]:
    """Revalidate Gold/Silver admission without trusting the serialized tier."""

    allowed = tuple(str(value) for value in accepted_quality_tiers)
    if not allowed or set(allowed) - {"gold", "silver"}:
        raise SFTDatasetError("accepted quality tiers must be a subset of gold,silver")
    strict = set(trajectory_rejection_reasons(record))
    if not isinstance(record, Mapping):
        return tuple(sorted(strict))
    # ``quality_tier`` is collection metadata, not an admission authority.
    # Infer the tier again from the trajectory evidence so a stale or forged
    # serialized label can neither upgrade Silver to Gold nor reject valid
    # evidence that was mislabeled by an older collector.
    if not strict:
        return () if "gold" in allowed else ("quality_tier_not_accepted",)

    # Silver records may relax only the same evidence failures admitted during
    # collection.  Every structural, protocol, answer and split check remains
    # enforced by the record-level validator above.
    structural = strict - _SILVER_RELAXABLE_RECORD_REASONS
    if structural:
        return tuple(sorted(structural))
    try:
        trajectory = TeacherTrajectory.from_record(record)
    except Exception:
        return ("invalid_silver_trajectory",)
    collection_reasons = collection_rejection_reasons(trajectory)
    if quality_tier_for_trajectory(trajectory, collection_reasons) != "silver":
        return ("invalid_silver_admission",)
    masked_assistant_turns = sum(
        isinstance(message, Mapping)
        and message.get("role") == "assistant"
        and message.get("loss_mask") is True
        for message in trajectory.messages
    )
    if trajectory.simulator_judgment_fallbacks > masked_assistant_turns:
        return ("unmasked_silver_judgment_fallback",)
    return () if "silver" in allowed else ("quality_tier_not_accepted",)


def prefix_admission_reasons(record: Any) -> tuple[str, ...]:
    """Validate a safe Stage-1 prefix without pretending it is terminal Gold/Silver.

    Prefix records are extracted from failed Teacher attempts only after every
    retained action received positive environment evidence.  The failed final
    answer is metadata and must not appear in ``messages``; the last retained
    assistant decision is therefore the successful search immediately before
    that answer.
    """

    if not isinstance(record, Mapping):
        return ("record_not_mapping",)
    reasons: set[str] = set()
    if record.get("schema_version") != PREFIX_SCHEMA_VERSION:
        reasons.add("legacy_or_unknown_prefix_schema")
    for field in ("task_id", "composition", "source_split"):
        if not isinstance(record.get(field), str) or not str(record[field]).strip():
            reasons.add(f"missing_{field}")
    if record.get("source_split") != "train":
        reasons.add("not_official_train")

    messages = record.get("messages")
    reasons.update(_message_reasons(messages))
    assistant_messages = (
        [
            message
            for message in messages
            if isinstance(message, Mapping) and message.get("role") == "assistant"
        ]
        if isinstance(messages, list)
        else []
    )
    if any(message.get("loss_mask") is True for message in assistant_messages):
        reasons.add("prefix_contains_loss_masked_action")
    action_count = record.get("prefix_action_count")
    if (
        not isinstance(action_count, int)
        or isinstance(action_count, bool)
        or action_count <= 0
        or action_count != len(assistant_messages)
    ):
        reasons.add("prefix_action_count_mismatch")

    assistant_choices: list[str | None] = []
    for assistant_message in assistant_messages:
        calls = assistant_message.get("tool_calls")
        call = calls[0] if isinstance(calls, list) and len(calls) == 1 else None
        function = call.get("function") if isinstance(call, Mapping) else None
        arguments = function.get("arguments") if isinstance(function, Mapping) else None
        try:
            parameters = json.loads(arguments) if isinstance(arguments, str) else None
        except json.JSONDecodeError:
            parameters = None
        assistant_choices.append(
            parameters.get("choice") if isinstance(parameters, Mapping) else None
        )
    last_choice = assistant_choices[-1] if assistant_choices else None
    if last_choice != "search":
        reasons.add("prefix_does_not_end_after_successful_search")

    evidence = record.get("retained_action_evidence")
    evidence_aspects: list[str | None] = []
    if not isinstance(evidence, list) or len(evidence) != len(assistant_messages):
        reasons.add("invalid_retained_action_evidence")
    else:
        for index, item in enumerate(evidence):
            if not isinstance(item, Mapping):
                reasons.add("invalid_retained_action_evidence")
                continue
            reward = item.get("reward")
            if (
                not isinstance(reward, (int, float))
                or isinstance(reward, bool)
                or not math.isfinite(float(reward))
                or float(reward) <= 0.0
            ):
                reasons.add("nonpositive_retained_action")
            if item.get("choice") != assistant_choices[index]:
                reasons.add("retained_action_evidence_mismatch")
            aspect = item.get("aspect")
            evidence_aspects.append(aspect if isinstance(aspect, str) else None)

    failures = record.get("source_failure_reasons")
    if (
        not isinstance(failures, list)
        or not failures
        or any(
            not isinstance(reason, str)
            or not any(
                reason.startswith(prefix)
                for prefix in _PREFIX_FINAL_ANSWER_FAILURE_PREFIXES
            )
            for reason in failures
        )
    ):
        reasons.add("invalid_prefix_source_failure")
    failed_answer = record.get("failed_answer")
    if not isinstance(failed_answer, Mapping):
        reasons.add("missing_removed_failed_answer")
    else:
        if not isinstance(failed_answer.get("aspect"), str) or not failed_answer.get(
            "aspect"
        ):
            reasons.add("invalid_removed_failed_answer_aspect")
        if not isinstance(failed_answer.get("content"), str) or not failed_answer.get(
            "content"
        ):
            reasons.add("invalid_removed_failed_answer_content")
        environment_turn = failed_answer.get("environment_turn")
        if (
            not isinstance(environment_turn, int)
            or isinstance(environment_turn, bool)
            or environment_turn <= 0
        ):
            reasons.add("invalid_removed_failed_answer_turn")
        if not evidence_aspects or evidence_aspects[-1] != failed_answer.get("aspect"):
            reasons.add("prefix_search_aspect_mismatch")
    return tuple(sorted(reasons))


def recovery_admission_reasons(record: Any) -> tuple[str, ...]:
    """Validate a rendered recovery record without reward-state admission.

    Recovery examples are one-step records. Historical assistant calls remain
    in the prompt with ``loss_mask=True`` and the final assistant call is the
    only unmasked target. A private synthetic response lets us reuse the
    ordinary paired-message/tool schema checker without serializing a result.
    """

    if not isinstance(record, Mapping):
        return ("record_not_mapping",)
    reasons: set[str] = set()
    if record.get("schema_version") != RECOVERY_SFT_SCHEMA_VERSION:
        reasons.add("legacy_or_unknown_recovery_schema")
    if record.get("source_schema_version") != "recovery-target-v1":
        reasons.add("wrong_recovery_source_schema")
    for field in (
        "task_id",
        "composition",
        "boundary_type",
        "project_split",
        "policy_version",
        "actor_policy_version",
    ):
        if not isinstance(record.get(field), str) or not str(record[field]).strip():
            reasons.add(f"missing_{field}")
    if record.get("project_split") == "evaluation":
        reasons.add("evaluation_record_not_trainable")
    if record.get("target_status") != "accepted":
        reasons.add("target_not_accepted")
    if not isinstance(record.get("public_state_before"), Mapping):
        reasons.add("missing_public_state")
    if not isinstance(record.get("target_assistant"), Mapping):
        reasons.add("missing_target_assistant")

    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        reasons.add("invalid_messages")
        return tuple(sorted(reasons))
    if [
        message.get("role")
        for message in messages[:2]
        if isinstance(message, Mapping)
    ] != ["system", "user"]:
        reasons.add("invalid_message_prefix")
    if not isinstance(messages[-1], Mapping) or messages[-1].get("role") != "assistant":
        reasons.add("recovery_target_must_be_final_assistant")
        return tuple(sorted(reasons))
    assistants = [
        message
        for message in messages
        if isinstance(message, Mapping) and message.get("role") == "assistant"
    ]
    if not assistants:
        reasons.add("missing_target_assistant_message")
        return tuple(sorted(reasons))
    if any(message.get("loss_mask") is not True for message in assistants[:-1]):
        reasons.add("historical_assistant_must_be_loss_masked")
    if assistants[-1].get("loss_mask") is True:
        reasons.add("target_assistant_must_not_be_loss_masked")
    if sum(message.get("loss_mask") is not True for message in assistants) != 1:
        reasons.add("recovery_requires_one_unmasked_assistant")

    validation_messages = copy.deepcopy(messages)
    target = validation_messages[-1]
    calls = target.get("tool_calls") if isinstance(target, Mapping) else None
    call_id = None
    if isinstance(calls, list) and len(calls) == 1 and isinstance(calls[0], Mapping):
        call_id = calls[0].get("id")
    validation_messages.append(
        {
            "role": "tool",
            "name": TOOL_NAME,
            "tool_call_id": call_id,
            "content": "",
        }
    )
    # These simulator feedback strings are public history. They are useful
    # recovery evidence and must not be treated as trajectory-level admission
    # failures for a one-step boundary target. Structural/tool errors remain
    # enforced by the shared validator.
    recovery_history_warnings = {
        "vague_action_feedback",
        "duplicate_recommendation_feedback",
        "invalid_option_id_feedback",
    }
    reasons.update(
        reason
        for reason in _message_reasons(validation_messages)
        if reason not in recovery_history_warnings
    )
    return tuple(sorted(reasons))


def sft_record_admission_reasons(
    record: Any,
    *,
    record_format: str = "trajectory",
    accepted_quality_tiers: Sequence[str] = ("gold", "silver"),
) -> tuple[str, ...]:
    if record_format not in SFT_RECORD_FORMATS:
        raise SFTDatasetError(
            f"record_format must be one of {', '.join(sorted(SFT_RECORD_FORMATS))}"
        )
    if record_format == "prefix":
        return prefix_admission_reasons(record)
    if record_format == "recovery":
        return recovery_admission_reasons(record)
    return sft_admission_reasons(
        record, accepted_quality_tiers=accepted_quality_tiers
    )


def audit_trajectory_file(
    path: str | Path,
    *,
    limit: int | None = None,
    accepted_quality_tiers: Sequence[str] = ("gold",),
    record_format: str = "trajectory",
) -> TrajectoryAudit:
    source = Path(path)
    if limit is not None and limit <= 0:
        raise SFTDatasetError("limit must be positive")
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SFTDatasetError(f"cannot read trajectory file: {source}") from exc
    accepted: list[dict[str, Any]] = []
    rejected: list[TrajectoryRejection] = []
    seen: set[str] = set()
    nonempty = 0
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        if limit is not None and nonempty >= limit:
            break
        nonempty += 1
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            rejected.append(TrajectoryRejection(line_number, None, ("invalid_json",)))
            continue
        task_id = record.get("task_id") if isinstance(record, Mapping) else None
        reasons = set(
            sft_record_admission_reasons(
                record,
                record_format=record_format,
                accepted_quality_tiers=accepted_quality_tiers,
            )
        )
        if isinstance(task_id, str):
            if task_id in seen:
                reasons.add("duplicate_task_id")
            seen.add(task_id)
        if reasons:
            rejected.append(
                TrajectoryRejection(
                    line_number,
                    task_id if isinstance(task_id, str) else None,
                    tuple(sorted(reasons)),
                )
            )
        else:
            accepted.append(dict(record))
    if nonempty == 0:
        raise SFTDatasetError(f"trajectory file is empty: {source}")
    return TrajectoryAudit(source.resolve(), tuple(accepted), tuple(rejected))


def load_sft_trajectories(
    path: str | Path,
    *,
    limit: int | None = None,
    accepted_quality_tiers: Sequence[str] = ("gold",),
    record_format: str = "trajectory",
) -> tuple[dict[str, Any], ...]:
    audit = audit_trajectory_file(
        path,
        limit=limit,
        accepted_quality_tiers=accepted_quality_tiers,
        record_format=record_format,
    )
    if audit.rejections:
        first = audit.rejections[0]
        raise SFTDatasetError(
            f"trajectory line {first.line_number} is not trainable: {', '.join(first.reasons)}"
        )
    return audit.records


def load_sft_trajectory_files(
    paths: Sequence[str | Path],
    *,
    limit: int | None = None,
    accepted_quality_tiers: Sequence[str] = ("gold", "silver"),
    record_format: str = "trajectory",
) -> tuple[dict[str, Any], ...]:
    """Load several tier artifacts while enforcing global task-ID uniqueness."""

    if not paths:
        raise SFTDatasetError("at least one SFT trajectory file is required")
    combined: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        values = load_sft_trajectories(
            path,
            limit=limit,
            accepted_quality_tiers=accepted_quality_tiers,
            record_format=record_format,
        )
        for record in values:
            task_id = str(record["task_id"])
            if task_id in seen:
                raise SFTDatasetError(f"duplicate SFT task across tier files: {task_id!r}")
            seen.add(task_id)
            combined.append(record)
    return tuple(combined)


def assert_sft_readiness(
    train: Sequence[Mapping[str, Any]],
    validation: Sequence[Mapping[str, Any]],
    *,
    # minimum_train: int = 400,
    # minimum_validation: int = 40,
    # required_compositions: Sequence[str] = (
    #     "22",
    #     "33",
    #     "44",
    #     "2222",
    #     "233",
    #     "333",
    #     "334",
    #     "444",
    # ),
    minimum_train: int = 50,
    minimum_validation: int = 5,
    required_compositions: Sequence[str] = (
    ),
) -> None:
    """Fail before formal SFT when data quantity or composition coverage is weak."""

    if len(train) < minimum_train:
        raise SFTDatasetError(
            f"formal SFT requires at least {minimum_train} train trajectories, found {len(train)}"
        )
    if len(validation) < minimum_validation:
        raise SFTDatasetError(
            "formal SFT requires at least "
            f"{minimum_validation} validation trajectories, found {len(validation)}"
        )
    for name, values in (("train", train), ("validation", validation)):
        observed = {str(record.get("composition")) for record in values}
        missing = sorted(set(required_compositions) - observed)
        if missing:
            raise SFTDatasetError(
                f"formal SFT {name} is missing compositions: {', '.join(missing)}"
            )


def assert_train_validation_disjoint(
    train: Sequence[Mapping[str, Any]], validation: Sequence[Mapping[str, Any]]
) -> None:
    overlap = {str(record["task_id"]) for record in train} & {
        str(record["task_id"]) for record in validation
    }
    if overlap:
        raise SFTDatasetError(
            f"SFT train/validation overlap at task {min(overlap)!r}"
        )


def assert_task_ids_within_split(
    records: Sequence[Mapping[str, Any]],
    allowed_task_ids: Sequence[str],
    *,
    split_name: str,
) -> None:
    """Prevent accepted trajectories from crossing frozen project splits."""

    allowed = set(allowed_task_ids)
    observed = {str(record["task_id"]) for record in records}
    outside = sorted(observed - allowed)
    if outside:
        raise SFTDatasetError(
            f"SFT {split_name} contains task outside its frozen split: {outside[0]!r}"
        )


def _token_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        if len(value) != 1:
            raise SFTDatasetError("chat template returned more than one sequence")
        value = value[0]
    if not isinstance(value, list) or not all(isinstance(token, int) for token in value):
        raise SFTDatasetError("chat template must return a list of token IDs")
    return value


def _render(
    tokenizer: ChatTemplateTokenizer,
    messages: Sequence[Mapping[str, Any]],
    tool_schema: Mapping[str, Any],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    try:
        rendered = tokenizer.apply_chat_template(
            _messages_for_qwen_template(messages),
            tools=[dict(tool_schema)],
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except Exception as exc:
        raise SFTDatasetError(f"chat-template rendering failed: {exc}") from exc
    return _token_ids(rendered)


def _messages_for_qwen_template(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Adapt OpenAI JSON-string arguments to Qwen3.5's mapping contract."""

    adapted = copy.deepcopy(list(messages))
    for message in adapted:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls", []):
            function = call.get("function") if isinstance(call, Mapping) else None
            if not isinstance(function, dict):
                raise SFTDatasetError("assistant tool call has no function mapping")
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    parsed = json.loads(arguments)
                except json.JSONDecodeError as exc:
                    raise SFTDatasetError("assistant tool arguments are invalid JSON") from exc
                if not isinstance(parsed, dict):
                    raise SFTDatasetError("assistant tool arguments must decode to a mapping")
                function["arguments"] = parsed
    return adapted


def build_action_only_examples(
    records: Sequence[Mapping[str, Any]],
    tokenizer: ChatTemplateTokenizer,
    tool_schema: Mapping[str, Any],
    *,
    max_sequence_length: int,
    accepted_quality_tiers: Sequence[str] = ("gold", "silver"),
    record_format: str = "trajectory",
) -> tuple[ActionOnlyExample, ...]:
    """Build exact per-assistant-turn masks using a verified token prefix."""

    if max_sequence_length <= 0:
        raise SFTDatasetError("max_sequence_length must be positive")
    examples: list[ActionOnlyExample] = []
    for trajectory_number, record in enumerate(records, start=1):
        reasons = sft_record_admission_reasons(
            record,
            record_format=record_format,
            accepted_quality_tiers=accepted_quality_tiers,
        )
        if reasons:
            raise SFTDatasetError(
                f"task {record.get('task_id')!r} is not trainable: {', '.join(reasons)}"
            )
        messages = record["messages"]
        assistant_index = 0
        for message_index, message in enumerate(messages):
            if message.get("role") != "assistant":
                continue
            assistant_index += 1
            if message.get("loss_mask") is True:
                # The environment consumed this turn, but its action failed
                # admission and must remain context-only for action SFT.
                continue
            context = messages[:message_index]
            prompt_ids = _render(
                tokenizer, context, tool_schema, add_generation_prompt=True
            )
            full_ids = _render(
                tokenizer,
                [*context, message],
                tool_schema,
                add_generation_prompt=False,
            )
            if len(full_ids) == len(prompt_ids) and full_ids == prompt_ids:
                raise SFTDatasetError("action-only example has no supervised tokens")
            if len(full_ids) < len(prompt_ids) or full_ids[: len(prompt_ids)] != prompt_ids:
                raise SFTDatasetError(
                    "chat template cannot produce a verified assistant completion prefix"
                )
            if len(full_ids) > max_sequence_length:
                raise SFTTrajectoryTooLongError(
                    f"task {record['task_id']!r} assistant turn {assistant_index} has "
                    f"{len(full_ids)} tokens, exceeding max_sequence_length={max_sequence_length}; "
                    "silent truncation is forbidden"
                )
            labels = [IGNORE_INDEX] * len(prompt_ids) + full_ids[len(prompt_ids) :]
            if not any(value != IGNORE_INDEX for value in labels):
                raise SFTDatasetError("action-only example has no supervised tokens")
            examples.append(
                ActionOnlyExample(
                    input_ids=tuple(full_ids),
                    attention_mask=(1,) * len(full_ids),
                    labels=tuple(labels),
                    task_id=str(record["task_id"]),
                    trajectory_id=f"{record['task_id']}#trajectory-{trajectory_number}",
                    assistant_turn_index=assistant_index,
                    composition=str(record["composition"]),
                )
            )
    if not examples:
        raise SFTDatasetError("no action-only examples were produced")
    return tuple(examples)


def build_action_only_dataset(
    records: Sequence[Mapping[str, Any]],
    tokenizer: ChatTemplateTokenizer,
    tool_schema: Mapping[str, Any],
    *,
    max_sequence_length: int,
    accepted_quality_tiers: Sequence[str] = ("gold", "silver"),
    record_format: str = "trajectory",
) -> tuple[tuple[ActionOnlyExample, ...], tuple[dict[str, Any], ...]]:
    """Render a split, dropping only whole overlong trajectories.

    Structural or gate failures still fail loudly.  An overlong turn discards
    every example from that trajectory and is returned as an explicit audit
    record; no prompt, tool call, or Observation is truncated.
    """

    examples: list[ActionOnlyExample] = []
    rejected: list[dict[str, Any]] = []
    for record in records:
        try:
            values = build_action_only_examples(
                [record],
                tokenizer,
                tool_schema,
                max_sequence_length=max_sequence_length,
                accepted_quality_tiers=accepted_quality_tiers,
                record_format=record_format,
            )
        except SFTTrajectoryTooLongError as exc:
            rejected.append(
                {
                    "task_id": str(record.get("task_id", "")),
                    "composition": str(record.get("composition", "")),
                    "reason": "trajectory_too_long",
                    "detail": str(exc),
                }
            )
            continue
        examples.extend(values)
    if not examples:
        raise SFTDatasetError("no action-only examples remain after overlong rejection")
    return tuple(examples), tuple(rejected)


@dataclass(frozen=True)
class ActionOnlyDataCollator:
    pad_token_id: int
    label_pad_token_id: int = IGNORE_INDEX
    padding_side: str = "right"

    def __call__(self, features: Sequence[Mapping[str, Any] | ActionOnlyExample]) -> dict[str, Any]:
        if not features:
            raise SFTDatasetError("cannot collate an empty batch")
        try:
            import torch
        except ImportError:  # Offline schema checks do not require training extras.
            import numpy as np

            tensor = np.asarray
        else:
            tensor = lambda value: torch.tensor(value, dtype=torch.long)
        normalized = [
            value.to_trainer_dict() if isinstance(value, ActionOnlyExample) else dict(value)
            for value in features
        ]
        width = max(len(value["input_ids"]) for value in normalized)
        rows: dict[str, list[list[int]]] = {"input_ids": [], "attention_mask": [], "labels": []}
        for value in normalized:
            length = len(value["input_ids"])
            if len(value["attention_mask"]) != length or len(value["labels"]) != length:
                raise SFTDatasetError("input_ids, attention_mask, and labels must align")
            pad = width - length
            if self.padding_side == "right":
                rows["input_ids"].append(list(value["input_ids"]) + [self.pad_token_id] * pad)
                rows["attention_mask"].append(list(value["attention_mask"]) + [0] * pad)
                rows["labels"].append(list(value["labels"]) + [self.label_pad_token_id] * pad)
            elif self.padding_side == "left":
                rows["input_ids"].append([self.pad_token_id] * pad + list(value["input_ids"]))
                rows["attention_mask"].append([0] * pad + list(value["attention_mask"]))
                rows["labels"].append([self.label_pad_token_id] * pad + list(value["labels"]))
            else:
                raise SFTDatasetError("padding_side must be 'left' or 'right'")
        return {key: tensor(value) for key, value in rows.items()}


def rendered_dataset_summary(examples: Sequence[ActionOnlyExample]) -> dict[str, Any]:
    if not examples:
        raise SFTDatasetError("cannot summarize an empty rendered dataset")
    lengths = sorted(value.sequence_length for value in examples)
    labels = sum(value.label_tokens for value in examples)

    def percentile(fraction: float) -> int:
        index = min(len(lengths) - 1, math.ceil(fraction * len(lengths)) - 1)
        return lengths[max(0, index)]

    return {
        "assistant_decisions": len(examples),
        "effective_label_tokens": labels,
        "sequence_length": {
            "min": lengths[0],
            "p50": percentile(0.50),
            "p90": percentile(0.90),
            "p95": percentile(0.95),
            "max": lengths[-1],
        },
        "composition_distribution": dict(
            sorted(Counter(value.composition for value in examples).items())
        ),
    }
