"""Construct and validate one-step recovery targets.

This module consumes ``recovery-boundary-v1`` records and emits a new,
target-bearing contract without consulting reward snapshots or hidden task
labels.  Accepted Teacher actions are reused only when they can be matched to
the actor-visible context prefix.  Other targets are short deterministic
public rewrites; ambiguous answer IDs and incomplete aspect transitions are
quarantined instead of guessed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from travel_grpo.data.recovery_boundaries import (
    ALL_SPLITS,
    BOUNDARY_TYPES,
    HIDDEN_FIELD_NAMES,
    SCHEMA_VERSION as BOUNDARY_SCHEMA_VERSION,
    TRAINING_SPLITS,
    VALIDATION_SPLITS,
    events_from_messages,
    extract_message_boundaries,
    extract_recovery_boundaries,
    load_task_split_map,
    normalize_actor_messages,
    parse_grpo_transcript,
)
from travel_grpo.envs.public_control import (
    extract_public_aspects,
    is_substantive_query_change,
    normalize_public_query,
)
from travel_grpo.envs.userbench_tools import (
    ActionChoice,
    FIELD_QUERY_HINTS,
    OPTION_ID,
    UserBenchAction,
    UserBenchActionError,
    action_field_matches,
    action_mentions_aspect,
    aspect_from_option_id,
)


TARGET_SCHEMA_VERSION = "recovery-target-v1"
TARGET_GENERATOR_VERSION = "recovery-target-generator-v1"
TARGET_STATUS_ACCEPTED = "accepted"
TARGET_STATUS_REJECTED = "rejected"
TARGET_STATUS_EXCLUDED_EVALUATION = "excluded_evaluation"

_HUMAN_ASPECT = {
    "flight": "flight",
    "hotel": "hotel",
    "apartment": "apartment",
    "rental_car": "rental car",
    "restaurant": "restaurant",
}

_FIELD_PHRASES = {
    "flight": {
        "company": "airline or carrier",
        "path": "route or nonstop preference",
        "time": "departure or arrival time",
        "amenities": "flight amenities",
        "service": "baggage or cabin service",
    },
    "hotel": {
        "name": "hotel or property name",
        "room": "room configuration",
        "amenities": "hotel amenities",
        "service": "hotel services",
        "rating": "minimum hotel rating",
    },
    "apartment": {
        "name": "apartment or property name",
        "room": "room configuration",
        "amenities": "apartment amenities",
        "service": "apartment services",
        "rating": "minimum apartment rating",
    },
    "rental_car": {
        "brand": "rental car brand or company",
        "model": "rental car model or vehicle type",
        "seats": "number of rental car seats",
        "insurance": "rental car insurance coverage",
        "service": "rental car services",
    },
    "restaurant": {
        "cuisine": "restaurant cuisine",
        "tags": "restaurant features or tags",
        "rating": "minimum restaurant rating",
        "expectation": "restaurant price range or budget",
    },
}

_GENERIC_THOUGHT = "public recovery"
_REQUIRED_ARGUMENTS = frozenset(("thought", "choice", "content"))
_SOURCE_MESSAGES_CACHE: dict[tuple[str, int], list[dict[str, Any]] | None] = {}


@dataclass(frozen=True)
class _SourceAction:
    action: UserBenchAction
    source: dict[str, Any]


@dataclass(frozen=True)
class TargetDecision:
    status: str
    target_assistant: dict[str, Any] | None
    target_provenance: dict[str, Any]
    quality_checks: dict[str, Any]
    rejection_reasons: tuple[str, ...] = ()


def _normalise(value: str) -> str:
    return " ".join(value.casefold().split())


def _relative(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _jsonl_row(path: Path, line: int) -> Mapping[str, Any] | None:
    try:
        with path.open(encoding="utf-8") as handle:
            for number, raw in enumerate(handle, start=1):
                if number != line:
                    continue
                value = json.loads(raw)
                return value if isinstance(value, Mapping) else None
    except (OSError, json.JSONDecodeError):
        return None
    return None


def _load_messages(path: Path, root: Path, line: int | None = None) -> list[dict[str, Any]] | None:
    """Load only actor-visible messages from one provenance artifact.

    Source prefixes recur across deduplicated contexts.  A small process-local
    cache avoids reparsing the same Teacher/GRPO row thousands of times while
    retaining only actor-visible messages.
    """

    cache_key = (str(path.resolve()), int(line or 0))
    if cache_key in _SOURCE_MESSAGES_CACHE:
        cached = _SOURCE_MESSAGES_CACHE[cache_key]
        return copy.deepcopy(cached) if cached is not None else None
    try:
        if path.suffix == ".jsonl":
            row = _jsonl_row(path, line or 1)
        else:
            row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(row, Mapping):
        _SOURCE_MESSAGES_CACHE[cache_key] = None
        return None
    messages = row.get("messages")
    if isinstance(messages, list):
        result = normalize_actor_messages(messages)
    else:
        transcript = row.get("visible_transcript")
        if isinstance(transcript, list):
            result = normalize_actor_messages(transcript)
        elif "input" in row or "output" in row:
            result = parse_grpo_transcript(str(row.get("input", "")), str(row.get("output", "")))
        else:
            result = None
    _SOURCE_MESSAGES_CACHE[cache_key] = copy.deepcopy(result) if result is not None else None
    return result


def _source_for_provenance(
    root: Path, provenance: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    """Resolve a source record, preferring the exact source context reference."""

    source_path = provenance.get("source_context_file") or provenance.get("path")
    source_line = provenance.get("source_context_line") or provenance.get("line")
    if not isinstance(source_path, str):
        return None
    path = root / source_path
    messages = _load_messages(path, root, int(source_line) if isinstance(source_line, int) else None)
    if messages is None:
        return None
    return messages, {
        "path": source_path,
        "line": source_line,
        "source_kind": provenance.get("source_kind", "unknown"),
    }


def _message_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if left.get("role") != right.get("role"):
        return False
    if str(left.get("content", "")) != str(right.get("content", "")):
        return False
    if left.get("role") != "assistant":
        return True
    left_calls = left.get("tool_calls")
    right_calls = right.get("tool_calls")
    if not isinstance(left_calls, list) or not isinstance(right_calls, list):
        return left_calls == right_calls
    if len(left_calls) != len(right_calls):
        return False
    for left_call, right_call in zip(left_calls, right_calls):
        left_function = left_call.get("function", {}) if isinstance(left_call, Mapping) else {}
        right_function = right_call.get("function", {}) if isinstance(right_call, Mapping) else {}
        if left_function.get("name") != right_function.get("name"):
            return False
        try:
            left_args = json.loads(left_function.get("arguments", "{}"))
            right_args = json.loads(right_function.get("arguments", "{}"))
        except (TypeError, json.JSONDecodeError):
            return False
        if left_args != right_args:
            return False
    return True


def _prefix_matches(
    context_messages: Sequence[Mapping[str, Any]],
    source_messages: Sequence[Mapping[str, Any]],
    end: int,
) -> bool:
    if end != len(context_messages) or end > len(source_messages):
        return False
    return all(_message_equal(left, right) for left, right in zip(context_messages, source_messages[:end]))


def _next_source_action(
    context: Mapping[str, Any], root: Path
) -> _SourceAction | None:
    context_messages = normalize_actor_messages(context.get("messages", []))
    for provenance in context.get("source_provenance", []):
        if not isinstance(provenance, Mapping):
            continue
        loaded = _source_for_provenance(root, provenance)
        if loaded is None:
            continue
        source_messages, source_meta = loaded
        if not _prefix_matches(context_messages, source_messages, len(context_messages)):
            continue
        for event in events_from_messages(source_messages):
            if event.assistant_index < len(context_messages):
                continue
            return _SourceAction(event.action, source_meta)
    return None


def _last_action(
    context: Mapping[str, Any], *, choice: ActionChoice | None = None
) -> UserBenchAction | None:
    events = events_from_messages(normalize_actor_messages(context.get("messages", [])))
    for event in reversed(events):
        if choice is None or event.action.choice is choice:
            return event.action
    return None


def _public_aspects(context: Mapping[str, Any]) -> tuple[str, ...]:
    messages = normalize_actor_messages(context.get("messages", []))
    initial = next(
        (str(item.get("content", "")) for item in messages if item.get("role") == "user"),
        "",
    )
    return extract_public_aspects(initial)


def _state(context: Mapping[str, Any]) -> Mapping[str, Any]:
    value = context.get("public_state_before")
    return value if isinstance(value, Mapping) else {}


def _current_aspect(context: Mapping[str, Any]) -> str | None:
    value = _state(context).get("current_aspect")
    return value if isinstance(value, str) and value else None


def _human_aspect(aspect: str) -> str:
    return _HUMAN_ASPECT.get(aspect, aspect.replace("_", " "))


def _field_prompt(aspect: str, field: str) -> str:
    phrase = _FIELD_PHRASES.get(aspect, {}).get(field, f"{_human_aspect(aspect)} preferences")
    article = "an" if phrase[:1].lower() in "aeiou" else "a"
    return f"What {phrase} do you prefer for {article} {_human_aspect(aspect)}?"


def _search_prompt(aspect: str) -> str:
    return f"Search for {_human_aspect(aspect)} options using the preferences stated by the user."


def _target_message(action: UserBenchAction, *, target_id: str) -> dict[str, Any]:
    arguments = json.dumps(
        {
            "thought": action.thought,
            "choice": action.choice.value,
            "content": action.content,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": target_id,
                "type": "function",
                "function": {
                    "name": "interact_with_env",
                    "arguments": arguments,
                },
            }
        ],
    }


def _generated_action(choice: str, content: str) -> UserBenchAction:
    return UserBenchAction.from_parameters(
        {"thought": _GENERIC_THOUGHT, "choice": choice, "content": content}
    )


def _asked_fields(context: Mapping[str, Any], aspect: str) -> set[str]:
    public_aspects = _public_aspects(context)
    asked: set[str] = set()
    for event in events_from_messages(normalize_actor_messages(context.get("messages", []))):
        if event.action.choice is not ActionChoice.ACTION:
            continue
        for found_aspect, field in action_field_matches(event.action.content, public_aspects):
            if found_aspect == aspect:
                asked.add(field)
    return asked


def _next_open_aspect(context: Mapping[str, Any]) -> str | None:
    current = _current_aspect(context)
    state = _state(context)
    answered = set(state.get("answered_aspects", ()))
    blocked = set(state.get("blocked_aspects", ()))
    for aspect in _public_aspects(context):
        if aspect != current and aspect not in answered and aspect not in blocked:
            return aspect
    return None


def _target_for_boundary(
    context: Mapping[str, Any], root: Path
) -> tuple[UserBenchAction | None, dict[str, Any], list[str]]:
    boundary_type = context.get("boundary_type")
    state = _state(context)
    aspect = _current_aspect(context)
    source_action = _next_source_action(context, root)
    source_meta = source_action.source if source_action is not None else {}
    reasons: list[str] = []

    if boundary_type == "visible_options_pending_answer":
        return None, {"method": "quarantine"}, ["answer_id_not_determinable_without_hidden_correctness"]

    if boundary_type in {"preference_complete_to_search", "valid_search_to_answer"}:
        if source_action is None:
            return None, {"method": "accepted_teacher_or_source_prefix"}, ["source_target_not_found"]
        action = source_action.action
        if boundary_type == "preference_complete_to_search":
            if action.choice is not ActionChoice.SEARCH:
                reasons.append("source_target_choice_not_search")
            if not aspect or not action_mentions_aspect(action.content, aspect):
                reasons.append("search_target_aspect_mismatch")
            if state.get("recovery_mode") not in {"ELICITING", "SEARCH_REQUIRED"}:
                reasons.append("search_phase_mismatch")
        else:
            if action.choice is not ActionChoice.ANSWER:
                reasons.append("source_target_choice_not_answer")
            ids = [item.strip() for item in action.content.split(",") if item.strip()]
            visible = set(str(item) for item in state.get("visible_option_ids", ()))
            if len(ids) != 1 or OPTION_ID.fullmatch(ids[0] or "") is None:
                reasons.append("answer_not_exactly_one_option_id")
            elif ids[0] not in visible:
                reasons.append("answer_id_not_visible")
            elif not aspect or aspect_from_option_id(ids[0]) != aspect:
                reasons.append("answer_aspect_mismatch")
            if state.get("recovery_mode") != "ANSWER_REQUIRED":
                reasons.append("answer_phase_mismatch")
        if reasons:
            return None, {"method": "accepted_teacher_or_source_prefix", **source_meta}, reasons
        return action, {"method": "accepted_teacher_or_source_prefix", **source_meta}, []

    if not aspect:
        return None, {"method": "deterministic_public_rule"}, ["no_current_public_aspect"]

    if boundary_type == "first_fallback":
        original = _last_action(context, choice=ActionChoice.SEARCH)
        if original is None:
            return None, {"method": "deterministic_query_rewrite"}, ["original_search_not_found"]
        if state.get("recovery_mode") != "SEARCH_RETRY_REQUIRED":
            reasons.append("retry_phase_mismatch")
        revised = f"Revised search for {_human_aspect(aspect)}: {original.content}"
        if not action_mentions_aspect(revised, aspect):
            reasons.append("rewritten_query_aspect_mismatch")
        if not is_substantive_query_change(original.content, revised):
            reasons.append("rewritten_query_not_substantive")
        if reasons:
            return None, {"method": "deterministic_query_rewrite"}, reasons
        return _generated_action("search", revised), {"method": "deterministic_query_rewrite"}, []

    if boundary_type == "second_fallback":
        if state.get("recovery_mode") != "SWITCH_ASPECT_REQUIRED":
            reasons.append("switch_phase_mismatch")
        if aspect not in set(state.get("blocked_aspects", ())):
            reasons.append("failed_aspect_not_publicly_blocked")
        next_aspect = _next_open_aspect(context)
        if next_aspect is None:
            reasons.append("no_next_open_public_aspect")
        if reasons:
            return None, {"method": "deterministic_aspect_switch"}, reasons
        asked = _asked_fields(context, next_aspect)
        field = next(
            (candidate for candidate in FIELD_QUERY_HINTS.get(next_aspect, {}) if candidate not in asked),
            None,
        )
        if field is None:
            return None, {"method": "deterministic_aspect_switch"}, ["no_safe_field_for_next_aspect"]
        return (
            _generated_action("action", _field_prompt(next_aspect, field)),
            {"method": "deterministic_aspect_switch", "next_aspect": next_aspect, "field": field},
            [],
        )

    if boundary_type == "repeated_no_progress_action":
        if state.get("recovery_mode") not in {"ELICITING", "SEARCH_REQUIRED"}:
            reasons.append("search_phase_mismatch")
        query = _search_prompt(aspect)
        if reasons:
            return None, {"method": "deterministic_public_search"}, reasons
        return _generated_action("search", query), {"method": "deterministic_public_search"}, []

    if boundary_type == "explicit_no_preference":
        if state.get("recovery_mode") == "SEARCH_REQUIRED":
            return (
                _generated_action("search", _search_prompt(aspect)),
                {"method": "deterministic_no_preference_search"},
                [],
            )
        if state.get("recovery_mode") != "ELICITING":
            return None, {"method": "deterministic_next_public_field"}, ["elicitation_phase_mismatch"]
        asked = _asked_fields(context, aspect)
        field = next(
            (candidate for candidate in FIELD_QUERY_HINTS.get(aspect, {}) if candidate not in asked),
            None,
        )
        if field is None:
            return (
                _generated_action("search", _search_prompt(aspect)),
                {"method": "deterministic_no_preference_search"},
                [],
            )
        return (
            _generated_action("action", _field_prompt(aspect, field)),
            {"method": "deterministic_next_public_field", "field": field},
            [],
        )

    return None, {"method": "unknown_boundary"}, ["unknown_boundary_type"]


def _target_hidden_key_hits(value: Any, path: str = "") -> list[str]:
    hits: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).casefold() in HIDDEN_FIELD_NAMES:
                hits.append(f"{path}/{key}")
            hits.extend(_target_hidden_key_hits(child, f"{path}/{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            hits.extend(_target_hidden_key_hits(child, f"{path}/{index}"))
    return hits


def validate_target(
    context: Mapping[str, Any], action: UserBenchAction
) -> tuple[bool, list[str], dict[str, Any]]:
    """Validate one candidate action using only its public boundary record."""

    reasons: list[str] = []
    state = _state(context)
    aspect = _current_aspect(context)
    if not aspect:
        reasons.append("no_current_public_aspect")
    if action.choice is ActionChoice.ANSWER:
        ids = [item.strip() for item in action.content.split(",") if item.strip()]
        visible = set(str(item) for item in state.get("visible_option_ids", ()))
        if len(ids) != 1 or OPTION_ID.fullmatch(ids[0] if ids else "") is None:
            reasons.append("answer_not_exactly_one_option_id")
        elif ids[0] not in visible:
            reasons.append("answer_id_not_visible")
        elif aspect_from_option_id(ids[0]) != aspect:
            reasons.append("answer_aspect_mismatch")
        if state.get("recovery_mode") != "ANSWER_REQUIRED":
            reasons.append("answer_phase_mismatch")
    elif action.choice is ActionChoice.SEARCH:
        if not aspect or not action_mentions_aspect(action.content, aspect):
            reasons.append("search_target_aspect_mismatch")
        mode = state.get("recovery_mode")
        if mode not in {"ELICITING", "SEARCH_REQUIRED", "SEARCH_RETRY_REQUIRED"}:
            reasons.append("search_phase_mismatch")
    elif action.choice is ActionChoice.ACTION:
        if not aspect or not action_mentions_aspect(action.content, aspect):
            # A switch target is allowed to mention the next public aspect;
            # callers verify that transition separately below.
            next_aspect = _next_open_aspect(context)
            if not next_aspect or not action_mentions_aspect(action.content, next_aspect):
                reasons.append("action_target_aspect_mismatch")
        if state.get("recovery_mode") not in {"ELICITING", "SWITCH_ASPECT_REQUIRED"}:
            reasons.append("action_phase_mismatch")
    else:
        reasons.append("unsupported_choice")

    if context.get("boundary_type") == "first_fallback":
        original = _last_action(context, choice=ActionChoice.SEARCH)
        if original is None or action.choice is not ActionChoice.SEARCH:
            reasons.append("fallback_retry_missing_original_search")
        elif not is_substantive_query_change(original.content, action.content):
            reasons.append("rewritten_query_not_substantive")
        elif _normalise(action.content) == _normalise(original.content):
            reasons.append("exact_search_repeat")
    if context.get("boundary_type") == "second_fallback":
        failed_aspect = aspect
        next_aspect = _next_open_aspect(context)
        if next_aspect is None:
            reasons.append("no_next_open_public_aspect")
        if failed_aspect and action_mentions_aspect(action.content, failed_aspect):
            reasons.append("target_reuses_blocked_aspect")
        if next_aspect and not action_mentions_aspect(action.content, next_aspect):
            reasons.append("target_does_not_switch_aspect")

    target_message = _target_message(action, target_id="recovery-target")
    hidden_hits = _target_hidden_key_hits(target_message)
    if hidden_hits:
        reasons.append("hidden_field_in_target")
    checks = {
        "single_tool_call": True,
        "tool_name": "interact_with_env",
        "tool_schema_valid": not bool(reasons),
        "assistant_content_empty": target_message.get("content") == "",
        "public_only": not hidden_hits,
        "public_state_consistent": not bool(reasons),
    }
    return not reasons, reasons, {"message": target_message, "checks": checks}


def construct_target(context: Mapping[str, Any], project_root: str | Path) -> TargetDecision:
    """Construct one target and classify it as accepted, rejected, or eval-excluded."""

    root = Path(project_root).resolve()
    try:
        action, provenance, construction_reasons = _target_for_boundary(context, root)
    except (TypeError, ValueError, UserBenchActionError) as exc:
        action, provenance, construction_reasons = None, {"method": "construction_error"}, [
            f"construction_error:{exc.__class__.__name__}"
        ]
    if action is None:
        status = TARGET_STATUS_REJECTED
        if context.get("project_split") == "evaluation":
            status = TARGET_STATUS_EXCLUDED_EVALUATION
        return TargetDecision(
            status=status,
            target_assistant=None,
            target_provenance=provenance,
            quality_checks={
                "single_tool_call": False,
                "tool_schema_valid": False,
                "assistant_content_empty": True,
                "public_only": True,
                "public_state_consistent": False,
            },
            rejection_reasons=tuple(construction_reasons),
        )
    valid, validation_reasons, target = validate_target(context, action)
    reasons = list(construction_reasons) + validation_reasons
    if reasons or not valid:
        status = TARGET_STATUS_REJECTED
        if context.get("project_split") == "evaluation":
            status = TARGET_STATUS_EXCLUDED_EVALUATION
        return TargetDecision(
            status=status,
            target_assistant=None,
            target_provenance=provenance,
            quality_checks=target["checks"],
            rejection_reasons=tuple(dict.fromkeys(reasons)),
        )
    status = (
        TARGET_STATUS_EXCLUDED_EVALUATION
        if context.get("project_split") == "evaluation"
        else TARGET_STATUS_ACCEPTED
    )
    return TargetDecision(
        status=status,
        target_assistant=target["message"],
        target_provenance=provenance,
        quality_checks={**target["checks"], "validated": True},
    )


def _target_record(context: Mapping[str, Any], decision: TargetDecision) -> dict[str, Any]:
    record = copy.deepcopy(dict(context))
    record["schema_version"] = TARGET_SCHEMA_VERSION
    record["boundary_schema_version"] = BOUNDARY_SCHEMA_VERSION
    record["target_assistant"] = copy.deepcopy(decision.target_assistant)
    record["target_status"] = decision.status
    record["target_provenance"] = copy.deepcopy(decision.target_provenance)
    record["target_quality_checks"] = copy.deepcopy(decision.quality_checks)
    record["target_rejection_reasons"] = list(decision.rejection_reasons)
    return record


def _read_contexts(path: Path) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            if raw.strip():
                value = json.loads(raw)
                if isinstance(value, dict):
                    contexts.append(value)
    return contexts


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def build_target_dataset(
    contexts: Sequence[Mapping[str, Any]], project_root: str | Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build train/validation/rejected records and a validation manifest."""

    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    by_boundary: dict[str, Counter[str]] = defaultdict(Counter)
    rejection_reasons: Counter[str] = Counter()
    hidden_hits = 0
    for context in contexts:
        boundary_type = str(context.get("boundary_type", "unknown"))
        decision = construct_target(context, project_root)
        enriched = _target_record(context, decision)
        bucket = by_boundary[boundary_type]
        bucket["total"] += 1
        if decision.target_assistant is not None:
            bucket["target_valid"] += 1
        if decision.status == TARGET_STATUS_ACCEPTED:
            bucket["accepted"] += 1
            if context.get("project_split") in TRAINING_SPLITS:
                train.append(enriched)
            elif context.get("project_split") in VALIDATION_SPLITS:
                validation.append(enriched)
            else:
                rejected.append(enriched)
        else:
            bucket["excluded_or_rejected"] += 1
            rejected.append(enriched)
        for reason in decision.rejection_reasons:
            rejection_reasons[reason] += 1
        hidden_hits += len(_target_hidden_key_hits(enriched))

    def finalize(value: Counter[str]) -> dict[str, Any]:
        total = int(value.get("total", 0))
        valid = int(value.get("target_valid", 0))
        accepted = int(value.get("accepted", 0))
        return {
            "total": total,
            "target_valid": valid,
            "accepted": accepted,
            "excluded_or_rejected": int(value.get("excluded_or_rejected", 0)),
            "target_valid_rate": valid / total if total else 0.0,
            "train_validation_acceptance_rate": accepted / total if total else 0.0,
        }

    by_boundary_final = {
        key: finalize(value) for key, value in sorted(by_boundary.items())
    }
    split_counts = Counter(str(context.get("project_split", "unknown")) for context in contexts)
    manifest = {
        "schema_version": TARGET_SCHEMA_VERSION,
        "boundary_schema_version": BOUNDARY_SCHEMA_VERSION,
        "generator_version": TARGET_GENERATOR_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_contract": {
            "assistant_role": "assistant",
            "tool_name": "interact_with_env",
            "one_tool_call_only": True,
            "assistant_content_must_be_empty": True,
            "target_generation": "deterministic_rules_and_accepted_source_actions",
            "teacher_api_used": False,
            "local_gpu_used": False,
        },
        "counts": {
            "contexts_seen": len(contexts),
            "train": len(train),
            "validation": len(validation),
            "rejected_or_quarantine": len(rejected),
            "by_boundary_type": by_boundary_final,
            "by_project_split": dict(sorted(split_counts.items())),
            "rejection_reasons": dict(sorted(rejection_reasons.items())),
        },
        "split_checks": {
            "task_id_assignment_before_target_generation": True,
            "sample_level_random_split": False,
            "evaluation_in_train": sum(
                record.get("project_split") == "evaluation" for record in train
            ),
            "evaluation_contexts_excluded": split_counts.get("evaluation", 0),
            "train_task_ids": sorted(
                {record["task_id"] for record in train}
            ),
            "validation_task_ids": sorted(
                {record["task_id"] for record in validation}
            ),
        },
        "quality_checks": {
            "target_hidden_key_hits": hidden_hits,
            "all_targets_single_tool_call": all(
                record.get("target_quality_checks", {}).get("single_tool_call") is True
                for record in train + validation
            ),
            "all_targets_public_only": all(
                record.get("target_quality_checks", {}).get("public_only") is True
                for record in train + validation
            ),
            "all_targets_deferred_in_source": all(
                record.get("boundary_schema_version") == BOUNDARY_SCHEMA_VERSION
                for record in train + validation + rejected
            ),
        },
    }
    return train, validation, rejected, manifest


def write_target_dataset(
    train: Sequence[Mapping[str, Any]],
    validation: Sequence[Mapping[str, Any]],
    rejected: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write derived target files and manifest under an ignored directory."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": directory / "train.jsonl",
        "validation": directory / "validation.jsonl",
        "rejected": directory / "rejected.jsonl",
        "manifest": directory / "manifest.json",
    }
    _write_jsonl(paths["train"], train)
    _write_jsonl(paths["validation"], validation)
    _write_jsonl(paths["rejected"], rejected)
    manifest_value = copy.deepcopy(dict(manifest))
    manifest_value["output"] = {key: str(path) for key, path in paths.items()}
    paths["manifest"].write_text(
        json.dumps(manifest_value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return paths


def build_targets_from_boundary_file(
    context_path: str | Path,
    project_root: str | Path,
    output_dir: str | Path,
) -> tuple[dict[str, Path], dict[str, Any]]:
    """Build targets from JSONL without retaining the full boundary set in RAM."""

    root = Path(project_root).resolve()
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": directory / "train.jsonl",
        "validation": directory / "validation.jsonl",
        "rejected": directory / "rejected.jsonl",
        "manifest": directory / "manifest.json",
    }
    by_boundary: dict[str, Counter[str]] = defaultdict(Counter)
    rejection_reasons: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    train_task_ids: set[str] = set()
    validation_task_ids: set[str] = set()
    contexts_seen = 0
    train_count = 0
    validation_count = 0
    rejected_count = 0
    hidden_hits = 0
    all_train_validation_single_call = True
    all_train_validation_public_only = True
    all_deferred = True

    with (
        Path(context_path).open(encoding="utf-8") as source_handle,
        paths["train"].open("w", encoding="utf-8") as train_handle,
        paths["validation"].open("w", encoding="utf-8") as validation_handle,
        paths["rejected"].open("w", encoding="utf-8") as rejected_handle,
    ):
        for raw in source_handle:
            if not raw.strip():
                continue
            context = json.loads(raw)
            if not isinstance(context, Mapping):
                continue
            contexts_seen += 1
            boundary_type = str(context.get("boundary_type", "unknown"))
            split = str(context.get("project_split", "unknown"))
            split_counts[split] += 1
            decision = construct_target(context, root)
            enriched = _target_record(context, decision)
            bucket = by_boundary[boundary_type]
            bucket["total"] += 1
            if decision.target_assistant is not None:
                bucket["target_valid"] += 1
            if decision.status == TARGET_STATUS_ACCEPTED and split in TRAINING_SPLITS:
                bucket["accepted"] += 1
                train_count += 1
                train_task_ids.add(str(context.get("task_id", "")))
                train_handle.write(json.dumps(enriched, ensure_ascii=False, sort_keys=True) + "\n")
            elif decision.status == TARGET_STATUS_ACCEPTED and split in VALIDATION_SPLITS:
                bucket["accepted"] += 1
                validation_count += 1
                validation_task_ids.add(str(context.get("task_id", "")))
                validation_handle.write(json.dumps(enriched, ensure_ascii=False, sort_keys=True) + "\n")
            else:
                bucket["excluded_or_rejected"] += 1
                rejected_count += 1
                rejected_handle.write(json.dumps(enriched, ensure_ascii=False, sort_keys=True) + "\n")
            for reason in decision.rejection_reasons:
                rejection_reasons[reason] += 1
            hidden_hits += len(_target_hidden_key_hits(enriched))
            if decision.status == TARGET_STATUS_ACCEPTED:
                checks = decision.quality_checks
                all_train_validation_single_call = all_train_validation_single_call and checks.get("single_tool_call") is True
                all_train_validation_public_only = all_train_validation_public_only and checks.get("public_only") is True
            all_deferred = all_deferred and enriched.get("boundary_schema_version") == BOUNDARY_SCHEMA_VERSION

    def finalize(value: Counter[str]) -> dict[str, Any]:
        total = int(value.get("total", 0))
        valid = int(value.get("target_valid", 0))
        accepted = int(value.get("accepted", 0))
        return {
            "total": total,
            "target_valid": valid,
            "accepted": accepted,
            "excluded_or_rejected": int(value.get("excluded_or_rejected", 0)),
            "target_valid_rate": valid / total if total else 0.0,
            "train_validation_acceptance_rate": accepted / total if total else 0.0,
        }

    manifest = {
        "schema_version": TARGET_SCHEMA_VERSION,
        "boundary_schema_version": BOUNDARY_SCHEMA_VERSION,
        "generator_version": TARGET_GENERATOR_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_contract": {
            "assistant_role": "assistant",
            "tool_name": "interact_with_env",
            "one_tool_call_only": True,
            "assistant_content_must_be_empty": True,
            "target_generation": "deterministic_rules_and_accepted_source_actions",
            "teacher_api_used": False,
            "local_gpu_used": False,
        },
        "counts": {
            "contexts_seen": contexts_seen,
            "train": train_count,
            "validation": validation_count,
            "rejected_or_quarantine": rejected_count,
            "by_boundary_type": {
                key: finalize(value) for key, value in sorted(by_boundary.items())
            },
            "by_project_split": dict(sorted(split_counts.items())),
            "rejection_reasons": dict(sorted(rejection_reasons.items())),
        },
        "split_checks": {
            "task_id_assignment_before_target_generation": True,
            "sample_level_random_split": False,
            "evaluation_in_train": 0,
            "evaluation_contexts_excluded": split_counts.get("evaluation", 0),
            "train_task_ids": sorted(train_task_ids),
            "validation_task_ids": sorted(validation_task_ids),
        },
        "quality_checks": {
            "target_hidden_key_hits": hidden_hits,
            "all_targets_single_tool_call": all_train_validation_single_call,
            "all_targets_public_only": all_train_validation_public_only,
            "all_targets_deferred_in_source": all_deferred,
        },
        "output": {key: str(path) for key, path in paths.items()},
    }
    paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return paths, manifest


__all__ = [
    "TARGET_GENERATOR_VERSION",
    "TARGET_SCHEMA_VERSION",
    "TARGET_STATUS_ACCEPTED",
    "TARGET_STATUS_EXCLUDED_EVALUATION",
    "TARGET_STATUS_REJECTED",
    "TargetDecision",
    "build_target_dataset",
    "build_targets_from_boundary_file",
    "construct_target",
    "validate_target",
    "write_target_dataset",
]
