#!/usr/bin/env python3
"""Reproducible SFT Actor inference gate.

This gate compares the historical runtime (A) with the production runtime
policy plus public control/phase guard (B).  Boundary probes are one-step
Actor calls over frozen, actor-visible contexts; the closed-loop portion runs
the same eight frozen UserBench task rows under both conditions.  The script
never updates model parameters and never serializes hidden reward state into
Actor prompts or probe artifacts.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import sys
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from travel_grpo.data.recovery_boundaries import normalize_actor_messages
from travel_grpo.envs.public_control import (
    PublicAspectStatus,
    RecoveryMode,
    advance_public_aspect,
    render_actor_control_info,
)
from travel_grpo.envs.userbench_context import UserBenchSessionState
from travel_grpo.envs.userbench_interaction import SimulatorRole, UserSimulatorRuntime
from travel_grpo.envs.userbench_tools import (
    ActionChoice,
    OPTION_ID,
    UserBenchAction,
    UserBenchActionError,
    action_mentions_aspect,
    normalized_action_signature,
)
from travel_grpo.envs.userbench_wrapper import UserBenchEnvironmentConfig, UserBenchWrapper
from travel_grpo.evaluation.artifacts import atomic_json
from travel_grpo.evaluation.rollout import rollout_task
from travel_grpo.models.openai_compatible import TeacherApiError, TeacherProtocolError
from travel_grpo.models.vllm_policy import ActorRuntime, OpenAICompatibleActorClient
from travel_grpo.prompts.actor_policy import (
    ACTOR_RUNTIME_POLICY,
    ACTOR_RUNTIME_POLICY_MARKER,
    ACTOR_RUNTIME_POLICY_VERSION,
    ensure_actor_runtime_policy,
    strip_actor_runtime_policy,
)
from travel_grpo.training.recovery_sft import public_state_from_payload


SCHEMA_VERSION = "inference-gate-v1"
PHASE_GUARD_VERSION = "public-control-v1"
BOUNDARY_SCHEMA_VERSION = "recovery-boundary-v1"
BOUNDARY_FILE = ROOT / "outputs/recovery_boundaries/recovery-boundary-v1/contexts.jsonl"
BOUNDARY_TYPES = {
    "normal_search_result": "valid_search_to_answer",
    "first_fallback": "first_fallback",
    "second_fallback": "second_fallback",
    "preference_complete": "preference_complete_to_search",
    "confused_history": "repeated_no_progress_action",
}
FIXED_COUNTS = {
    "normal_search_result": 24,
    "first_fallback": 24,
    "second_fallback": 24,
    "preference_complete": 24,
    "confused_history": 32,
}
EXCLUDED_COMPOSITIONS = ("333", "334", "444", "2222")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    )


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def _source_info(record: Mapping[str, Any]) -> dict[str, Any]:
    provenance = record.get("source_provenance")
    first = provenance[0] if isinstance(provenance, list) and provenance else {}
    if not isinstance(first, Mapping):
        first = {}
    return {
        "source_kind": first.get("source_kind", "unknown"),
        "source_split": record.get("project_split", "unknown"),
        "source_path": first.get("path"),
        "source_line": first.get("line"),
        "formal_evaluation": bool(first.get("formal_evaluation", False)),
    }


def load_boundary_records(path: Path = BOUNDARY_FILE) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path}; build recovery-boundary-v1 before running the gate"
        )
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or value.get("schema_version") != BOUNDARY_SCHEMA_VERSION:
                continue
            value["_line"] = line_number
            records.append(value)
    return records


def _stable_record_key(record: Mapping[str, Any]) -> str:
    quality = record.get("quality_checks")
    dedupe = quality.get("dedupe_key") if isinstance(quality, Mapping) else None
    return _sha256_json(
        {
            "task_id": record.get("task_id"),
            "boundary_type": record.get("boundary_type"),
            "dedupe_key": dedupe,
            "line": record.get("_line"),
        }
    )


def _is_grpo_source(record: Mapping[str, Any]) -> bool:
    provenance = record.get("source_provenance")
    return isinstance(provenance, list) and any(
        isinstance(item, Mapping) and item.get("source_kind") == "grpo_failed"
        for item in provenance
    )


def choose_probe_records(
    records: Sequence[Mapping[str, Any]],
    *,
    boundary_type: str,
    count: int,
    confused: bool = False,
) -> list[dict[str, Any]]:
    """Choose fixed non-formal-evaluation contexts by a stable hash."""

    eligible = [
        dict(value)
        for value in records
        if value.get("boundary_type") == boundary_type
        and value.get("project_split") != "evaluation"
        and (not confused or _is_grpo_source(value))
    ]
    # The extractor retains a boundary label for provenance, while the
    # public snapshot is the authoritative moment seen by the next Actor
    # call. Keep only snapshots that actually expose the requested phase.
    if boundary_type == "first_fallback":
        eligible = [
            value for value in eligible
            if isinstance(value.get("public_state_before"), Mapping)
            and int(value["public_state_before"].get("fallback_count", 0)) == 1
            and str(value["public_state_before"].get("recovery_mode", "")).upper()
            == "SEARCH_RETRY_REQUIRED"
        ]
    elif boundary_type == "valid_search_to_answer":
        # A valid search-result boundary must still expose an open aspect and
        # visible option IDs.  The extractor can retain later terminal
        # snapshots for the same task; those are not answer@1 contexts.
        valid: list[dict[str, Any]] = []
        for value in eligible:
            try:
                state = public_state_from_payload(
                    value.get("public_state_before", {}),
                    value.get("messages", []),
                    phase_hint=str(value.get("boundary_type", "")),
                )
            except Exception:
                continue
            if (
                state.phase is RecoveryMode.ANSWER_REQUIRED
                and state.current is not None
                and bool(state.current.visible_option_ids)
            ):
                valid.append(value)
        eligible = valid
    elif boundary_type == "second_fallback":
        eligible = [
            value for value in eligible
            if isinstance(value.get("public_state_before"), Mapping)
            and int(value["public_state_before"].get("fallback_count", 0)) >= 2
            and str(value["public_state_before"].get("recovery_mode", "")).upper()
            == "SWITCH_ASPECT_REQUIRED"
        ]
    if boundary_type in {"preference_complete_to_search", "second_fallback"}:
        # Do not spend probe budget on a context that is already terminal.
        # Such a record is still useful for quarantine/audit, but it cannot
        # measure search@1 or an aspect switch. The check uses only the public
        # snapshot and public messages.
        def _eligible_open_transition(value: Mapping[str, Any]) -> bool:
            try:
                state = public_state_from_payload(
                    value.get("public_state_before", {}),
                    value.get("messages", []),
                    phase_hint=str(value.get("boundary_type", "")),
                )
            except Exception:
                return False
            if boundary_type == "preference_complete_to_search":
                return state.current is not None and state.current.status is PublicAspectStatus.OPEN
            return any(item.status is PublicAspectStatus.OPEN for item in state.aspects)

        eligible = [value for value in eligible if _eligible_open_transition(value)]
    eligible.sort(key=_stable_record_key)
    # Prefer one context per task. If a phase has fewer than ``count`` unique
    # tasks (second-fallback is intentionally small), fill the remainder with
    # additional deterministic contexts and record their task IDs unchanged.
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in eligible:
        task_id = str(value.get("task_id", ""))
        if task_id in seen:
            continue
        seen.add(task_id)
        unique.append(value)
        if len(unique) == count:
            break
    if len(unique) < count:
        selected_keys = {_stable_record_key(value) for value in unique}
        unique.extend(
            value for value in eligible
            if _stable_record_key(value) not in selected_keys
        )
    if len(unique) < count:
        raise ValueError(
            f"only {len(unique)} eligible {boundary_type} contexts; need {count}"
        )
    return unique[:count]


def load_frozen_tasks(path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(path)
    expected = ("task_id", "composition", "difficulty", "source_split", "prompt")
    if tuple(table.column_names) != expected:
        raise ValueError(f"unexpected evaluation schema: {table.column_names}")
    rows = table.to_pylist()
    if len(rows) != 471:
        raise ValueError(f"expected frozen 471-row test set, found {len(rows)}")
    return rows


def choose_closed_loop_tasks(rows: Sequence[Mapping[str, Any]], count: int = 8) -> list[dict[str, Any]]:
    """Use the existing balanced, reproducible task selection rule."""

    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        composition = str(row["composition"])
        if composition in EXCLUDED_COMPOSITIONS:
            continue
        groups.setdefault(composition, []).append(row)
    preferred = [key for key in ("22", "33", "44") if key in groups]
    selected: list[Mapping[str, Any]] = []
    while len(selected) < count:
        progressed = False
        for composition in preferred:
            index = sum(str(item["composition"]) == composition for item in selected)
            if index < len(groups[composition]):
                selected.append(groups[composition][index])
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            raise ValueError("not enough eligible frozen evaluation tasks")
    return [dict(value) for value in selected]


def _initial_user_content(messages: Sequence[Mapping[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            return str(message["content"])
    raise ValueError("public context has no user message")


def _extract_actions(messages: Sequence[Mapping[str, Any]]) -> list[UserBenchAction]:
    actions: list[UserBenchAction] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls")
        if not isinstance(calls, list) or len(calls) != 1:
            continue
        call = calls[0]
        function = call.get("function") if isinstance(call, Mapping) else None
        if not isinstance(function, Mapping) or function.get("name") != "interact_with_env":
            continue
        raw = function.get("arguments")
        try:
            parameters = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(parameters, Mapping):
                actions.append(UserBenchAction.from_parameters(parameters))
        except (json.JSONDecodeError, UserBenchActionError, TypeError):
            continue
    return actions


def _append_control_note(messages: list[dict[str, Any]], note: str) -> list[dict[str, Any]]:
    if not messages:
        raise ValueError("cannot render control note on an empty prompt")
    last = messages[-1]
    if last.get("role") not in {"tool", "user"}:
        raise ValueError("public context must end in a tool or user message")
    content = str(last.get("content", ""))
    if note not in content:
        last["content"] = f"{content}\n\n{note}" if content else note
    return messages


def _public_messages(record: Mapping[str, Any], condition: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base = normalize_actor_messages(record.get("messages", []))
    if not base or base[0].get("role") != "system":
        raise ValueError("boundary context must begin with a system message")
    if condition == "A":
        return strip_actor_runtime_policy(base), dict(record.get("public_state_before", {}))
    state = public_state_from_payload(
        record.get("public_state_before", {}),
        base,
        phase_hint=str(record.get("boundary_type", "")),
    )
    # The runtime prepares a terminal aspect before the next generation. This
    # is essential for second-fallback contexts: B must see the next aspect,
    # never a blocked one.
    if state.phase is RecoveryMode.SWITCH_ASPECT_REQUIRED and state.current is not None:
        if state.current.status in {PublicAspectStatus.ANSWERED, PublicAspectStatus.BLOCKED}:
            state = advance_public_aspect(state)
    messages = ensure_actor_runtime_policy(base)
    _append_control_note(messages, render_actor_control_info(state))
    payload = {
        "current_aspect": state.current_aspect,
        "recovery_mode": state.phase.name,
        "fallback_count": state.current.search_fallbacks if state.current else 0,
        "visible_option_ids": sorted(state.current.visible_option_ids) if state.current else [],
        "preference_complete_aspects": list(
            item.aspect for item in state.aspects if item.preferences_complete
        ),
        "last_transition_aspect": state.last_transition_aspect,
        "last_transition_status": (
            state.last_transition_status.value.upper()
            if state.last_transition_status is not None
            else None
        ),
    }
    return messages, payload


def _answer_ids(content: str) -> list[str]:
    return [value.strip() for value in content.split(",") if value.strip()]


def _action_record(call: Any) -> dict[str, Any]:
    action = getattr(call, "action", None)
    if not isinstance(action, UserBenchAction):
        return {"protocol_valid": False, "error": "missing_action"}
    return {
        "protocol_valid": True,
        "choice": action.choice.value,
        "content": action.content,
        "thought": action.thought,
    }


def _classify_probe(
    category: str,
    record: Mapping[str, Any],
    action: UserBenchAction | None,
    *,
    state_payload: Mapping[str, Any],
    previous_actions: Sequence[UserBenchAction],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "protocol_valid": action is not None,
        "answer_at_1": False,
        "visible_id_only": False,
        "search_at_1": False,
        "clean_search_at_1": False,
        "search_targets_current_aspect": False,
        "changed_query": False,
        "exact_query_repeat": False,
        "switch_aspect": False,
        "same_aspect_search": False,
        "repeated_action": False,
        "repeated_search": False,
    }
    if action is None:
        return result
    previous_searches = [item for item in previous_actions if item.choice is ActionChoice.SEARCH]
    current = str(state_payload.get("current_aspect") or "")
    if category == "normal_search_result":
        result["answer_at_1"] = action.choice is ActionChoice.ANSWER
        ids = _answer_ids(action.content)
        visible = set(str(value) for value in state_payload.get("visible_option_ids", ()))
        result["visible_id_only"] = (
            len(ids) == 1 and OPTION_ID.fullmatch(ids[0]) is not None and ids[0] in visible
        )
    elif category == "preference_complete":
        result["search_at_1"] = action.choice is ActionChoice.SEARCH
        result["search_targets_current_aspect"] = (
            result["search_at_1"]
            and bool(current)
            and action_mentions_aspect(action.content, current)
        )
        # ``clean_search_at_1`` is the requested primary rate: exactly one
        # search tool call, with the stricter aspect-targeting sub-rate kept
        # separately for diagnosis.
        result["clean_search_at_1"] = result["search_at_1"]
    elif category == "first_fallback":
        if action.choice is ActionChoice.SEARCH and previous_searches:
            from travel_grpo.envs.public_control import is_substantive_query_change, normalize_public_query

            previous = previous_searches[-1].content
            result["changed_query"] = is_substantive_query_change(previous, action.content)
            candidate = normalize_public_query(action.content)
            result["exact_query_repeat"] = any(
                candidate == normalize_public_query(item.content) for item in previous_searches
            )
    elif category == "second_fallback":
        # For B, ``state_payload`` has already advanced to the next aspect.
        # The same-aspect metric must still compare with the failed aspect in
        # the pre-advance public snapshot.
        failed = str((record.get("public_state_before") or {}).get("current_aspect") or current)
        public_aspects: list[str] = []
        try:
            state = public_state_from_payload(
                record.get("public_state_before", {}),
                record.get("messages", []),
                phase_hint=str(record.get("boundary_type", "")),
            )
            public_aspects = list(state.public_aspects)
        except Exception:
            public_aspects = []
        next_aspect = None
        if failed in public_aspects:
            index = public_aspects.index(failed)
            next_aspect = next(
                (
                    item.aspect
                    for item in state.aspects[index + 1 :]
                    if item.status is PublicAspectStatus.OPEN
                ),
                None,
            )
        result["switch_aspect"] = bool(next_aspect) and action_mentions_aspect(action.content, next_aspect)
        result["same_aspect_search"] = (
            action.choice is ActionChoice.SEARCH and bool(failed) and action_mentions_aspect(action.content, failed)
        )
    elif category == "confused_history":
        signature = normalized_action_signature(action)
        result["repeated_action"] = signature in {normalized_action_signature(item) for item in previous_actions}
        result["repeated_search"] = (
            action.choice is ActionChoice.SEARCH
            and any(item.choice is ActionChoice.SEARCH and normalized_action_signature(item) == signature for item in previous_actions)
        )
    return result


async def run_one_step_probes(
    *,
    samples: Mapping[str, Sequence[Mapping[str, Any]]],
    actor: OpenAICompatibleActorClient,
    output: Path,
    conditions: Sequence[str] = ("A", "B"),
) -> dict[str, Any]:
    all_records: list[dict[str, Any]] = []
    requested = tuple(dict.fromkeys(str(value).upper() for value in conditions))
    for condition in requested:
        condition_dir = output / condition / "probes"
        condition_dir.mkdir(parents=True, exist_ok=True)
        for category, values in samples.items():
            for index, record in enumerate(values, start=1):
                messages, state_payload = _public_messages(record, condition)
                prompt_hash = _sha256_json(messages)
                previous_actions = _extract_actions(messages)
                entry: dict[str, Any] = {
                    "schema_version": SCHEMA_VERSION,
                    "condition": condition,
                    "category": category,
                    "task_id": record.get("task_id"),
                    "composition": record.get("composition"),
                    "project_split": record.get("project_split"),
                    "source": _source_info(record),
                    "prompt_hash": prompt_hash,
                    "actor_policy_version": ACTOR_RUNTIME_POLICY_VERSION if condition == "B" else "none",
                    "phase_guard_version": PHASE_GUARD_VERSION if condition == "B" else "none",
                }
                try:
                    call = await actor.generate_action(messages)
                    action = call.action
                    entry["action"] = _action_record(call)
                    entry["metrics"] = _classify_probe(
                        category, record, action,
                        state_payload=state_payload,
                        previous_actions=previous_actions,
                    )
                except (TeacherApiError, TeacherProtocolError, UserBenchActionError, ValueError) as exc:
                    entry["action"] = {"protocol_valid": False, "error": exc.__class__.__name__}
                    entry["metrics"] = _classify_probe(
                        category, record, None,
                        state_payload=state_payload,
                        previous_actions=previous_actions,
                    )
                atomic_json(condition_dir / f"{category}-{index:02d}.json", entry)
                all_records.append(entry)
                print(
                    f"condition={condition} category={category} sample={index}/{len(values)} "
                    f"choice={entry.get('action', {}).get('choice')} protocol={entry.get('action', {}).get('protocol_valid')}",
                    flush=True,
                )
    summaries: dict[str, Any] = {}
    for condition in requested:
        rows = [item for item in all_records if item["condition"] == condition]
        summary: dict[str, Any] = {"samples": len(rows), "categories": {}}
        for category in samples:
            values = [item for item in rows if item["category"] == category]
            summary["categories"][category] = _mean_flags(values)
        atomic_json(output / condition / "probes-summary.json", summary)
        summaries[condition] = summary
    atomic_json(output / "probe-records.json", all_records)
    return summaries


def _mean_flags(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = (
        "answer_at_1", "visible_id_only", "search_at_1", "clean_search_at_1",
        "search_targets_current_aspect", "changed_query", "exact_query_repeat",
        "switch_aspect", "same_aspect_search", "repeated_action", "repeated_search",
    )
    return {
        "n": len(rows),
        **{
            key: sum(bool((row.get("metrics") or {}).get(key)) for row in rows) / len(rows)
            if rows else None
            for key in keys
        },
        "protocol_valid_rate": sum(bool((row.get("action") or {}).get("protocol_valid")) for row in rows) / len(rows)
        if rows else None,
    }


def _public_reward_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only closed-loop audit fields; never copy hidden correctness labels."""

    return {
        key: report.get(key)
        for key in (
            "reward_valid", "completion_rate", "environment_steps",
            "actor_attempts", "effective_steps", "accepted_actor_attempts",
            "correct_answer_rate", "answer_submission_rate", "answer_quality",
            "preference_coverage", "phase_transition_score",
            "phase_transition_breakdown", "guard_rejections",
            "guard_rejection_rate", "blocked_aspects", "reward_degraded",
            "invalid_actions", "exact_repeats",
            "semantic_repeats", "infrastructure_invalid", "infrastructure_errors",
            "simulator_fallback_counts", "termination_reason",
        )
        if key in report
    }


def _sanitize_transcript(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return normalize_actor_messages(messages)


async def guarded_rollout_task(
    task: Mapping[str, Any],
    *,
    actor: OpenAICompatibleActorClient,
    simulator: UserSimulatorRuntime,
    source_root: Path,
) -> dict[str, Any]:
    """Run B with public pre-environment rejection and no hidden phase input."""

    task_id = str(task["task_id"])
    prompt = normalize_actor_messages(task["prompt"])
    initial = _initial_user_content(prompt)
    messages = ensure_actor_runtime_policy(prompt)
    wrapper = UserBenchWrapper(task_id, simulator, UserBenchEnvironmentConfig(), source_root=source_root)
    session: UserBenchSessionState | None = None
    guard_rejections = 0
    guard_rejection_reasons: Counter[str] = Counter()
    try:
        await wrapper.areset()
        session = UserBenchSessionState(
            request_id=f"gate-{uuid.uuid4().hex}", task_id=task_id, wrapper=wrapper,
            reward_task=wrapper.reward_task(), reward_snapshot=wrapper.reward_snapshot(),
            public_initial_message=initial,
        )
        for _ in range(20):
            if session.done:
                break
            before_phase = session.public_control_state.phase if session.public_control_state else None
            session.prepare_public_action()
            after_phase = session.public_control_state.phase if session.public_control_state else None
            if before_phase != after_phase and session.public_control_state is not None:
                messages.append({"role": "user", "content": session.render_actor_feedback("")})
            session.actor_attempts += 1
            try:
                call = await actor.generate_action(messages)
                action = UserBenchAction.from_parameters(call.parameters)
            except (TeacherApiError, TeacherProtocolError, UserBenchActionError, ValueError) as exc:
                session.invalid_actions += 1
                session.protocol_error = "invalid_actor_tool_call"
                messages.append({"role": "user", "content": f"Error: invalid tool call ({exc.__class__.__name__}). Emit one valid interact_with_env call."})
                continue
            messages.append(call.to_assistant_message())
            reason = session.validate_public_action(action)
            if reason is not None:
                guard_rejections += 1
                guard_rejection_reasons[reason] += 1
                session.invalid_actions += 1
                session.record_public_guard_rejection(reason)
                session.record_public_non_progress(reason)
                messages.append({"role": "user", "content": session.render_actor_feedback(f"Error: public control rejected this call: {reason}")})
                continue
            try:
                result = await wrapper.astep(action)
                snapshot = wrapper.reward_snapshot()
            except Exception as exc:
                session.infrastructure_errors.append(f"simulator_{exc.__class__.__name__}")
                session.termination_reason = "simulator_infrastructure_failure"
                break
            session.record_step(result, action, snapshot)
            messages.append({
                "role": "tool", "tool_call_id": call.call_id, "name": "interact_with_env",
                "content": session.render_actor_feedback(result.observation.feedback),
            })
            if session.done:
                break
        if not session.done and session.termination_reason is None:
            session.termination_reason = "actor_turn_limit"
        report = session.reward_report()
        return {
            "schema_version": SCHEMA_VERSION,
            "actor_policy_version": ACTOR_RUNTIME_POLICY_VERSION,
            "phase_guard_version": PHASE_GUARD_VERSION,
            "task_id": task_id,
            "composition": str(task["composition"]),
            "guard_rejections": guard_rejections,
            "guard_rejection_reasons": dict(guard_rejection_reasons),
            "actor_attempts": session.actor_attempts,
            "environment_steps": session.num_tool_calls,
            "termination_reason": session.termination_reason,
            "reward": _public_reward_summary(report),
            "visible_transcript": _sanitize_transcript(messages),
        }
    finally:
        wrapper.close()


def _old_result_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    report = result.get("reward") if isinstance(result.get("reward"), Mapping) else {}
    return {
        "schema_version": SCHEMA_VERSION,
        "actor_policy_version": "none",
        "phase_guard_version": "none",
        "task_id": result.get("task_id"),
        "composition": result.get("composition"),
        "guard_rejections": 0,
        "guard_rejection_reasons": {},
        "actor_attempts": result.get("actor_attempts", 0),
        "environment_steps": result.get("environment_steps", 0),
        "termination_reason": result.get("termination_reason"),
        "reward": _public_reward_summary(report),
        "visible_transcript": _sanitize_transcript(result.get("visible_transcript", [])),
    }


async def run_closed_loop(
    *,
    tasks: Sequence[Mapping[str, Any]],
    actor: OpenAICompatibleActorClient,
    simulator: UserSimulatorRuntime,
    output: Path,
    conditions: Sequence[str] = ("A", "B"),
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    requested = tuple(dict.fromkeys(str(value).upper() for value in conditions))
    for condition in requested:
        condition_dir = output / condition / "closed_loop"
        condition_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        for index, task in enumerate(tasks, start=1):
            if condition == "A":
                raw = await rollout_task(
                    task, actor=actor, simulator=simulator,
                    source_root=ROOT / "environments/UserBench",
                    apply_actor_policy=False,
                )
                result = _old_result_summary(raw)
            else:
                result = await guarded_rollout_task(
                    task, actor=actor, simulator=simulator,
                    source_root=ROOT / "environments/UserBench",
                )
            result["prompt_hash"] = _sha256_json(task["prompt"])
            atomic_json(condition_dir / f"{index:02d}.json", result)
            rows.append(result)
            print(
                f"condition={condition} closed_loop={index}/{len(tasks)} id={task['task_id']} "
                f"steps={result.get('environment_steps')} termination={result.get('termination_reason')}",
                flush=True,
            )
        summary = summarize_closed_loop(rows)
        atomic_json(condition_dir / "summary.json", summary)
        summaries[condition] = summary
    return summaries


def _transcript_choice_counts(transcript: Any) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not isinstance(transcript, list):
        return counts
    for message in transcript:
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, Mapping):
                continue
            function = call.get("function")
            if not isinstance(function, Mapping):
                continue
            raw = function.get("arguments")
            try:
                value = json.loads(raw) if isinstance(raw, str) else raw
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(value, Mapping) and isinstance(value.get("choice"), str):
                counts[value["choice"]] += 1
    return counts


def _guard_reason_count(
    rows: Sequence[Mapping[str, Any]],
    predicate: Any,
) -> tuple[int, int]:
    total = 0
    tasks = 0
    for row in rows:
        reasons = row.get("guard_rejection_reasons")
        if not isinstance(reasons, Mapping):
            continue
        matched = sum(
            int(value)
            for key, value in reasons.items()
            if isinstance(value, int) and not isinstance(value, bool) and predicate(str(key))
        )
        total += matched
        tasks += bool(matched)
    return total, tasks


def summarize_closed_loop(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rewards = [row.get("reward", {}) for row in rows]
    completion = [float(value.get("completion_rate", 0.0)) for value in rewards if isinstance(value, Mapping)]
    exact_repeats = [int(value.get("exact_repeats", 0)) for value in rewards if isinstance(value, Mapping)]
    semantic_repeats = [int(value.get("semantic_repeats", 0)) for value in rewards if isinstance(value, Mapping)]
    choice_counts = [_transcript_choice_counts(row.get("visible_transcript")) for row in rows]
    answer_calls = [counts.get("answer", 0) for counts in choice_counts]
    wrong_aspect_rejections, wrong_aspect_tasks = _guard_reason_count(
        rows,
        lambda reason: (
            "must target the current public aspect" in reason
            or "different public aspect" in reason
        ),
    )
    terminal_aspect_rejections, terminal_aspect_tasks = _guard_reason_count(
        rows,
        lambda reason: "public aspect" in reason and "terminal" in reason,
    )
    repeated_query_rejections, repeated_query_tasks = _guard_reason_count(
        rows,
        lambda reason: (
            "search query was already attempted" in reason
            or "retry query must materially change" in reason
        ),
    )
    termination_reasons = Counter(str(row.get("termination_reason")) for row in rows)
    reward_valid = [
        bool(value.get("reward_valid", False))
        for value in rewards
        if isinstance(value, Mapping)
    ]
    return {
        "tasks": len(rows),
        "completion": sum(value > 0 for value in completion) / len(rows) if rows else None,
        "tasks_with_nonzero_completion": sum(value > 0 for value in completion),
        "mean_completion_rate": sum(completion) / len(completion) if completion else 0.0,
        "reward_valid_rate": sum(reward_valid) / len(reward_valid) if reward_valid else None,
        "reward_valid_tasks": sum(reward_valid),
        "reward_degraded": sum(bool(value.get("reward_degraded", False)) for value in rewards if isinstance(value, Mapping)) / len(rows) if rows else None,
        "max_steps": termination_reasons.get("max_steps", 0) / len(rows) if rows else None,
        "actor_turn_limit": termination_reasons.get("actor_turn_limit", 0) / len(rows) if rows else None,
        "public_control_complete": termination_reasons.get("public_control_complete", 0) / len(rows) if rows else None,
        "mean_environment_steps": sum(int(row.get("environment_steps", 0)) for row in rows) / len(rows) if rows else 0.0,
        "answer_calls_total": sum(answer_calls),
        "tasks_with_answer_call": sum(value > 0 for value in answer_calls),
        "answer_call_rate": sum(value > 0 for value in answer_calls) / len(rows) if rows else None,
        "repeated_action_or_search": sum(
            exact + semantic > 0 for exact, semantic in zip(exact_repeats, semantic_repeats)
        ) / len(rows) if rows else None,
        "exact_repeats_total": sum(exact_repeats),
        "semantic_repeats_total": sum(semantic_repeats),
        "exact_repeats_mean": sum(exact_repeats) / len(exact_repeats) if exact_repeats else 0.0,
        "semantic_repeats_mean": sum(semantic_repeats) / len(semantic_repeats) if semantic_repeats else 0.0,
        "repeated_query_rejections_total": repeated_query_rejections,
        "tasks_with_repeated_query_rejection": repeated_query_tasks,
        "wrong_aspect_rejections_total": wrong_aspect_rejections,
        "tasks_with_wrong_aspect_rejection": wrong_aspect_tasks,
        "terminal_aspect_rejections_total": terminal_aspect_rejections,
        "tasks_with_terminal_aspect_rejection": terminal_aspect_tasks,
        "termination_reasons": dict(sorted(termination_reasons.items())),
        "guard_rejections": sum(int(row.get("guard_rejections", 0)) for row in rows),
        "guard_rejections_per_task": sum(int(row.get("guard_rejections", 0)) for row in rows) / len(rows) if rows else 0.0,
        "failure_samples": [
            {
                "task_id": row.get("task_id"),
                "termination_reason": row.get("termination_reason"),
                "completion_rate": (row.get("reward") or {}).get("completion_rate"),
                "environment_steps": row.get("environment_steps"),
                "guard_rejections": row.get("guard_rejections", 0),
            }
            for row in rows
            if float((row.get("reward") or {}).get("completion_rate", 0.0)) < 1.0
            or row.get("termination_reason") not in {"environment_terminated", "public_control_complete"}
        ],
    }


def build_comparison_report(output: Path) -> dict[str, Any]:
    """Assemble a compact A/B report from immutable per-sample summaries."""

    def load(relative: str) -> dict[str, Any] | None:
        path = output / relative
        if not path.is_file():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "actor_policy_version": ACTOR_RUNTIME_POLICY_VERSION,
        "phase_guard_version": PHASE_GUARD_VERSION,
        "model_path": "outputs/models/sft-merged",
        "parameter_updates": False,
        "grpo": False,
        "single_step": {
            "A": (load("A/probes-summary.json") or {}).get("categories", {}),
            "B": (load("B/probes-summary.json") or {}).get("categories", {}),
        },
        "closed_loop": {
            "A": load("A/closed_loop/summary.json"),
            "B": load("B/closed_loop/summary.json"),
        },
    }
    report["failure_samples"] = {
        condition: (report["closed_loop"].get(condition) or {}).get("failure_samples", [])
        for condition in ("A", "B")
    }
    return report

def build_manifest(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    records = load_boundary_records(args.boundary_file)
    samples: dict[str, list[dict[str, Any]]] = {}
    for category, boundary in BOUNDARY_TYPES.items():
        samples[category] = choose_probe_records(
            records, boundary_type=boundary, count=FIXED_COUNTS[category], confused=category == "confused_history"
        )
    task_count = int(getattr(args, "closed_loop_task_count", 8))
    if task_count < 1:
        raise ValueError("closed_loop_task_count must be >= 1")
    tasks = choose_closed_loop_tasks(load_frozen_tasks(args.dataset), count=task_count)
    index: list[dict[str, Any]] = []
    for category, values in samples.items():
        for value in values:
            index.append({
                "kind": "probe", "category": category, "task_id": value.get("task_id"),
                "composition": value.get("composition"), "project_split": value.get("project_split"),
                "source": _source_info(value), "source_line": value.get("_line"),
            })
    for value in tasks:
        index.append({
            "kind": "closed_loop", "task_id": value.get("task_id"),
            "composition": str(value.get("composition")), "source_split": value.get("source_split"),
        })
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "boundary_schema_version": BOUNDARY_SCHEMA_VERSION,
        "model_path": str(Path(args.model)),
        "model_merge_manifest": _relative(Path(args.model) / "merge_manifest.json"),
        "model_merge_manifest_sha256": _sha256_file(Path(args.model) / "merge_manifest.json"),
        "dataset": _relative(args.dataset),
        "boundary_file": _relative(args.boundary_file),
        "actor_policy_version": ACTOR_RUNTIME_POLICY_VERSION,
        "actor_policy_hash": _sha256_bytes(ACTOR_RUNTIME_POLICY.encode()),
        "phase_guard_version": PHASE_GUARD_VERSION,
        "inference_config": {
            "temperature": 0.0, "max_tokens": int(args.max_tokens), "max_steps": 20,
            "one_step_probe": True, "parameter_updates": False, "grpo": False,
        },
        "probe_counts": FIXED_COUNTS,
        "conditions_run": list(_requested_conditions(args)),
        "baseline_output_path": (
            _relative(Path(args.baseline_output))
            if getattr(args, "baseline_output", None) is not None
            else None
        ),
        "closed_loop_task_count": len(tasks),
        "closed_loop_task_ids": [value["task_id"] for value in tasks],
        "probes_executed": not bool(getattr(args, "closed_loop_only", False)),
        "closed_loop_executed": not bool(getattr(args, "probes_only", False)),
        "closed_loop_compositions": [str(value["composition"]) for value in tasks],
        "excluded_compositions": list(EXCLUDED_COMPOSITIONS),
        "probe_task_ids": {key: [value["task_id"] for value in values] for key, values in samples.items()},
        "output_path": _relative(args.output),
        "actor_endpoint_env": "ACTOR_BASE_URL",
        "simulator_endpoint_env": "EVAL_USER_SIM_BASE_URL",
        "actor_endpoint_not_recorded": True,
        "api_keys_not_recorded": True,
        "sample_index_path": _relative(args.output / "sample-index.jsonl"),
    }
    return manifest, samples, tasks


def _requested_conditions(args: argparse.Namespace) -> tuple[str, ...]:
    values = getattr(args, "conditions", None)
    if values is None:
        values = ("A", "B")
    normalized = tuple(dict.fromkeys(str(value).upper() for value in values))
    if not normalized or any(value not in {"A", "B"} for value in normalized):
        raise ValueError(f"conditions must be a non-empty subset of A/B, got {values!r}")
    return normalized


def preserve_baseline_condition(args: argparse.Namespace, conditions: Sequence[str]) -> None:
    """Copy only the prior A artifacts into a B-only rerun directory.

    The copy is intentionally limited to condition ``A``. It makes the new
    comparison self-contained while ensuring no A inference request is made
    during a B-only rerun.
    """

    if "A" in conditions:
        return
    baseline = getattr(args, "baseline_output", None)
    if baseline is None:
        return
    source = Path(baseline) / "A"
    destination = Path(args.output) / "A"
    if not source.is_dir():
        raise FileNotFoundError(f"baseline A artifacts not found: {source}")
    if destination.exists():
        source_files = sorted(path.relative_to(source) for path in source.rglob("*") if path.is_file())
        destination_files = sorted(path.relative_to(destination) for path in destination.rglob("*") if path.is_file())
        if source_files == destination_files and all(
            _sha256_file(source / relative) == _sha256_file(destination / relative)
            for relative in source_files
        ):
            return
        raise FileExistsError(
            f"refusing to overwrite non-identical preserved A artifacts: {destination}"
        )
    shutil.copytree(source, destination)


async def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    conditions = _requested_conditions(args)
    preserve_baseline_condition(args, conditions)
    manifest, samples, tasks = build_manifest(args)
    atomic_json(args.output / "manifest.json", manifest)
    with (args.output / "sample-index.jsonl").open("w", encoding="utf-8") as handle:
        for category, values in samples.items():
            for value in values:
                handle.write(json.dumps({
                    "kind": "probe", "category": category, "task_id": value.get("task_id"),
                    "line": value.get("_line"), "source": _source_info(value),
                }, ensure_ascii=False, sort_keys=True) + "\n")
        for value in tasks:
            handle.write(json.dumps({
                "kind": "closed_loop", "task_id": value.get("task_id"),
                "composition": value.get("composition"),
            }, ensure_ascii=False, sort_keys=True) + "\n")
    if args.dry_run:
        print(json.dumps({"manifest": str(args.output / "manifest.json"), "dry_run": True}, ensure_ascii=False))
        return
    actor_runtime = ActorRuntime.from_environment()
    actor_runtime.require_model(args.model)
    actor = OpenAICompatibleActorClient(actor_runtime)
    try:
        if not args.closed_loop_only:
            await run_one_step_probes(
                samples=samples, actor=actor, output=args.output, conditions=conditions
            )
        if not args.probes_only:
            simulator = UserSimulatorRuntime.from_environment(SimulatorRole.EVAL)
            await run_closed_loop(
                tasks=tasks, actor=actor, simulator=simulator,
                output=args.output, conditions=conditions,
            )
        atomic_json(args.output / "comparison.json", build_comparison_report(args.output))
    finally:
        await actor.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/evaluation/tasks.parquet")
    parser.add_argument("--boundary-file", type=Path, default=BOUNDARY_FILE)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/evaluation/inference_gate_sft_merged")
    parser.add_argument("--model", default="outputs/models/sft-merged")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--probes-only", action="store_true")
    parser.add_argument("--closed-loop-only", action="store_true")
    parser.add_argument(
        "--closed-loop-task-count", type=int, default=8,
        help="number of deterministic closed-loop tasks (8 for the gate; 32 for validation)",
    )
    parser.add_argument(
        "--conditions", nargs="+", choices=("A", "B"), default=("A", "B"),
        help="conditions to execute; use --conditions B with --baseline-output for a B-only rerun",
    )
    parser.add_argument(
        "--baseline-output", type=Path,
        help="prior gate output whose A artifacts are copied into a B-only rerun",
    )
    args = parser.parse_args()
    asyncio.run(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
