"""Render and audit one-step recovery records for action-only SFT.

The renderer keeps the boundary history public, injects the same production
Actor policy used by GRPO/evaluation, appends the public control note emitted
by the runtime control renderer, and appends one unmasked target call. All
historical assistant calls are retained with ``loss_mask=True`` so the
existing action-only SFT renderer remains the single source of loss masking.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from travel_grpo.data.recovery.boundaries import HIDDEN_FIELD_NAMES
from travel_grpo.protocols.actor_messages import normalize_actor_messages
from travel_grpo.envs.public_control import (
    PublicAspectState,
    PublicAspectStatus,
    PublicControlState,
    PublicObservationKind,
    RecoveryMode,
    extract_public_aspects,
    mark_public_preference_complete,
    render_actor_control_info,
)
from travel_grpo.envs.userbench_tools import (
    ActionChoice,
    OPTION_ID,
    UserBenchAction,
    UserBenchActionError,
)
from travel_grpo.prompts.actor_policy import (
    ACTOR_RUNTIME_POLICY,
    ACTOR_RUNTIME_POLICY_MARKER,
    ACTOR_RUNTIME_POLICY_VERSION,
    LEGACY_TEACHER_ACTOR_POLICY,
    TEACHER_GENERATION_INSTRUCTION,
    TEACHER_GENERATION_INSTRUCTION_MARKER,
    ensure_actor_runtime_policy,
)
from travel_grpo.training.sft.dataset import (
    ActionOnlyExample,
    SFTDatasetError,
    SFTTrajectoryTooLongError,
    build_action_only_examples,
    load_tool_schema,
    recovery_admission_reasons,
)

RECOVERY_SFT_SCHEMA_VERSION = "recovery-sft-v1"
RECOVERY_TARGET_SCHEMA_VERSION = "recovery-target-v1"
RECOVERY_SFT_GENERATOR_VERSION = "recovery-sft-renderer-v1"
TRAINING_SPLITS = frozenset(("sft_train", "grpo_train"))
VALIDATION_SPLITS = frozenset(("sft_validation", "grpo_validation"))


class RecoverySFTError(ValueError):
    """Raised when a recovery record cannot be rendered safely."""


@dataclass(frozen=True)
class RecoveryAuditResult:
    """One rendered record's audit result."""

    valid: bool
    reasons: tuple[str, ...]
    metrics: dict[str, Any]
    sample_hash: str | None = None


class CPUChatTemplateTokenizer:
    """Small CPU-only adapter around ``tokenizers`` and the saved Jinja file.

    Importing Transformers can load optional model dependencies and exceed the
    project's constrained worker memory. This adapter renders the same saved
    Qwen chat template and only loads ``tokenizer.json``.
    """

    padding_side = "right"

    # [项目注释] 功能：`__init__`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。
    # [项目注释] 输入：`tokenizer`: Any；`template`: Any；`pad_token_id`: int | None；`eos_token_id`: int | None。
    # [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
    def __init__(self, tokenizer: Any, template: Any, *, pad_token_id: int | None, eos_token_id: int | None):
        self._tokenizer = tokenizer
        self._template = template
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

    # [项目注释] 功能：`apply_chat_template`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：render, list, RecoverySFTError,
    # [项目注释]    encode。
    # [项目注释] 输入：`conversation`: Sequence[Mapping[str, Any]]；`tools`: Sequence[Mapping[str, Any]]；`tokenize`:
    # [项目注释]    bool；`add_generation_prompt`: bool；`enable_thinking`: bool；**`_`。
    # [项目注释] 输出：标注返回 `list[int]`；具体值由各分支决定。
    def apply_chat_template(
        self,
        conversation: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]],
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
        **_: Any,
    ) -> list[int]:
        if not tokenize:
            raise RecoverySFTError("CPU chat template adapter requires tokenize=True")
        rendered = self._template.render(
            messages=conversation,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
            add_vision_id=False,
        )
        return list(self._tokenizer.encode(rendered, add_special_tokens=False).ids)

    @property
    # [项目注释] 功能：`vocab_size`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：int, get_vocab_size。
    # [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
    # [项目注释] 输出：标注返回 `int`；具体值由各分支决定。
    def vocab_size(self) -> int:
        return int(self._tokenizer.get_vocab_size())


def load_cpu_chat_template_tokenizer(path: str | Path) -> CPUChatTemplateTokenizer:
    """Load a local tokenizer and Qwen chat template without model weights."""

    try:
        from jinja2 import Environment
        from tokenizers import Tokenizer
    except ImportError as exc:  # pragma: no cover - dependency is project-local.
        raise RecoverySFTError("jinja2 and tokenizers are required for CPU rendering") from exc
    source = Path(path)
    directory = source if source.is_dir() else source.parent
    tokenizer_path = source / "tokenizer.json" if source.is_dir() else source
    template_path = directory / "chat_template.jinja"
    config_path = directory / "tokenizer_config.json"
    if not tokenizer_path.is_file():
        raise RecoverySFTError(f"missing tokenizer.json: {tokenizer_path}")
    if not template_path.is_file():
        raise RecoverySFTError(f"missing chat_template.jinja: {template_path}")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    environment = Environment(trim_blocks=False, lstrip_blocks=False)
    template = environment.from_string(template_path.read_text(encoding="utf-8"))
    config: Mapping[str, Any] = {}
    if config_path.is_file():
        try:
            value = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(value, Mapping):
                config = value
        except json.JSONDecodeError:
            config = {}
    vocab = tokenizer.get_vocab()
    pad_token = config.get("pad_token")
    eos_token = config.get("eos_token")
    pad_id = vocab.get(pad_token) if isinstance(pad_token, str) else None
    eos_id = vocab.get(eos_token) if isinstance(eos_token, str) else None
    return CPUChatTemplateTokenizer(tokenizer, template, pad_token_id=pad_id, eos_token_id=eos_id)


# [项目注释] 功能：`_jsonl_records`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：open, enumerate, RecoverySFTError,
# [项目注释]    strip。
# [项目注释] 输入：`path`: Path。
# [项目注释] 输出：标注返回 `Iterable[tuple[int, Mapping[str, Any] | None, str | None]]`；具体值由各分支决定。
def _jsonl_records(path: Path) -> Iterable[tuple[int, Mapping[str, Any] | None, str | None]]:
    try:
        handle = path.open(encoding="utf-8")
    except OSError as exc:
        raise RecoverySFTError(f"cannot read target file: {path}") from exc
    with handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                yield line_number, None, "invalid_json"
                continue
            if not isinstance(value, Mapping):
                yield line_number, None, "record_not_mapping"
                continue
            yield line_number, value, None


# [项目注释] 功能：`_sha256`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：sha256, hexdigest, open, iter。
# [项目注释] 输入：`path`: Path。
# [项目注释] 输出：标注返回 `str`；具体值由各分支决定。
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# [项目注释] 功能：`_initial_user_message`：把协议/状态数据转换为模型、用户或日志可见的文本表示。 主要协作调用：RecoverySFTError, isinstance, str。
# [项目注释] 输入：`messages`: Sequence[Mapping[str, Any]]。
# [项目注释] 输出：标注返回 `str`；具体值由各分支决定。
def _initial_user_message(messages: Sequence[Mapping[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            return str(message["content"])
    raise RecoverySFTError("recovery history has no public user message")


# [项目注释] 功能：`_coerce_recovery_mode`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：isinstance, casefold,
# [项目注释]    RecoverySFTError, strip。
# [项目注释] 输入：`value`: Any。
# [项目注释] 输出：标注返回 `RecoveryMode`；具体值由各分支决定。
def _coerce_recovery_mode(value: Any) -> RecoveryMode:
    if isinstance(value, RecoveryMode):
        return value
    raw = str(value or "none").strip().casefold()
    aliases = {
        "none": RecoveryMode.NONE,
        "eliciting": RecoveryMode.NONE,
        "search_required": RecoveryMode.SEARCH_REQUIRED,
        "search_retry_required": RecoveryMode.SEARCH_RETRY_REQUIRED,
        "answer_required": RecoveryMode.ANSWER_REQUIRED,
        "switch_aspect_required": RecoveryMode.SWITCH_ASPECT_REQUIRED,
        "blocked": RecoveryMode.SWITCH_ASPECT_REQUIRED,
        "answered": RecoveryMode.ANSWERED,
    }
    if raw not in aliases:
        raise RecoverySFTError(f"unknown public recovery mode: {value!r}")
    return aliases[raw]


def public_state_from_payload(
    payload: Mapping[str, Any],
    messages: Sequence[Mapping[str, Any]],
    *,
    phase_hint: str | None = None,
) -> PublicControlState:
    """Rebuild public renderer state from a boundary snapshot only.

    ``phase_hint`` is an offline, public-boundary label. All known boundary
    labels are accepted as no-op compatibility hints; only
    ``preference_complete_to_search`` changes the reconstructed phase. It is
    not a hidden reward signal and is never read from a reward/task ledger.
    """

    public_aspects = extract_public_aspects(_initial_user_message(messages))
    current = payload.get("current_aspect")
    if current is not None and current not in public_aspects:
        raise RecoverySFTError("current aspect is not named in the public user message")
    answered = set(payload.get("answered_aspects", ()))
    blocked = set(payload.get("blocked_aspects", ()))
    complete = set(payload.get("preference_complete_aspects", ()))
    if not answered <= set(public_aspects) or not blocked <= set(public_aspects):
        raise RecoverySFTError("public state mentions an aspect absent from public history")
    if not complete <= set(public_aspects):
        raise RecoverySFTError("preference completion mentions an absent public aspect")
    hint = str(phase_hint or "").strip().casefold()
    allowed_hints = {
        "preference_complete_to_search",
        "valid_search_to_answer",
        "first_fallback",
        "second_fallback",
        "repeated_no_progress_action",
        "explicit_no_preference",
        "visible_options_pending_answer",
    }
    if hint and hint not in allowed_hints:
        raise RecoverySFTError(f"unsupported public phase hint: {phase_hint!r}")
    if hint == "preference_complete_to_search" and current is not None:
        complete.add(str(current))
    visible = payload.get("visible_option_ids", ())
    if not isinstance(visible, (list, tuple, set)):
        raise RecoverySFTError("visible_option_ids must be a sequence")
    visible_ids = frozenset(str(value) for value in visible)
    if any(OPTION_ID.fullmatch(value) is None for value in visible_ids):
        raise RecoverySFTError("public state contains an invalid visible option ID")
    try:
        fallback_count = int(payload.get("fallback_count", 0))
        search_attempts = int(payload.get("search_attempts", 0))
        consecutive = int(payload.get("consecutive_no_progress", 0))
    except (TypeError, ValueError) as exc:
        raise RecoverySFTError("public state counters must be integers") from exc
    if min(fallback_count, search_attempts, consecutive) < 0:
        raise RecoverySFTError("public state counters must be non-negative")
    transition_aspect = payload.get("last_transition_aspect")
    if transition_aspect is not None and transition_aspect not in public_aspects:
        raise RecoverySFTError("last transition mentions an absent public aspect")
    transition_raw = payload.get("last_transition_status")
    transition_status = None
    if transition_raw not in (None, ""):
        try:
            transition_status = PublicAspectStatus(str(transition_raw).strip().casefold())
        except ValueError as exc:
            raise RecoverySFTError("invalid public transition status") from exc
    aspects = tuple(
        PublicAspectState(
            aspect,
            status=(
                PublicAspectStatus.ANSWERED
                if aspect in answered
                else PublicAspectStatus.BLOCKED
                if aspect in blocked
                else PublicAspectStatus.OPEN
            ),
            visible_option_ids=visible_ids if aspect == current else frozenset(),
            normal_search_seen=bool(payload.get("normal_search_seen", False)) if aspect == current else False,
            search_attempts=search_attempts if aspect == current else 0,
            search_fallbacks=fallback_count if aspect == current else 0,
            preferences_complete=aspect in complete,
        )
        for aspect in public_aspects
    )
    terminal = bool(aspects) and all(item.status is not PublicAspectStatus.OPEN for item in aspects)
    state = PublicControlState(
        aspects=aspects,
        current_aspect=current,
        recovery_mode=_coerce_recovery_mode(payload.get("recovery_mode")),
        consecutive_no_progress=consecutive,
        episode_done=terminal,
        last_transition_aspect=transition_aspect,
        last_transition_status=transition_status,
    )
    if hint == "preference_complete_to_search" and current is not None:
        state = mark_public_preference_complete(state, str(current))
    return state


# [项目注释] 功能：`_append_public_control_note`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：str, RecoverySFTError。
# [项目注释] 输入：`messages`: list[dict[str, Any]]；`note`: str。
# [项目注释] 输出：标注返回 `str`；具体值由各分支决定。
def _append_public_control_note(
    messages: list[dict[str, Any]], note: str
) -> str:
    if not messages:
        raise RecoverySFTError("cannot append control note to empty history")
    last = messages[-1]
    role = last.get("role")
    if role not in {"tool", "user"}:
        raise RecoverySFTError("recovery history must end in a public user/tool message")
    content = str(last.get("content", ""))
    if note not in content:
        last["content"] = f"{content}\n\n{note}" if content else note
    return str(role)


# [项目注释] 功能：`_existing_call_ids`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：set, isinstance, add, str。
# [项目注释] 输入：`messages`: Sequence[Mapping[str, Any]]。
# [项目注释] 输出：标注返回 `set[str]`；具体值由各分支决定。
def _existing_call_ids(messages: Sequence[Mapping[str, Any]]) -> set[str]:
    result: set[str] = set()
    for message in messages:
        if message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls")
        if isinstance(calls, list):
            for call in calls:
                if isinstance(call, Mapping) and isinstance(call.get("id"), str):
                    result.add(str(call["id"]))
    return result


# [项目注释] 功能：`_normalise_target`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：normalize_actor_messages, str, pop,
# [项目注释]    RecoverySFTError。
# [项目注释] 输入：`target`: Mapping[str, Any]；`existing_ids`: set[str]；`task_id`: str；`boundary`: str。
# [项目注释] 输出：标注返回 `dict[str, Any]`；具体值由各分支决定。
def _normalise_target(target: Mapping[str, Any], existing_ids: set[str], task_id: str, boundary: str) -> dict[str, Any]:
    values = normalize_actor_messages([target])
    if len(values) != 1 or values[0].get("role") != "assistant":
        raise RecoverySFTError("target_assistant must be one assistant message")
    result = values[0]
    if result.get("content") not in (None, ""):
        raise RecoverySFTError("target assistant content must be empty")
    calls = result.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        raise RecoverySFTError("target assistant must contain one tool call")
    call = calls[0]
    call_id = str(call.get("id", "recovery-target"))
    if call_id in existing_ids:
        digest = hashlib.sha256(f"{task_id}|{boundary}".encode()).hexdigest()[:12]
        call_id = f"recovery-target-{digest}"
    call["id"] = call_id
    result.pop("loss_mask", None)
    return result


def render_recovery_record(target_record: Mapping[str, Any]) -> dict[str, Any]:
    """Render one accepted recovery-target record as recovery-sft-v1."""

    if target_record.get("schema_version") != RECOVERY_TARGET_SCHEMA_VERSION:
        raise RecoverySFTError("input is not recovery-target-v1")
    if target_record.get("target_status") != "accepted":
        raise RecoverySFTError("only accepted recovery targets can be rendered")
    raw_messages = normalize_actor_messages(target_record.get("messages", []))
    if not raw_messages or raw_messages[0].get("role") != "system":
        raise RecoverySFTError("recovery history must begin with a system message")
    state = public_state_from_payload(
        target_record.get("public_state_before", {}),
        raw_messages,
        phase_hint=str(target_record.get("boundary_type", "")),
    )
    note = render_actor_control_info(state)
    messages = ensure_actor_runtime_policy(raw_messages)
    placement = _append_public_control_note(messages, note)
    for message in messages:
        if message.get("role") == "assistant":
            message["loss_mask"] = True
    target = _normalise_target(
        target_record.get("target_assistant", {}),
        _existing_call_ids(messages),
        str(target_record.get("task_id", "")),
        str(target_record.get("boundary_type", "")),
    )
    messages.append(target)
    rendered = {
        "schema_version": RECOVERY_SFT_SCHEMA_VERSION,
        "source_schema_version": RECOVERY_TARGET_SCHEMA_VERSION,
        "boundary_schema_version": target_record.get("boundary_schema_version", "recovery-boundary-v1"),
        "task_id": str(target_record.get("task_id", "")),
        "boundary_type": str(target_record.get("boundary_type", "")),
        "composition": str(target_record.get("composition", "unknown")),
        "project_split": str(target_record.get("project_split", "unknown")),
        "target_status": "accepted",
        "policy_version": str(target_record.get("policy_version", "unknown")),
        "actor_policy_version": ACTOR_RUNTIME_POLICY_VERSION,
        "messages": messages,
        "public_state_before": copy.deepcopy(dict(target_record.get("public_state_before", {}))),
        "control_note": note,
        "control_note_placement": placement,
        "target_assistant": copy.deepcopy(target),
        "source_provenance": copy.deepcopy(target_record.get("source_provenance", [])),
        "target_provenance": copy.deepcopy(target_record.get("target_provenance", {})),
        "quality_checks": {
            "production_policy_injected": True,
            "policy_version_parity": True,
            "teacher_instruction_removed": True,
            "public_control_note_rendered": True,
            "historical_assistants_loss_masked": True,
            "only_final_assistant_unmasked": True,
            "target_deferred_from_public_state": True,
        },
    }
    return rendered


# [项目注释] 功能：`_leakage_hits`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：isinstance, items, extend, enumerate。
# [项目注释] 输入：`value`: Any；`path`: str。
# [项目注释] 输出：标注返回 `list[str]`；具体值由各分支决定。
def _leakage_hits(value: Any, path: str = "") -> list[str]:
    hits: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).casefold() in HIDDEN_FIELD_NAMES:
                hits.append(f"{path}/{key}")
            hits.extend(_leakage_hits(child, f"{path}/{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            hits.extend(_leakage_hits(child, f"{path}/{index}"))
    elif isinstance(value, str):
        if TEACHER_GENERATION_INSTRUCTION_MARKER in value or TEACHER_GENERATION_INSTRUCTION in value:
            hits.append(f"{path}:teacher_instruction")
        if LEGACY_TEACHER_ACTOR_POLICY.strip() and LEGACY_TEACHER_ACTOR_POLICY.strip() in value:
            hits.append(f"{path}:legacy_teacher_policy")
    return hits


# [项目注释] 功能：`_target_action`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：isinstance, RecoverySFTError,
# [项目注释]    from_parameters, loads。
# [项目注释] 输入：`record`: Mapping[str, Any]。
# [项目注释] 输出：标注返回 `UserBenchAction`；具体值由各分支决定。
def _target_action(record: Mapping[str, Any]) -> UserBenchAction:
    target = record.get("target_assistant")
    if not isinstance(target, Mapping):
        raise RecoverySFTError("missing target assistant")
    calls = target.get("tool_calls")
    function = calls[0].get("function") if isinstance(calls, list) and len(calls) == 1 else None
    arguments = function.get("arguments") if isinstance(function, Mapping) else None
    if not isinstance(arguments, str):
        raise RecoverySFTError("target tool arguments must be JSON text")
    try:
        return UserBenchAction.from_parameters(json.loads(arguments))
    except (json.JSONDecodeError, UserBenchActionError) as exc:
        raise RecoverySFTError("target tool arguments are invalid") from exc


# [项目注释] 功能：`_answer_visibility`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：_target_action,
# [项目注释]    public_state_from_payload, strip, frozenset。
# [项目注释] 输入：`record`: Mapping[str, Any]。
# [项目注释] 输出：标注返回 `tuple[bool, str | None]`；具体值由各分支决定。
def _answer_visibility(record: Mapping[str, Any]) -> tuple[bool, str | None]:
    action = _target_action(record)
    if action.choice is not ActionChoice.ANSWER:
        return True, None
    ids = [value.strip() for value in action.content.split(",") if value.strip()]
    state = public_state_from_payload(
        record["public_state_before"],
        record["messages"][:-1],
        phase_hint=str(record.get("boundary_type", "")),
    )
    visible = state.current.visible_option_ids if state.current is not None else frozenset()
    if len(ids) != 1 or OPTION_ID.fullmatch(ids[0] if ids else "") is None:
        return False, "answer_not_exactly_one_visible_option_id"
    if ids[0] not in visible:
        return False, "answer_id_not_visible"
    return True, None


# [项目注释] 功能：`_sample_hash`：按固定约束拆分、采样或选择输入集合，保持确定性和边界条件。 主要协作调用：encode, hexdigest, dumps, sha256。
# [项目注释] 输入：`record`: Mapping[str, Any]。
# [项目注释] 输出：标注返回 `str`；具体值由各分支决定。
def _sample_hash(record: Mapping[str, Any]) -> str:
    payload = {
        "boundary_type": record.get("boundary_type"),
        "composition": record.get("composition"),
        "actor_policy_version": record.get("actor_policy_version"),
        "messages": record.get("messages"),
        "public_state_before": record.get("public_state_before"),
        "target_assistant": record.get("target_assistant"),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def audit_rendered_record(
    record: Mapping[str, Any],
    tokenizer: Any,
    tool_schema: Mapping[str, Any],
    *,
    max_sequence_length: int,
) -> RecoveryAuditResult:
    """Run schema, parity, leakage, answer, and action-only mask checks."""

    reasons = list(recovery_admission_reasons(record))
    messages = record.get("messages") if isinstance(record.get("messages"), list) else []
    system = messages[0].get("content", "") if messages and isinstance(messages[0], Mapping) else ""
    if not isinstance(system, str) or system.count(ACTOR_RUNTIME_POLICY_MARKER) != 1 or system.count(ACTOR_RUNTIME_POLICY) != 1:
        reasons.append("actor_policy_parity_failure")
    if TEACHER_GENERATION_INSTRUCTION_MARKER in system or TEACHER_GENERATION_INSTRUCTION in system:
        reasons.append("teacher_instruction_in_actor_prompt")
    note = record.get("control_note")
    if not isinstance(note, str) or not note:
        reasons.append("missing_control_note")
    else:
        try:
            expected = render_actor_control_info(
                public_state_from_payload(
                    record["public_state_before"],
                    messages[:-1],
                    phase_hint=str(record.get("boundary_type", "")),
                )
            )
            if note != expected:
                reasons.append("control_note_parity_failure")
            occurrences = sum(
                note in str(message.get("content", ""))
                for message in messages
                if isinstance(message, Mapping)
            )
            if occurrences != 1:
                reasons.append("control_note_duplicate_or_missing")
        except RecoverySFTError:
            reasons.append("public_state_render_failure")
    leakage = _leakage_hits(record)
    if leakage:
        reasons.append("hidden_state_leakage")
    try:
        answer_ok, answer_reason = _answer_visibility(record)
        if not answer_ok and answer_reason:
            reasons.append(answer_reason)
    except RecoverySFTError:
        reasons.append("target_action_parse_failure")
    examples: tuple[ActionOnlyExample, ...] = ()
    overlong = False
    try:
        examples = build_action_only_examples(
            [record],
            tokenizer,
            tool_schema,
            max_sequence_length=max_sequence_length,
            record_format="recovery",
        )
    except SFTTrajectoryTooLongError:
        overlong = True
        reasons.append("overlong_sample")
    except SFTDatasetError as exc:
        reasons.append(f"action_only_render_failure:{str(exc).split(':', 1)[0]}")
    if len(examples) != 1:
        reasons.append("expected_one_action_only_example")
    assistants = [
        message
        for message in messages
        if isinstance(message, Mapping) and message.get("role") == "assistant"
    ]
    if not assistants or any(message.get("loss_mask") is not True for message in assistants[:-1]) or assistants[-1].get("loss_mask") is True:
        reasons.append("loss_mask_not_final_only")
    metrics: dict[str, Any] = {
        "policy_parity": "actor_policy_parity_failure" not in reasons,
        "teacher_instruction_absent": "teacher_instruction_in_actor_prompt" not in reasons,
        "control_note_parity": "control_note_parity_failure" not in reasons,
        "hidden_state_leakage": bool(leakage),
        "answer_id_visible": not any(reason in reasons for reason in ("answer_id_not_visible", "answer_not_exactly_one_visible_option_id")),
        "loss_mask_final_only": "loss_mask_not_final_only" not in reasons and "expected_one_action_only_example" not in reasons,
        "overlong": overlong,
        "sequence_length": examples[0].sequence_length if examples else None,
        "label_tokens": examples[0].label_tokens if examples else None,
        "assistant_turn_index": examples[0].assistant_turn_index if examples else None,
        "leakage_hits": leakage,
    }
    unique_reasons = tuple(dict.fromkeys(reasons))
    return RecoveryAuditResult(
        valid=not unique_reasons,
        reasons=unique_reasons,
        metrics=metrics,
        sample_hash=_sample_hash(record),
    )


# [项目注释] 功能：`_percentile`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：sorted, min, int, max。
# [项目注释] 输入：`values`: Sequence[int]；`fraction`: float。
# [项目注释] 输出：标注返回 `int | None`；具体值由各分支决定。
def _percentile(values: Sequence[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return int(ordered[index])


# [项目注释] 功能：`_length_summary`：计算奖励、指标或聚合统计，供训练、评测或报告使用。 主要协作调用：len, int, sum, _percentile。
# [项目注释] 输入：`lengths`: Sequence[int]；`labels`: Sequence[int]。
# [项目注释] 输出：标注返回 `dict[str, Any]`；具体值由各分支决定。
def _length_summary(lengths: Sequence[int], labels: Sequence[int]) -> dict[str, Any]:
    return {
        "examples": len(lengths),
        "effective_label_tokens": int(sum(labels)),
        "sequence_length": {
            "min": min(lengths) if lengths else None,
            "p50": _percentile(lengths, 0.50),
            "p90": _percentile(lengths, 0.90),
            "p95": _percentile(lengths, 0.95),
            "max": max(lengths) if lengths else None,
        },
    }


# [项目注释] 功能：`_rejection_record`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：str, list, isinstance, fromkeys。
# [项目注释] 输入：`value`: Mapping[str, Any] | None；`line`: int；`reasons`: Sequence[str]；`path`: Path。
# [项目注释] 输出：标注返回 `dict[str, Any]`；具体值由各分支决定。
def _rejection_record(value: Mapping[str, Any] | None, *, line: int, reasons: Sequence[str], path: Path) -> dict[str, Any]:
    return {
        "schema_version": RECOVERY_SFT_SCHEMA_VERSION,
        "source_schema_version": RECOVERY_TARGET_SCHEMA_VERSION,
        "source_file": str(path),
        "source_line": line,
        "task_id": str(value.get("task_id", "")) if isinstance(value, Mapping) else "",
        "boundary_type": str(value.get("boundary_type", "unknown")) if isinstance(value, Mapping) else "unknown",
        "project_split": str(value.get("project_split", "unknown")) if isinstance(value, Mapping) else "unknown",
        "target_status": "rejected",
        "rejection_reasons": list(dict.fromkeys(str(reason) for reason in reasons)),
    }


# [项目注释] 功能：`_preflight_task_issues`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：defaultdict, Counter, items,
# [项目注释]    _jsonl_records。
# [项目注释] 输入：`paths`: Mapping[str, Path]。
# [项目注释] 输出：标注返回 `tuple[dict[str, list[tuple[str, int, str]]], dict[str, int]]`；具体值由各分支决定。
def _preflight_task_issues(paths: Mapping[str, Path]) -> tuple[dict[str, list[tuple[str, int, str]],], dict[str, int]]:
    occurrences: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    invalid: dict[str, int] = Counter()
    for source_name, path in paths.items():
        for line, value, error in _jsonl_records(path):
            if error is not None or value is None:
                invalid[f"{source_name}:{error}"] += 1
                continue
            task_id = str(value.get("task_id", ""))
            split = str(value.get("project_split", "unknown"))
            occurrences[task_id].append((source_name, line, split))
    return occurrences, dict(invalid)


def build_recovery_sft_dataset(
    target_dir: str | Path,
    output_dir: str | Path,
    tokenizer: Any,
    tool_schema: Mapping[str, Any],
    *,
    max_sequence_length: int = 16384,
) -> tuple[dict[str, Path], dict[str, Any]]:
    """Render train/validation targets and write a reproducible audit."""

    target_root = Path(target_dir).resolve()
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    paths_in = {
        "train": target_root / "train.jsonl",
        "validation": target_root / "validation.jsonl",
    }
    for path in paths_in.values():
        if not path.is_file():
            raise RecoverySFTError(f"missing recovery target split: {path}")
    output_paths = {
        "train": output_root / "train.jsonl",
        "validation": output_root / "validation.jsonl",
        "rejected": output_root / "rejected.jsonl",
        "manifest": output_root / "manifest.json",
        "audit": output_root / "audit.json",
    }
    occurrences, invalid_input = _preflight_task_issues(paths_in)
    task_issues: dict[str, tuple[str, ...]] = {}
    for task_id, items in occurrences.items():
        reasons: list[str] = []
        # Multiple boundary contexts for one task are expected. The invariant
        # is task-level split ownership, not one sample per task.
        if len({item[2] for item in items}) > 1:
            reasons.append("train_validation_task_overlap")
        task_issues[task_id] = tuple(reasons)

    counts: Counter[str] = Counter()
    boundary_counts: dict[str, Counter[str]] = defaultdict(Counter)
    composition_counts: dict[str, Counter[str]] = defaultdict(Counter)
    rejection_reasons: Counter[str] = Counter()
    lengths: dict[str, list[int]] = {"train": [], "validation": []}
    labels: dict[str, list[int]] = {"train": [], "validation": []}
    accepted_task_ids: dict[str, set[str]] = {"train": set(), "validation": set()}
    sample_hashes: set[str] = set()
    quality = Counter()
    control_note_placements: Counter[str] = Counter()
    rejected_records = 0

    with (
        output_paths["train"].open("w", encoding="utf-8") as train_handle,
        output_paths["validation"].open("w", encoding="utf-8") as validation_handle,
        output_paths["rejected"].open("w", encoding="utf-8") as rejected_handle,
    ):
        handles = {"train": train_handle, "validation": validation_handle}
        for source_name, source_path in paths_in.items():
            for line, value, parse_error in _jsonl_records(source_path):
                counts[f"input_{source_name}"] += 1
                if parse_error is not None or value is None:
                    rejection_reasons[parse_error or "invalid_record"] += 1
                    rejected_records += 1
                    rejected_handle.write(json.dumps(_rejection_record(value, line=line, reasons=(parse_error or "invalid_record",), path=source_path), ensure_ascii=False, sort_keys=True) + "\n")
                    continue
                task_id = str(value.get("task_id", ""))
                split = str(value.get("project_split", "unknown"))
                reasons = list(task_issues.get(task_id, ()))
                expected = TRAINING_SPLITS if source_name == "train" else VALIDATION_SPLITS
                if split not in expected:
                    reasons.append("source_file_split_mismatch")
                if split == "evaluation":
                    reasons.append("evaluation_excluded")
                if split not in TRAINING_SPLITS | VALIDATION_SPLITS:
                    reasons.append("unknown_project_split")
                if reasons:
                    for reason in reasons:
                        rejection_reasons[reason] += 1
                    rejected_records += 1
                    rejected_handle.write(json.dumps(_rejection_record(value, line=line, reasons=reasons, path=source_path), ensure_ascii=False, sort_keys=True) + "\n")
                    continue
                try:
                    rendered = render_recovery_record(value)
                except (RecoverySFTError, TypeError, ValueError) as exc:
                    reason = f"render_failure:{exc.__class__.__name__}"
                    rejection_reasons[reason] += 1
                    rejected_records += 1
                    rejected_handle.write(json.dumps(_rejection_record(value, line=line, reasons=(reason,), path=source_path), ensure_ascii=False, sort_keys=True) + "\n")
                    continue
                audit = audit_rendered_record(rendered, tokenizer, tool_schema, max_sequence_length=max_sequence_length)
                if not audit.valid:
                    for reason in audit.reasons:
                        rejection_reasons[reason] += 1
                    rejected_records += 1
                    rejected_handle.write(json.dumps(_rejection_record(value, line=line, reasons=audit.reasons, path=source_path), ensure_ascii=False, sort_keys=True) + "\n")
                    continue
                if audit.sample_hash in sample_hashes:
                    rejection_reasons["duplicate_sample_hash"] += 1
                    rejected_records += 1
                    rejected_handle.write(json.dumps(_rejection_record(value, line=line, reasons=("duplicate_sample_hash",), path=source_path), ensure_ascii=False, sort_keys=True) + "\n")
                    continue
                sample_hashes.add(str(audit.sample_hash))
                output_name = "train" if split in TRAINING_SPLITS else "validation"
                rendered["render_audit"] = dict(audit.metrics)
                rendered["sample_hash"] = audit.sample_hash
                handles[output_name].write(json.dumps(rendered, ensure_ascii=False, sort_keys=True) + "\n")
                counts[f"accepted_{output_name}"] += 1
                boundary_counts[output_name][str(rendered["boundary_type"])] += 1
                composition_counts[output_name][str(rendered["composition"])] += 1
                control_note_placements[str(rendered["control_note_placement"])] += 1
                accepted_task_ids[output_name].add(task_id)
                lengths[output_name].append(int(audit.metrics["sequence_length"]))
                labels[output_name].append(int(audit.metrics["label_tokens"]))
                quality["policy_parity"] += int(audit.metrics["policy_parity"])
                quality["loss_mask"] += int(audit.metrics["loss_mask_final_only"])
                quality["answer_visibility"] += int(audit.metrics["answer_id_visible"])
                quality["hidden_clean"] += int(not audit.metrics["hidden_state_leakage"])

    train_ids = accepted_task_ids["train"]
    validation_ids = accepted_task_ids["validation"]
    overlap = sorted(train_ids & validation_ids)
    output_hashes = {
        name: _sha256(path)
        for name, path in output_paths.items()
        if name not in {"manifest", "audit"}
    }
    source_hashes = {name: _sha256(path) for name, path in paths_in.items()}
    source_manifest = target_root / "manifest.json"
    if source_manifest.is_file():
        source_hashes["manifest"] = _sha256(source_manifest)
    total_accepted = counts["accepted_train"] + counts["accepted_validation"]
    audit_report = {
        "schema_version": RECOVERY_SFT_SCHEMA_VERSION,
        "generator_version": RECOVERY_SFT_GENERATOR_VERSION,
        "actor_policy_version": ACTOR_RUNTIME_POLICY_VERSION,
        "counts": {
            "input_train": counts["input_train"],
            "input_validation": counts["input_validation"],
            "accepted_train": counts["accepted_train"],
            "accepted_validation": counts["accepted_validation"],
            "rejected": rejected_records,
            "accepted_total": total_accepted,
        },
        "boundary_type_distribution": {
            split: dict(sorted(values.items())) for split, values in boundary_counts.items()
        },
        "composition_distribution": {
            split: dict(sorted(values.items())) for split, values in composition_counts.items()
        },
        "control_note_placement_distribution": dict(sorted(control_note_placements.items())),
        "token_length": {
            "train": _length_summary(lengths["train"], labels["train"]),
            "validation": _length_summary(lengths["validation"], labels["validation"]),
            "max_sequence_length": max_sequence_length,
            "overlong_rejected": rejection_reasons.get("overlong_sample", 0),
        },
        "quality_checks": {
            "policy_parity": quality["policy_parity"] == total_accepted,
            "loss_mask_final_only": quality["loss_mask"] == total_accepted,
            "answer_id_visibility": quality["answer_visibility"] == total_accepted,
            "hidden_state_leakage": quality["hidden_clean"] != total_accepted,
            "hidden_state_leakage_free": quality["hidden_clean"] == total_accepted,
            # Duplicate samples are quarantined before writing accepted
            # files; this flag describes the final dataset, while the count
            # below preserves the audit finding.
            "complete_duplicate_samples": True,
            "duplicate_samples_quarantined": rejection_reasons.get("duplicate_sample_hash", 0),
            "train_validation_task_overlap": not overlap,
            "evaluation_in_train": 0,
        },
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "task_split_checks": {
            "train_validation_overlap": overlap,
            "task_id_assignment_before_render": True,
            "formal_evaluation_excluded": True,
        },
        "source_file_hashes": source_hashes,
        "output_file_hashes": output_hashes,
        "input_schema_version": RECOVERY_TARGET_SCHEMA_VERSION,
        "output_schema_version": RECOVERY_SFT_SCHEMA_VERSION,
    }
    manifest = {
        "schema_version": RECOVERY_SFT_SCHEMA_VERSION,
        "generator_version": RECOVERY_SFT_GENERATOR_VERSION,
        "input_schema_version": RECOVERY_TARGET_SCHEMA_VERSION,
        "actor_policy_version": ACTOR_RUNTIME_POLICY_VERSION,
        "target_policy_version_field_preserved": True,
        "tokenizer": {
            "type": type(tokenizer).__name__,
            "vocab_size": getattr(tokenizer, "vocab_size", None),
            "max_sequence_length": max_sequence_length,
            "local_cpu_only": True,
        },
        "paths": {name: str(path) for name, path in output_paths.items()},
        "audit": audit_report,
        "source_file_hashes": source_hashes,
        "output_file_hashes": output_hashes,
        "invalid_input_records": invalid_input,
    }
    output_paths["audit"].write_text(json.dumps(audit_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["audit_file_hash"] = _sha256(output_paths["audit"])
    output_paths["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_paths, manifest


__all__ = [
    "CPUChatTemplateTokenizer",
    "RECOVERY_SFT_GENERATOR_VERSION",
    "RECOVERY_SFT_SCHEMA_VERSION",
    "RecoveryAuditResult",
    "RecoverySFTError",
    "audit_rendered_record",
    "build_recovery_sft_dataset",
    "load_cpu_chat_template_tokenizer",
    "public_state_from_payload",
    "render_recovery_record",
]
