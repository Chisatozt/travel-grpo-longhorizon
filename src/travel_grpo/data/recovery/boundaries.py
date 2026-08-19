"""Extract public recovery-boundary contexts from existing trajectories.

The extractor is deliberately an offline, CPU-only data utility.  It reads
existing JSONL/JSON artifacts, replays only actor-visible messages through the
public control reducer, and writes derived records with deferred targets.  It
does not read reward snapshots, correctness labels, or hidden preference
values when constructing a context.

The output contract is ``recovery-boundary-v1``.  Source artifacts are never
modified; provenance and split checks are recorded in the generated manifest.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from travel_grpo.envs.public_control import (
    PublicAspectStatus,
    PublicControlState,
    PublicObservationKind,
    RecoveryMode,
    advance_public_aspect,
    classify_public_observation,
    mark_public_preference_complete,
    new_public_control_state,
    reduce_public_feedback,
)
from travel_grpo.envs.userbench_tools import (
    ActionChoice,
    UserBenchAction,
    UserBenchActionError,
    action_mentions_aspect,
    normalized_action_signature,
    aspect_from_option_id,
)
from travel_grpo.protocols.actor_messages import normalize_actor_messages


SCHEMA_VERSION = "recovery-boundary-v1"
GENERATOR_VERSION = "recovery-boundary-extractor-v1"

BOUNDARY_TYPES = (
    "preference_complete_to_search",
    "explicit_no_preference",
    "valid_search_to_answer",
    "first_fallback",
    "second_fallback",
    "repeated_no_progress_action",
    "visible_options_pending_answer",
)

TRAINING_SPLITS = frozenset(("sft_train", "grpo_train"))
VALIDATION_SPLITS = frozenset(("sft_validation", "grpo_validation"))
EVALUATION_SPLITS = frozenset(("evaluation",))
ALL_SPLITS = TRAINING_SPLITS | VALIDATION_SPLITS | EVALUATION_SPLITS

# These names are never copied from source record-level metadata into public
# contexts.  The message normalizer also drops arbitrary keys, so a hidden
# reward field cannot accidentally become part of a derived message.
HIDDEN_FIELD_NAMES = frozenset(
    {
        "remaining_preference_ids",
        "correct_ids",
        "best_ids",
        "reward_snapshot",
        "reward_delta",
        "reward",
        "gold_itinerary",
        "correct_itinerary",
        "hidden_preference",
        "hidden_preferences",
        "oracle_next_search",
        "gts",
    }
)

_GRPO_PARAMETER_RE = re.compile(
    r"<parameter=(thought|choice|content)>\s*(.*?)\s*</parameter>",
    re.IGNORECASE | re.DOTALL,
)
_GRPO_CALL_RE = re.compile(
    r"<tool_call>\s*<function=interact_with_env>\s*(.*?)\s*</function>\s*</tool_call>",
    re.IGNORECASE | re.DOTALL,
)
_GRPO_RESPONSE_RE = re.compile(
    r"<tool_response>\s*(.*?)\s*</tool_response>",
    re.IGNORECASE | re.DOTALL,
)
_NO_PREFERENCE_MARKERS = (
    "no specific preference",
    "do not have specific preference",
    "don't have specific preference",
    "no particular preference",
    "no preference",
    "without a preference",
    "any preference is fine",
    "open to any",
)
_PUBLIC_FALLBACK_MARKERS = (
    "searching backend is experiencing some issues",
    "search backend is experiencing some issues",
    "search backend is unavailable",
    "search is temporarily unavailable",
    "normally simulate a system error",
    "by default will return n/a",
)


@dataclass(frozen=True)
class SourceSpec:
    """One input artifact discovered by :func:`discover_sources`."""

    path: Path
    kind: str
    split_hint: str | None = None
    formal_evaluation: bool = False


@dataclass(frozen=True)
class _Event:
    """One actor action and its immediately following public tool response."""

    assistant_index: int
    tool_index: int | None
    action: UserBenchAction
    feedback: str | None
    before_messages: list[dict[str, Any]]
    through_feedback: list[dict[str, Any]]


@dataclass
class _Candidate:
    """Internal candidate before schema normalization and deduplication."""

    task_id: str
    boundary_type: str
    policy_version: str
    messages: list[dict[str, Any]]
    public_state_before: dict[str, Any]
    source_provenance: dict[str, Any]
    composition: str
    project_split: str
    replay_ok: bool


# [项目注释] 功能：`_normalise_text`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：join, split, casefold。
# [项目注释] 输入：`value`: str。
# [项目注释] 输出：标注返回 `str`；具体值由各分支决定。
def _normalise_text(value: str) -> str:
    return " ".join(value.casefold().split())


# [项目注释] 功能：`_sha256_file`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：sha256, hexdigest, open, iter。
# [项目注释] 输入：`path`: Path。
# [项目注释] 输出：标注返回 `str`；具体值由各分支决定。
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# [项目注释] 功能：`_relative_path`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：str, relative_to, resolve。
# [项目注释] 输入：`path`: Path；`root`: Path。
# [项目注释] 输出：标注返回 `str`；具体值由各分支决定。
def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _jsonl_rows(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    """Yield JSONL rows with one-based source line numbers."""

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                yield line_number, value


def load_task_split_map(project_root: str | Path) -> dict[str, dict[str, Any]]:
    """Load the authoritative task-ID split assignment.

    A task appearing in more than one project split is rejected.  This check
    happens before trajectory extraction, preventing sample-level random
    splitting and making formal evaluation leakage explicit.
    """

    root = Path(project_root)
    paths = {
        "sft_train": root / "data/sft/tasks_train.jsonl",
        "sft_validation": root / "data/sft/tasks_validation.jsonl",
        "grpo_train": root / "data/grpo/train.jsonl",
        "grpo_validation": root / "data/grpo/validation.jsonl",
        "evaluation": root / "data/evaluation/tasks.jsonl",
    }
    assignments: dict[str, dict[str, Any]] = {}
    for split, path in paths.items():
        if not path.exists():
            continue
        for line_number, row in _jsonl_rows(path):
            task_id = row.get("task_id")
            if not isinstance(task_id, str) or not task_id:
                continue
            previous = assignments.get(task_id)
            if previous is not None and previous["project_split"] != split:
                raise ValueError(
                    f"task_id {task_id!r} appears in both "
                    f"{previous['project_split']} and {split}"
                )
            assignments[task_id] = {
                "project_split": split,
                "composition": str(row.get("composition", "unknown")),
                "source_path": str(path),
                "source_line": line_number,
            }
    return assignments


def discover_sources(project_root: str | Path) -> list[SourceSpec]:
    """Discover the requested accepted, GRPO-failure, and A/B artifacts."""

    root = Path(project_root)
    sources: list[SourceSpec] = []
    # The current SFT train/holdout inventories are input to split
    # assignment.  They do not contain assistant/tool turns, so they produce
    # no boundary records but remain visible in the manifest as auditable
    # sources.
    for name, hint in ((
        "data/sft/tasks_train.jsonl",
        "sft_train",
    ), (
        "data/sft/tasks_validation.jsonl",
        "sft_validation",
    )):
        path = root / name
        if path.exists():
            sources.append(SourceSpec(path, "sft_task_inventory", hint))

    teacher_dir = root / "outputs/teacher_trajectories"
    for name, hint in (
        ("sft_train.accepted.jsonl", "sft_train"),
        ("sft_validation.from_train.accepted.jsonl", "sft_validation"),
    ):
        path = teacher_dir / name
        if path.exists():
            sources.append(SourceSpec(path, "teacher_accepted", hint))

    grpo_root = root / "outputs/models/grpo-baseline-20"
    for split_name, hint in (
        ("training_rollouts", "grpo_train"),
        ("validation_rollouts", "grpo_validation"),
    ):
        for path in sorted((grpo_root / split_name).glob("*.jsonl")):
            sources.append(SourceSpec(path, "grpo_failed", hint))

    evaluation_root = root / "outputs/evaluation"
    for directory in (
        "ab_prompt_test",
        "search_answer_probe",
        "preference_boundary_probe",
        "fallback_recovery_probe",
    ):
        path = evaluation_root / directory
        if not path.exists():
            continue
        for record_path in sorted(path.rglob("*.json")):
            if record_path.name in {"manifest.json", "summary.json"}:
                continue
            sources.append(SourceSpec(record_path, "ab_offline", "evaluation", True))
    return sources


def _is_failure_record(record: Mapping[str, Any]) -> bool:
    """Identify a failed/unfinished 20-step actor rollout without using it in state."""

    if record.get("correct_itinerary") is False:
        return True
    try:
        if float(record.get("completion_rate", 1.0)) < 1.0:
            return True
    except (TypeError, ValueError):
        pass
    if any((record.get(name) or 0) > 0 for name in ("exact_repeats", "semantic_repeats")):
        return True
    termination = str(record.get("termination_reason", ""))
    return termination not in {"environment_terminated", "completed"}


# [项目注释] 功能：`_safe_json`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：dumps, str。
# [项目注释] 输入：`value`: Any。
# [项目注释] 输出：标注返回 `Any`；具体值由各分支决定。
def _safe_json(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except (TypeError, ValueError):
        return str(value)




# [项目注释] 功能：`_action_from_message`：把协议/状态数据转换为模型、用户或日志可见的文本表示。 主要协作调用：isinstance, from_parameters, len,
# [项目注释]    loads。
# [项目注释] 输入：`message`: Mapping[str, Any]。
# [项目注释] 输出：标注返回 `UserBenchAction | None`；具体值由各分支决定。
def _action_from_message(message: Mapping[str, Any]) -> UserBenchAction | None:
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        return None
    call = calls[0]
    if not isinstance(call, Mapping):
        return None
    function = call.get("function")
    if not isinstance(function, Mapping) or function.get("name") != "interact_with_env":
        return None
    raw = function.get("arguments")
    try:
        parameters = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(parameters, Mapping):
            return None
        return UserBenchAction.from_parameters(parameters)
    except (json.JSONDecodeError, UserBenchActionError, TypeError):
        return None


def events_from_messages(messages: Sequence[Mapping[str, Any]]) -> list[_Event]:
    """Extract action/tool pairs while preserving only normalized messages."""

    normalized = normalize_actor_messages(messages)
    events: list[_Event] = []
    for index, message in enumerate(normalized):
        if message.get("role") != "assistant":
            continue
        action = _action_from_message(message)
        if action is None:
            continue
        tool_index: int | None = None
        feedback: str | None = None
        if index + 1 < len(normalized) and normalized[index + 1].get("role") == "tool":
            tool_index = index + 1
            value = normalized[index + 1].get("content")
            feedback = value if isinstance(value, str) else None
        end = index + 1 if tool_index is None else tool_index + 1
        events.append(
            _Event(
                assistant_index=index,
                tool_index=tool_index,
                action=action,
                feedback=feedback,
                before_messages=copy.deepcopy(normalized[:index]),
                through_feedback=copy.deepcopy(normalized[:end]),
            )
        )
    return events


def parse_grpo_transcript(input_text: str, output_text: str) -> list[dict[str, Any]]:
    """Parse the lightweight veRL transcript format into public messages."""

    combined = f"{input_text or ''}{output_text or ''}"
    system_match = re.search(r"(?:^|\n)system\n(.*?)\nuser\n(.*?)\nassistant\n", combined, re.DOTALL)
    if system_match is None:
        return []
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_match.group(1)},
        {"role": "user", "content": system_match.group(2)},
    ]
    cursor = system_match.end()
    call_index = 0
    while True:
        call_match = _GRPO_CALL_RE.search(combined, cursor)
        if call_match is None:
            break
        parameters = {
            name: value.strip()
            for name, value in _GRPO_PARAMETER_RE.findall(call_match.group(1))
        }
        if set(parameters) != {"thought", "choice", "content"}:
            cursor = call_match.end()
            continue
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"offline-grpo-call-{call_index}",
                        "type": "function",
                        "function": {
                            "name": "interact_with_env",
                            "arguments": json.dumps(parameters, ensure_ascii=False, separators=(",", ":")),
                        },
                    }
                ],
            }
        )
        call_index += 1
        response_match = _GRPO_RESPONSE_RE.search(combined, call_match.end())
        if response_match is None:
            break
        messages.append(
            {
                "role": "tool",
                "name": "interact_with_env",
                "tool_call_id": f"offline-grpo-call-{call_index - 1}",
                "content": response_match.group(1).strip(),
            }
        )
        cursor = response_match.end()
    return normalize_actor_messages(messages)


# [项目注释] 功能：`_initial_user_message`：把协议/状态数据转换为模型、用户或日志可见的文本表示。 主要协作调用：isinstance, str。
# [项目注释] 输入：`messages`: Sequence[Mapping[str, Any]]。
# [项目注释] 输出：标注返回 `str`；具体值由各分支决定。
def _initial_user_message(messages: Sequence[Mapping[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            return str(message["content"])
    return ""


# [项目注释] 功能：`_advance_if_terminal`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：advance_public_aspect, replace。
# [项目注释] 输入：`state`: PublicControlState。
# [项目注释] 输出：标注返回 `PublicControlState`；具体值由各分支决定。
def _advance_if_terminal(state: PublicControlState) -> PublicControlState:
    if state.current is not None and state.current.status in {
        PublicAspectStatus.ANSWERED,
        PublicAspectStatus.BLOCKED,
    } and not state.episode_done:
        try:
            advanced = advance_public_aspect(state)
            # Historical trajectories can complete aspects out of the
            # initial order.  The public transcript still tells us which
            # aspects remain open, so restore the first open one instead of
            # treating an earlier open aspect as a hidden terminal signal.
            if advanced.episode_done and advanced.open_aspects:
                return replace(
                    advanced,
                    current_aspect=advanced.open_aspects[0],
                    recovery_mode=RecoveryMode.NONE,
                    episode_done=False,
                    consecutive_no_progress=0,
                )
            return advanced
        except Exception:
            return state
    return state


class _PublicReplay:
    """Fail-closed public reducer wrapper used by the offline extractor."""

    # [项目注释] 功能：`__init__`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：new_public_control_state, bool。
    # [项目注释] 输入：`initial_user_message`: str。
    # [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
    def __init__(self, initial_user_message: str) -> None:
        self.state = new_public_control_state(initial_user_message)
        self.ok = bool(initial_user_message)

    # [项目注释] 功能：`_target_aspect`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：_target_aspect_from_action, strip,
    # [项目注释]    split, aspect_from_option_id。
    # [项目注释] 输入：`action`: UserBenchAction。
    # [项目注释] 输出：标注返回 `str | None`；具体值由各分支决定。
    def _target_aspect(self, action: UserBenchAction) -> str | None:
        target = _target_aspect_from_action(action, self.state.public_aspects)
        if action.choice is ActionChoice.ANSWER:
            for option_id in (item.strip() for item in action.content.split(",")):
                target = aspect_from_option_id(option_id) or target
                if target is not None:
                    break
        return target if target in self.state.public_aspects else None

    def prepare(self, action: UserBenchAction) -> PublicControlState:
        """Align replay with a historically explicit public aspect mention.

        Early Teacher/GRPO artifacts sometimes search a named aspect before
        the previous aspect is terminal.  This is not a hidden-task lookup: it
        is the aspect literally named in the actor call (or option ID).
        """

        self.state = _advance_if_terminal(self.state)
        target = self._target_aspect(action)
        if target is not None and target != self.state.current_aspect:
            target_state = next(
                (item for item in self.state.aspects if item.aspect == target),
                None,
            )
            if target_state is not None and target_state.status is PublicAspectStatus.OPEN:
                self.state = replace(
                    self.state,
                    current_aspect=target,
                    recovery_mode=RecoveryMode.NONE,
                    consecutive_no_progress=0,
                )
        return self.state

    # [项目注释] 功能：`before_event`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：_advance_if_terminal。
    # [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
    # [项目注释] 输出：标注返回 `PublicControlState`；具体值由各分支决定。
    def before_event(self) -> PublicControlState:
        self.state = _advance_if_terminal(self.state)
        return self.state

    # [项目注释] 功能：`apply`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：prepare, reduce_public_feedback。
    # [项目注释] 输入：`action`: UserBenchAction；`feedback`: str | None。
    # [项目注释] 输出：标注返回 `PublicControlState`；具体值由各分支决定。
    def apply(self, action: UserBenchAction, feedback: str | None) -> PublicControlState:
        self.prepare(action)
        if feedback is None:
            self.ok = False
            return self.state
        try:
            self.state = reduce_public_feedback(self.state, action, feedback)
        except Exception:
            self.ok = False
        return self.state


# [项目注释] 功能：`_phase_label`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。
# [项目注释] 输入：`state`: PublicControlState。
# [项目注释] 输出：标注返回 `str`；具体值由各分支决定。
def _phase_label(state: PublicControlState) -> str:
    phase = state.phase
    if phase is RecoveryMode.NONE and not state.episode_done:
        return "ELICITING"
    return phase.name


def public_state_payload(state: PublicControlState) -> dict[str, Any]:
    """Serialize only public control evidence (never reward or task labels)."""

    current = state.current
    return {
        "current_aspect": state.current_aspect,
        "recovery_mode": _phase_label(state),
        "fallback_count": current.search_fallbacks if current is not None else 0,
        "visible_option_ids": sorted(current.visible_option_ids) if current is not None else [],
        "answered_aspects": list(state.answered_aspects),
        "blocked_aspects": list(state.blocked_aspects),
        "preference_complete_aspects": [
            item.aspect for item in state.aspects if item.preferences_complete
        ],
        "search_attempts": current.search_attempts if current is not None else 0,
        "normal_search_seen": bool(current.normal_search_seen) if current is not None else False,
        "consecutive_no_progress": state.consecutive_no_progress,
        "last_transition_aspect": state.last_transition_aspect,
        "last_transition_status": (
            state.last_transition_status.value.upper()
            if state.last_transition_status is not None
            else None
        ),
    }


# [项目注释] 功能：`_is_no_preference`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：_normalise_text, any, isinstance。
# [项目注释] 输入：`text`: str | None。
# [项目注释] 输出：标注返回 `bool`；具体值由各分支决定。
def _is_no_preference(text: str | None) -> bool:
    if not isinstance(text, str):
        return False
    normalized = _normalise_text(text)
    return any(marker in normalized for marker in _NO_PREFERENCE_MARKERS)


# [项目注释] 功能：`_observation_kind`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：classify_public_observation。
# [项目注释] 输入：`action`: UserBenchAction；`feedback`: str | None。
# [项目注释] 输出：标注返回 `PublicObservationKind | None`；具体值由各分支决定。
def _observation_kind(action: UserBenchAction, feedback: str | None) -> PublicObservationKind | None:
    if feedback is None:
        return None
    try:
        return classify_public_observation(feedback, choice=action.choice).kind
    except Exception:
        return None


# [项目注释] 功能：`_target_aspect_from_action`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：len, action_mentions_aspect。
# [项目注释] 输入：`action`: UserBenchAction；`aspects`: Sequence[str]。
# [项目注释] 输出：标注返回 `str | None`；具体值由各分支决定。
def _target_aspect_from_action(action: UserBenchAction, aspects: Sequence[str]) -> str | None:
    mentions = [aspect for aspect in aspects if action_mentions_aspect(action.content, aspect)]
    if len(mentions) == 1:
        return mentions[0]
    return None


# [项目注释] 功能：`_candidate`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：_Candidate, ValueError,
# [项目注释]    mark_public_preference_complete, normalize_actor_messages。
# [项目注释] 输入：`task_id`: str；`boundary_type`: str；`policy_version`: str；`messages`: Sequence[Mapping[str,
# [项目注释]    Any]]；`state`: PublicControlState；`provenance`: Mapping[str, Any]；`composition`:
# [项目注释]    str；`project_split`: str；`replay_ok`: bool。
# [项目注释] 输出：标注返回 `_Candidate`；具体值由各分支决定。
def _candidate(
    *,
    task_id: str,
    boundary_type: str,
    policy_version: str,
    messages: Sequence[Mapping[str, Any]],
    state: PublicControlState,
    provenance: Mapping[str, Any],
    composition: str,
    project_split: str,
    replay_ok: bool,
) -> _Candidate:
    if boundary_type not in BOUNDARY_TYPES:
        raise ValueError(f"unsupported boundary type {boundary_type!r}")
    # A preference-complete boundary is an explicit public phase hint derived
    # from the accepted, actor-visible search transition. Persist it in the
    # derived record so offline renderers do not fall back to ELICITING.
    if boundary_type == "preference_complete_to_search" and state.current_aspect is not None:
        state = mark_public_preference_complete(state, state.current_aspect)
    return _Candidate(
        task_id=task_id,
        boundary_type=boundary_type,
        policy_version=policy_version or "unknown",
        messages=normalize_actor_messages(messages),
        public_state_before=public_state_payload(state),
        source_provenance=dict(provenance),
        composition=composition or "unknown",
        project_split=project_split,
        replay_ok=replay_ok,
    )


def extract_message_boundaries(
    *,
    task_id: str,
    messages: Sequence[Mapping[str, Any]],
    policy_version: str,
    provenance: Mapping[str, Any],
    composition: str,
    project_split: str,
) -> list[_Candidate]:
    """Extract boundary contexts from a public message sequence."""

    normalized = normalize_actor_messages(messages)
    replay = _PublicReplay(_initial_user_message(normalized))
    events = events_from_messages(normalized)
    candidates: list[_Candidate] = []
    previous_event: _Event | None = None
    previous_state: PublicControlState | None = None
    previous_feedback: str | None = None
    seen_action_signatures: set[str] = set()
    fallback_counts: defaultdict[str, int] = defaultdict(int)

    for event in events:
        # Preserve the state created by the preceding normal search before
        # considering a potentially wrong next action.  Otherwise preparing
        # that action could move ``current_aspect`` and hide the visible
        # options that define this boundary.
        unprepared_before = replay.before_event()
        replay.prepare(event.action)
        before = replay.before_event()
        # A search that follows public elicitation is the observable
        # preference-complete boundary.  Search retries are classified by
        # their fallback boundary instead of being mislabeled here.
        target = _target_aspect_from_action(event.action, before.public_aspects)
        current_aspect = target or before.current_aspect
        if (
            event.action.choice is ActionChoice.SEARCH
            and before.recovery_mode is not RecoveryMode.SEARCH_RETRY_REQUIRED
            and (current_aspect is not None)
        ):
            candidates.append(
                _candidate(
                    task_id=task_id,
                    boundary_type="preference_complete_to_search",
                    policy_version=policy_version,
                    messages=event.before_messages,
                    state=before,
                    provenance=provenance,
                    composition=composition,
                    project_split=project_split,
                    replay_ok=replay.ok,
                )
            )

        action_signature = normalized_action_signature(event.action)
        repeated = action_signature in seen_action_signatures
        if previous_feedback is not None and event.feedback is not None:
            repeated = repeated or _normalise_text(event.feedback) == _normalise_text(previous_feedback)
        if repeated:
            candidates.append(
                _candidate(
                    task_id=task_id,
                    boundary_type="repeated_no_progress_action",
                    policy_version=policy_version,
                    messages=event.before_messages,
                    state=before,
                    provenance=provenance,
                    composition=composition,
                    project_split=project_split,
                    replay_ok=replay.ok,
                )
            )
        seen_action_signatures.add(action_signature)

        if (
            previous_event is not None
            and previous_state is not None
            and previous_state.current is not None
            and bool(previous_state.current.visible_option_ids)
            and previous_event.action.choice is ActionChoice.SEARCH
            and _observation_kind(previous_event.action, previous_event.feedback)
            is PublicObservationKind.SEARCH_NORMAL
            and event.action.choice is ActionChoice.ANSWER
        ):
            candidates.append(
                _candidate(
                    task_id=task_id,
                    boundary_type="valid_search_to_answer",
                    policy_version=policy_version,
                    messages=event.before_messages,
                    state=before,
                    provenance=provenance,
                    composition=composition,
                    project_split=project_split,
                    replay_ok=replay.ok,
                )
            )
        elif (
            previous_event is not None
            and previous_state is not None
            and previous_state.current is not None
            and bool(previous_state.current.visible_option_ids)
            and previous_event.action.choice is ActionChoice.SEARCH
            and _observation_kind(previous_event.action, previous_event.feedback)
            is PublicObservationKind.SEARCH_NORMAL
            and event.action.choice is not ActionChoice.ANSWER
        ):
            candidates.append(
                _candidate(
                    task_id=task_id,
                    boundary_type="visible_options_pending_answer",
                    policy_version=policy_version,
                    messages=event.before_messages,
                    state=unprepared_before,
                    provenance=provenance,
                    composition=composition,
                    project_split=project_split,
                    replay_ok=replay.ok,
                )
            )

        after = replay.apply(event.action, event.feedback)
        kind = _observation_kind(event.action, event.feedback)
        if _is_no_preference(event.feedback):
            candidates.append(
                _candidate(
                    task_id=task_id,
                    boundary_type="explicit_no_preference",
                    policy_version=policy_version,
                    messages=event.through_feedback,
                    state=after,
                    provenance=provenance,
                    composition=composition,
                    project_split=project_split,
                    replay_ok=replay.ok,
                )
            )

        if event.action.choice is ActionChoice.SEARCH and kind in {
            PublicObservationKind.SEARCH_FALLBACK,
            PublicObservationKind.SEARCH_EMPTY,
        }:
            aspect = current_aspect or before.current_aspect or "unknown"
            fallback_counts[aspect] += 1
            boundary_type = "first_fallback" if fallback_counts[aspect] == 1 else "second_fallback"
            if fallback_counts[aspect] <= 2:
                candidates.append(
                    _candidate(
                        task_id=task_id,
                        boundary_type=boundary_type,
                        policy_version=policy_version,
                        messages=event.through_feedback,
                        state=after,
                        provenance=provenance,
                        composition=composition,
                        project_split=project_split,
                        replay_ok=replay.ok,
                    )
                )

        previous_event = event
        previous_state = after
        previous_feedback = event.feedback
    return candidates


# [项目注释] 功能：`_split_for_task`：按固定约束拆分、采样或选择输入集合，保持确定性和边界条件。 主要协作调用：str。
# [项目注释] 输入：`task_id`: str；`assignments`: Mapping[str, Mapping[str, Any]]；`source`: SourceSpec。
# [项目注释] 输出：标注返回 `tuple[str, str, bool]`；具体值由各分支决定。
def _split_for_task(
    task_id: str,
    assignments: Mapping[str, Mapping[str, Any]],
    source: SourceSpec,
) -> tuple[str, str, bool]:
    entry = assignments.get(task_id)
    # A context created by a formal evaluation/probe artifact is never a
    # training sample, even when its task ID also appears in an upstream SFT
    # inventory (offline probes intentionally reuse a few train tasks).
    if source.formal_evaluation:
        composition = (
            str(entry.get("composition", "unknown"))
            if entry is not None
            else "unknown"
        )
        return "evaluation", composition, entry is not None
    if entry is not None:
        split = str(entry["project_split"])
        composition = str(entry.get("composition", "unknown"))
        return split, composition, True
    if source.formal_evaluation:
        return "evaluation", "unknown", False
    hint = source.split_hint if source.split_hint in ALL_SPLITS else "unresolved"
    return hint, "unknown", False


# [项目注释] 功能：`_policy_version`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：isinstance。
# [项目注释] 输入：`record`: Mapping[str, Any]；`source`: SourceSpec。
# [项目注释] 输出：标注返回 `str`；具体值由各分支决定。
def _policy_version(record: Mapping[str, Any], source: SourceSpec) -> str:
    value = record.get("policy_version")
    if isinstance(value, str) and value:
        return value
    if source.kind == "ab_offline":
        condition = record.get("ab_condition")
        if condition == "B":
            return "actor-runtime-v1"
        if condition == "A":
            return "actor-runtime-unversioned"
    return "unknown"


# [项目注释] 功能：`_provenance`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：_relative_path, bool, items, casefold。
# [项目注释] 输入：`source`: SourceSpec；`root`: Path；`line`: int | None；`record_index`: int | None；`extra`:
# [项目注释]    Mapping[str, Any] | None。
# [项目注释] 输出：标注返回 `dict[str, Any]`；具体值由各分支决定。
def _provenance(
    source: SourceSpec,
    root: Path,
    *,
    line: int | None = None,
    record_index: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "source_kind": source.kind,
        "path": _relative_path(source.path, root),
        "formal_evaluation": bool(source.formal_evaluation),
    }
    if line is not None:
        value["line"] = line
    if record_index is not None:
        value["record_index"] = record_index
    if extra:
        for key, item in extra.items():
            if key.casefold() not in HIDDEN_FIELD_NAMES:
                value[key] = _safe_json(item)
    return value


# [项目注释] 功能：`_record_to_context`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：normalize_actor_messages, deepcopy,
# [项目注释]    bool, dict。
# [项目注释] 输入：`candidate`: _Candidate；`dedupe_key`: str。
# [项目注释] 输出：标注返回 `dict[str, Any]`；具体值由各分支决定。
def _record_to_context(candidate: _Candidate, *, dedupe_key: str) -> dict[str, Any]:
    training_eligible = candidate.project_split in TRAINING_SPLITS and not bool(
        candidate.source_provenance.get("formal_evaluation")
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": candidate.task_id,
        "boundary_type": candidate.boundary_type,
        "policy_version": candidate.policy_version,
        "messages": normalize_actor_messages(candidate.messages),
        "public_state_before": copy.deepcopy(candidate.public_state_before),
        "target_assistant": None,
        "source_provenance": [dict(candidate.source_provenance)],
        "composition": candidate.composition,
        "project_split": candidate.project_split,
        "quality_checks": {
            "messages_actor_visible": True,
            "public_state_only": True,
            "target_deferred": True,
            "task_split_resolved": candidate.project_split in ALL_SPLITS,
            "training_eligible": training_eligible,
            "formal_eval_excluded_from_training": not training_eligible
            if candidate.project_split == "evaluation"
            else True,
            "public_replay_ok": candidate.replay_ok,
            "dedupe_key": dedupe_key,
            "source_duplicate_count": 1,
        },
    }


# [项目注释] 功能：`_dedupe_candidates`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：list, sort, hexdigest, values。
# [项目注释] 输入：`candidates`: Sequence[_Candidate]。
# [项目注释] 输出：标注返回 `tuple[list[dict[str, Any]], int]`；具体值由各分支决定。
def _dedupe_candidates(candidates: Sequence[_Candidate]) -> tuple[list[dict[str, Any]], int]:
    records: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        fingerprint = {
            "task_id": candidate.task_id,
            "boundary_type": candidate.boundary_type,
            "messages": normalize_actor_messages(candidate.messages),
            "public_state_before": candidate.public_state_before,
        }
        key = hashlib.sha256(
            json.dumps(fingerprint, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        record = records.get(key)
        if record is None:
            records[key] = _record_to_context(candidate, dedupe_key=key)
            continue
        record["source_provenance"].append(dict(candidate.source_provenance))
        checks = record["quality_checks"]
        checks["source_duplicate_count"] = int(checks["source_duplicate_count"]) + 1
        checks["public_replay_ok"] = bool(checks["public_replay_ok"]) and candidate.replay_ok
        if record["policy_version"] != candidate.policy_version:
            record["policy_version"] = "mixed"
    result = list(records.values())
    result.sort(key=lambda value: (value["task_id"], value["boundary_type"], value["quality_checks"]["dedupe_key"]))
    return result, len(candidates) - len(result)


# [项目注释] 功能：`_load_record_at`：读取并解析外部数据，将其转换为项目内部可消费的结构。 主要协作调用：_jsonl_rows。
# [项目注释] 输入：`path`: Path；`line`: int。
# [项目注释] 输出：标注返回 `Mapping[str, Any] | None`；具体值由各分支决定。
def _load_record_at(path: Path, line: int) -> Mapping[str, Any] | None:
    try:
        for line_number, row in _jsonl_rows(path):
            if line_number == line:
                return row
    except (OSError, json.JSONDecodeError):
        return None
    return None


# [项目注释] 功能：`_probe_metadata`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：exists, loads, read_text, isinstance。
# [项目注释] 输入：`root`: Path。
# [项目注释] 输出：标注返回 `dict[tuple[str, str, int], dict[str, Any]]`；具体值由各分支决定。
def _probe_metadata(root: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    # Context IDs are only unique inside one probe manifest; include the
    # probe name so search-answer, preference-boundary, and fallback metadata
    # cannot overwrite one another.
    metadata: dict[tuple[str, str, int], dict[str, Any]] = {}
    for name in ("search_answer_probe", "preference_boundary_probe", "fallback_recovery_probe"):
        manifest_path = root / "outputs/evaluation" / name / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for value in manifest.get("contexts_metadata", []):
            if not isinstance(value, Mapping):
                continue
            task_id = value.get("task_id")
            context_id = value.get("context_id")
            if isinstance(task_id, str) and isinstance(context_id, int):
                item = dict(value)
                item["probe_name"] = name
                item["manifest_fallback_text"] = manifest.get("fallback_text")
                metadata[(name, task_id, context_id)] = item
    return metadata


# [项目注释] 功能：`_load_ab_source_messages`：读取并解析外部数据，将其转换为项目内部可消费的结构。 主要协作调用：_load_record_at,
# [项目注释]    normalize_actor_messages, isinstance。
# [项目注释] 输入：`root`: Path；`metadata`: Mapping[str, Any]。
# [项目注释] 输出：标注返回 `tuple[list[dict[str, Any]], Mapping[str, Any]] | None`；具体值由各分支决定。
def _load_ab_source_messages(root: Path, metadata: Mapping[str, Any]) -> tuple[list[dict[str, Any]], Mapping[str, Any]] | None:
    source_file = metadata.get("source_file")
    source_line = metadata.get("source_line")
    if not isinstance(source_file, str) or not isinstance(source_line, int):
        return None
    path = root / source_file
    record = _load_record_at(path, source_line)
    if not isinstance(record, Mapping) or not isinstance(record.get("messages"), list):
        return None
    return normalize_actor_messages(record["messages"]), record


# [项目注释] 功能：`_find_event`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：isinstance, next, action_mentions_aspect。
# [项目注释] 输入：`events`: Sequence[_Event]；`metadata`: Mapping[str, Any]。
# [项目注释] 输出：标注返回 `_Event | None`；具体值由各分支决定。
def _find_event(events: Sequence[_Event], metadata: Mapping[str, Any]) -> _Event | None:
    message_index = metadata.get("message_index")
    aspect = metadata.get("aspect")
    if isinstance(message_index, int):
        for event in events:
            if event.assistant_index == message_index:
                return event
    for event in events:
        if event.action.choice is ActionChoice.SEARCH and isinstance(aspect, str):
            if action_mentions_aspect(event.action.content, aspect):
                return event
    return next((event for event in events if event.action.choice is ActionChoice.SEARCH), None)


# [项目注释] 功能：`_synthetic_fallback_messages`：把协议/状态数据转换为模型、用户或日志可见的文本表示。 主要协作调用：normalize_actor_messages,
# [项目注释]    append_call, str, dumps。
# [项目注释] 输入：`base_messages`: Sequence[Mapping[str, Any]]；`original_search`: Mapping[str,
# [项目注释]    Any]；`retry_search`: Mapping[str, Any] | None；`fallback_text`: str。
# [项目注释] 输出：标注返回 `list[dict[str, Any]]`；具体值由各分支决定。
def _synthetic_fallback_messages(
    base_messages: Sequence[Mapping[str, Any]],
    original_search: Mapping[str, Any],
    retry_search: Mapping[str, Any] | None,
    fallback_text: str,
) -> list[dict[str, Any]]:
    messages = normalize_actor_messages(base_messages)

    # [项目注释] 功能：`append_call`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：str, dumps。
    # [项目注释] 输入：`parameters`: Mapping[str, Any]；`index`: int。
    # [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
    def append_call(parameters: Mapping[str, Any], index: int) -> None:
        public = {
            key: str(parameters.get(key, ""))
            for key in ("thought", "choice", "content")
        }
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"offline-fallback-call-{index}",
                        "type": "function",
                        "function": {
                            "name": "interact_with_env",
                            "arguments": json.dumps(public, ensure_ascii=False, separators=(",", ":")),
                        },
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "name": "interact_with_env",
                "tool_call_id": f"offline-fallback-call-{index}",
                "content": fallback_text,
            }
        )

    append_call(original_search, 0)
    if retry_search is not None:
        append_call(retry_search, 1)
    return normalize_actor_messages(messages)


# [项目注释] 功能：`_extract_ab_file`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：_split_for_task, _policy_version,
# [项目注释]    _provenance, isinstance。
# [项目注释] 输入：`source`: SourceSpec；`root`: Path；`record`: Mapping[str, Any]；`line`: int；`assignments`:
# [项目注释]    Mapping[str, Mapping[str, Any]]；`metadata_index`: Mapping[tuple[str, str, int], Mapping[str,
# [项目注释]    Any]]。
# [项目注释] 输出：标注返回 `list[_Candidate]`；具体值由各分支决定。
def _extract_ab_file(
    *,
    source: SourceSpec,
    root: Path,
    record: Mapping[str, Any],
    line: int,
    assignments: Mapping[str, Mapping[str, Any]],
    metadata_index: Mapping[tuple[str, str, int], Mapping[str, Any]],
) -> list[_Candidate]:
    task_id = record.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        return []
    project_split, composition, _ = _split_for_task(task_id, assignments, source)
    if composition == "unknown" and isinstance(record.get("composition"), str):
        composition = str(record["composition"])
    policy = _policy_version(record, source)
    try:
        probe_name = str(source.path.relative_to(root / "outputs/evaluation").parts[0])
    except (ValueError, IndexError):
        probe_name = "ab_offline"
    provenance = _provenance(
        source,
        root,
        line=line,
        extra={
            "condition": record.get("ab_condition"),
            "context_id": record.get("context_id"),
            "probe": probe_name,
        },
    )
    visible_transcript = record.get("visible_transcript")
    if isinstance(visible_transcript, list):
        return extract_message_boundaries(
            task_id=task_id,
            messages=visible_transcript,
            policy_version=policy,
            provenance=provenance,
            composition=composition,
            project_split=project_split,
        )

    context_id = record.get("context_id")
    metadata = (
        metadata_index.get((probe_name, task_id, context_id))
        if isinstance(context_id, int)
        else None
    )
    if metadata is None:
        return []
    loaded = _load_ab_source_messages(root, metadata)
    if loaded is None:
        return []
    base_messages, source_record = loaded
    probe_name = str(metadata.get("probe_name", ""))
    source_events = events_from_messages(base_messages)
    event = _find_event(source_events, metadata)
    if event is None:
        return []
    provenance.update(
        {
            "context_id": context_id,
            "source_context_file": metadata.get("source_file"),
            "source_context_line": metadata.get("source_line"),
        }
    )
    if probe_name == "search_answer_probe":
        replay = _PublicReplay(_initial_user_message(base_messages))
        for candidate_event in source_events:
            replay.prepare(candidate_event.action)
            before = replay.before_event()
            after = replay.apply(candidate_event.action, candidate_event.feedback)
            if candidate_event.assistant_index == event.assistant_index:
                return [
                    _candidate(
                        task_id=task_id,
                        boundary_type="valid_search_to_answer",
                        policy_version=policy,
                        messages=event.through_feedback,
                        state=after,
                        provenance=provenance,
                        composition=composition,
                        project_split=project_split,
                        replay_ok=replay.ok,
                    )
                ]
        return []
    if probe_name == "preference_boundary_probe":
        replay = _PublicReplay(_initial_user_message(base_messages))
        for candidate_event in source_events:
            replay.prepare(candidate_event.action)
            before = replay.before_event()
            if candidate_event.assistant_index == event.assistant_index:
                return [
                    _candidate(
                        task_id=task_id,
                        boundary_type="preference_complete_to_search",
                        policy_version=policy,
                        messages=event.before_messages,
                        state=before,
                        provenance=provenance,
                        composition=composition,
                        project_split=project_split,
                        replay_ok=replay.ok,
                    )
                ]
            replay.apply(candidate_event.action, candidate_event.feedback)
        return []
    if probe_name == "fallback_recovery_probe":
        original_search = record.get("original_search")
        if not isinstance(original_search, Mapping):
            return []
        retry_search = record.get("parameters") if record.get("scenario") == "double" else None
        if retry_search is not None and not isinstance(retry_search, Mapping):
            retry_search = None
        fallback_text = metadata.get("manifest_fallback_text")
        if not isinstance(fallback_text, str) or not fallback_text:
            fallback_text = "Currently the searching backend is experiencing some issues. Please try again later."
        base_prefix = event.before_messages
        synthetic = _synthetic_fallback_messages(base_prefix, original_search, retry_search, fallback_text)
        synthetic_events = events_from_messages(synthetic)
        replay = _PublicReplay(_initial_user_message(synthetic))
        final_event: _Event | None = None
        final_state: PublicControlState | None = None
        for synthetic_event in synthetic_events:
            replay.prepare(synthetic_event.action)
            replay.before_event()
            final_state = replay.apply(synthetic_event.action, synthetic_event.feedback)
            final_event = synthetic_event
        if final_event is None or final_state is None:
            return []
        boundary_type = "second_fallback" if record.get("scenario") == "double" else "first_fallback"
        return [
            _candidate(
                task_id=task_id,
                boundary_type=boundary_type,
                policy_version=policy,
                messages=final_event.through_feedback,
                state=final_state,
                provenance=provenance,
                composition=composition,
                project_split=project_split,
                replay_ok=replay.ok,
            )
        ]
    return []


# [项目注释] 功能：`_source_inventory_entry`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：exists, _relative_path,
# [项目注释]    _sha256_file, stat。
# [项目注释] 输入：`source`: SourceSpec；`root`: Path。
# [项目注释] 输出：标注返回 `dict[str, Any]`；具体值由各分支决定。
def _source_inventory_entry(source: SourceSpec, root: Path) -> dict[str, Any]:
    value: dict[str, Any] = {
        "source_kind": source.kind,
        "path": _relative_path(source.path, root),
        "exists": source.path.exists(),
        "formal_evaluation": source.formal_evaluation,
        "split_hint": source.split_hint,
    }
    if source.path.exists():
        value["sha256"] = _sha256_file(source.path)
        value["bytes"] = source.path.stat().st_size
    return value


def extract_recovery_boundaries(
    project_root: str | Path,
    *,
    sources: Sequence[SourceSpec] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract, deduplicate, and return records plus a reproducible manifest."""

    root = Path(project_root).resolve()
    assignments = load_task_split_map(root)
    source_list = list(sources) if sources is not None else discover_sources(root)
    metadata_index = _probe_metadata(root)
    candidates: list[_Candidate] = []
    source_stats: dict[str, dict[str, Any]] = {}

    for source in source_list:
        inventory = _source_inventory_entry(source, root)
        inventory.update({"records_seen": 0, "failure_records_selected": 0, "candidates": 0})
        source_key = inventory["path"]
        source_stats[source_key] = inventory
        if not source.path.exists():
            continue
        if source.kind == "sft_task_inventory":
            inventory["records_seen"] = sum(1 for _ in _jsonl_rows(source.path))
        elif source.kind in {"teacher_accepted", "grpo_failed"}:
            for line, record in _jsonl_rows(source.path):
                inventory["records_seen"] += 1
                if source.kind == "grpo_failed" and not _is_failure_record(record):
                    continue
                if source.kind == "grpo_failed":
                    inventory["failure_records_selected"] += 1
                task_id = record.get("task_id")
                messages: list[Mapping[str, Any]]
                if source.kind == "teacher_accepted":
                    value = record.get("messages")
                    messages = value if isinstance(value, list) else []
                else:
                    messages = parse_grpo_transcript(
                        str(record.get("input", "")), str(record.get("output", ""))
                    )
                if not isinstance(task_id, str) or not messages:
                    continue
                split, composition, _ = _split_for_task(task_id, assignments, source)
                if composition == "unknown" and isinstance(record.get("composition"), str):
                    composition = str(record["composition"])
                provenance = _provenance(
                    source,
                    root,
                    line=line,
                    extra={"source_split": record.get("source_split")},
                )
                extracted = extract_message_boundaries(
                    task_id=task_id,
                    messages=messages,
                    policy_version=_policy_version(record, source),
                    provenance=provenance,
                    composition=composition,
                    project_split=split,
                )
                candidates.extend(extracted)
                inventory["candidates"] += len(extracted)
        elif source.kind == "ab_offline":
            try:
                record = json.loads(source.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            inventory["records_seen"] = 1
            extracted = _extract_ab_file(
                source=source,
                root=root,
                record=record,
                line=1,
                assignments=assignments,
                metadata_index=metadata_index,
            )
            candidates.extend(extracted)
            inventory["candidates"] += len(extracted)

    unique_records, duplicates_removed = _dedupe_candidates(candidates)
    counts_by_boundary = Counter(str(record["boundary_type"]) for record in unique_records)
    counts_by_split = Counter(str(record["project_split"]) for record in unique_records)
    counts_by_composition = Counter(str(record.get("composition", "unknown")) for record in unique_records)
    counts_by_source: Counter[str] = Counter()
    for record in unique_records:
        for provenance in record.get("source_provenance", []):
            counts_by_source[str(provenance.get("source_kind", "unknown"))] += 1
    eval_in_training = sum(
        bool(record["quality_checks"].get("training_eligible"))
        for record in unique_records
        if record["project_split"] == "evaluation"
    )
    unresolved = sorted(
        {
            record["task_id"]
            for record in unique_records
            if record["project_split"] not in ALL_SPLITS
        }
    )
    policy_versions = sorted({str(record["policy_version"]) for record in unique_records})
    source_inventory: dict[str, dict[str, int]] = defaultdict(
        lambda: {"files": 0, "records_seen": 0, "candidates": 0}
    )
    for item in source_stats.values():
        bucket = source_inventory[str(item["source_kind"])]
        bucket["files"] += 1
        bucket["records_seen"] += int(item.get("records_seen", 0))
        bucket["candidates"] += int(item.get("candidates", 0))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "schema_fields": [
            "schema_version",
            "task_id",
            "boundary_type",
            "policy_version",
            "messages",
            "public_state_before",
            "target_assistant",
            "source_provenance",
            "quality_checks",
        ],
        "sources": list(source_stats.values()),
        "source_inventory_summary": {
            key: dict(value) for key, value in sorted(source_inventory.items())
        },
        "counts": {
            "raw_candidates": len(candidates),
            "unique_contexts": len(unique_records),
            "duplicates_removed": duplicates_removed,
            "by_boundary_type": dict(sorted(counts_by_boundary.items())),
            "by_project_split": dict(sorted(counts_by_split.items())),
            "by_composition": dict(sorted(counts_by_composition.items())),
            "by_source_kind": dict(sorted(counts_by_source.items())),
        },
        "policy_versions": policy_versions,
        "deduplication": {
            "key_fields": ["task_id", "boundary_type", "messages", "public_state_before"],
            "algorithm": "sha256(canonical-json)",
            "duplicates_removed": duplicates_removed,
        },
        "split_checks": {
            "assignment_basis": "task_id before extraction",
            "task_map_size": len(assignments),
            "evaluation_contexts_marked_training": eval_in_training,
            "unresolved_task_ids": unresolved,
            "training_contexts": sum(
                bool(record["quality_checks"].get("training_eligible"))
                for record in unique_records
            ),
            "evaluation_contexts": sum(
                record["project_split"] == "evaluation" for record in unique_records
            ),
            "sample_level_random_split": False,
        },
        "target_generation": {"status": "deferred", "target_assistant_is_null": True},
        "hidden_data_policy": {
            "record_level_reward_fields_copied": False,
            "hidden_preference_fields_copied": False,
            "public_messages_only": True,
        },
    }
    return unique_records, manifest


def write_extraction(
    records: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Write derived JSONL and manifest under an ignored output directory."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    contexts_path = directory / "contexts.jsonl"
    manifest_path = directory / "manifest.json"
    with contexts_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    manifest_value = copy.deepcopy(dict(manifest))
    manifest_value["output"] = {
        "contexts": str(contexts_path),
        "manifest": str(manifest_path),
    }
    manifest_path.write_text(
        json.dumps(manifest_value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return contexts_path, manifest_path


__all__ = [
    "ALL_SPLITS",
    "BOUNDARY_TYPES",
    "EVALUATION_SPLITS",
    "GENERATOR_VERSION",
    "HIDDEN_FIELD_NAMES",
    "SCHEMA_VERSION",
    "SourceSpec",
    "discover_sources",
    "events_from_messages",
    "extract_message_boundaries",
    "extract_recovery_boundaries",
    "load_task_split_map",
    "normalize_actor_messages",
    "parse_grpo_transcript",
    "public_state_payload",
    "write_extraction",
]
