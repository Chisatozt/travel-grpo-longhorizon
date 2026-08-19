"""Extract safe decision prefixes from failed Teacher attempts for Stage-1 SFT.

This is intentionally separate from the formal Gold/Silver SFT admission path.
It consumes diagnostic ``partial_trajectory`` records and keeps only attempts
whose last action was an unsuccessful ``answer`` after a successful search.
The failed answer (assistant tool call and its observation) is removed, so the
output teaches the model how to elicit and search without supervising a bad
answer.  The output uses ``userbench-teacher-prefix-v1`` rather than pretending
to be a complete, terminal UserBench trajectory.

The default command reads the current collection diagnostics and writes:

    outputs/teacher_trajectories/sft_stage1_prefix.jsonl
    outputs/teacher_trajectories/sft_stage1_prefix.manifest.json

Example::

    python scripts/train/sft/prepare_stage1_prefix_sft.py

Use ``--keep-all`` to retain multiple qualifying attempts for one task.  The
default deduplicates task IDs and keeps the longest valid prefix.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from travel_grpo.envs.userbench_tools import (  # noqa: E402
    TOOL_NAME,
    UserBenchAction,
    UserBenchActionError,
)
from travel_grpo.training.sft.dataset import PREFIX_SCHEMA_VERSION  # noqa: E402

DEFAULT_DIAGNOSTICS = ROOT / "outputs/teacher_trajectories/sft_train.diagnostics.jsonl"
DEFAULT_OUTPUT = ROOT / "outputs/teacher_trajectories/sft_stage1_prefix.jsonl"
DEFAULT_MANIFEST = ROOT / "outputs/teacher_trajectories/sft_stage1_prefix.manifest.json"

FINAL_ANSWER_FAILURE_PREFIXES = (
    "environment.wrong_answer",
    "environment.answer_not_recorded",
    "environment.answer_not_matching_public_requirement",
)


# [项目注释] 功能：`_read_jsonl`：读取并解析外部数据，将其转换为项目内部可消费的结构。 主要协作调用：enumerate, splitlines, FileNotFoundError, strip。
# [项目注释] 输入：`path`: Path。
# [项目注释] 输出：标注返回 `tuple[list[dict[str, Any]], list[dict[str, Any]]]`；具体值由各分支决定。
def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise FileNotFoundError(f"cannot read diagnostics file: {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(
                {"line": line_number, "reason": "invalid_json", "detail": str(exc)}
            )
            continue
        if not isinstance(value, Mapping):
            errors.append({"line": line_number, "reason": "record_not_mapping"})
            continue
        records.append({"line": line_number, "record": dict(value)})
    return records, errors


def _valid_tool_pairs(messages: Any) -> tuple[bool, str | None]:
    """Validate the retained prefix without checking hidden reward labels."""

    if not isinstance(messages, list) or len(messages) < 4:
        return False, "invalid_prefix_messages"
    if len(messages[2:]) % 2:
        return False, "unpaired_prefix_messages"
    for offset in range(2, len(messages), 2):
        assistant = messages[offset]
        tool = messages[offset + 1]
        if not isinstance(assistant, Mapping) or assistant.get("role") != "assistant":
            return False, "invalid_assistant_message"
        calls = assistant.get("tool_calls")
        if not isinstance(calls, list) or len(calls) != 1:
            return False, "assistant_must_contain_one_tool_call"
        call = calls[0]
        if not isinstance(call, Mapping) or call.get("type") != "function":
            return False, "invalid_tool_call_type"
        call_id = call.get("id")
        function = call.get("function")
        if not isinstance(call_id, str) or not call_id:
            return False, "invalid_tool_call_id"
        if not isinstance(function, Mapping) or function.get("name") != TOOL_NAME:
            return False, "wrong_function_name"
        arguments = function.get("arguments")
        try:
            parameters = json.loads(arguments) if isinstance(arguments, str) else None
        except json.JSONDecodeError:
            parameters = None
        if not isinstance(parameters, Mapping):
            return False, "invalid_tool_arguments_json"
        try:
            UserBenchAction.from_parameters(parameters)
        except UserBenchActionError:
            return False, "invalid_tool_arguments"
        if not isinstance(tool, Mapping) or tool.get("role") != "tool":
            return False, "invalid_tool_message"
        if tool.get("tool_call_id") != call_id or tool.get("name") != TOOL_NAME:
            return False, "tool_call_id_or_name_mismatch"
        if not isinstance(tool.get("content"), str):
            return False, "invalid_tool_message_content"
    return True, None


# [项目注释] 功能：`_action_parameters`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：isinstance, loads, len。
# [项目注释] 输入：`message`: Mapping[str, Any]。
# [项目注释] 输出：标注返回 `Mapping[str, Any] | None`；具体值由各分支决定。
def _action_parameters(message: Mapping[str, Any]) -> Mapping[str, Any] | None:
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        return None
    call = calls[0]
    function = call.get("function") if isinstance(call, Mapping) else None
    arguments = function.get("arguments") if isinstance(function, Mapping) else None
    if not isinstance(arguments, str):
        return None
    try:
        value = json.loads(arguments)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, Mapping) else None


# [项目注释] 功能：`_composition`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：split, join, isdigit。
# [项目注释] 输入：`task_id`: str。
# [项目注释] 输出：标注返回 `str | None`；具体值由各分支决定。
def _composition(task_id: str) -> str | None:
    parts: list[str] = []
    for item in task_id.split("|"):
        try:
            difficulty = item.split(":", 1)[1].split("-", 1)[0]
        except (IndexError, AttributeError):
            return None
        if not difficulty.isdigit():
            return None
        parts.append(difficulty)
    return "".join(parts) if parts else None


# [项目注释] 功能：`_has_final_answer_failure`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：any, isinstance, startswith。
# [项目注释] 输入：`reasons`: Sequence[Any]。
# [项目注释] 输出：标注返回 `bool`；具体值由各分支决定。
def _has_final_answer_failure(reasons: Sequence[Any]) -> bool:
    return any(
        isinstance(reason, str)
        and any(reason.startswith(prefix) for prefix in FINAL_ANSWER_FAILURE_PREFIXES)
        for reason in reasons
    )


# [项目注释] 功能：`_candidate`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：any, _valid_tool_pairs, isinstance,
# [项目注释]    _action_parameters。
# [项目注释] 输入：`envelope`: Mapping[str, Any]；`line_number`: int。
# [项目注释] 输出：标注返回 `tuple[dict[str, Any] | None, str | None]`；具体值由各分支决定。
def _candidate(
    envelope: Mapping[str, Any],
    *,
    line_number: int,
) -> tuple[dict[str, Any] | None, str | None]:
    if envelope.get("accepted") is not False:
        return None, "diagnostic_not_rejected"
    reasons = envelope.get("rejection_reasons")
    if not isinstance(reasons, list) or not _has_final_answer_failure(reasons):
        return None, "not_final_answer_failure"
    # A mixed failure (for example, a simulator fallback plus wrong answer) is
    # not a clean decision prefix and should not be promoted to Stage-1 SFT.
    if any(
        not isinstance(reason, str)
        or not any(reason.startswith(prefix) for prefix in FINAL_ANSWER_FAILURE_PREFIXES)
        for reason in reasons
    ):
        return None, "mixed_failure_reasons"
    partial = envelope.get("partial_trajectory")
    if not isinstance(partial, Mapping):
        return None, "missing_partial_trajectory"
    actions = partial.get("committed_actions")
    if not isinstance(actions, list) or len(actions) < 2:
        return None, "missing_action_prefix"
    failed_action = actions[-1]
    if not isinstance(failed_action, Mapping) or failed_action.get("choice") != "answer":
        return None, "last_action_not_answer"
    if any(
        not isinstance(action, Mapping)
        or not isinstance(action.get("reward"), (int, float))
        or isinstance(action.get("reward"), bool)
        or not math.isfinite(float(action.get("reward")))
        or float(action.get("reward")) <= 0.0
        for action in actions[:-1]
    ):
        return None, "earlier_action_not_successful"
    for field in ("simulator_fallbacks", "simulator_search_fallbacks", "simulator_judgment_fallbacks"):
        if partial.get(field, 0):
            return None, "simulator_fallback"

    failed_aspect = failed_action.get("aspect")
    searches = [
        action
        for action in actions[:-1]
        if isinstance(action, Mapping)
        and action.get("choice") == "search"
        and action.get("aspect") == failed_aspect
        and isinstance(action.get("reward"), (int, float))
        and not isinstance(action.get("reward"), bool)
        and float(action.get("reward")) > 0.0
    ]
    if not searches:
        return None, "no_successful_search_for_failed_aspect"

    messages = partial.get("messages")
    valid, message_error = _valid_tool_pairs(messages)
    if not valid:
        return None, message_error
    assert isinstance(messages, list)
    if len(messages) < 4:
        return None, "missing_prefix_messages"
    final_parameters = _action_parameters(messages[-2])
    if not isinstance(final_parameters, Mapping) or final_parameters.get("choice") != "answer":
        return None, "failed_answer_not_aligned_with_messages"
    prefix_messages = messages[:-2]
    prefix_valid, prefix_error = _valid_tool_pairs(prefix_messages)
    if not prefix_valid:
        return None, prefix_error

    task_id = partial.get("task_id") or envelope.get("task_id")
    if not isinstance(task_id, str) or not task_id.strip():
        return None, "missing_task_id"
    output = {
        "schema_version": PREFIX_SCHEMA_VERSION,
        "source_schema_version": envelope.get("schema_version"),
        "task_id": task_id,
        "composition": _composition(task_id),
        "source_split": "train",
        "attempt": envelope.get("attempt"),
        "attempt_strategy": envelope.get("attempt_strategy"),
        "source_diagnostic_line": line_number,
        "source_failure_reasons": list(reasons),
        "prefix_action_count": len(actions) - 1,
        "prefix_environment_steps": partial.get("environment_steps_completed"),
        "retained_action_evidence": [
            {
                "choice": action.get("choice"),
                "aspect": action.get("aspect"),
                "environment_turn": action.get("environment_turn"),
                "reward": action.get("reward"),
            }
            for action in actions[:-1]
        ],
        "expected_aspects": partial.get("expected_aspects", []),
        "answered_aspects": partial.get("answered_aspects", []),
        "failed_answer": {
            "aspect": failed_action.get("aspect"),
            "content": failed_action.get("content"),
            "environment_turn": failed_action.get("environment_turn"),
        },
        "messages": prefix_messages,
    }
    return output, None


# [项目注释] 功能：`prepare`：根据输入配置和中间状态构建或生成新的项目产物。 主要协作调用：_read_jsonl, Counter, mkdir, write_text。
# [项目注释] 输入：`diagnostics`: Path；`output`: Path；`manifest`: Path；`keep_all`: bool。
# [项目注释] 输出：标注返回 `dict[str, Any]`；具体值由各分支决定。
def prepare(
    diagnostics: Path,
    output: Path,
    manifest: Path,
    *,
    keep_all: bool = False,
) -> dict[str, Any]:
    envelopes, load_errors = _read_jsonl(diagnostics)
    selected: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    for item in envelopes:
        candidate, reason = _candidate(item["record"], line_number=int(item["line"]))
        if candidate is None:
            skipped[str(reason)] += 1
        else:
            selected.append(candidate)

    if not keep_all:
        by_task: dict[str, dict[str, Any]] = {}
        for candidate in selected:
            task_id = str(candidate["task_id"])
            existing = by_task.get(task_id)
            rank = (
                int(candidate.get("prefix_action_count") or 0),
                candidate.get("attempt_strategy") == "canonical",
                -int(candidate.get("source_diagnostic_line") or 0),
            )
            if existing is None or rank > (
                int(existing.get("prefix_action_count") or 0),
                existing.get("attempt_strategy") == "canonical",
                -int(existing.get("source_diagnostic_line") or 0),
            ):
                by_task[task_id] = candidate
        selected = sorted(by_task.values(), key=lambda value: str(value["task_id"]))

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in selected),
        encoding="utf-8",
    )
    composition_distribution = Counter(
        str(value.get("composition")) for value in selected
    )
    report = {
        "schema_version": PREFIX_SCHEMA_VERSION,
        "source": str(diagnostics),
        "output": str(output),
        "deduplicated_by_task": not keep_all,
        "candidate_prefixes": len(selected),
        "unique_task_ids": len({value["task_id"] for value in selected}),
        "composition_distribution": dict(sorted(composition_distribution.items())),
        "selection_criteria": {
            "last_committed_action": "answer",
            "failure_reasons": list(FINAL_ANSWER_FAILURE_PREFIXES),
            "all_prior_actions_positive_reward": True,
            "successful_search_for_failed_aspect": True,
            "simulator_fallbacks_allowed": False,
            "failed_answer_removed": True,
            "tool_call_pairs_validated": True,
        },
        "skipped": dict(sorted(skipped.items())),
        "load_errors": load_errors,
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


# [项目注释] 功能：`build_parser`：读取并解析外部数据，将其转换为项目内部可消费的结构。 主要协作调用：ArgumentParser, add_argument。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `argparse.ArgumentParser`；具体值由各分支决定。
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--keep-all",
        action="store_true",
        help="retain multiple qualifying attempts for the same task ID",
    )
    return parser


# [项目注释] 功能：`main`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：prepare, print, dumps, vars。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def main() -> None:
    report = prepare(**vars(build_parser().parse_args()))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
