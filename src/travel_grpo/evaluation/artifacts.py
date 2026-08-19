"""Atomic, resumable evaluation artifact storage."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


# [项目注释] 功能：`atomic_json`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：mkdir, mkstemp, replace, exists。
# [项目注释] 输入：`path`: Path；`value`: Any。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


# [项目注释] 功能：`task_path`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：replace。
# [项目注释] 输入：`root`: Path；`task_id`: str。
# [项目注释] 输出：标注返回 `Path`；具体值由各分支决定。
def task_path(root: Path, task_id: str) -> Path:
    safe = task_id.replace(":", "_").replace("/", "_").replace("\\", "_")
    return root / "tasks" / f"{safe}.json"


# [项目注释] 功能：`load_completed`：读取并解析外部数据，将其转换为项目内部可消费的结构。 主要协作调用：is_dir, sorted, loads, glob。
# [项目注释] 输入：`root`: Path。
# [项目注释] 输出：标注返回 `dict[str, dict[str, Any]]`；具体值由各分支决定。
def load_completed(root: Path) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    for path in sorted((root / "tasks").glob("*.json")) if (root / "tasks").is_dir() else ():
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or not isinstance(value.get("task_id"), str):
            raise ValueError(f"invalid task checkpoint: {path}")
        if value["task_id"] in completed:
            raise ValueError(f"duplicate task checkpoint for {value['task_id']}")
        completed[value["task_id"]] = value
    return completed


# [项目注释] 功能：`write_results_jsonl`：把内部结果序列化或导出到指定介质，并保持项目约定的格式。 主要协作调用：mkdir, mkstemp, replace, exists。
# [项目注释] 输入：`path`: Path；`records`: Sequence[Mapping[str, Any]]。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def write_results_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def attach_attempt_history(
    result: dict[str, Any], previous: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Keep compact diagnostics for every explicitly retried task attempt."""

    history = list(previous.get("attempt_history", ())) if previous else []
    if previous and not history:
        history.append(
            {
                "infrastructure_valid": previous.get("infrastructure_valid") is True,
                "actor_attempts": int(previous.get("actor_attempts", 0)),
                "environment_steps": int(previous.get("environment_steps", 0)),
                "termination_reason": previous.get("termination_reason"),
                "infrastructure_errors": list(
                    previous.get("reward", {}).get("infrastructure_errors", ())
                ),
            }
        )
    history.append(
        {
            "infrastructure_valid": result.get("infrastructure_valid") is True,
            "actor_attempts": int(result.get("actor_attempts", 0)),
            "environment_steps": int(result.get("environment_steps", 0)),
            "termination_reason": result.get("termination_reason"),
            "infrastructure_errors": list(
                result.get("reward", {}).get("infrastructure_errors", ())
            ),
        }
    )
    result["attempt_history"] = history
    return result
