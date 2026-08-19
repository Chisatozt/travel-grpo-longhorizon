"""Deterministic conversion from canonical UserBench splits to veRL rows."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from travel_grpo.prompts.actor_policy import (
    ACTOR_RUNTIME_POLICY_VERSION,
    ensure_actor_runtime_policy,
)

VERL_DATASET_VERSION = "userbench-verl-grpo-v1"
VERL_RUNTIME_VERSION = "0.8.0"
VERL_DATA_GENERATOR_VERSION = "travel-grpo-verl-data-generator-v1"
VERL_DATA_SOURCE = "userbench_travel"
VERL_ABILITY = "travel_user_alignment"
VERL_AGENT_NAME = "userbench_tool_agent"
ENVIRONMENT_NAME = "TravelGym"
EXPECTED_SPLIT_COUNTS = {"train": 1723, "validation": 132}
CANONICAL_COLUMNS = (
    "task_id",
    "composition",
    "difficulty",
    "source_split",
    "prompt",
)


class GRPODataError(ValueError):
    """Raised when canonical or derived GRPO data violates its contract."""


# [项目注释] 功能：`_pyarrow`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：RuntimeError。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：返回运行时计算结果；没有显式返回值的分支返回 `None`。
def _pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - exercised by core-only installs.
        raise RuntimeError("GRPO data preparation requires `pip install -e .[data]`") from exc
    return pa, pq


# [项目注释] 功能：`sha256_file`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：sha256, hexdigest, open, iter。
# [项目注释] 输入：`path`: str | Path。
# [项目注释] 输出：标注返回 `str`；具体值由各分支决定。
def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# [项目注释] 功能：`_non_empty`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：strip, GRPODataError, isinstance。
# [项目注释] 输入：`value`: Any；`name`: str。
# [项目注释] 输出：标注返回 `str`；具体值由各分支决定。
def _non_empty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GRPODataError(f"{name} must be a non-empty string")
    return value.strip()


# [项目注释] 功能：`_validate_prompt`：执行输入、状态或产物校验，并在不满足约束时返回诊断或抛出异常。 主要协作调用：enumerate, isinstance, GRPODataError,
# [项目注释]    dict。
# [项目注释] 输入：`prompt`: Any；`task_id`: str。
# [项目注释] 输出：标注返回 `list[dict[str, str]]`；具体值由各分支决定。
def _validate_prompt(prompt: Any, task_id: str) -> list[dict[str, str]]:
    if not isinstance(prompt, Sequence) or isinstance(prompt, (str, bytes)):
        raise GRPODataError(f"task {task_id!r} prompt must be a message sequence")
    messages = [dict(value) for value in prompt if isinstance(value, Mapping)]
    if len(messages) != 2 or [value.get("role") for value in messages] != [
        "system",
        "user",
    ]:
        raise GRPODataError(f"task {task_id!r} prompt roles must be system,user")
    for index, message in enumerate(messages):
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise GRPODataError(
                f"task {task_id!r} prompt message {index} has empty content"
            )
    return [
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in messages
    ]


def build_verl_records(
    source_path: str | Path,
    *,
    project_split: str,
) -> tuple[dict[str, Any], ...]:
    """Load one canonical split and return hidden-label-free veRL 0.8 rows."""

    if project_split not in EXPECTED_SPLIT_COUNTS:
        raise GRPODataError(f"unsupported GRPO project split: {project_split!r}")
    _, pq = _pyarrow()
    source = Path(source_path).resolve()
    try:
        table = pq.read_table(source)
    except Exception as exc:
        raise GRPODataError(f"cannot read canonical GRPO data: {source}") from exc
    if tuple(table.column_names) != CANONICAL_COLUMNS:
        raise GRPODataError(
            f"canonical columns drifted: expected {CANONICAL_COLUMNS}, found {tuple(table.column_names)}"
        )
    expected = EXPECTED_SPLIT_COUNTS[project_split]
    if table.num_rows != expected:
        raise GRPODataError(
            f"{project_split} source count drifted: expected {expected}, found {table.num_rows}"
        )

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row_index, raw in enumerate(table.to_pylist()):
        task_id = _non_empty(raw.get("task_id"), f"row {row_index} task_id")
        if task_id in seen:
            raise GRPODataError(f"duplicate task ID in {project_split}: {task_id!r}")
        seen.add(task_id)
        source_split = _non_empty(raw.get("source_split"), "source_split")
        if source_split != "train":
            raise GRPODataError(
                f"GRPO {project_split} task {task_id!r} must come from official train"
            )
        prompt = ensure_actor_runtime_policy(
            _validate_prompt(raw.get("prompt"), task_id)
        )
        create_kwargs = {"env_name": ENVIRONMENT_NAME, "id": task_id}
        records.append(
            {
                "data_source": VERL_DATA_SOURCE,
                "prompt": prompt,
                "ability": VERL_ABILITY,
                "agent_name": VERL_AGENT_NAME,
                "reward_model": {
                    "style": "rule",
                    "env_name": ENVIRONMENT_NAME,
                    "id": task_id,
                    # The environment loads private labels by task ID.  Keeping
                    # this empty prevents best/correct option leakage through
                    # the rollout dataset.
                    "ground_truth": "",
                },
                "extra_info": {
                    # veRL uses this stable per-split row index to distinguish
                    # prompt trajectories when assigning rollout_n.
                    "index": row_index,
                    "task_id": task_id,
                    "composition": _non_empty(raw.get("composition"), "composition"),
                    "difficulty": _non_empty(raw.get("difficulty"), "difficulty"),
                    "source_split": source_split,
                    "project_split": project_split,
                    "actor_policy_version": ACTOR_RUNTIME_POLICY_VERSION,
                    "need_tools_kwargs": True,
                    "tools_kwargs": {
                        "interact_with_env": {"create_kwargs": create_kwargs}
                    },
                },
            }
        )
    return tuple(records)


# [项目注释] 功能：`_validate_derived_records`：执行输入、状态或产物校验，并在不满足约束时返回诊断或抛出异常。 主要协作调用：set, enumerate, len,
# [项目注释]    GRPODataError。
# [项目注释] 输入：`records`: Sequence[Mapping[str, Any]]；`project_split`: str。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def _validate_derived_records(
    records: Sequence[Mapping[str, Any]], *, project_split: str
) -> None:
    expected = EXPECTED_SPLIT_COUNTS[project_split]
    if len(records) != expected:
        raise GRPODataError(
            f"derived {project_split} count drifted: expected {expected}, found {len(records)}"
        )
    seen: set[str] = set()
    seen_indices: set[int] = set()
    for index, record in enumerate(records):
        if record.get("data_source") != VERL_DATA_SOURCE:
            raise GRPODataError(f"derived row {index} has wrong data_source")
        if record.get("agent_name") != VERL_AGENT_NAME:
            raise GRPODataError(f"derived row {index} has wrong agent_name")
        reward = record.get("reward_model")
        extra = record.get("extra_info")
        if not isinstance(reward, Mapping) or not isinstance(extra, Mapping):
            raise GRPODataError(f"derived row {index} is missing reward_model/extra_info")
        tool = extra.get("tools_kwargs")
        tool = tool.get("interact_with_env") if isinstance(tool, Mapping) else None
        create = tool.get("create_kwargs") if isinstance(tool, Mapping) else None
        task_ids = (
            reward.get("id"),
            extra.get("task_id"),
            create.get("id") if isinstance(create, Mapping) else None,
        )
        if not all(isinstance(value, str) and value for value in task_ids):
            raise GRPODataError(f"derived row {index} has an empty task ID")
        if len(set(task_ids)) != 1:
            raise GRPODataError(f"derived row {index} has mismatched task IDs")
        task_id = str(task_ids[0])
        if task_id in seen:
            raise GRPODataError(f"duplicate derived task ID: {task_id!r}")
        seen.add(task_id)
        row_index = extra.get("index")
        if not isinstance(row_index, int) or isinstance(row_index, bool):
            raise GRPODataError(f"derived task {task_id!r} has invalid extra_info.index")
        if row_index in seen_indices:
            raise GRPODataError(f"duplicate derived extra_info.index: {row_index}")
        seen_indices.add(row_index)
        if reward.get("ground_truth") != "":
            raise GRPODataError(f"derived task {task_id!r} leaks ground_truth")
        if create.get("env_name") != ENVIRONMENT_NAME:
            raise GRPODataError(f"derived task {task_id!r} has wrong env_name")
        prompt = _validate_prompt(record.get("prompt"), task_id)
        if ensure_actor_runtime_policy(prompt) != prompt:
            raise GRPODataError(
                f"derived task {task_id!r} does not contain exactly one current Actor policy"
            )
        if extra.get("actor_policy_version") != ACTOR_RUNTIME_POLICY_VERSION:
            raise GRPODataError(
                f"derived task {task_id!r} has an incompatible Actor policy version"
            )


# [项目注释] 功能：`_atomic_write_parquet`：把内部结果序列化或导出到指定介质，并保持项目约定的格式。 主要协作调用：_pyarrow, mkdir, mkstemp, close。
# [项目注释] 输入：`records`: Sequence[Mapping[str, Any]]；`destination`: Path。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def _atomic_write_parquet(records: Sequence[Mapping[str, Any]], destination: Path) -> None:
    pa, pq = _pyarrow()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        table = pa.Table.from_pylist([dict(value) for value in records])
        pq.write_table(table, temporary, compression="zstd", version="2.6")
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


# [项目注释] 功能：`_atomic_write_json`：把内部结果序列化或导出到指定介质，并保持项目约定的格式。 主要协作调用：mkdir, mkstemp, Path, replace。
# [项目注释] 输入：`document`: Mapping[str, Any]；`destination`: Path。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def _atomic_write_json(document: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(document, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def prepare_verl_datasets(
    *,
    train_source: str | Path,
    validation_source: str | Path,
    output_root: str | Path,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Build both GRPO splits and a hash manifest, or only return a dry-run summary."""

    sources = {
        "train": Path(train_source).resolve(),
        "validation": Path(validation_source).resolve(),
    }
    records = {
        split: build_verl_records(path, project_split=split)
        for split, path in sources.items()
    }
    for split, values in records.items():
        _validate_derived_records(values, project_split=split)
    output = Path(output_root).resolve()
    destinations = {
        "train": output / "train.parquet",
        "validation": output / "validation.parquet",
        "manifest": output / "manifest.json",
    }
    existing = [path for path in destinations.values() if path.exists()]
    if existing and not force and not dry_run:
        raise FileExistsError(f"GRPO data artifact already exists: {existing[0]}")
    summary: dict[str, Any] = {
        "dataset_version": VERL_DATASET_VERSION,
        "dry_run": dry_run,
        "counts": {key: len(value) for key, value in records.items()},
        "sources": {key: str(value) for key, value in sources.items()},
        "output_root": str(output),
        "actor_policy_version": ACTOR_RUNTIME_POLICY_VERSION,
    }
    if dry_run:
        return summary

    for split in ("train", "validation"):
        _atomic_write_parquet(records[split], destinations[split])
    pa, pq = _pyarrow()
    manifest = {
        "dataset_version": VERL_DATASET_VERSION,
        "generator_version": VERL_DATA_GENERATOR_VERSION,
        "verl_version": VERL_RUNTIME_VERSION,
        "data_source": VERL_DATA_SOURCE,
        "agent_name": VERL_AGENT_NAME,
        "actor_policy_version": ACTOR_RUNTIME_POLICY_VERSION,
        "pyarrow_version": pa.__version__,
        "sources": {
            split: {
                "path": str(path),
                "rows": EXPECTED_SPLIT_COUNTS[split],
                "sha256": sha256_file(path),
            }
            for split, path in sources.items()
        },
        "artifacts": {
            split: {
                "path": destinations[split].name,
                "rows": pq.read_metadata(destinations[split]).num_rows,
                "sha256": sha256_file(destinations[split]),
                "schema": str(pq.read_schema(destinations[split])),
            }
            for split in ("train", "validation")
        },
        "hidden_ground_truth_embedded": False,
    }
    _atomic_write_json(manifest, destinations["manifest"])
    verify_verl_datasets(output)
    return {**summary, "manifest": str(destinations["manifest"]), "valid": True}


def verify_verl_datasets(output_root: str | Path) -> dict[str, Any]:
    """Revalidate existing veRL artifacts without rewriting them."""

    _, pq = _pyarrow()
    output = Path(output_root).resolve()
    manifest_path = output / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GRPODataError(f"cannot read GRPO data manifest: {manifest_path}") from exc
    if manifest.get("dataset_version") != VERL_DATASET_VERSION:
        raise GRPODataError("GRPO dataset version drifted")
    if manifest.get("generator_version") != VERL_DATA_GENERATOR_VERSION:
        raise GRPODataError("GRPO data generator version drifted")
    if manifest.get("verl_version") != VERL_RUNTIME_VERSION:
        raise GRPODataError("GRPO veRL version drifted")
    if manifest.get("pyarrow_version") != _pyarrow()[0].__version__:
        raise GRPODataError("GRPO PyArrow version drifted")
    if manifest.get("hidden_ground_truth_embedded") is not False:
        raise GRPODataError("GRPO manifest does not prove hidden-label isolation")
    if manifest.get("actor_policy_version") != ACTOR_RUNTIME_POLICY_VERSION:
        raise GRPODataError("GRPO manifest Actor policy version drifted")
    for split in ("train", "validation"):
        source_entry = manifest.get("sources", {}).get(split, {})
        if source_entry.get("rows") != EXPECTED_SPLIT_COUNTS[split]:
            raise GRPODataError(f"GRPO {split} source row count drifted")
        source = Path(str(source_entry.get("path", "")))
        if not source.is_file() or sha256_file(source) != source_entry.get("sha256"):
            raise GRPODataError(f"GRPO {split} canonical source hash drifted")
        artifact_entry = manifest.get("artifacts", {}).get(split, {})
        if artifact_entry.get("path") != f"{split}.parquet":
            raise GRPODataError(f"GRPO {split} artifact path drifted")
        artifact = output / str(artifact_entry.get("path", ""))
        if not artifact.is_file() or sha256_file(artifact) != artifact_entry.get("sha256"):
            raise GRPODataError(f"GRPO {split} artifact hash drifted")
        table = pq.read_table(artifact)
        records = table.to_pylist()
        _validate_derived_records(records, project_split=split)
        if table.num_rows != artifact_entry.get("rows"):
            raise GRPODataError(f"GRPO {split} manifest row count drifted")
        if str(table.schema) != artifact_entry.get("schema"):
            raise GRPODataError(f"GRPO {split} schema drifted")
    return {
        "dataset_version": VERL_DATASET_VERSION,
        "counts": dict(EXPECTED_SPLIT_COUNTS),
        "manifest": str(manifest_path),
        "valid": True,
    }
