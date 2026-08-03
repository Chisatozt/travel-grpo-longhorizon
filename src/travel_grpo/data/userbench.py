"""Load pinned UserBench Parquet tasks and build disjoint project splits.

The upstream one-choice Parquet files contain environment prompts rather than
supervised assistant trajectories.  The generated records deliberately expose
only the compact task contract demonstrated by ``data/example.jsonl``:
``task_id``, ``composition``, ``difficulty``, ``source_split``, and ``prompt``.
Consequently, the SFT outputs are teacher-collection task pools, not action-only
SFT examples.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 compatibility.
    import tomli as tomllib

SOURCE_COLUMNS = (
    "data_source",
    "prompt",
    "ability",
    "reward_model",
    "extra_info",
)
OUTPUT_COLUMNS = (
    "task_id",
    "composition",
    "difficulty",
    "source_split",
    "prompt",
)
TRAIN_PROJECT_SPLITS = (
    "sft_train",
    "sft_validation",
    "grpo_train",
    "grpo_validation",
)
ALL_PROJECT_SPLITS = TRAIN_PROJECT_SPLITS + ("evaluation",)
ARTIFACT_PATHS = {
    "sft_train": ("sft/tasks_train.jsonl", "sft/tasks_train.parquet"),
    "sft_validation": (
        "sft/tasks_validation.jsonl",
        "sft/tasks_validation.parquet",
    ),
    "grpo_train": ("grpo/train.jsonl", "grpo/train.parquet"),
    "grpo_validation": ("grpo/validation.jsonl", "grpo/validation.parquet"),
    "evaluation": ("evaluation/tasks.jsonl", "evaluation/tasks.parquet"),
}
MANIFEST_NAME = "split_manifest.json"


class DatasetSplitError(ValueError):
    """Raised when source data or generated split artifacts violate a contract."""


@dataclass(frozen=True)
class CompositionSpec:
    """Expected source counts and fixed project quotas for one composition."""

    name: str
    train_count: int
    test_count: int
    quotas: Mapping[str, int]


@dataclass(frozen=True)
class SplitSpec:
    """Complete, versioned configuration for a deterministic split."""

    schema_version: str
    variant: str
    split_version: str
    hash_seed: str
    source_label: str
    upstream_commit: str
    target_ratios: Mapping[str, float]
    compositions: tuple[CompositionSpec, ...]


@dataclass(frozen=True)
class LoadedTaskSet:
    """Validated rows and provenance for one upstream Parquet file."""

    composition: str
    upstream_split: str
    records: tuple[dict[str, Any], ...]
    source_path: str
    source_sha256: str
    schema_signature: str


@dataclass(frozen=True)
class SplitBundle:
    """All five project splits plus the source-derived manifest content."""

    spec: SplitSpec
    records: Mapping[str, tuple[dict[str, Any], ...]]
    manifest_base: Mapping[str, Any]


def _require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on installation mode
        raise RuntimeError(
            "Dataset splitting requires PyArrow. Install the project with "
            "`pip install -e .[data]`."
        ) from exc
    return pa, pq


def _require_string(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise DatasetSplitError(
            f"configuration field {key!r} must be a non-empty string"
        )
    return value


def _require_non_negative_int(mapping: Mapping[str, Any], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise DatasetSplitError(
            f"configuration field {key!r} must be a non-negative integer"
        )
    return value


def load_split_spec(path: str | Path) -> SplitSpec:
    """Read and validate a TOML split configuration."""

    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    names = raw.get("compositions")
    if (
        not isinstance(names, list)
        or not names
        or not all(isinstance(v, str) for v in names)
    ):
        raise DatasetSplitError("compositions must be a non-empty list of strings")
    if len(set(names)) != len(names):
        raise DatasetSplitError("compositions contains duplicates")

    raw_ratios = raw.get("target_ratios")
    if not isinstance(raw_ratios, Mapping) or set(raw_ratios) != set(
        TRAIN_PROJECT_SPLITS
    ):
        raise DatasetSplitError(
            f"target_ratios must contain exactly {', '.join(TRAIN_PROJECT_SPLITS)}"
        )
    ratios = {key: float(raw_ratios[key]) for key in TRAIN_PROJECT_SPLITS}
    if (
        any(value < 0.0 for value in ratios.values())
        or abs(sum(ratios.values()) - 1.0) > 1e-12
    ):
        raise DatasetSplitError("target_ratios must be non-negative and sum to 1.0")

    raw_compositions = raw.get("composition")
    if not isinstance(raw_compositions, Mapping):
        raise DatasetSplitError("configuration is missing [composition.*] tables")

    compositions = []
    for name in names:
        values = raw_compositions.get(name)
        if not isinstance(values, Mapping):
            raise DatasetSplitError(f"configuration is missing [composition.{name!r}]")
        train_count = _require_non_negative_int(values, "train")
        test_count = _require_non_negative_int(values, "test")
        quotas = {
            split: _require_non_negative_int(values, split)
            for split in TRAIN_PROJECT_SPLITS
        }
        if sum(quotas.values()) != train_count:
            raise DatasetSplitError(
                f"composition {name} quotas sum to {sum(quotas.values())}, "
                f"expected train count {train_count}"
            )
        compositions.append(
            CompositionSpec(
                name=name,
                train_count=train_count,
                test_count=test_count,
                quotas=quotas,
            )
        )

    return SplitSpec(
        schema_version=_require_string(raw, "schema_version"),
        variant=_require_string(raw, "variant"),
        split_version=_require_string(raw, "split_version"),
        hash_seed=_require_string(raw, "hash_seed"),
        source_label=_require_string(raw, "source_label").rstrip("/"),
        upstream_commit=_require_string(raw, "upstream_commit"),
        target_ratios=ratios,
        compositions=tuple(compositions),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(record: Mapping[str, Any]) -> str:
    return json.dumps(
        record,
        ensure_ascii=False,
    )


def compute_jsonl_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    """Return the SHA-256 of the canonical JSONL bytes without writing a file."""

    digest = hashlib.sha256()
    for record in records:
        digest.update(_canonical_json(record).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _nested(mapping: Mapping[str, Any], path: Sequence[str], *, context: str) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            joined = ".".join(path)
            raise DatasetSplitError(f"{context} is missing {joined}")
        value = value[key]
    return value


def _validate_source_row(
    row: Mapping[str, Any],
    *,
    task_data: Mapping[str, Any],
    composition: str,
    upstream_split: str,
    source_path: str,
    source_row_index: int,
) -> dict[str, Any]:
    context = f"{source_path} row {source_row_index}"
    if row.get("data_source") != "interact_travelgym":
        raise DatasetSplitError(f"{context} has unexpected data_source")
    if row.get("ability") != "interaction":
        raise DatasetSplitError(f"{context} has unexpected ability")

    prompt = row.get("prompt")
    if not isinstance(prompt, list) or not all(
        isinstance(message, Mapping) for message in prompt
    ):
        raise DatasetSplitError(f"{context} prompt must be a list of message mappings")
    if [message.get("role") for message in prompt] != ["system", "user"]:
        raise DatasetSplitError(f"{context} prompt roles must be exactly system,user")
    if any(
        not isinstance(message, Mapping) or not isinstance(message.get("content"), str)
        for message in prompt
    ):
        raise DatasetSplitError(
            f"{context} prompt messages must contain string content"
        )

    reward_model = row.get("reward_model")
    if not isinstance(reward_model, Mapping):
        raise DatasetSplitError(f"{context} reward_model must be a mapping")
    if (
        reward_model.get("env_name") != "TravelGym"
        or reward_model.get("style") != "rule"
    ):
        raise DatasetSplitError(f"{context} has an unexpected reward_model contract")
    task_id = reward_model.get("id")
    if not isinstance(task_id, str) or not task_id:
        raise DatasetSplitError(f"{context} reward_model.id must be a non-empty string")

    extra_info = row.get("extra_info")
    if not isinstance(extra_info, Mapping):
        raise DatasetSplitError(f"{context} extra_info must be a mapping")
    if extra_info.get("split") != upstream_split:
        raise DatasetSplitError(
            f"{context} extra_info.split={extra_info.get('split')!r}, "
            f"expected {upstream_split!r}"
        )
    tool_task_id = _nested(
        extra_info,
        ("tools_kwargs", "interact_with_env", "create_kwargs", "id"),
        context=context,
    )
    if tool_task_id != task_id:
        raise DatasetSplitError(
            f"{context} task ID does not match tools_kwargs create ID"
        )

    scenario = task_data.get(task_id)
    if not isinstance(scenario, Mapping):
        raise DatasetSplitError(
            f"{context} task ID {task_id!r} is absent from task data"
        )
    dimensions = scenario.get("dimensions")
    if not isinstance(dimensions, list) or not dimensions:
        raise DatasetSplitError(f"{context} task dimensions must be a non-empty list")
    try:
        ground_truth = ast.literal_eval(str(reward_model.get("ground_truth")))
    except (SyntaxError, ValueError) as exc:
        raise DatasetSplitError(
            f"{context} ground_truth is not a Python string list"
        ) from exc
    if not isinstance(ground_truth, list) or not all(
        isinstance(value, str) for value in ground_truth
    ):
        raise DatasetSplitError(f"{context} ground_truth must be a list of strings")
    expected_best = []
    for dimension in dimensions:
        dimension_data = scenario.get(dimension)
        if not isinstance(dimension_data, Mapping) or not isinstance(
            dimension_data.get("best_id"), str
        ):
            raise DatasetSplitError(f"{context} is missing best_id for {dimension!r}")
        expected_best.append(dimension_data["best_id"])
    if len(ground_truth) != len(expected_best) or set(ground_truth) != set(
        expected_best
    ):
        raise DatasetSplitError(
            f"{context} ground_truth does not match the scenario best IDs"
        )

    difficulty = scenario.get("difficulty")
    if not isinstance(difficulty, str) or not difficulty:
        raise DatasetSplitError(
            f"{context} scenario difficulty must be a non-empty string"
        )

    # Keep this insertion order aligned with data/example.jsonl.  The project
    # split is represented by the artifact path; source_split intentionally
    # retains the immutable official UserBench train/test boundary.
    return {
        "task_id": task_id,
        "composition": composition,
        "difficulty": difficulty,
        "source_split": upstream_split,
        "prompt": row["prompt"],
    }


def load_onechoice_tasks(
    source_root: str | Path,
    composition: str,
    upstream_split: str,
    *,
    expected_count: int | None = None,
    source_label: str = "environments/UserBench",
) -> LoadedTaskSet:
    """Load and validate one pinned UserBench one-choice Parquet file."""

    if upstream_split not in {"train", "test"}:
        raise DatasetSplitError("upstream_split must be 'train' or 'test'")
    _, pq = _require_pyarrow()
    source_root = Path(source_root)
    relative = (
        Path("data")
        / f"travel{composition}_multiturn_onechoice"
        / (f"{upstream_split}.parquet")
    )
    source_file = source_root / relative.relative_to("data")
    if not source_file.is_file():
        raise DatasetSplitError(f"missing source Parquet: {source_file}")
    table = pq.read_table(source_file)
    if tuple(table.column_names) != SOURCE_COLUMNS:
        raise DatasetSplitError(
            f"{source_file} columns are {table.column_names}, expected {list(SOURCE_COLUMNS)}"
        )
    if expected_count is not None and table.num_rows != expected_count:
        raise DatasetSplitError(
            f"{source_file} contains {table.num_rows} rows, expected {expected_count}"
        )

    task_data_file = (
        source_root.parent
        / "travelgym"
        / "data"
        / (f"travelgym_data_{composition}.json")
    )
    if not task_data_file.is_file():
        raise DatasetSplitError(f"missing task data JSON: {task_data_file}")
    with task_data_file.open(encoding="utf-8") as handle:
        task_data = json.load(handle)
    if not isinstance(task_data, Mapping):
        raise DatasetSplitError(f"{task_data_file} must contain a JSON object")

    source_path = (Path(source_label) / relative).as_posix()
    records = []
    seen_ids = set()
    for index, row in enumerate(table.to_pylist()):
        record = _validate_source_row(
            row,
            task_data=task_data,
            composition=composition,
            upstream_split=upstream_split,
            source_path=source_path,
            source_row_index=index,
        )
        task_id = record["task_id"]
        if task_id in seen_ids:
            raise DatasetSplitError(
                f"{source_file} contains duplicate task ID {task_id!r}"
            )
        seen_ids.add(task_id)
        records.append(record)

    return LoadedTaskSet(
        composition=composition,
        upstream_split=upstream_split,
        records=tuple(records),
        source_path=source_path,
        source_sha256=_sha256_file(source_file),
        schema_signature=str(table.schema.remove_metadata()),
    )


def _stable_task_key(task_id: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}\0{task_id}".encode("utf-8")).hexdigest()


def _as_output_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {column: record[column] for column in OUTPUT_COLUMNS}


def _pairwise_intersections(
    records: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, int]:
    result = {}
    names = list(ALL_PROJECT_SPLITS)
    id_sets = {name: {row["task_id"] for row in records[name]} for name in names}
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            result[f"{left}:{right}"] = len(id_sets[left] & id_sets[right])
    return result


def _read_embedded_source(source_root: Path, expected_commit: str) -> dict[str, Any]:
    path = source_root.parent / "EMBEDDED_SOURCE.json"
    if not path.is_file():
        raise DatasetSplitError(f"missing embedded source manifest: {path}")
    with path.open(encoding="utf-8") as handle:
        source = json.load(handle)
    if source.get("upstream_commit") != expected_commit:
        raise DatasetSplitError(
            f"UserBench commit is {source.get('upstream_commit')!r}, expected {expected_commit!r}"
        )
    return source


def build_dataset_splits(spec: SplitSpec, source_root: str | Path) -> SplitBundle:
    """Build all disjoint task splits in memory from the pinned upstream files."""

    source_root = Path(source_root)
    embedded_source = _read_embedded_source(source_root, spec.upstream_commit)
    output: dict[str, list[dict[str, Any]]] = {name: [] for name in ALL_PROJECT_SPLITS}
    sources = []
    source_schema: str | None = None
    upstream_train_ids: set[str] = set()
    upstream_test_ids: set[str] = set()
    by_composition: dict[str, dict[str, int]] = {}

    for composition in spec.compositions:
        train = load_onechoice_tasks(
            source_root,
            composition.name,
            "train",
            expected_count=composition.train_count,
            source_label=spec.source_label,
        )
        test = load_onechoice_tasks(
            source_root,
            composition.name,
            "test",
            expected_count=composition.test_count,
            source_label=spec.source_label,
        )
        for loaded in (train, test):
            if source_schema is None:
                source_schema = loaded.schema_signature
            elif loaded.schema_signature != source_schema:
                raise DatasetSplitError(
                    f"source schema drift detected in {loaded.source_path}"
                )
            sources.append(
                {
                    "composition": loaded.composition,
                    "upstream_split": loaded.upstream_split,
                    "path": loaded.source_path,
                    "rows": len(loaded.records),
                    "sha256": loaded.source_sha256,
                }
            )

        train_ids = {record["task_id"] for record in train.records}
        test_ids = {record["task_id"] for record in test.records}
        if upstream_train_ids & train_ids or upstream_test_ids & test_ids:
            raise DatasetSplitError("task IDs overlap across compositions")
        upstream_train_ids.update(train_ids)
        upstream_test_ids.update(test_ids)

        ordered_train = sorted(
            train.records,
            key=lambda row: (
                _stable_task_key(row["task_id"], spec.hash_seed),
                row["task_id"],
            ),
        )
        cursor = 0
        composition_counts = {}
        for project_split in TRAIN_PROJECT_SPLITS:
            count = composition.quotas[project_split]
            selected = ordered_train[cursor : cursor + count]
            cursor += count
            output[project_split].extend(
                _as_output_record(record) for record in selected
            )
            composition_counts[project_split] = len(selected)
        if cursor != len(ordered_train):
            raise DatasetSplitError(
                f"composition {composition.name} consumed {cursor} train tasks, "
                f"expected {len(ordered_train)}"
            )
        output["evaluation"].extend(
            _as_output_record(record) for record in test.records
        )
        composition_counts["evaluation"] = len(test.records)
        by_composition[composition.name] = composition_counts

    if upstream_train_ids & upstream_test_ids:
        raise DatasetSplitError("official UserBench train and test task IDs overlap")
    intersections = _pairwise_intersections(output)
    if any(intersections.values()):
        raise DatasetSplitError(f"project split overlap detected: {intersections}")

    frozen_records = {name: tuple(output[name]) for name in ALL_PROJECT_SPLITS}
    counts = {name: len(frozen_records[name]) for name in ALL_PROJECT_SPLITS}
    manifest_base = {
        "schema_version": "travel-dataset-split-manifest-v2",
        "output_schema": {
            "reference": "data/example.jsonl",
            "columns": list(OUTPUT_COLUMNS),
            "project_split_from_artifact_path": True,
            "source_split_values": ["train", "test"],
        },
        "split_spec": {
            "config_schema_version": spec.schema_version,
            "variant": spec.variant,
            "split_version": spec.split_version,
            "hash_seed": spec.hash_seed,
            "source_label": spec.source_label,
            "target_ratios": dict(spec.target_ratios),
            "compositions": [composition.name for composition in spec.compositions],
            "quotas": {
                composition.name: {
                    **dict(composition.quotas),
                    "evaluation": composition.test_count,
                }
                for composition in spec.compositions
            },
        },
        "upstream": {
            "name": embedded_source.get("name"),
            "url": embedded_source.get("upstream_url"),
            "commit": embedded_source.get("upstream_commit"),
            "license": embedded_source.get("license"),
        },
        "source_files": sources,
        "counts": {
            "total": sum(counts.values()),
            "upstream_train": len(upstream_train_ids),
            "upstream_test": len(upstream_test_ids),
            "project_splits": counts,
            "by_composition": by_composition,
        },
        "checks": {
            "upstream_train_test_overlap": len(upstream_train_ids & upstream_test_ids),
            "project_split_intersections": intersections,
        },
    }
    return SplitBundle(spec=spec, records=frozen_records, manifest_base=manifest_base)


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(_canonical_json(record))
            handle.write("\n")


def _temporary_path(target: Path) -> Path:
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    return Path(raw_path)


def write_dataset_splits(
    bundle: SplitBundle,
    output_root: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Write JSONL, Parquet, and a hash manifest using per-file atomic replacement."""

    pa, pq = _require_pyarrow()
    output_root = Path(output_root)
    targets = [output_root / MANIFEST_NAME]
    for paths in ARTIFACT_PATHS.values():
        targets.extend(output_root / path for path in paths)
    existing = [path for path in targets if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "refusing to overwrite existing split artifacts without force: "
            + ", ".join(str(path) for path in existing)
        )
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)

    temporary: dict[Path, Path] = {}
    artifacts: dict[str, Any] = {}
    try:
        for project_split in ALL_PROJECT_SPLITS:
            records = bundle.records[project_split]
            jsonl_rel, parquet_rel = ARTIFACT_PATHS[project_split]
            jsonl_target = output_root / jsonl_rel
            parquet_target = output_root / parquet_rel
            jsonl_temp = _temporary_path(jsonl_target)
            parquet_temp = _temporary_path(parquet_target)
            temporary[jsonl_target] = jsonl_temp
            temporary[parquet_target] = parquet_temp
            _write_jsonl(jsonl_temp, records)
            expected_jsonl_hash = compute_jsonl_sha256(records)
            actual_jsonl_hash = _sha256_file(jsonl_temp)
            if actual_jsonl_hash != expected_jsonl_hash:
                raise DatasetSplitError(
                    f"canonical JSONL hash mismatch while writing {project_split}"
                )
            table = pa.Table.from_pylist(list(records))
            if tuple(table.column_names) != OUTPUT_COLUMNS:
                raise DatasetSplitError(
                    f"generated {project_split} columns do not match the output contract"
                )
            pq.write_table(
                table,
                parquet_temp,
                compression="zstd",
                use_dictionary=True,
                write_statistics=True,
                version="2.6",
            )
            artifacts[project_split] = {
                "rows": len(records),
                "jsonl": {
                    "path": Path(jsonl_rel).as_posix(),
                    "sha256": actual_jsonl_hash,
                },
                "parquet": {
                    "path": Path(parquet_rel).as_posix(),
                    "sha256": _sha256_file(parquet_temp),
                },
            }

        manifest = dict(bundle.manifest_base)
        manifest["builder"] = {"pyarrow_version": pa.__version__}
        manifest["artifacts"] = artifacts
        manifest_target = output_root / MANIFEST_NAME
        manifest_temp = _temporary_path(manifest_target)
        temporary[manifest_target] = manifest_temp
        with manifest_temp.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")

        for target, temp in temporary.items():
            os.replace(temp, target)
        return manifest
    finally:
        for temp in temporary.values():
            if temp.exists():
                temp.unlink()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetSplitError(
                    f"invalid JSON in {path} line {line_number}"
                ) from exc
            if not isinstance(record, dict):
                raise DatasetSplitError(
                    f"{path} line {line_number} is not a JSON object"
                )
            records.append(record)
    return records


def _safe_artifact_path(output_root: Path, relative: str) -> Path:
    candidate = (output_root / relative).resolve()
    try:
        candidate.relative_to(output_root.resolve())
    except ValueError as exc:
        raise DatasetSplitError(
            f"artifact path escapes output root: {relative!r}"
        ) from exc
    return candidate


def verify_dataset_splits(
    spec: SplitSpec,
    source_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Rebuild the expected split in memory and verify every saved artifact."""

    _, pq = _require_pyarrow()
    output_root = Path(output_root)
    manifest_path = output_root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise DatasetSplitError(f"missing split manifest: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, Mapping):
        raise DatasetSplitError("split manifest must contain a JSON object")

    expected = build_dataset_splits(spec, source_root)
    for key in (
        "schema_version",
        "output_schema",
        "split_spec",
        "upstream",
        "source_files",
        "counts",
        "checks",
    ):
        if manifest.get(key) != expected.manifest_base.get(key):
            raise DatasetSplitError(
                f"manifest field {key!r} does not match source data"
            )

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(ALL_PROJECT_SPLITS):
        raise DatasetSplitError(
            "manifest artifacts do not cover exactly the five project splits"
        )

    verified: dict[str, tuple[dict[str, Any], ...]] = {}
    for project_split in ALL_PROJECT_SPLITS:
        details = artifacts[project_split]
        if not isinstance(details, Mapping):
            raise DatasetSplitError(f"manifest artifact {project_split} is invalid")
        if details.get("rows") != len(expected.records[project_split]):
            raise DatasetSplitError(f"manifest row count mismatch for {project_split}")
        format_records = {}
        for format_name in ("jsonl", "parquet"):
            format_details = details.get(format_name)
            if not isinstance(format_details, Mapping):
                raise DatasetSplitError(
                    f"manifest artifact {project_split}.{format_name} is invalid"
                )
            artifact_path = _safe_artifact_path(
                output_root, str(format_details.get("path"))
            )
            if not artifact_path.is_file():
                raise DatasetSplitError(f"missing split artifact: {artifact_path}")
            if _sha256_file(artifact_path) != format_details.get("sha256"):
                raise DatasetSplitError(f"SHA-256 mismatch for {artifact_path}")
            if format_name == "jsonl":
                rows = _read_jsonl(artifact_path)
            else:
                table = pq.read_table(artifact_path)
                if tuple(table.column_names) != OUTPUT_COLUMNS:
                    raise DatasetSplitError(f"column mismatch in {artifact_path}")
                rows = table.to_pylist()
            format_records[format_name] = rows

        jsonl_canonical = [_canonical_json(row) for row in format_records["jsonl"]]
        parquet_canonical = [_canonical_json(row) for row in format_records["parquet"]]
        expected_canonical = [
            _canonical_json(row) for row in expected.records[project_split]
        ]
        if jsonl_canonical != parquet_canonical:
            raise DatasetSplitError(f"JSONL and Parquet differ for {project_split}")
        if jsonl_canonical != expected_canonical:
            raise DatasetSplitError(
                f"saved records differ from source split for {project_split}"
            )
        verified[project_split] = tuple(format_records["jsonl"])

    intersections = _pairwise_intersections(verified)
    if any(intersections.values()):
        raise DatasetSplitError(f"verified artifacts overlap: {intersections}")
    return {
        "valid": True,
        "manifest": MANIFEST_NAME,
        "counts": {name: len(verified[name]) for name in ALL_PROJECT_SPLITS},
        "total": sum(len(verified[name]) for name in ALL_PROJECT_SPLITS),
        "project_split_intersections": intersections,
    }
