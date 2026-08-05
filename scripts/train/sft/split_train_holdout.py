"""Create a deterministic internal SFT validation holdout from train trajectories.

This is intentionally distinct from the repository's frozen ``sft_validation``
split.  It is useful when no separately collected validation trajectories are
available, but it must not be reported as frozen-split or benchmark validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "outputs/teacher_trajectories"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gold-source",
        type=Path,
        default=DEFAULT_OUTPUT / "sft_train.accepted.jsonl",
    )
    parser.add_argument(
        "--silver-source",
        type=Path,
        default=DEFAULT_OUTPUT / "sft_train.silver.jsonl",
    )
    parser.add_argument(
        "--task-source",
        type=Path,
        default=ROOT / "data/sft/tasks_train.parquet",
    )
    parser.add_argument(
        "--validation-gold",
        type=Path,
        default=DEFAULT_OUTPUT / "sft_validation.from_train.accepted.jsonl",
    )
    parser.add_argument(
        "--validation-silver",
        type=Path,
        default=DEFAULT_OUTPUT / "sft_validation.from_train.silver.jsonl",
    )
    parser.add_argument(
        "--train-tasks-output",
        type=Path,
        default=DEFAULT_OUTPUT / "sft_train.from_train_holdout.tasks.parquet",
    )
    parser.add_argument(
        "--validation-tasks-output",
        type=Path,
        default=DEFAULT_OUTPUT / "sft_validation.from_train.tasks.parquet",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_OUTPUT / "sft_validation.from_train.manifest.json",
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=DEFAULT_OUTPUT / "backups/pre_sft_validation_from_train",
    )
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", default="sft-validation-from-train-v1")
    return parser


def _load_jsonl(path: Path, tier: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read trajectory file: {path}") from exc
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"trajectory row must be an object at {path}:{line_number}")
        if not value.get("task_id") or not value.get("composition"):
            raise ValueError(f"trajectory is missing task_id/composition at {path}:{line_number}")
        value["_holdout_tier"] = tier
        value["_original_line"] = line
        records.append(value)
    return records


def _largest_remainder(
    weights: dict[str, int], total: int, capacities: dict[str, int]
) -> dict[str, int]:
    if total < 0 or total > sum(capacities.values()):
        raise ValueError("requested allocation exceeds capacity")
    allocation = {key: 0 for key in weights}
    if total == 0:
        return allocation
    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        raise ValueError("allocation weights must be positive")
    ideals = {key: total * value / weight_sum for key, value in weights.items()}
    for key, ideal in ideals.items():
        allocation[key] = min(capacities[key], math.floor(ideal))
    remaining = total - sum(allocation.values())
    order = sorted(
        weights,
        key=lambda key: (-(ideals[key] - math.floor(ideals[key])), key),
    )
    while remaining:
        eligible = [key for key in order if allocation[key] < capacities[key]]
        if not eligible:
            raise ValueError("cannot complete bounded allocation")
        for key in eligible:
            if not remaining:
                break
            allocation[key] += 1
            remaining -= 1
    return allocation


def _composition_quotas(records: list[dict[str, Any]], count: int) -> dict[str, int]:
    totals = Counter(str(record["composition"]) for record in records)
    if count < len(totals):
        return _largest_remainder(dict(totals), count, dict(totals))
    quotas = {composition: 1 for composition in totals}
    remaining = count - len(quotas)
    residual = {key: value - 1 for key, value in totals.items()}
    extra = _largest_remainder(residual, remaining, residual)
    return {key: quotas[key] + extra[key] for key in quotas}


def _gold_quotas(
    records: list[dict[str, Any]], composition_quotas: dict[str, int], gold_target: int
) -> dict[str, int]:
    by_composition: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        by_composition[str(record["composition"])][str(record["_holdout_tier"])] += 1

    minimum: dict[str, int] = {}
    maximum: dict[str, int] = {}
    ideals: dict[str, float] = {}
    allocation: dict[str, int] = {}
    for composition, quota in composition_quotas.items():
        gold = by_composition[composition]["gold"]
        silver = by_composition[composition]["silver"]
        minimum[composition] = max(0, quota - silver)
        maximum[composition] = min(quota, gold)
        ideals[composition] = quota * gold / (gold + silver)
        allocation[composition] = min(
            maximum[composition], max(minimum[composition], math.floor(ideals[composition]))
        )

    while sum(allocation.values()) < gold_target:
        eligible = [key for key in allocation if allocation[key] < maximum[key]]
        if not eligible:
            raise ValueError("cannot satisfy Gold holdout target")
        key = min(eligible, key=lambda value: (allocation[value] - ideals[value], value))
        allocation[key] += 1
    while sum(allocation.values()) > gold_target:
        eligible = [key for key in allocation if allocation[key] > minimum[key]]
        if not eligible:
            raise ValueError("cannot reduce Gold holdout allocation")
        key = max(eligible, key=lambda value: (allocation[value] - ideals[value], value))
        allocation[key] -= 1
    return allocation


def _rank(seed: str, record: dict[str, Any]) -> str:
    material = f"{seed}\0{record['_holdout_tier']}\0{record['task_id']}".encode()
    return hashlib.sha256(material).hexdigest()


def _select(records: list[dict[str, Any]], count: int, seed: str) -> list[dict[str, Any]]:
    if count <= 0 or count >= len(records):
        raise ValueError("--count must be positive and smaller than the trajectory count")
    compositions = _composition_quotas(records, count)
    tier_totals = Counter(str(record["_holdout_tier"]) for record in records)
    tier_targets = _largest_remainder(
        dict(tier_totals), count, dict(tier_totals)
    )
    gold = _gold_quotas(records, compositions, tier_targets["gold"])

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["composition"]), str(record["_holdout_tier"]))].append(record)
    selected: list[dict[str, Any]] = []
    for composition, quota in sorted(compositions.items()):
        tier_quotas = {"gold": gold[composition], "silver": quota - gold[composition]}
        for tier, tier_quota in tier_quotas.items():
            candidates = sorted(grouped[(composition, tier)], key=lambda row: _rank(seed, row))
            if len(candidates) < tier_quota:
                raise ValueError(f"not enough {composition}/{tier} trajectories")
            selected.extend(candidates[:tier_quota])
    if len(selected) != count:
        raise AssertionError("holdout selection count mismatch")
    return selected


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _write_task_splits(
    source: Path, train_output: Path, validation_output: Path, selected_ids: set[str]
) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("task split writing requires pyarrow") from exc
    table = pq.read_table(source)
    task_ids = [str(value) for value in table.column("task_id").to_pylist()]
    missing = selected_ids - set(task_ids)
    if missing:
        raise ValueError(f"selected task is absent from train task source: {min(missing)!r}")
    validation_mask = pa.array([task_id in selected_ids for task_id in task_ids])
    train_mask = pa.array([task_id not in selected_ids for task_id in task_ids])
    for output, subset in (
        (train_output, table.filter(train_mask)),
        (validation_output, table.filter(validation_mask)),
    ):
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp")
        pq.write_table(subset, temporary)
        os.replace(temporary, output)


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = [
        args.gold_source,
        args.silver_source,
        args.validation_gold,
        args.validation_silver,
        args.train_tasks_output,
        args.validation_tasks_output,
        args.manifest,
    ]
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("input and output paths must be distinct")
    existing_outputs = [
        path
        for path in (
            args.validation_gold,
            args.validation_silver,
            args.train_tasks_output,
            args.validation_tasks_output,
            args.manifest,
            args.backup_dir,
        )
        if path.exists()
    ]
    if existing_outputs:
        raise ValueError(f"refusing to overwrite existing output: {existing_outputs[0]}")

    gold_records = _load_jsonl(args.gold_source, "gold")
    silver_records = _load_jsonl(args.silver_source, "silver")
    records = [*gold_records, *silver_records]
    task_ids = [str(record["task_id"]) for record in records]
    duplicates = [task_id for task_id, amount in Counter(task_ids).items() if amount > 1]
    if duplicates:
        raise ValueError(f"duplicate trajectory task ID: {min(duplicates)!r}")
    selected = _select(records, args.count, args.seed)
    selected_ids = {str(record["task_id"]) for record in selected}

    args.backup_dir.mkdir(parents=True)
    gold_backup = args.backup_dir / args.gold_source.name
    silver_backup = args.backup_dir / args.silver_source.name
    shutil.copy2(args.gold_source, gold_backup)
    shutil.copy2(args.silver_source, silver_backup)

    remaining_gold = [
        str(record["_original_line"])
        for record in gold_records
        if str(record["task_id"]) not in selected_ids
    ]
    remaining_silver = [
        str(record["_original_line"])
        for record in silver_records
        if str(record["task_id"]) not in selected_ids
    ]
    validation_gold = [
        str(record["_original_line"])
        for record in gold_records
        if str(record["task_id"]) in selected_ids
    ]
    validation_silver = [
        str(record["_original_line"])
        for record in silver_records
        if str(record["task_id"]) in selected_ids
    ]
    _atomic_write_lines(args.gold_source, remaining_gold)
    _atomic_write_lines(args.silver_source, remaining_silver)
    _atomic_write_lines(args.validation_gold, validation_gold)
    _atomic_write_lines(args.validation_silver, validation_silver)
    _write_task_splits(
        args.task_source,
        args.train_tasks_output,
        args.validation_tasks_output,
        selected_ids,
    )

    selected_summary = [
        {
            "task_id": str(record["task_id"]),
            "quality_tier": str(record["_holdout_tier"]),
            "composition": str(record["composition"]),
        }
        for record in sorted(selected, key=lambda row: str(row["task_id"]))
    ]
    manifest = {
        "schema_version": "sft-internal-validation-holdout-v1",
        "warning": (
            "Derived from frozen sft_train tasks; this is internal validation and "
            "must not be reported as frozen-split or benchmark validation."
        ),
        "seed": args.seed,
        "selection_strategy": (
            "composition coverage plus largest-remainder allocation; tier-proportional; "
            "SHA-256 task ranking"
        ),
        "source": {
            "gold": str(args.gold_source.resolve()),
            "silver": str(args.silver_source.resolve()),
            "task_pool": str(args.task_source.resolve()),
            "trajectory_count_before": len(records),
            "backup_gold": str(gold_backup.resolve()),
            "backup_silver": str(silver_backup.resolve()),
        },
        "outputs": {
            "train_gold": str(args.gold_source.resolve()),
            "train_silver": str(args.silver_source.resolve()),
            "validation_gold": str(args.validation_gold.resolve()),
            "validation_silver": str(args.validation_silver.resolve()),
            "train_tasks": str(args.train_tasks_output.resolve()),
            "validation_tasks": str(args.validation_tasks_output.resolve()),
        },
        "counts": {
            "train": len(records) - len(selected),
            "validation": len(selected),
            "validation_quality_tiers": dict(
                sorted(Counter(row["quality_tier"] for row in selected_summary).items())
            ),
            "validation_compositions": dict(
                sorted(Counter(row["composition"] for row in selected_summary).items())
            ),
        },
        "selected": selected_summary,
    }
    _atomic_write_json(args.manifest, manifest)
    manifest["sha256"] = {
        "train_gold": _sha256(args.gold_source),
        "train_silver": _sha256(args.silver_source),
        "validation_gold": _sha256(args.validation_gold),
        "validation_silver": _sha256(args.validation_silver),
        "train_tasks": _sha256(args.train_tasks_output),
        "validation_tasks": _sha256(args.validation_tasks_output),
    }
    _atomic_write_json(args.manifest, manifest)
    return manifest


def main() -> None:
    try:
        print(json.dumps(run(build_parser().parse_args()), ensure_ascii=False, indent=2))
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"SFT holdout error: {exc}") from exc


if __name__ == "__main__":
    main()
