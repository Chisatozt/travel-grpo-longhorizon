"""Task-pool validation and deterministic SFT sampling plans."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from travel_grpo.training.sft.errors import TeacherCollectionError


def load_teacher_task_pool(
    path: str | Path, *, expected_source_split: str = "train"
) -> tuple[dict[str, Any], ...]:
    """Load the five-field SFT task contract produced by the split pipeline."""

    source = Path(path)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise TeacherCollectionError(
            f"cannot read teacher task pool: {source}"
        ) from exc
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TeacherCollectionError(
                f"invalid JSON in teacher task pool at line {index}"
            ) from exc
        if not isinstance(record, dict):
            raise TeacherCollectionError(f"teacher task line {index} must be an object")
        task_id = record.get("task_id")
        prompt = record.get("prompt")
        if not isinstance(task_id, str) or not task_id:
            raise TeacherCollectionError(f"teacher task line {index} has no task_id")
        if task_id in seen:
            raise TeacherCollectionError(f"duplicate teacher task ID {task_id!r}")
        if not isinstance(prompt, list) or [
            message.get("role") for message in prompt if isinstance(message, dict)
        ] != ["system", "user"]:
            raise TeacherCollectionError(
                f"teacher task {task_id!r} prompt must contain system,user roles"
            )
        for key in ("composition", "difficulty", "source_split"):
            if not isinstance(record.get(key), str) or not record[key]:
                raise TeacherCollectionError(
                    f"teacher task {task_id!r} is missing {key}"
                )
        if record["source_split"] != expected_source_split:
            raise TeacherCollectionError(
                f"task {task_id!r} must originate from official {expected_source_split}"
            )
        seen.add(task_id)
        records.append(record)
    if not records:
        raise TeacherCollectionError("teacher task pool is empty")
    return tuple(records)


def _largest_remainder_quota(
    counts: Mapping[str, int], target: int
) -> dict[str, int]:
    """Allocate an integer target proportionally, deterministically."""

    if target <= 0:
        raise TeacherCollectionError("stratified target must be positive")
    total = sum(int(value) for value in counts.values())
    if total <= 0:
        raise TeacherCollectionError("cannot stratify an empty task pool")
    if target > total:
        raise TeacherCollectionError(
            f"stratified target {target} exceeds task pool size {total}"
        )
    quotas = {
        str(key): target * int(value) // total for key, value in counts.items()
    }
    remainder = target - sum(quotas.values())
    ranked = sorted(
        (str(key) for key in counts),
        key=lambda key: (
            -(target * int(counts[key]) % total),
            key,
        ),
    )
    for key in ranked[:remainder]:
        quotas[key] += 1
    return quotas


def build_stratified_task_plan(
    tasks: Sequence[Mapping[str, Any]],
    *,
    target: int,
    field: str = "composition",
    seed: str = "sft-stratified-v1",
) -> tuple[tuple[dict[str, Any], ...], dict[str, int]]:
    """Build a reproducible, composition-stratified candidate order.

    The returned order is grouped by ``field`` and deterministically shuffled
    within each group.  The quota is the exact largest-remainder allocation
    for ``target`` tasks; rejected tasks remain in the candidate order so a
    later wave can refill the same stratum without changing other strata.
    """

    if not isinstance(field, str) or not field:
        raise TeacherCollectionError("stratification field must be non-empty")
    if not isinstance(seed, str) or not seed:
        raise TeacherCollectionError("stratification seed must be non-empty")
    groups: dict[str, list[dict[str, Any]]] = {}
    for task in tasks:
        value = task.get(field)
        if not isinstance(value, str) or not value:
            raise TeacherCollectionError(
                f"task {task.get('task_id', '<unknown>')!r} is missing stratification field {field!r}"
            )
        task_id = task.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise TeacherCollectionError("stratified task is missing task_id")
        groups.setdefault(value, []).append(dict(task))
    counts = {key: len(value) for key, value in groups.items()}
    quotas = _largest_remainder_quota(counts, target)
    ordered: list[dict[str, Any]] = []
    for value in sorted(groups):
        ordered.extend(
            sorted(
                groups[value],
                key=lambda task: (
                    hashlib.sha256(
                        f"{seed}\0{task['task_id']}".encode("utf-8")
                    ).hexdigest(),
                    str(task["task_id"]),
                ),
            )
        )
    return tuple(ordered), quotas


def select_stratified_task_wave(
    tasks: Sequence[Mapping[str, Any]],
    *,
    quotas: Mapping[str, int],
    attempted_task_ids: set[str],
    accepted_task_ids: set[str],
    field: str = "composition",
    wave_size: int = 32,
) -> tuple[dict[str, Any], ...]:
    """Select the next wave, proportional to each stratum's remaining deficit.

    Accepted means Gold or Silver.  Rejected and infrastructure-invalid tasks
    are only marked attempted, so the next call naturally refills that same
    stratum from its remaining candidates.
    """

    if wave_size <= 0:
        raise TeacherCollectionError("stratified wave size must be positive")
    groups: dict[str, list[dict[str, Any]]] = {}
    accepted_by_stratum: Counter[str] = Counter()
    for task in tasks:
        task_id = str(task["task_id"])
        value = str(task[field])
        if task_id not in attempted_task_ids:
            groups.setdefault(value, []).append(dict(task))
        if task_id in accepted_task_ids:
            accepted_by_stratum[value] += 1
    deficits = {
        value: max(int(quotas.get(value, 0)) - accepted_by_stratum[value], 0)
        for value in quotas
    }
    capacities = {
        value: min(deficits[value], len(groups.get(value, ())))
        for value in deficits
        if deficits[value] > 0 and groups.get(value)
    }
    if not capacities:
        return ()
    total = min(wave_size, sum(capacities.values()))
    allocations = {value: 0 for value in capacities}
    # D'Hondt allocation gives an integer proportional split and handles a
    # stratum whose remaining candidate pool is smaller than its quota.
    for _ in range(total):
        available = [
            value for value in sorted(capacities) if allocations[value] < capacities[value]
        ]
        if not available:
            break
        value = max(
            available,
            key=lambda item: (
                capacities[item] / (allocations[item] + 1),
                capacities[item],
                item,
            ),
        )
        allocations[value] += 1
    selected: list[dict[str, Any]] = []
    for value in sorted(allocations):
        selected.extend(groups[value][: allocations[value]])
    return tuple(selected)


def assert_disjoint_from_evaluation(
    tasks: Sequence[Mapping[str, Any]], evaluation_path: str | Path
) -> None:
    """Reject a task pool that overlaps the frozen evaluation split."""

    evaluation = load_teacher_task_pool(evaluation_path, expected_source_split="test")
    task_ids = {str(task["task_id"]) for task in tasks}
    overlap = task_ids & {str(task["task_id"]) for task in evaluation}
    if overlap:
        example = min(overlap)
        raise TeacherCollectionError(
            f"teacher task pool overlaps frozen evaluation task {example!r}"
        )
