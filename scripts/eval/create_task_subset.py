#!/usr/bin/env python3
"""Create a reproducible composition-stratified UserBench test subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_COLUMNS = ("task_id", "composition", "difficulty", "source_split", "prompt")
DEFAULT_COUNT = 200
DEFAULT_SEED = 47120042


# [项目注释] 功能：`_sha256`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：sha256, hexdigest, open, iter。
# [项目注释] 输入：`path`: Path。
# [项目注释] 输出：标注返回 `str`；具体值由各分支决定。
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# [项目注释] 功能：`_allocate_quotas`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：sum, items, dict, values。
# [项目注释] 输入：`counts`: Counter[str]；`target`: int。
# [项目注释] 输出：标注返回 `dict[str, int]`；具体值由各分支决定。
def _allocate_quotas(counts: Counter[str], target: int) -> dict[str, int]:
    total = sum(counts.values())
    if target <= 0:
        raise ValueError("target count must be positive")
    if target > total:
        raise ValueError(f"target count {target} exceeds source count {total}")

    floors: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for composition, count in counts.items():
        exact = count * target / total
        floors[composition] = int(exact)
        remainders.append((exact - floors[composition], composition))

    remaining = target - sum(floors.values())
    for _, composition in sorted(remainders, key=lambda item: (-item[0], item[1]))[:remaining]:
        floors[composition] += 1
    return dict(sorted(floors.items()))


# [项目注释] 功能：`create_subset`：根据输入配置和中间状态构建或生成新的项目产物。 主要协作调用：read_table, to_pylist, any, Counter。
# [项目注释] 输入：`source`: Path；`output`: Path；`manifest_path`: Path；`count`: int；`seed`: int。
# [项目注释] 输出：标注返回 `dict`；具体值由各分支决定。
def create_subset(
    source: Path,
    output: Path,
    manifest_path: Path,
    *,
    count: int = DEFAULT_COUNT,
    seed: int = DEFAULT_SEED,
) -> dict:
    table = pq.read_table(source)
    if tuple(table.column_names) != EXPECTED_COLUMNS:
        raise ValueError(f"evaluation Parquet schema drift: {table.column_names}")
    rows = table.to_pylist()
    if not rows:
        raise ValueError("source evaluation dataset is empty")
    if any(row.get("source_split") != "test" for row in rows):
        raise ValueError("subset source must contain only official test rows")

    task_ids = [str(row["task_id"]) for row in rows]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("source evaluation task IDs must be unique")

    source_counts = Counter(str(row["composition"]) for row in rows)
    quotas = _allocate_quotas(source_counts, count)
    by_composition: defaultdict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_composition[str(row["composition"])].append(row)

    rng = random.Random(seed)
    selected_ids: set[str] = set()
    for composition, quota in quotas.items():
        candidates = list(by_composition[composition])
        if quota > len(candidates):
            raise ValueError(
                f"composition {composition} needs {quota} tasks, "
                f"but only {len(candidates)} are available"
            )
        rng.shuffle(candidates)
        selected_ids.update(str(row["task_id"]) for row in candidates[:quota])

    selected = [row for row in rows if str(row["task_id"]) in selected_ids]
    selected_counts = Counter(str(row["composition"]) for row in selected)
    if len(selected) != count or dict(selected_counts) != quotas:
        raise RuntimeError(
            f"selected subset mismatch: count={len(selected)}, "
            f"composition={dict(selected_counts)}, quotas={quotas}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(selected, schema=table.schema), output)

    manifest = {
        "schema_version": "travel-evaluation-subset-v1",
        "source": str(source),
        "source_sha256": _sha256(source),
        "source_task_count": len(rows),
        "selected_task_count": len(selected),
        "sampling": "composition-stratified random sampling without replacement",
        "seed": seed,
        "quotas": quotas,
        "counts": dict(selected_counts),
        "task_ids": [str(row["task_id"]) for row in selected],
        "compositions": [str(row["composition"]) for row in selected],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


# [项目注释] 功能：`main`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：ArgumentParser, add_argument, parse_args,
# [项目注释]    create_subset。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `int`；具体值由各分支决定。
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path, default=ROOT / "data/evaluation/tasks.parquet"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    output = args.output or ROOT / f"outputs/evaluation/subsets/tasks_{args.count}_proportional_v1.parquet"
    manifest_path = args.manifest or output.with_suffix(".json")
    manifest = create_subset(
        args.source,
        output,
        manifest_path,
        count=args.count,
        seed=args.seed,
    )
    print(json.dumps({"output": str(output), "manifest": str(manifest_path), **manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
