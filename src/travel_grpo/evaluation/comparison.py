"""Paired three-stage comparison over one frozen contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from travel_grpo.evaluation.metrics import result_metrics


# [项目注释] 功能：`compare_stage_results`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：len, any, tuple, ValueError。
# [项目注释] 输入：`stages`: Mapping[str, Mapping[str, Any]]；`allow_subset`: bool。
# [项目注释] 输出：标注返回 `dict[str, Any]`；具体值由各分支决定。
def compare_stage_results(
    stages: Mapping[str, Mapping[str, Any]], *, allow_subset: bool = False
) -> dict[str, Any]:
    required = ("baseline", "sft", "grpo")
    if tuple(sorted(stages)) != tuple(sorted(required)):
        raise ValueError("comparison requires baseline, sft, and grpo")
    hashes = {value["contract"]["contract_hash"] for value in stages.values()}
    if len(hashes) != 1:
        raise ValueError("stage evaluation contract hashes differ")
    contracts = [value["contract"] for value in stages.values()]
    modes = {contract.get("evaluation_mode", "formal") for contract in contracts}
    ids = [tuple(contract["task_ids"]) for contract in contracts]
    expected_count = len(ids[0])
    if any(value != ids[0] for value in ids[1:]):
        raise ValueError("stage evaluation task IDs differ")
    if allow_subset:
        if modes != {"subset"} or expected_count <= 0:
            raise ValueError("subset comparison requires three subset contracts")
    elif modes != {"formal"} or expected_count != 471:
        raise ValueError("formal comparison requires identical complete 471 task IDs")
    maps = {stage: {row["task_id"]: row for row in value["results"]} for stage, value in stages.items()}
    if any(set(stage_map) != set(ids[0]) for stage_map in maps.values()):
        raise ValueError(
            f"all three stages must contain {expected_count} task results"
        )
    pairs = {}
    for left, right in (("baseline", "sft"), ("sft", "grpo")):
        deltas = []
        for task_id in ids[0]:
            l, r = result_metrics(maps[left][task_id]), result_metrics(maps[right][task_id])
            deltas.append({key: r.get(key, 0.0) - l.get(key, 0.0) for key in set(l) | set(r)})
        keys = sorted(set().union(*(value.keys() for value in deltas)))
        pairs[f"{left}_to_{right}"] = {
            key: sum(value.get(key, 0.0) for value in deltas) / expected_count
            for key in keys
        }
    return {
        "schema_version": "travel-stage-comparison-v1",
        "evaluation_mode": "subset" if allow_subset else "formal",
        "expected_tasks": expected_count,
        "contract_hash": hashes.pop(),
        "paired_deltas": pairs,
    }
