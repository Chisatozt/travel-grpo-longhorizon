"""Paired three-stage comparison over one frozen contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from travel_grpo.evaluation.metrics import result_metrics


def compare_stage_results(stages: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    required = ("baseline", "sft", "grpo")
    if tuple(sorted(stages)) != tuple(sorted(required)):
        raise ValueError("comparison requires baseline, sft, and grpo")
    hashes = {value["contract"]["contract_hash"] for value in stages.values()}
    if len(hashes) != 1:
        raise ValueError("stage evaluation contract hashes differ")
    ids = [tuple(value["contract"]["task_ids"]) for value in stages.values()]
    if any(value != ids[0] for value in ids[1:]) or len(ids[0]) != 471:
        raise ValueError("formal comparison requires identical complete 471 task IDs")
    maps = {stage: {row["task_id"]: row for row in value["results"]} for stage, value in stages.items()}
    if any(set(stage_map) != set(ids[0]) for stage_map in maps.values()):
        raise ValueError("all three stages must contain 471 task results")
    pairs = {}
    for left, right in (("baseline", "sft"), ("sft", "grpo")):
        deltas = []
        for task_id in ids[0]:
            l, r = result_metrics(maps[left][task_id]), result_metrics(maps[right][task_id])
            deltas.append({key: r.get(key, 0.0) - l.get(key, 0.0) for key in set(l) | set(r)})
        keys = sorted(set().union(*(value.keys() for value in deltas)))
        pairs[f"{left}_to_{right}"] = {key: sum(value.get(key, 0.0) for value in deltas) / 471 for key in keys}
    return {"schema_version": "travel-stage-comparison-v1", "contract_hash": hashes.pop(), "paired_deltas": pairs}
