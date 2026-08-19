"""Fixed-denominator evaluation aggregation."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from travel_grpo.evaluation.metrics import RESULT_METRIC_KEYS, result_metrics


# [项目注释] 功能：`_aggregate`：计算奖励、指标或聚合统计，供训练、评测或报告使用。 主要协作调用：Counter, defaultdict, averages, isinstance。
# [项目注释] 输入：`records`: Sequence[Mapping[str, Any]]；`denominator`: int。
# [项目注释] 输出：标注返回 `dict[str, Any]`；具体值由各分支决定。
def _aggregate(records: Sequence[Mapping[str, Any]], denominator: int) -> dict[str, Any]:
    sums: Counter[str] = Counter()
    valid = 0
    terminations: Counter[str] = Counter()
    guard_rejection_reasons: Counter[str] = Counter()
    guard_rejections_total = 0
    tasks_with_guard_rejection = 0
    aspects: dict[str, list[float]] = defaultdict(list)
    for result in records:
        guard_count = result.get("guard_rejections", 0)
        if isinstance(guard_count, int) and not isinstance(guard_count, bool):
            guard_rejections_total += max(0, guard_count)
            tasks_with_guard_rejection += bool(guard_count > 0)
        raw_guard_reasons = result.get("guard_rejection_reasons", {})
        if isinstance(raw_guard_reasons, Mapping):
            for reason, count in raw_guard_reasons.items():
                if isinstance(count, int) and not isinstance(count, bool) and count > 0:
                    guard_rejection_reasons[str(reason)] += count
        metrics = result_metrics(result)
        if metrics:
            valid += 1
            sums.update(metrics)
            reward = result.get("reward", {})
            for aspect, value in reward.get("quality_by_aspect", {}).items():
                aspects[str(aspect)].append(float(value))
        terminations[str(result.get("termination_reason") or "missing")] += 1
    # [项目注释] 功能：`averages`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：pop。
    # [项目注释] 输入：`divisor`: int。
    # [项目注释] 输出：标注返回 `dict[str, float]`；具体值由各分支决定。
    def averages(divisor: int) -> dict[str, float]:
        values = {key: sums[key] / divisor for key in RESULT_METRIC_KEYS}
        values["avg_number_of_1"] = values.pop("number_of_1")
        values["avg_number_of_08"] = values.pop("number_of_08")
        return values

    fixed = averages(denominator)
    return {
        "denominator": denominator,
        "valid_tasks": valid,
        "infrastructure_valid_rate": valid / denominator,
        "fixed_denominator": fixed,
        "valid_only": averages(valid) if valid else averages(1),
        "aspect_option_quality": {key: sum(values) / len(values) for key, values in sorted(aspects.items())},
        "termination_reasons": dict(sorted(terminations.items())),
        "guard_rejections_total": guard_rejections_total,
        "guard_rejections_per_task": (
            guard_rejections_total / denominator if denominator else 0.0
        ),
        "tasks_with_guard_rejection": tasks_with_guard_rejection,
        "guard_rejection_reasons": dict(sorted(guard_rejection_reasons.items())),
    }


# [项目注释] 功能：`summarize_results`：计算奖励、指标或聚合统计，供训练、评测或报告使用。 主要协作调用：tuple, dict, defaultdict, Counter。
# [项目注释] 输入：`records`: Sequence[Mapping[str, Any]]；`expected_task_ids`:
# [项目注释]    Sequence[str]；`expected_compositions`: Sequence[str] | None。
# [项目注释] 输出：标注返回 `dict[str, Any]`；具体值由各分支决定。
def summarize_results(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_task_ids: Sequence[str],
    expected_compositions: Sequence[str] | None = None,
) -> dict[str, Any]:
    expected = tuple(expected_task_ids)
    by_id = {str(value["task_id"]): value for value in records}
    if len(by_id) != len(records):
        raise ValueError("evaluation results contain duplicate task IDs")
    if expected_compositions is not None and len(expected_compositions) != len(expected):
        raise ValueError("expected compositions must align with task IDs")
    compositions_by_id = dict(zip(expected, expected_compositions or ("unknown",) * len(expected), strict=True))
    ordered = [
        by_id.get(
            task_id,
            {
                "task_id": task_id,
                "composition": compositions_by_id[task_id],
                "infrastructure_valid": False,
                "termination_reason": "missing",
            },
        )
        for task_id in expected
    ]
    compositions: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    expected_compositions: Counter[str] = Counter()
    for result in ordered:
        composition = str(result.get("composition", "unknown"))
        compositions[composition].append(result)
        expected_compositions[composition] += 1
    return {
        "schema_version": "travel-evaluation-summary-v1",
        "expected_tasks": len(expected),
        "completed_tasks": len(set(expected) & set(by_id)),
        **_aggregate(ordered, len(expected)),
        "by_composition": {key: _aggregate(values, expected_compositions[key]) for key, values in sorted(compositions.items())},
    }
