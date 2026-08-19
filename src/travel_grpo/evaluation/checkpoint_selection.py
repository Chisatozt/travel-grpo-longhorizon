"""Deterministic GRPO checkpoint selection over the frozen 132-task validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


# [项目注释] 功能：`_fixed`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：ValueError, isinstance。
# [项目注释] 输入：`summary`: Mapping[str, Any]。
# [项目注释] 输出：标注返回 `Mapping[str, Any]`；具体值由各分支决定。
def _fixed(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    value = summary.get("fixed_denominator")
    if not isinstance(value, Mapping) or summary.get("denominator") != 132:
        raise ValueError("checkpoint summary must use the fixed 132-task denominator")
    return value


# [项目注释] 功能：`select_checkpoint`：执行输入、状态或产物校验，并在不满足约束时返回诊断或抛出异常。 主要协作调用：_fixed, max, float, int。
# [项目注释] 输入：`candidates`: Sequence[Mapping[str, Any]]；`sft_summary`: Mapping[str, Any]。
# [项目注释] 输出：标注返回 `dict[str, Any]`；具体值由各分支决定。
def select_checkpoint(candidates: Sequence[Mapping[str, Any]], sft_summary: Mapping[str, Any]) -> dict[str, Any]:
    reference = _fixed(sft_summary)
    audited = []
    eligible = []
    for candidate in candidates:
        metrics = _fixed(candidate["summary"])
        reasons = []
        valid_rate = float(candidate["summary"].get("infrastructure_valid_rate", 0.0))
        if valid_rate < 0.98:
            reasons.append("valid_rate_below_0.98")
        for metric in ("correct_itinerary", "user_aligned_success"):
            if float(metrics.get(metric, 0.0)) < float(reference.get(metric, 0.0)) - 0.01:
                reasons.append(f"{metric}_regressed_more_than_0.01")
        row = {"step": int(candidate["step"]), "summary_path": candidate.get("summary_path"), "eligible": not reasons, "rejection_reasons": reasons, "metrics": dict(metrics), "valid_rate": valid_rate}
        audited.append(row)
        if not reasons:
            eligible.append(row)
    if not eligible:
        return {"schema_version": "travel-checkpoint-selection-v1", "passed": False, "selected_step": None, "candidates": audited}
    selected = max(eligible, key=lambda row: (float(row["metrics"].get("terminal_reward", 0.0)), float(row["metrics"].get("correct_itinerary", 0.0)), float(row["metrics"].get("user_aligned_success", 0.0)), float(row["metrics"].get("efficiency", 0.0)), -row["step"]))
    return {"schema_version": "travel-checkpoint-selection-v1", "passed": True, "selected_step": selected["step"], "selected_summary_path": selected["summary_path"], "candidates": audited}
