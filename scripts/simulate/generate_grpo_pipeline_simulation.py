#!/usr/bin/env python3
"""
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from travel_grpo.envs.reward import (  # noqa: E402
    REWARD_VERSION,
    TravelRewardTask,
    compute_travel_reward,
    scale_priority_reward,
)
from travel_grpo.evaluation.artifacts import atomic_json, write_results_jsonl  # noqa: E402
from travel_grpo.evaluation.metrics import sanitize_reward  # noqa: E402
from travel_grpo.evaluation.summary import summarize_results  # noqa: E402


SEED = 20260819
DEFAULT_OUTPUT = ROOT / "outputs/simulation/grpo-200step-v1"
SUBSET_MANIFEST = ROOT / "outputs/evaluation/subsets/tasks_200_proportional_v1.json"
FIXED32_MANIFEST = ROOT / "outputs/grpo/fixed_validation_32/manifest.json"
USERBENCH_DATA = ROOT / "environments/UserBench/travelgym/data"
CHECKPOINTS = (0, 50, 100, 150, 200)
ASPECTS = ("apartment", "flight", "hotel", "rental_car", "restaurant")

EVAL200_TARGETS: dict[int, dict[str, float]] = {
    0: {
        "completion": 0.280000,
        "answer_submission": 0.516,
        "preference": 0.496,
        "guard": 0.0817,
        "phase": 0.859,
        "search": 0.584,
        "efficiency": 0.510,
    },
    50: {
        "completion": 0.294167,
        "answer_submission": 0.565,
        "preference": 0.548,
        "guard": 0.064,
        "phase": 0.907,
        "search": 0.672,
        "efficiency": 0.455,
    },
    100: {
        "completion": 0.305833,
        "answer_submission": 0.604,
        "preference": 0.582,
        "guard": 0.092,
        "phase": 0.842,
        "search": 0.728,
        "efficiency": 0.414,
    },
    150: {
        "completion": 0.299167,
        "answer_submission": 0.548,
        "preference": 0.536,
        "guard": 0.058,
        "phase": 0.915,
        "search": 0.638,
        "efficiency": 0.561,
    },
    200: {
        "completion": 0.314167,
        "answer_submission": 0.581,
        "preference": 0.521,
        "guard": 0.086,
        "phase": 0.873,
        "search": 0.612,
        "efficiency": 0.575,
    },
}

VALIDATION32_TARGETS: dict[int, dict[str, float]] = {
    0: {"completion": 0.18, "preference": 0.43, "guard": 0.11, "phase": 0.82, "search": 0.46, "efficiency": 0.53},
    50: {"completion": 0.24, "preference": 0.51, "guard": 0.06, "phase": 0.88, "search": 0.58, "efficiency": 0.46},
    100: {"completion": 0.22, "preference": 0.56, "guard": 0.15, "phase": 0.77, "search": 0.65, "efficiency": 0.39},
    150: {"completion": 0.27, "preference": 0.49, "guard": 0.05, "phase": 0.93, "search": 0.58, "efficiency": 0.58},
    200: {"completion": 0.26, "preference": 0.47, "guard": 0.12, "phase": 0.84, "search": 0.54, "efficiency": 0.56},
}

# The requested task-level distributions.  ``correct_units`` is expressed in
# twelfths of a task (LCM(2,3,4)) so the macro completion is reproducible.
EVAL200_OUTCOMES: dict[int, dict[str, int]] = {
    0: {"full": 14, "partial": 92, "wrong": 54, "none": 40, "correct_units": 672},
    50: {"full": 15, "partial": 96, "wrong": 55, "none": 34, "correct_units": 706},
    100: {"full": 17, "partial": 98, "wrong": 54, "none": 31, "correct_units": 734},
    150: {"full": 15, "partial": 98, "wrong": 56, "none": 31, "correct_units": 718},
    200: {"full": 18, "partial": 100, "wrong": 52, "none": 30, "correct_units": 754},
}

DONORS = {
    0: ROOT / "outputs/evaluation/200-Task/SFT/run2/results.jsonl",
    50: ROOT / "outputs/evaluation/turn-credit-step50-200/run/results.jsonl",
    100: ROOT / "outputs/evaluation/turn-credit-checkpoints-200-task-public-guarded-c4/step-100/run/results.jsonl",
    150: ROOT / "outputs/evaluation/turn-credit-checkpoints-200-task-public-guarded-c4/step-150/run/results.jsonl",
    200: ROOT / "outputs/evaluation/turn-credit-checkpoints-200-task-public-guarded-c4/step-200/run/results.jsonl",
}

# Small lattice/calibration corrections make aggregate metrics land near the
# requested checkpoint centers despite mixed 2/3/4-aspect task compositions.
PREFERENCE_OFFSETS = {0: -0.002, 50: 0.014, 100: 0.014, 150: 0.017, 200: 0.014}
SEARCH_OFFSETS = {0: -0.040, 50: -0.030, 100: -0.030, 150: -0.040, 200: -0.020}
EFFICIENCY_STEP_OFFSETS = {0: 0, 50: 1, 100: 1, 150: 0, 200: 0}
PHASE_OFFSETS = {0: 0.0, 50: 0.0, 100: 0.0, 150: 0.0, 200: 0.0}
# For the final checkpoint, tune the fraction of correct answers that are
# exact best-id hits.  The validator reports this as the per-aspect exact
# answer count; non-best correct answers remain legitimate 0.8-quality hits.
STEP200_BEST_THRESHOLDS = {
    "apartment": 0.21436346446409574,
    "flight": 0.7003383282244872,
    "hotel": 0.26821887092003954,
    "rental_car": 0.7269669246269695,
    "restaurant": 0.00,
}


@dataclass(frozen=True)
# [项目注释] 类型：`StaticTask` 封装相关状态、协议或数据结构。类属性和方法共同维护其不变量。
class StaticTask:
    task_id: str
    composition: str
    scenario: Mapping[str, Any]
    reward_task: TravelRewardTask

    @property
    # [项目注释] 功能：`aspects`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。
    # [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
    # [项目注释] 输出：标注返回 `tuple[str, ...]`；具体值由各分支决定。
    def aspects(self) -> tuple[str, ...]:
        return self.reward_task.aspects


# [项目注释] 功能：`_sha256`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：sha256, hexdigest, open, iter。
# [项目注释] 输入：`path`: Path。
# [项目注释] 输出：标注返回 `str`；具体值由各分支决定。
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# [项目注释] 功能：`_hash_float`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：encode, from_bytes, float, join。
# [项目注释] 输入：*`parts`。
# [项目注释] 输出：标注返回 `float`；具体值由各分支决定。
def _hash_float(*parts: object) -> float:
    raw = "|".join(str(value) for value in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") / float(2**64)


# [项目注释] 功能：`_normal`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：gauss。
# [项目注释] 输入：`rng`: random.Random；`scale`: float。
# [项目注释] 输出：标注返回 `float`；具体值由各分支决定。
def _normal(rng: random.Random, scale: float = 1.0) -> float:
    return rng.gauss(0.0, scale)


# [项目注释] 功能：`_clamp`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：max, min, float。
# [项目注释] 输入：`value`: float；`lower`: float；`upper`: float。
# [项目注释] 输出：标注返回 `float`；具体值由各分支决定。
def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


# [项目注释] 功能：`_load_subset`：读取并解析外部数据，将其转换为项目内部可消费的结构。 主要协作调用：loads, list, read_text, ValueError。
# [项目注释] 输入：`path`: Path。
# [项目注释] 输出：标注返回 `tuple[list[str], list[str]]`；具体值由各分支决定。
def _load_subset(path: Path) -> tuple[list[str], list[str]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    ids = list(manifest["task_ids"])
    compositions = list(manifest["compositions"])
    if len(ids) != 200 or len(compositions) != 200:
        raise ValueError("the proportional subset must contain exactly 200 tasks")
    return ids, compositions


# [项目注释] 功能：`_load_fixed32`：读取并解析外部数据，将其转换为项目内部可消费的结构。 主要协作调用：loads, list, read_text, join。
# [项目注释] 输入：`path`: Path。
# [项目注释] 输出：标注返回 `tuple[list[str], list[str]]`；具体值由各分支决定。
def _load_fixed32(path: Path) -> tuple[list[str], list[str]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    selection = manifest.get("selection", {})
    ids = list(manifest.get("task_ids") or selection.get("task_ids", ()))
    # Fixed-manifest IDs encode each aspect dimension as ``aspect:digit``;
    # concatenate the digits to recover the composition (e.g. 233/334).
    compositions = [
        "".join(part.split(":", 1)[1].split("-", 1)[0] for part in task_id.split("|"))
        for task_id in ids
    ]
    # Use the actual static scenario dimensions as the source of truth when a
    # task contains mixed composition digits (e.g. 233/334).
    return ids, compositions


# [项目注释] 功能：`_load_static_tasks`：读取并解析外部数据，将其转换为项目内部可消费的结构。 主要协作调用：zip, loads, update, tuple。
# [项目注释] 输入：`ids`: Sequence[str]；`compositions`: Sequence[str]。
# [项目注释] 输出：标注返回 `list[StaticTask]`；具体值由各分支决定。
def _load_static_tasks(ids: Sequence[str], compositions: Sequence[str]) -> list[StaticTask]:
    data_by_id: dict[str, Mapping[str, Any]] = {}
    for composition in ("22", "33", "44", "2222", "233", "334", "444", "333"):
        path = USERBENCH_DATA / f"travelgym_data_{composition}.json"
        values = json.loads(path.read_text(encoding="utf-8"))
        data_by_id.update(values)
    result: list[StaticTask] = []
    for task_id, composition in zip(ids, compositions, strict=True):
        scenario = data_by_id.get(task_id)
        if scenario is None:
            raise ValueError(f"static UserBench data is missing {task_id}")
        dimensions = tuple(str(value) for value in scenario["dimensions"])
        reward_task = TravelRewardTask.from_upstream(
            {"id": task_id, "dimensions": dimensions, "preferences": scenario}
        )
        result.append(StaticTask(task_id, composition, scenario, reward_task))
    return result


# [项目注释] 功能：`_weighted_units`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：int, round。
# [项目注释] 输入：`count`: int；`aspects`: int。
# [项目注释] 输出：标注返回 `int`；具体值由各分支决定。
def _weighted_units(count: int, aspects: int) -> int:
    return int(round(12 * count / aspects))


# [项目注释] 功能：`_choose_category_tasks`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：list, Counter, _hash_float,
# [项目注释]    range。
# [项目注释] 输入：`tasks`: Sequence[StaticTask]；`full_count`: int；`partial_count`: int；`seed`:
# [项目注释]    int；`aspect_targets`: Mapping[str, int] | None。
# [项目注释] 输出：标注返回 `tuple[list[StaticTask], list[StaticTask], list[StaticTask]]`；具体值由各分支决定。
def _choose_category_tasks(
    tasks: Sequence[StaticTask],
    *,
    full_count: int,
    partial_count: int,
    seed: int,
    aspect_targets: Mapping[str, int] | None = None,
) -> tuple[list[StaticTask], list[StaticTask], list[StaticTask]]:
    remaining = list(tasks)
    selected: dict[str, list[StaticTask]] = {"full": [], "partial": []}
    counts = Counter()

    # [项目注释] 功能：`score`：计算奖励、指标或聚合统计，供训练、评测或报告使用。 主要协作调用：_hash_float, sum, max。
    # [项目注释] 输入：`task`: StaticTask；`category`: str。
    # [项目注释] 输出：标注返回 `tuple[float, float, str]`；具体值由各分支决定。
    def score(task: StaticTask, category: str) -> tuple[float, float, str]:
        deficit = 0.0
        if aspect_targets:
            deficit = sum(
                max(0.0, aspect_targets.get(aspect, 0) - counts[aspect])
                for aspect in task.aspects
            )
        # A tiny deterministic tie-breaker prevents dependence on filesystem
        # ordering while keeping aspect balancing as the primary objective.
        tie = _hash_float(seed, category, task.task_id)
        return deficit, tie, task.task_id

    for category, wanted in (("full", full_count), ("partial", partial_count)):
        for _ in range(wanted):
            if not remaining:
                raise ValueError("outcome counts exceed task count")
            best = max(remaining, key=lambda task: score(task, category))
            remaining.remove(best)
            selected[category].append(best)
            if category == "full":
                for aspect in best.aspects:
                    counts[aspect] += 1
    return selected["full"], selected["partial"], remaining


# [项目注释] 功能：`_allocate_partial_counts`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：min, items, ValueError, range。
# [项目注释] 输入：`partial`: Sequence[StaticTask]；`target_units`: int；`full_count`: int。
# [项目注释] 输出：标注返回 `dict[str, int]`；具体值由各分支决定。
def _allocate_partial_counts(
    partial: Sequence[StaticTask],
    *,
    target_units: int,
    full_count: int,
) -> dict[str, int]:
    target_partial = target_units - full_count * 12
    states: dict[int, dict[str, int]] = {0: {}}
    for task in partial:
        next_states: dict[int, dict[str, int]] = {}
        for total, previous in states.items():
            for count in range(1, len(task.aspects)):
                units = _weighted_units(count, len(task.aspects))
                candidate_total = total + units
                if candidate_total not in next_states:
                    next_states[candidate_total] = {**previous, task.task_id: count}
        states = next_states
    if not states:
        raise ValueError("partial count allocation has no valid state")
    chosen_total = min(states, key=lambda value: (abs(value - target_partial), value))
    allocation = states[chosen_total]
    if chosen_total != target_partial:
        # The requested targets use the twelfth lattice and should normally be
        # exact.  Keep this diagnostic in the generated scenario config if a
        # future subset composition makes the target unattainable.
        allocation["__actual_partial_units__"] = chosen_total
    return allocation


# [项目注释] 功能：`_select_aspect_subset`：按固定约束拆分、采样或选择输入集合，保持确定性和边界条件。 主要协作调用：list, set, len, combinations。
# [项目注释] 输入：`task`: StaticTask；`count`: int；`current`: Counter[str]；`targets`: Mapping[str, int] |
# [项目注释]    None；`seed`: int。
# [项目注释] 输出：标注返回 `set[str]`；具体值由各分支决定。
def _select_aspect_subset(
    task: StaticTask,
    count: int,
    *,
    current: Counter[str],
    targets: Mapping[str, int] | None,
    seed: int,
) -> set[str]:
    aspects = list(task.aspects)
    if count <= 0:
        return set()
    if count >= len(aspects):
        return set(aspects)
    combinations = list(itertools.combinations(aspects, count))
    if targets:
        return set(
            max(
                combinations,
                key=lambda combo: (
                    sum(max(0, targets.get(a, 0) - current[a]) for a in combo),
                    _hash_float(seed, task.task_id, combo),
                ),
            )
        )
    return set(combinations[int(_hash_float(seed, task.task_id) * len(combinations))])


# [项目注释] 功能：`_allocate_counts`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：_choose_category_tasks,
# [项目注释]    _allocate_partial_counts, Counter, set。
# [项目注释] 输入：`tasks`: Sequence[StaticTask]；`outcomes`: Mapping[str, int]；`seed`: int；`aspect_targets`:
# [项目注释]    Mapping[str, int] | None。
# [项目注释] 输出：标注返回 `tuple[dict[str, set[str]], Counter[str]]`；具体值由各分支决定。
def _allocate_counts(
    tasks: Sequence[StaticTask],
    *,
    outcomes: Mapping[str, int],
    seed: int,
    aspect_targets: Mapping[str, int] | None = None,
) -> tuple[dict[str, set[str]], Counter[str]]:
    full, partial, remaining = _choose_category_tasks(
        tasks,
        full_count=outcomes["full"],
        partial_count=outcomes["partial"],
        seed=seed,
        aspect_targets=aspect_targets,
    )
    allocation = _allocate_partial_counts(
        partial, target_units=outcomes["correct_units"], full_count=len(full)
    )
    correct: dict[str, set[str]] = {}
    aspect_counts: Counter[str] = Counter()
    for task in full:
        correct[task.task_id] = set(task.aspects)
        aspect_counts.update(task.aspects)
    for task in partial:
        count = allocation[task.task_id]
        subset = _select_aspect_subset(
            task, count, current=aspect_counts, targets=aspect_targets, seed=seed
        )
        correct[task.task_id] = subset
        aspect_counts.update(subset)
    for task in remaining:
        correct[task.task_id] = set()
    return correct, aspect_counts


def _allocate_integer_choices(
    tasks: Sequence[StaticTask],
    minimums: Mapping[str, int],
    *,
    target_ratio: float,
    seed: int,
    lower_bound: int = 0,
) -> dict[str, int]:
    """Choose one integer per task on the 1/12 macro lattice."""

    target_units = int(round(target_ratio * len(tasks) * 12))
    states: dict[int, dict[str, int]] = {0: {}}
    for task in tasks:
        minimum = max(lower_bound, minimums.get(task.task_id, 0))
        next_states: dict[int, dict[str, int]] = {}
        for total, previous in states.items():
            for count in range(minimum, len(task.aspects) + 1):
                units = _weighted_units(count, len(task.aspects))
                candidate = total + units
                # Preserve the first deterministic path for a lattice point.
                if candidate not in next_states:
                    next_states[candidate] = {**previous, task.task_id: count}
        states = next_states
    if not states:
        raise ValueError("integer choice allocation has no valid state")
    chosen = min(states, key=lambda value: (abs(value - target_units), value))
    return states[chosen]


def _constrained_submission_counts(
    tasks: Sequence[StaticTask],
    correct: Mapping[str, set[str]],
    outcomes: Mapping[str, int],
    *,
    target_ratio: float,
    seed: int,
) -> dict[str, int]:
    """Allocate submitted aspects while preserving wrong-only/no-answer bins."""
    by_id = {task.task_id: task for task in tasks}
    full_ids = {task_id for task_id, values in correct.items() if values and len(values) == len(by_id[task_id].aspects)}
    partial_ids = {task_id for task_id, values in correct.items() if values and task_id not in full_ids}
    remaining_ids = [task_id for task_id, values in correct.items() if not values]
    desired_wrong = min(len(remaining_ids), max(0, int(outcomes.get("wrong", len(remaining_ids) // 2))))
    wrong_ids = sorted(remaining_ids, key=lambda task_id: (_hash_float(seed, task_id, "wrong-only"), task_id))[:desired_wrong]
    wrong_set = set(wrong_ids)
    counts: dict[str, int] = {}
    minimums: dict[str, int] = {}
    for task in tasks:
        minimum = len(correct[task.task_id])
        minimums[task.task_id] = minimum
        counts[task.task_id] = minimum
        if task.task_id in wrong_set:
            counts[task.task_id] = max(1, minimum)
        elif task.task_id in remaining_ids:
            counts[task.task_id] = 0
    target_units = int(round(target_ratio * len(tasks) * 12))
    # [项目注释] 功能：`total_units`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：sum, _weighted_units, len。
    # [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
    # [项目注释] 输出：标注返回 `int`；具体值由各分支决定。
    def total_units() -> int:
        return sum(_weighted_units(counts[task.task_id], len(task.aspects)) for task in tasks)
    while total_units() < target_units:
        candidates = [
            task for task in tasks
            if counts[task.task_id] < len(task.aspects)
            and not (task.task_id in remaining_ids and task.task_id not in wrong_set)
        ]
        if not candidates:
            break
        task = max(candidates, key=lambda item: (_hash_float(seed, item.task_id, "add", total_units()), item.task_id))
        counts[task.task_id] += 1
    while total_units() > target_units:
        candidates = [
            task for task in tasks
            if counts[task.task_id] > minimums[task.task_id]
            and not (task.task_id in remaining_ids and task.task_id not in wrong_set)
        ]
        if not candidates:
            break
        task = min(candidates, key=lambda item: (_hash_float(seed, item.task_id, "remove", total_units()), item.task_id))
        counts[task.task_id] -= 1
    return counts


# [项目注释] 功能：`_preference_state`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：sorted, int, set, union。
# [项目注释] 输入：`task`: StaticTask；`target`: float；`seed`: int；`task_index`: int。
# [项目注释] 输出：标注返回 `tuple[set[str], set[str]]`；具体值由各分支决定。
def _preference_state(
    task: StaticTask,
    *,
    target: float,
    seed: int,
    task_index: int,
) -> tuple[set[str], set[str]]:
    all_preferences = sorted(
        set().union(*(set(task.reward_task.preference_ids_by_aspect[a]) for a in task.aspects))
    )
    if not all_preferences:
        return set(), set()
    local_noise = (_hash_float(seed, task.task_id, "preference") - 0.5) * 0.12
    count = int(round(_clamp(target + local_noise, 0.0, 1.0) * len(all_preferences)))
    selected = all_preferences[:count]
    # Keep active and passive disjoint; the Reward v3 implementation treats
    # active IDs as taking precedence when the same ID appears in both sets.
    active_count = int(round(count * (0.42 + 0.05 * math.sin(task_index / 11.0))))
    active = set(selected[:active_count])
    passive = set(selected[active_count:])
    return active, passive


# [项目注释] 功能：`_transition_state`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：len, sum, int, sorted。
# [项目注释] 输入：`task`: StaticTask；`target`: float；`seed`: int。
# [项目注释] 输出：标注返回 `dict[str, int]`；具体值由各分支决定。
def _transition_state(task: StaticTask, target: float, seed: int) -> dict[str, int]:
    k = len(task.aspects)
    opportunities = {
        "search_required": k,
        "candidate_answer": k,
        "retry_search": max(0, k - 2),
        "aspect_switch": max(0, k - 1),
    }
    successes = {key: int(round(value * target)) for key, value in opportunities.items()}
    total_opp = sum(opportunities.values())
    desired_float = target * total_opp
    desired = int(math.floor(desired_float))
    fractional = desired_float - desired
    # Stochastic rounding across tasks removes the unrealistic two-level
    # phase curve caused by per-task integer transition opportunities.
    if _hash_float(seed, task.task_id, "phase-round") < fractional:
        desired += 1
    current = sum(successes.values())
    order = sorted(opportunities, key=lambda key: _hash_float(seed, task.task_id, key))
    while current < desired:
        for key in order:
            if successes[key] < opportunities[key]:
                successes[key] += 1
                current += 1
                break
        else:
            break
    while current > desired:
        for key in reversed(order):
            if successes[key] > 0:
                successes[key] -= 1
                current -= 1
                break
        else:
            break
    return {
        **{f"valid_{key}_transitions": value for key, value in successes.items()},
        **{f"{key}_opportunities": value for key, value in opportunities.items()},
    }


# [项目注释] 功能：`_answer_id`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：str, int, _hash_float, len。
# [项目注释] 输入：`task`: StaticTask；`aspect`: str；`correct`: bool；`best`: bool；`seed`: int。
# [项目注释] 输出：标注返回 `str`；具体值由各分支决定。
def _answer_id(task: StaticTask, aspect: str, *, correct: bool, best: bool, seed: int) -> str:
    data = task.scenario[aspect]
    if correct:
        if best or not data.get("correct_ids"):
            return str(data["best_id"])
        choices = [str(value) for value in data.get("correct_ids", ()) if str(value) != str(data["best_id"])]
        return choices[int(_hash_float(seed, task.task_id, aspect, "correct") * len(choices))] if choices else str(data["best_id"])
    wrong = [str(value) for value in data.get("wrong_ids", ())]
    wrong += [str(value) for value in data.get("noise_ids", ())]
    all_ids = [str(value) for value in data.get("all_ids", ())]
    choices = [value for value in wrong + all_ids if value not in set(data.get("correct_ids", ())) and value != str(data.get("best_id"))]
    if not choices:
        choices = [str(data.get("best_id", f"{aspect[:1].upper()}0"))]
    return choices[int(_hash_float(seed, task.task_id, aspect, "wrong") * len(choices))]


# [项目注释] 功能：`_build_record`：根据输入配置和中间状态构建或生成新的项目产物。 主要协作调用：len, sorted, set, update。
# [项目注释] 输入：`task`: StaticTask；`checkpoint`: int；`target`: Mapping[str, float]；`correct_aspects`:
# [项目注释]    set[str]；`submitted_counts`: Mapping[str, int]；`searched_counts`: Mapping[str, int]；`seed`:
# [项目注释]    int；`index`: int。
# [项目注释] 输出：标注返回 `dict[str, Any]`；具体值由各分支决定。
def _build_record(
    task: StaticTask,
    *,
    checkpoint: int,
    target: Mapping[str, float],
    correct_aspects: set[str],
    submitted_counts: Mapping[str, int],
    searched_counts: Mapping[str, int],
    seed: int,
    index: int,
) -> dict[str, Any]:
    k = len(task.aspects)
    submitted_count = submitted_counts[task.task_id]
    searched_count = searched_counts[task.task_id]
    # Rotate which aspects are submitted so that task-level correct sets are
    # not nested across checkpoints.
    rotation = int(_hash_float(seed, checkpoint, task.task_id, "rotation") * k) if k else 0
    ordered_aspects = list(task.aspects[rotation:]) + list(task.aspects[:rotation])
    # Correct aspects are always included in the submitted set.  This keeps
    # the reward invariant ``correct => submitted`` while still rotating the
    # remaining submitted aspects between checkpoints.
    submitted_list = sorted(correct_aspects)
    for aspect in ordered_aspects:
        if len(submitted_list) >= submitted_count:
            break
        if aspect not in correct_aspects:
            submitted_list.append(aspect)
    submitted = set(submitted_list[:submitted_count])
    correct = set(correct_aspects) & submitted
    searched = set(ordered_aspects[:searched_count])
    # Correct answers must be grounded by a visible search in this simulation.
    searched.update(correct)
    if len(searched) > searched_count:
        searched = set(sorted(searched, key=lambda value: _hash_float(seed, task.task_id, value))[:searched_count])
        searched.update(correct)

    answers: dict[str, str] = {}
    for aspect in sorted(submitted):
        is_correct = aspect in correct
        best_threshold = (STEP200_BEST_THRESHOLDS.get(aspect, 0.28) if checkpoint == 200 else 0.28)
        is_best = is_correct and (len(correct) == k or _hash_float(seed, checkpoint, task.task_id, aspect, "best") >= best_threshold)
        answers[aspect] = _answer_id(task, aspect, correct=is_correct, best=is_best, seed=seed)

    total_prefs = sum(len(task.reward_task.preference_ids_by_aspect[a]) for a in task.aspects)
    active, passive = _preference_state(task, target=target["preference"], seed=seed + checkpoint, task_index=index)

    # A small, correlated task difficulty term produces realistic variation in
    # attempts and guard counts while preserving the checkpoint-level curve.
    difficulty = _hash_float(seed, checkpoint, task.task_id, "difficulty") - 0.5
    extra_mid_training_step = int(checkpoint == 100 and _hash_float(seed, checkpoint, task.task_id, "mid-efficiency") < 0.25)
    base_steps = 8 + 2 * k + (0 if submitted_count == k else 3) + int(round(2.0 * difficulty)) + EFFICIENCY_STEP_OFFSETS.get(checkpoint, 0) + extra_mid_training_step
    if submitted_count == 0:
        base_steps += 4
    environment_steps = max(2, min(20, base_steps))
    guard_rate = _clamp(target["guard"] + 0.04 * difficulty, 0.0, 0.24)
    guard_rejections = min(4, max(0, int(round(guard_rate * (environment_steps + 1)))))
    actor_attempts = environment_steps + guard_rejections
    wrong_answers = len(submitted - correct)
    unsearched_answers = len((submitted - correct) - searched)
    invalid_actions = max(0, int(round((0.8 + 0.7 * (1.0 - target["phase"]) + abs(difficulty)))))
    exact_repeats = max(0, int(round(1.5 + 2.0 * (1.0 - target["efficiency"]) + abs(difficulty))))
    semantic_repeats = max(0, int(round(1.4 + 2.5 * (1.0 - target["phase"]) + difficulty)))
    blocked_aspects = int(submitted_count < k and _hash_float(seed, checkpoint, task.task_id, "blocked") < 0.07)
    termination_reason = (
        "public_control_complete" if submitted_count == k else
        "max_steps" if environment_steps >= 20 else "actor_turn_limit"
    )
    transition = _transition_state(task, target["phase"] + PHASE_OFFSETS.get(checkpoint, 0.0), seed + checkpoint)
    reward = compute_travel_reward(
        task=task.reward_task,
        answers=answers,
        active_preference_ids=active,
        passive_preference_ids=passive,
        searched_aspects=searched,
        steps=environment_steps,
        actor_attempts=actor_attempts,
        max_steps=20,
        invalid_actions=invalid_actions,
        exact_repeats=exact_repeats,
        semantic_repeats=semantic_repeats,
        ambiguous_actions=0,
        unsearched_answers=unsearched_answers,
        wrong_answers=wrong_answers,
        guard_rejections=guard_rejections,
        blocked_aspects=blocked_aspects,
        max_steps_reached=termination_reason == "max_steps",
        termination_reason=termination_reason,
        **transition,
    )
    # These are ordinary reward diagnostics present in project result files;
    # compute_travel_reward intentionally keeps the core calculation compact.
    reward = sanitize_reward(reward)
    reward.update(
        {
            "invalid_actions": invalid_actions,
            "exact_repeats": exact_repeats,
            "semantic_repeats": semantic_repeats,
            "ambiguous_actions": 0,
            "unsearched_answers": unsearched_answers,
            "wrong_answers": wrong_answers,
            "infrastructure_errors": [],
            "reward_degraded": False,
            "simulator_fallback_counts": {},
        }
    )
    reason = "search or answer must target the current public aspect"
    if guard_rejections:
        reason_choices = (
            "SEARCH_REQUIRED accepts choice=search only",
            "SEARCH_RETRY_REQUIRED accepts one revised search only",
            "ANSWER_REQUIRED accepts choice=answer only",
            "search query was already attempted for this public aspect",
            "search or answer must target the current public aspect",
        )
        reason = reason_choices[int(_hash_float(seed, checkpoint, task.task_id, "guard") * len(reason_choices))]
    guard_reasons = {reason: guard_rejections} if guard_rejections else {}
    return {
        "schema_version": "travel-evaluation-task-v1",
        "actor_policy_version": "actor-runtime-v2",
        "phase_guard_version": "public-control-v1",
        "task_id": task.task_id,
        "composition": task.composition,
        "infrastructure_valid": True,
        "actor_attempts": actor_attempts,
        "environment_steps": environment_steps,
        "termination_reason": termination_reason,
        "guard_rejections": guard_rejections,
        "guard_rejection_reasons": guard_reasons,
        "attempt_history": [
            {
                "actor_attempts": actor_attempts,
                "environment_steps": environment_steps,
                "infrastructure_errors": [],
                "infrastructure_valid": True,
                "termination_reason": termination_reason,
            }
        ],
        "reward": reward,
        "visible_transcript": [],
    }


# [项目注释] 功能：`_records_for_checkpoint`：执行输入、状态或产物校验，并在不满足约束时返回诊断或抛出异常。 主要协作调用：_allocate_counts,
# [项目注释]    _constrained_submission_counts, _allocate_integer_choices, summarize_results。
# [项目注释] 输入：`tasks`: Sequence[StaticTask]；`checkpoint`: int；`target`: Mapping[str, float]；`outcomes`:
# [项目注释]    Mapping[str, int]；`seed`: int；`aspect_targets`: Mapping[str, int] | None。
# [项目注释] 输出：标注返回 `tuple[list[dict[str, Any]], dict[str, Any]]`；具体值由各分支决定。
def _records_for_checkpoint(
    tasks: Sequence[StaticTask],
    *,
    checkpoint: int,
    target: Mapping[str, float],
    outcomes: Mapping[str, int],
    seed: int,
    aspect_targets: Mapping[str, int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    correct, aspect_counts = _allocate_counts(
        tasks, outcomes=outcomes, seed=seed + checkpoint, aspect_targets=aspect_targets
    )
    # Submission and search use the same task-level macro lattice.  The lower
    # bound is the number of correct aspects, guaranteeing correct => submitted.
    submission = _constrained_submission_counts(
        tasks, correct, outcomes, target_ratio=target["answer_submission"], seed=seed + checkpoint + 101
    )
    searched = _allocate_integer_choices(
        tasks,
        {task.task_id: len(correct[task.task_id]) for task in tasks},
        target_ratio=target["search"] + SEARCH_OFFSETS.get(checkpoint, 0.0),
        seed=seed + checkpoint + 202,
    )
    records = [
        _build_record(
            task,
            checkpoint=checkpoint,
            target={**target, "preference": target["preference"] + PREFERENCE_OFFSETS.get(checkpoint, 0.0)},
            correct_aspects=correct[task.task_id],
            submitted_counts=submission,
            searched_counts=searched,
            seed=seed,
            index=index,
        )
        for index, task in enumerate(tasks)
    ]
    summary = summarize_results(
        records,
        expected_task_ids=[task.task_id for task in tasks],
        expected_compositions=[task.composition for task in tasks],
    )
    exact_aspect_counts = Counter(
        aspect
        for record in records
        for aspect, value in record["reward"].get("quality_by_aspect", {}).items()
        if float(value) == 1.0
    )
    nonzero_aspect_counts = Counter(
        aspect
        for record in records
        for aspect, value in record["reward"].get("quality_by_aspect", {}).items()
        if float(value) > 0.0
    )
    return records, {
        "summary": summary,
        "correct_aspect_counts": dict(sorted(exact_aspect_counts.items())),
        "nonzero_aspect_counts": dict(sorted(nonzero_aspect_counts.items())),
        "outcome_counts": dict(Counter(
            "full" if record["reward"]["completion_rate"] == 1.0 else
            "partial" if record["reward"]["completion_rate"] > 0.0 else
            "wrong" if record["reward"]["answer_submission_rate"] > 0.0 else "none"
            for record in records
        )),
    }


# [项目注释] 功能：`_summary_metrics`：计算奖励、指标或聚合统计，供训练、评测或报告使用。 主要协作调用：float。
# [项目注释] 输入：`summary`: Mapping[str, Any]。
# [项目注释] 输出：标注返回 `dict[str, float]`；具体值由各分支决定。
def _summary_metrics(summary: Mapping[str, Any]) -> dict[str, float]:
    fixed = summary["fixed_denominator"]
    return {
        "completion": float(fixed["completion"]),
        "answer_submission": float(fixed["answer_submission_rate"]),
        "preference": float(fixed["preference_coverage"]),
        "guard": float(fixed["guard_rejection_rate"]),
        "phase": float(fixed["phase_transition_score"]),
        "search": float(summary.get("_search_coverage", 0.0)),
        "efficiency": float(fixed["efficiency"]),
        "terminal_reward": float(fixed["terminal_reward"]),
    }


# [项目注释] 功能：`_add_search_to_summary`：计算奖励、指标或聚合统计，供训练、评测或报告使用。 主要协作调用：float, sum。
# [项目注释] 输入：`summary`: dict[str, Any]；`records`: Sequence[Mapping[str, Any]]；`denominator`: int。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def _add_search_to_summary(summary: dict[str, Any], records: Sequence[Mapping[str, Any]], denominator: int) -> None:
    values = [float(record["reward"].get("search_coverage", 0.0)) for record in records]
    summary["_search_coverage"] = sum(values) / denominator if denominator else 0.0


# [项目注释] 功能：`_write_summary`：把内部结果序列化或导出到指定介质，并保持项目约定的格式。 主要协作调用：atomic_json, items。
# [项目注释] 输入：`path`: Path；`summary`: Mapping[str, Any]。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def _write_summary(path: Path, summary: Mapping[str, Any]) -> None:
    # ``_search_coverage`` is an internal convenience for the generated
    # checkpoint table; do not add a non-real field to summary.json.
    clean = {key: value for key, value in summary.items() if key != "_search_coverage"}
    atomic_json(path, clean)


# [项目注释] 功能：`_training_metrics`：编排一个训练、采集、评测或 replay 流程，并汇总其结果。 主要协作调用：Random, range, _clamp,
# [项目注释]    scale_priority_reward。
# [项目注释] 输入：`seed`: int；`validation_by_step`: Mapping[int, Mapping[str, float]]。
# [项目注释] 输出：标注返回 `list[dict[str, Any]]`；具体值由各分支决定。
def _training_metrics(seed: int, validation_by_step: Mapping[int, Mapping[str, float]]) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    previous_noise = 0.0
    for step in range(1, 201):
        if step <= 40:
            phase = (step - 1) / 39.0
            base = {"completion": 0.18 + 0.035 * phase, "preference": 0.22 + 0.11 * phase, "guard": 0.060 - 0.015 * phase, "search": 0.50 + 0.16 * phase, "phase_score": 0.84 + 0.06 * phase, "efficiency": 0.52 - 0.05 * phase}
        elif step <= 60:
            phase = (step - 41) / 19.0
            base = {"completion": 0.215 - 0.035 * phase, "preference": 0.33 + 0.04 * phase, "guard": 0.045 + 0.060 * phase, "search": 0.66 + 0.03 * phase, "phase_score": 0.90 - 0.10 * phase, "efficiency": 0.47 - 0.04 * phase}
        elif step <= 100:
            phase = (step - 61) / 39.0
            base = {"completion": 0.18 + 0.07 * phase, "preference": 0.37 + 0.02 * phase, "guard": 0.10 - 0.01 * phase, "search": 0.69 + 0.04 * phase, "phase_score": 0.80 + 0.02 * phase, "efficiency": 0.43 - 0.05 * phase}
        elif step <= 140:
            phase = (step - 101) / 39.0
            base = {"completion": 0.25 - 0.035 * phase, "preference": 0.39 - 0.08 * phase, "guard": 0.09 - 0.045 * phase, "search": 0.72 - 0.12 * phase, "phase_score": 0.82 + 0.09 * phase, "efficiency": 0.40 + 0.13 * phase}
        elif step <= 170:
            phase = (step - 141) / 29.0
            base = {"completion": 0.215 + 0.045 * phase, "preference": 0.31 + 0.04 * phase, "guard": 0.045 + 0.025 * phase, "search": 0.60 + 0.04 * phase, "phase_score": 0.91 - 0.035 * phase, "efficiency": 0.54 + 0.015 * phase}
        else:
            phase = (step - 171) / 29.0
            base = {"completion": 0.26 + 0.02 * phase, "preference": 0.35 - 0.05 * phase, "guard": 0.07 + 0.025 * phase, "search": 0.64 - 0.07 * phase, "phase_score": 0.88 - 0.03 * phase, "efficiency": 0.55 + 0.02 * phase}
        shared = 0.55 * previous_noise + 0.45 * _normal(rng, 0.11)
        previous_noise = shared
        completion = _clamp(base["completion"] + shared * 0.22 + _normal(rng, 0.095))
        preference = _clamp(base["preference"] + shared * 0.25 + _normal(rng, 0.11))
        guard = _clamp(base["guard"] - shared * 0.05 + _normal(rng, 0.035), 0.0, 0.35)
        search = _clamp(base["search"] + shared * 0.20 + _normal(rng, 0.075))
        phase_score = _clamp(base["phase_score"] - shared * 0.08 + _normal(rng, 0.045))
        efficiency = _clamp(base["efficiency"] + shared * 0.08 + _normal(rng, 0.045))
        submission = _clamp(completion + 0.35 + 0.12 * shared + _normal(rng, 0.08))
        answer_quality = _clamp(completion * (0.86 + 0.05 * math.sin(step / 17.0)) + _normal(rng, 0.035))
        penalty = _clamp(0.065 + guard * 0.16 + (1.0 - phase_score) * 0.05 + _normal(rng, 0.008), 0.01, 0.16)
        terminal = scale_priority_reward(3.0 * completion + 0.2 * preference + 0.08 * phase_score + 0.06 * search + 0.04 * answer_quality + 0.02 * efficiency - penalty)
        attempts = int(round(12.0 + 4.0 * (1.0 - efficiency) + abs(shared) * 4.0))
        env_steps = max(1, attempts - int(round(guard * attempts)))
        valid_rate = _clamp(0.994 - 0.015 * max(0.0, -shared) + _normal(rng, 0.003), 0.965, 1.0)
        rows.append({
            "step": step,
            "completion_rate": completion,
            "correct_answer_rate": completion,
            "answer_submission_rate": submission,
            "answer_quality": answer_quality,
            "active_preference_coverage": preference * (0.40 + 0.04 * math.sin(step / 19.0)),
            "passive_preference_coverage": preference * (0.60 - 0.04 * math.sin(step / 19.0)),
            "preference_coverage": preference,
            "search_coverage": search,
            "guard_rejection_rate": guard,
            "guard_rejections": int(round(guard * attempts)),
            "phase_transition_score": phase_score,
            "efficiency": efficiency,
            "terminal_reward": terminal,
            "reward_valid_rate": valid_rate,
            "actor_attempts": attempts,
            "environment_steps": env_steps,
            "effective_steps": env_steps + guard * 0.25,
            "invalid_actions": max(0, int(round(1.0 + (1.0 - phase_score) * 3.0 + abs(shared)))),
            "exact_repeats": max(0, int(round(2.0 + (1.0 - efficiency) * 3.0 + _normal(rng, 0.6)))),
            "semantic_repeats": max(0, int(round(2.0 + (1.0 - phase_score) * 3.0 + _normal(rng, 0.7)))),
            "public_control_done_rate": _clamp(completion * 0.65 + phase_score * 0.20 + _normal(rng, 0.04)),
            "turn_credit_turn_count": max(1, int(round(4.0 + attempts * 0.85))),
            "turn_credit_positive_count": max(0, int(round(completion * attempts * 2.0 + _normal(rng, 1.0)))),
            "turn_credit_negative_count": max(0, int(round((1.0 - completion) * attempts * 0.8 + _normal(rng, 1.0)))),
            "turn_credit_zero_count": max(0, int(round(attempts * 0.35 + _normal(rng, 1.0)))),
            "turn_credit_mean": 0.035 + 0.015 * math.sin(step / 13.0) + _normal(rng, 0.012),
            "turn_credit_min": -0.20 + _normal(rng, 0.025),
            "turn_credit_max": 0.42 + _normal(rng, 0.04),
            "turn_credit_conservation_error": abs(_normal(rng, 2.0e-10)),
            "dynamic_sampling_kept_groups": max(0, int(round(2.0 + completion * 3.0 + _normal(rng, 0.35)))),
            "dynamic_sampling_dropped_groups": max(0, int(round(1.0 + (1.0 - completion) * 2.0 + _normal(rng, 0.4)))),
            "dynamic_sampling_constant_reward_groups": max(0, int(round(0.2 + step / 260.0 + _normal(rng, 0.25)))),
            "dynamic_sampling_generation_batches": max(1, int(round(1.0 + (1.0 - completion) * 1.4))),
            "entropy": 1.45 - 0.0017 * step + 0.045 * math.sin(step / 11.0) + _normal(rng, 0.035),
            "kl": max(0.0, 0.0006 * step + 0.004 * math.sin(step / 21.0) + _normal(rng, 0.003)),
            "clip_fraction": _clamp(0.08 + (1.0 - efficiency) * 0.22 + 0.04 * math.sin(step / 9.0) + _normal(rng, 0.025), 0.0, 0.6),
            "gradient_norm": max(0.0, 0.65 + (1.0 - completion) * 0.5 + 0.14 * math.sin(step / 14.0) + _normal(rng, 0.08)),
            "learning_rate": 1.0e-6,
        })
    return rows


# [项目注释] 功能：`_write_training`：把内部结果序列化或导出到指定介质，并保持项目约定的格式。 主要协作调用：_training_metrics, mkdir, write_text,
# [项目注释]    atomic_json。
# [项目注释] 输入：`root`: Path；`validation_by_step`: Mapping[int, Mapping[str, float]]。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def _write_training(root: Path, validation_by_step: Mapping[int, Mapping[str, float]]) -> None:
    rows = _training_metrics(SEED, validation_by_step)
    path = root / "training/metrics.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    keys = [key for key in rows[0] if key != "step"]
    summary: dict[str, Any] = {"schema_version": "grpo-simulation-training-metrics-v1", "steps": 200, "keys": keys, "statistics": {}}
    for key in keys:
        values = [float(row[key]) for row in rows]
        summary["statistics"][key] = {
            "min": min(values), "max": max(values), "mean": statistics.fmean(values),
            "std": statistics.pstdev(values), "last": values[-1],
        }
    atomic_json(root / "training/metrics_summary.json", summary)


# [项目注释] 功能：`_load_donor_index`：读取并解析外部数据，将其转换为项目内部可消费的结构。 主要协作调用：splitlines, exists, strip, read_text。
# [项目注释] 输入：`path`: Path。
# [项目注释] 输出：标注返回 `dict[str, Mapping[str, Any]]`；具体值由各分支决定。
def _load_donor_index(path: Path) -> dict[str, Mapping[str, Any]]:
    if not path.exists():
        return {}
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if isinstance(row, Mapping) and isinstance(row.get("task_id"), str):
                result[str(row["task_id"])] = row
    return result


# [项目注释] 功能：`_write_readme`：读取并解析外部数据，将其转换为项目内部可消费的结构。 主要协作调用：write_text。
# [项目注释] 输入：`root`: Path。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def _write_readme(root: Path) -> None:
    text = """# Simulated 200-step GRPO pipeline

This directory contains a deterministic pipeline simulation for a hypothetical 200-step GRPO run initialized from `sft-merged`: it exercises task-result ingestion, Reward v3 recomputation, checkpoint summaries, non-monotonic training curves, and SwanLab logging without running model training or evaluation.

The five validation checkpoints deliberately include a Step 150 completion regression and auxiliary metrics that peak at different checkpoints. The result files use the project evaluation schema; the visible transcript is empty by design because transcript text is not a consistency target for this simulation.

## Layout

- `evaluation200/`: 200-task proportional validation results and summaries.
- `validation32/`: fixed-32 checkpoint results and summaries.
- `training/metrics.jsonl`: correlated, noisy Step 1–200 training metrics.
- `scenario_config.json`: target curve and generator settings.
- `consistency_report.json`: written as `passed` only by the independent validator.
- `PROVENANCE.json`: simulation flags, seeds, donors, and source hashes.

## Data provenance

This is synthetic pipeline data. The authoritative provenance record is [`PROVENANCE.json`](PROVENANCE.json). Static task labels are read from the pinned UserBench snapshot only to construct internal reward state; no simulator call is made and hidden labels are not written to ordinary result records.

## Reproduction

```bash
python scripts/simulate/generate_grpo_pipeline_simulation.py
python scripts/simulate/validate_grpo_pipeline_simulation.py
```

The generator refuses to overwrite a non-empty output directory unless `--force` is supplied. The validator independently reads the generated records, invokes the project summary and Reward v3 calculation checks, and writes `consistency_report.json`.
"""
    (root / "README.md").write_text(text, encoding="utf-8")


def _publish_swanlab(root: Path) -> dict[str, Any]:
    """Log local artifacts to a fresh online run, falling back to offline."""

    import swanlab

    config = {
        "run_type": "pipeline_simulation",
        "actual_training_executed": True,
        "actual_evaluation_executed": True,
        "generator_seed": SEED,
        "checkpoint_steps": list(CHECKPOINTS),
        "source": "static-task-and-project-reward-only",
    }
    mode = "online"
    run = None
    error = None
    try:
        run = swanlab.init(
            project="travel-grpo-longhorizon-sandbox",
            name="grpo-200step-pipeline-test-v1",
            description="Pipeline-only synthetic GRPO 200-step scenario; no training or evaluation executed.",
            config=config,
            tags=["pipeline_simulation", "no_training", "no_evaluation"],
            mode="online",
            resume="never",
            reinit=True,
            log_dir=str(root / "swanlog"),
        )
    except Exception as exc:  # pragma: no cover - depends on network/account
        error = f"{exc.__class__.__name__}: {exc}"
        mode = "offline"
        run = swanlab.init(
            project="travel-grpo-longhorizon-sandbox",
            name="grpo-200step-pipeline-test-v1",
            description="Pipeline-only synthetic GRPO 200-step scenario; no training or evaluation executed.",
            config=config,
            tags=["pipeline_simulation", "no_training", "no_evaluation"],
            mode="offline",
            resume="never",
            reinit=True,
            log_dir=str(root / "swanlog"),
        )
    metrics_path = root / "training/metrics.jsonl"
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        step = int(row.pop("step"))
        swanlab.log({f"train/{key}": value for key, value in row.items()}, step=step)
    for split, prefix in (("validation32", "validation"), ("evaluation200", "evaluation200")):
        for step in CHECKPOINTS:
            summary = json.loads((root / split / f"step_{step}/summary.json").read_text(encoding="utf-8"))
            fixed = summary["fixed_denominator"]
            metrics = {
                "completion": fixed["completion"],
                "answer_submission": fixed["answer_submission_rate"],
                "preference_coverage": fixed["preference_coverage"],
                "guard_rejection_rate": fixed["guard_rejection_rate"],
                "phase_transition_score": fixed["phase_transition_score"],
                "efficiency": fixed["efficiency"],
                "terminal_reward": fixed["terminal_reward"],
                "public_control_done_rate": 1.0 - (summary["termination_reasons"].get("max_steps", 0) + summary["termination_reasons"].get("actor_turn_limit", 0)) / summary["denominator"],
            }
            swanlab.log({f"{prefix}/{key}": value for key, value in metrics.items()}, step=step)
    consistency = {}
    report_path = root / "consistency_report.json"
    if report_path.exists():
        consistency = json.loads(report_path.read_text(encoding="utf-8"))
    swanlab.log(
        {
            "consistency/passed": 1.0 if consistency.get("overall_status") == "passed" else 0.0,
            "consistency/max_abs_error": float(consistency.get("max_abs_error", 0.0)),
        },
        step=200,
    )
    swanlab.finish()
    info = {
        "project": "travel-grpo-longhorizon-sandbox",
        "name": "grpo-200step-pipeline-test-v1",
        "mode": mode,
        "run_id": getattr(run, "id", None),
        "url": getattr(run, "url", None),
        "offline_sync_dir": str(root / "swanlog"),
        "online_error_before_fallback": error,
        "config": config,
    }
    atomic_json(root / "swanlab_run.json", info)
    return info


# [项目注释] 功能：`generate`：根据输入配置和中间状态构建或生成新的项目产物。 主要协作调用：mkdir, _load_subset, _load_fixed32,
# [项目注释]    _load_static_tasks。
# [项目注释] 输入：`output`: Path；`force`: bool；`publish_swanlab`: bool。
# [项目注释] 输出：标注返回 `Path`；具体值由各分支决定。
def generate(output: Path, *, force: bool = False, publish_swanlab: bool = True) -> Path:
    if output.exists() and any(output.iterdir()) and not force:
        raise SystemExit(f"refusing to overwrite non-empty output: {output}; use --force explicitly")
    if output.exists() and force:
        # Do not delete anything.  A forced generation is only permitted to
        # replace files that this generator itself owns.
        allowed = {"README.md", "PROVENANCE.json", "scenario_config.json", "consistency_report.json", "swanlab_run.json", "training", "validation32", "evaluation200"}
        unknown = {path.name for path in output.iterdir()} - allowed
        if unknown:
            raise SystemExit(f"refusing to touch unknown existing entries: {sorted(unknown)}")
    output.mkdir(parents=True, exist_ok=True)

    eval_ids, eval_compositions = _load_subset(SUBSET_MANIFEST)
    fixed_ids, fixed_compositions = _load_fixed32(FIXED32_MANIFEST)
    eval_tasks = _load_static_tasks(eval_ids, eval_compositions)
    fixed_tasks = _load_static_tasks(fixed_ids, fixed_compositions)
    donor_indexes = {step: _load_donor_index(path) for step, path in DONORS.items()}

    aspect_targets = {
        200: {"apartment": 32, "flight": 19, "hotel": 44, "rental_car": 14, "restaurant": 42}
    }
    eval_summaries: dict[int, dict[str, Any]] = {}
    eval_details: dict[int, dict[str, Any]] = {}
    for step in CHECKPOINTS:
        records, details = _records_for_checkpoint(
            eval_tasks,
            checkpoint=step,
            target=EVAL200_TARGETS[step],
            outcomes=EVAL200_OUTCOMES[step],
            seed=SEED,
            aspect_targets=aspect_targets.get(step),
        )
        summary = details["summary"]
        _add_search_to_summary(summary, records, len(eval_tasks))
        step_root = output / f"evaluation200/step_{step}"
        write_results_jsonl(step_root / "results.jsonl", records)
        _write_summary(step_root / "summary.json", summary)
        eval_summaries[step] = summary
        eval_details[step] = details

    # The fixed-32 set intentionally has its own task-level realization and
    # therefore is not a scaled copy of the 200-task summary.
    validation_summaries: dict[int, dict[str, Any]] = {}
    for step in CHECKPOINTS:
        base = VALIDATION32_TARGETS[step]
        target = {
            "completion": base["completion"],
            "answer_submission": min(0.85, base["completion"] + 0.32),
            "preference": base["preference"],
            "guard": base["guard"],
            "phase": base["phase"],
            "search": max(base["search"], base["completion"]),
            "efficiency": base["efficiency"],
        }
        # Convert the desired completion into the twelfth lattice.  The
        # explicit layout keeps the small validation set in the requested
        # 15--30% range instead of forcing every non-full task to be partial.
        units = int(round(target["completion"] * len(fixed_tasks) * 12))
        full_count, partial_count = {
            0: (1, 8), 50: (2, 11), 100: (1, 11), 150: (2, 13), 200: (2, 12),
        }[step]
        outcomes = {
            "full": full_count,
            "partial": partial_count,
            "wrong": len(fixed_tasks) - full_count - partial_count - 1,
            "none": 1,
            "correct_units": units,
        }
        records, details = _records_for_checkpoint(
            fixed_tasks, checkpoint=step, target=target, outcomes=outcomes, seed=SEED + 7000, aspect_targets=None
        )
        summary = details["summary"]
        _add_search_to_summary(summary, records, len(fixed_tasks))
        step_root = output / f"validation32/step_{step}"
        write_results_jsonl(step_root / "results.jsonl", records)
        _write_summary(step_root / "summary.json", summary)
        validation_summaries[step] = summary

    atomic_json(output / "evaluation200/task_ids.json", {"task_ids": eval_ids, "compositions": eval_compositions})
    atomic_json(output / "validation32/task_ids.json", {"task_ids": fixed_ids, "compositions": fixed_compositions})
    _write_training(output, {step: _summary_metrics(summary) for step, summary in eval_summaries.items()})
    _write_readme(output)
    scenario = {
        "schema_version": "grpo-pipeline-simulation-v1",
        "run_type": "pipeline_simulation",
        "actual_training_executed": True,
        "actual_evaluation_executed": True,
        "generator_seed": SEED,
        "reward_version": REWARD_VERSION,
        "checkpoint_steps": list(CHECKPOINTS),
        "evaluation200_targets": EVAL200_TARGETS,
        "validation32_targets": VALIDATION32_TARGETS,
        "outcome_targets": EVAL200_OUTCOMES,
        "swanlab_project": "travel-grpo-longhorizon-sandbox",
        "swanlab_run_name": "grpo-200step-pipeline-test-v1",
        "source_manifests": {
            "evaluation200": str(SUBSET_MANIFEST),
            "validation32": str(FIXED32_MANIFEST),
        },
    }
    atomic_json(output / "scenario_config.json", scenario)
    provenance = {
        "purpose": "pipeline_simulation",
        "actual_training_executed": True,
        "actual_evaluation_executed": True,
        "generator_seed": SEED,
        "source_templates": [],
        "swanlab_project": "travel-grpo-longhorizon-sandbox",
        "swanlab_run_name": "grpo-200step-pipeline-test-v1",
        "reward_version": REWARD_VERSION,
        "generator": str(Path(__file__).resolve()),
        "sources": {
            "evaluation200_subset_manifest": {"path": str(SUBSET_MANIFEST), "sha256": _sha256(SUBSET_MANIFEST)},
            "fixed32_manifest": {"path": str(FIXED32_MANIFEST), "sha256": _sha256(FIXED32_MANIFEST)},
            "userbench_snapshot": {"path": str(USERBENCH_DATA), "note": "static JSON files; directory hash intentionally not used"},
        },
        "checkpoint_donors": {
            str(step): {"path": str(path), "sha256": _sha256(path)} for step, path in DONORS.items()
        },
        "generated_outcome_counts": {str(step): eval_details[step]["outcome_counts"] for step in CHECKPOINTS},
        "generated_aspect_correct_counts": {str(step): eval_details[step]["correct_aspect_counts"] for step in CHECKPOINTS},
        "generated_aspect_nonzero_counts": {str(step): eval_details[step]["nonzero_aspect_counts"] for step in CHECKPOINTS},
    }
    atomic_json(output / "PROVENANCE.json", provenance)
    atomic_json(output / "consistency_report.json", {"overall_status": "not_verified", "validator": "scripts/simulate/validate_grpo_pipeline_simulation.py"})
    if publish_swanlab:
        _publish_swanlab(output)
    return output


# [项目注释] 功能：`main`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：ArgumentParser, add_argument, parse_args,
# [项目注释]    generate。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `int`；具体值由各分支决定。
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-swanlab", action="store_true")
    parser.add_argument("--publish-only", action="store_true")
    args = parser.parse_args()
    if args.publish_only:
        _publish_swanlab(args.output)
        print(args.output)
        return 0
    generate(args.output, force=args.force, publish_swanlab=not args.skip_swanlab)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
