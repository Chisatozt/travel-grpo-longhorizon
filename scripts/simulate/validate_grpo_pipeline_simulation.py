#!/usr/bin/env python3
"""Independently validate the deterministic GRPO pipeline simulation.

The validator intentionally reads generated records rather than importing the
simulation's private allocation functions.  It re-runs the project summary
aggregation and checks Reward v3's public scalar invariants from the sanitized
records.  A ``passed`` report is written only when every check succeeds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from travel_grpo.envs.reward import REWARD_VERSION, scale_priority_reward  # noqa: E402
from travel_grpo.evaluation.summary import summarize_results  # noqa: E402

CHECKPOINTS = (0, 50, 100, 150, 200)
ASPECT_TARGETS = {"apartment": 32, "flight": 19, "hotel": 44, "rental_car": 14, "restaurant": 42}
EVAL_TARGETS = {
    0: {"completion": .280000, "preference": .496, "guard": .0817, "phase": .859, "search": .584, "efficiency": .510},
    50: {"completion": .294167, "preference": .548, "guard": .064, "phase": .907, "search": .672, "efficiency": .455},
    100: {"completion": .305833, "preference": .582, "guard": .092, "phase": .842, "search": .728, "efficiency": .414},
    150: {"completion": .299167, "preference": .536, "guard": .058, "phase": .915, "search": .638, "efficiency": .561},
    200: {"completion": .314167, "preference": .521, "guard": .086, "phase": .873, "search": .612, "efficiency": .575},
}
FIXED32_RANGES = {
    0: (.15, .20), 50: (.21, .26), 100: (.19, .25), 150: (.24, .29), 200: (.24, .30),
}
OUTCOME_TARGETS = {
    100: {"full": 17, "partial": 98, "wrong": 54, "none": 31},
    150: {"full": 15, "partial": 98, "wrong": 56, "none": 31},
    200: {"full": 18, "partial": 100, "wrong": 52, "none": 30},
}
HIDDEN_KEYS = {"answers", "best_by_aspect", "correct_by_aspect", "searched_aspects", "grounding_by_aspect", "grounded_quality_by_aspect", "active_coverage_by_aspect", "penalty_components"}


# [项目注释] 功能：`_sha256`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：sha256, hexdigest, open, iter。
# [项目注释] 输入：`path`: Path。
# [项目注释] 输出：标注返回 `str`；具体值由各分支决定。
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# [项目注释] 功能：`_read_json`：读取并解析外部数据，将其转换为项目内部可消费的结构。 主要协作调用：loads, read_text。
# [项目注释] 输入：`path`: Path。
# [项目注释] 输出：标注返回 `Any`；具体值由各分支决定。
def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# [项目注释] 功能：`_read_jsonl`：读取并解析外部数据，将其转换为项目内部可消费的结构。 主要协作调用：enumerate, splitlines, strip, loads。
# [项目注释] 输入：`path`: Path。
# [项目注释] 输出：标注返回 `list[dict[str, Any]]`；具体值由各分支决定。
def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


# [项目注释] 功能：`_close`：清理运行时资源或恢复边界状态，保证后续调用不会继承脏状态。 主要协作调用：abs, float。
# [项目注释] 输入：`actual`: float；`expected`: float；`tol`: float。
# [项目注释] 输出：标注返回 `bool`；具体值由各分支决定。
def _close(actual: float, expected: float, tol: float = 1e-9) -> bool:
    return abs(float(actual) - float(expected)) <= tol


# [项目注释] 功能：`_finite`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：isinstance, isfinite, float。
# [项目注释] 输入：`value`: Any。
# [项目注释] 输出：标注返回 `bool`；具体值由各分支决定。
def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


# [项目注释] 功能：`_json_close`：清理运行时资源或恢复边界状态，保证后续调用不会继承脏状态。 主要协作调用：isinstance, enumerate, set, extend。
# [项目注释] 输入：`actual`: Any；`expected`: Any；`path`: str；`tol`: float。
# [项目注释] 输出：标注返回 `list[str]`；具体值由各分支决定。
def _json_close(actual: Any, expected: Any, path: str = "", tol: float = 1e-9) -> list[str]:
    errors: list[str] = []
    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        if set(actual) != set(expected):
            errors.append(f"{path}: keys differ: {sorted(set(actual) ^ set(expected))}")
            return errors
        for key in expected:
            errors.extend(_json_close(actual[key], expected[key], f"{path}/{key}", tol))
        return errors
    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            return [f"{path}: list lengths {len(actual)} != {len(expected)}"]
        for index, (a, e) in enumerate(zip(actual, expected)):
            errors.extend(_json_close(a, e, f"{path}/{index}", tol))
        return errors
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)) and not isinstance(actual, bool) and not isinstance(expected, bool):
        if not _close(float(actual), float(expected), tol):
            errors.append(f"{path}: {actual!r} != {expected!r}")
        return errors
    if actual != expected:
        errors.append(f"{path}: {actual!r} != {expected!r}")
    return errors


# [项目注释] 功能：`_task_aspects`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：tuple, split。
# [项目注释] 输入：`task_id`: str。
# [项目注释] 输出：标注返回 `tuple[str, ...]`；具体值由各分支决定。
def _task_aspects(task_id: str) -> tuple[str, ...]:
    return tuple(part.split(":", 1)[0] for part in task_id.split("|"))


# [项目注释] 功能：`_penalty_from_public_fields`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：len, max, str, int。
# [项目注释] 输入：`record`: Mapping[str, Any]；`reward`: Mapping[str, Any]；`aspects`: Sequence[str]。
# [项目注释] 输出：标注返回 `float`；具体值由各分支决定。
def _penalty_from_public_fields(record: Mapping[str, Any], reward: Mapping[str, Any], aspects: Sequence[str]) -> float:
    k = len(aspects)
    guard = max(0, int(reward.get("guard_rejections", 0)))
    blocked = max(0, int(reward.get("blocked_aspects", 0)))
    invalid = max(0, int(reward.get("invalid_actions", 0)))
    exact = max(0, int(reward.get("exact_repeats", 0)))
    semantic = max(0, int(reward.get("semantic_repeats", 0)))
    ambiguous = max(0, int(reward.get("ambiguous_actions", 0)))
    unsearched = max(0, int(reward.get("unsearched_answers", 0)))
    wrong = max(0, int(reward.get("wrong_answers", 0)))
    termination = str(record.get("termination_reason") or reward.get("termination_reason") or "")
    return (
        0.08 * min(guard, 4) / 4
        + 0.08 * min(blocked, k) / k
        + 0.03 * min(invalid, 4) / 4
        + 0.02 * min(exact, 4) / 4
        + 0.02 * min(semantic, 4) / 4
        + 0.02 * min(ambiguous, k) / k
        + 0.03 * min(unsearched, k) / k
        + 0.04 * min(wrong, k) / k
        + (0.02 if termination == "max_steps" else 0.0)
    )


# [项目注释] 功能：`_validate_record`：执行输入、状态或产物校验，并在不满足约束时返回诊断或抛出异常。 主要协作调用：_task_aspects, float, bool, int。
# [项目注释] 输入：`record`: Mapping[str, Any]；`expected_id`: str。
# [项目注释] 输出：标注返回 `list[str]`；具体值由各分支决定。
def _validate_record(record: Mapping[str, Any], expected_id: str) -> list[str]:
    errors: list[str] = []
    if record.get("task_id") != expected_id:
        errors.append(f"task order/id mismatch: {record.get('task_id')} != {expected_id}")
    if record.get("schema_version") != "travel-evaluation-task-v1":
        errors.append(f"{expected_id}: bad result schema")
    if record.get("infrastructure_valid") is not True:
        errors.append(f"{expected_id}: infrastructure_valid is not true")
    if not isinstance(record.get("visible_transcript"), list) or record.get("visible_transcript"):
        errors.append(f"{expected_id}: visible_transcript must be empty")
    reward = record.get("reward")
    if not isinstance(reward, Mapping):
        return errors + [f"{expected_id}: reward is not an object"]
    leaked = HIDDEN_KEYS & set(reward)
    if leaked:
        errors.append(f"{expected_id}: hidden reward keys leaked: {sorted(leaked)}")
    aspects = _task_aspects(expected_id)
    qualities = reward.get("quality_by_aspect")
    if not isinstance(qualities, Mapping) or set(qualities) != set(aspects):
        errors.append(f"{expected_id}: quality_by_aspect keys do not match task aspects")
        qualities = {}
    values = []
    for aspect in aspects:
        value = qualities.get(aspect)
        if not _finite(value) or float(value) not in {0.0, 0.8, 1.0}:
            errors.append(f"{expected_id}/{aspect}: invalid quality {value!r}")
        else:
            values.append(float(value))
    completion = float(reward.get("completion_rate", -1.0))
    expected_completion = sum(value > 0.0 for value in values) / len(aspects)
    if not _close(completion, expected_completion):
        errors.append(f"{expected_id}: completion does not match quality indicators")
    if not _close(float(reward.get("correct_answer_rate", -1.0)), completion):
        errors.append(f"{expected_id}: correct_answer_rate != completion_rate")
    submission = float(reward.get("answer_submission_rate", -1.0))
    if not 0.0 <= submission <= 1.0 or submission + 1e-9 < completion:
        errors.append(f"{expected_id}: submission/completion invariant failed")
    correct_itinerary = bool(reward.get("correct_itinerary"))
    if correct_itinerary != (_close(completion, 1.0) and _close(submission, 1.0)):
        errors.append(f"{expected_id}: correct_itinerary invariant failed")
    if bool(reward.get("gold_itinerary")) and not correct_itinerary:
        errors.append(f"{expected_id}: gold_itinerary without correct_itinerary")
    active = float(reward.get("active_preference_coverage", -1.0))
    passive = float(reward.get("passive_preference_coverage", -1.0))
    preference = float(reward.get("preference_coverage", -1.0))
    if not all(0.0 <= value <= 1.0 for value in (active, passive, preference)):
        errors.append(f"{expected_id}: preference coverage outside [0,1]")
    elif not _close(preference, min(1.0, active + passive)):
        errors.append(f"{expected_id}: preference union mismatch")
    search = float(reward.get("search_coverage", -1.0))
    if not 0.0 <= search <= 1.0:
        errors.append(f"{expected_id}: search coverage outside [0,1]")
    breakdown = reward.get("phase_transition_breakdown")
    if not isinstance(breakdown, Mapping):
        errors.append(f"{expected_id}: missing phase breakdown")
    else:
        successes = opportunities = 0
        for name, item in breakdown.items():
            if not isinstance(item, Mapping):
                errors.append(f"{expected_id}: malformed phase item {name}")
                continue
            success = int(item.get("successes", -1)); opportunity = int(item.get("opportunities", -1))
            rate = float(item.get("rate", -1.0))
            if success < 0 or opportunity < 0 or success > opportunity or not 0.0 <= rate <= 1.0:
                errors.append(f"{expected_id}: invalid phase item {name}")
            if opportunity == 0 and not _close(rate, 1.0):
                errors.append(f"{expected_id}: empty phase item is not neutral")
            elif opportunity > 0 and not _close(rate, success / opportunity):
                errors.append(f"{expected_id}: phase rate mismatch {name}")
            successes += success; opportunities += opportunity
        phase = float(reward.get("phase_transition_score", -1.0))
        expected_phase = 1.0 if opportunities == 0 else successes / opportunities
        if not _close(phase, expected_phase):
            errors.append(f"{expected_id}: phase score mismatch")
    attempts = int(record.get("actor_attempts", -1)); env_steps = int(record.get("environment_steps", -1))
    guard = int(record.get("guard_rejections", -1))
    if attempts < 0 or env_steps < 0 or guard < 0 or guard > attempts or attempts < env_steps:
        errors.append(f"{expected_id}: attempts/steps/guard invariant failed")
    else:
        if int(reward.get("actor_attempts", attempts)) != attempts or int(reward.get("environment_steps", env_steps)) != env_steps:
            errors.append(f"{expected_id}: reward and result step counters differ")
        expected_effective = max(env_steps, attempts - min(guard, attempts)) + 0.25 * min(guard, attempts)
        if not _close(float(reward.get("effective_steps", -1.0)), expected_effective):
            errors.append(f"{expected_id}: effective_steps mismatch")
        if not _close(float(reward.get("guard_rejection_rate", -1.0)), guard / attempts if attempts else 0.0):
            errors.append(f"{expected_id}: guard rejection rate mismatch")
    for key in ("terminal_reward", "raw_terminal_reward", "policy_penalty", "answer_quality", "efficiency"):
        if not _finite(reward.get(key)):
            errors.append(f"{expected_id}: non-finite {key}")
    if all(_finite(reward.get(key)) for key in ("completion_rate", "preference_coverage", "phase_transition_score", "search_coverage", "answer_quality", "efficiency", "policy_penalty")):
        expected_raw = (
            3.0 * float(reward["completion_rate"])
            + 0.2 * float(reward["preference_coverage"])
            + 0.08 * float(reward["phase_transition_score"])
            + 0.06 * float(reward["search_coverage"])
            + 0.04 * float(reward["answer_quality"])
            + 0.02 * float(reward["efficiency"])
            - _penalty_from_public_fields(record, reward, aspects)
        )
        if not _close(float(reward["policy_penalty"]), _penalty_from_public_fields(record, reward, aspects), 1e-8):
            errors.append(f"{expected_id}: policy penalty mismatch")
        if not _close(float(reward["raw_terminal_reward"]), expected_raw, 1e-8):
            errors.append(f"{expected_id}: raw reward mismatch")
        if not _close(float(reward["terminal_reward"]), scale_priority_reward(expected_raw), 1e-8):
            errors.append(f"{expected_id}: terminal reward scaling mismatch")
    if record.get("termination_reason") != reward.get("termination_reason"):
        errors.append(f"{expected_id}: termination reason mismatch")
    return errors


# [项目注释] 功能：`_summary_and_records`：计算奖励、指标或聚合统计，供训练、评测或报告使用。 主要协作调用：_read_jsonl, zip, summarize_results,
# [项目注释]    _read_json。
# [项目注释] 输入：`root`: Path；`split`: str；`step`: int；`task_ids`: Sequence[str]；`compositions`: Sequence[str]。
# [项目注释] 输出：标注返回 `tuple[dict[str, Any], list[dict[str, Any]], list[str]]`；具体值由各分支决定。
def _summary_and_records(root: Path, split: str, step: int, task_ids: Sequence[str], compositions: Sequence[str]) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    result_path = root / split / f"step_{step}" / "results.jsonl"
    summary_path = root / split / f"step_{step}" / "summary.json"
    records = _read_jsonl(result_path)
    errors: list[str] = []
    if [record.get("task_id") for record in records] != list(task_ids):
        errors.append(f"{split}/step_{step}: result order is not task_ids.json order")
    if len({record.get("task_id") for record in records}) != len(records):
        errors.append(f"{split}/step_{step}: duplicate task IDs")
    for record, expected_id in zip(records, task_ids):
        errors.extend(_validate_record(record, expected_id))
    regenerated = summarize_results(records, expected_task_ids=task_ids, expected_compositions=compositions)
    stored = _read_json(summary_path)
    errors.extend(_json_close(stored, regenerated, f"{split}/step_{step}/summary", 1e-9))
    return stored, records, errors


# [项目注释] 功能：`_search_mean`：计算奖励、指标或聚合统计，供训练、评测或报告使用。 主要协作调用：sum, len, float。
# [项目注释] 输入：`records`: Sequence[Mapping[str, Any]]。
# [项目注释] 输出：标注返回 `float`；具体值由各分支决定。
def _search_mean(records: Sequence[Mapping[str, Any]]) -> float:
    return sum(float(record["reward"].get("search_coverage", 0.0)) for record in records) / len(records)


# [项目注释] 功能：`_category_counts`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：float, _close。
# [项目注释] 输入：`records`: Sequence[Mapping[str, Any]]。
# [项目注释] 输出：标注返回 `dict[str, int]`；具体值由各分支决定。
def _category_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    result = {"full": 0, "partial": 0, "wrong": 0, "none": 0}
    for record in records:
        reward = record["reward"]
        completion = float(reward["completion_rate"]); submission = float(reward["answer_submission_rate"])
        category = "full" if _close(completion, 1.0) else "partial" if completion > 0 else "wrong" if submission > 0 else "none"
        result[category] += 1
    return result


# [项目注释] 功能：`_check_training`：执行输入、状态或产物校验，并在不满足约束时返回诊断或抛出异常。 主要协作调用：_read_jsonl, values, set, items。
# [项目注释] 输入：`root`: Path。
# [项目注释] 输出：标注返回 `list[str]`；具体值由各分支决定。
def _check_training(root: Path) -> list[str]:
    errors: list[str] = []
    rows = _read_jsonl(root / "training" / "metrics.jsonl")
    if len(rows) != 200 or [row.get("step") for row in rows] != list(range(1, 201)):
        errors.append("training metrics must contain exactly steps 1..200")
        return errors
    required = {"completion_rate", "correct_answer_rate", "answer_submission_rate", "answer_quality", "active_preference_coverage", "passive_preference_coverage", "preference_coverage", "search_coverage", "guard_rejection_rate", "guard_rejections", "phase_transition_score", "efficiency", "terminal_reward", "reward_valid_rate", "actor_attempts", "environment_steps", "effective_steps", "invalid_actions", "exact_repeats", "semantic_repeats", "public_control_done_rate", "turn_credit_turn_count", "turn_credit_positive_count", "turn_credit_negative_count", "turn_credit_zero_count", "turn_credit_mean", "turn_credit_min", "turn_credit_max", "turn_credit_conservation_error", "dynamic_sampling_kept_groups", "dynamic_sampling_dropped_groups", "dynamic_sampling_constant_reward_groups", "dynamic_sampling_generation_batches", "entropy", "kl", "clip_fraction", "gradient_norm", "learning_rate"}
    missing = required - set(rows[0])
    if missing:
        errors.append(f"training metrics missing keys: {sorted(missing)}")
    for row in rows:
        for key, value in row.items():
            if key != "step" and not _finite(value):
                errors.append(f"training {key} contains non-finite value")
                break
    # [项目注释] 功能：`values`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：float。
    # [项目注释] 输入：`key`: str。
    # [项目注释] 输出：标注返回 `list[float]`；具体值由各分支决定。
    def values(key: str) -> list[float]: return [float(row[key]) for row in rows]
    # [项目注释] 功能：`std`：计算奖励、指标或聚合统计，供训练、评测或报告使用。 主要协作调用：values, sqrt, sum, len。
    # [项目注释] 输入：`key`: str。
    # [项目注释] 输出：标注返回 `float`；具体值由各分支决定。
    def std(key: str) -> float:
        vals = values(key); mean = sum(vals) / len(vals); return math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
    if not .09 <= std("completion_rate") <= .14:
        errors.append(f"completion noise std out of range: {std('completion_rate')}")
    if not .11 <= std("preference_coverage") <= .17:
        errors.append(f"preference noise std out of range: {std('preference_coverage')}")
    terminal = values("terminal_reward")
    if sum(b < a for a, b in zip(terminal, terminal[1:])) < 40:
        errors.append("terminal reward is not sufficiently non-monotonic")
    entropy = values("entropy"); kl = values("kl")
    if not entropy[-1] < entropy[0] or not any(b > a for a, b in zip(entropy, entropy[1:])):
        errors.append("entropy lacks overall decline plus local rebounds")
    if not kl[-1] > kl[0] or not any(b < a for a, b in zip(kl, kl[1:])):
        errors.append("KL lacks overall growth plus local reversion")
    constant = values("dynamic_sampling_constant_reward_groups")
    if max(constant[-20:]) < max(constant[:20]):
        errors.append("constant-reward group count never rises")
    return errors


# [项目注释] 功能：`validate`：执行输入、状态或产物校验，并在不满足约束时返回诊断或抛出异常。 主要协作调用：_read_json, check, list, items。
# [项目注释] 输入：`root`: Path。
# [项目注释] 输出：标注返回 `dict[str, Any]`；具体值由各分支决定。
def validate(root: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    # [项目注释] 功能：`check`：执行输入、状态或产物校验，并在不满足约束时返回诊断或抛出异常。
    # [项目注释] 输入：`name`: str；`condition`: bool；`detail`: str。
    # [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
    def check(name: str, condition: bool, detail: str = "") -> None:
        checks.append({"name": name, "status": "passed" if condition else "failed", "detail": detail})
        if not condition:
            errors.append(f"{name}: {detail or 'failed'}")

    scenario = _read_json(root / "scenario_config.json")
    provenance = _read_json(root / "PROVENANCE.json")
    check("simulation_flags", scenario.get("actual_training_executed") is False and scenario.get("actual_evaluation_executed") is False and provenance.get("actual_training_executed") is False and provenance.get("actual_evaluation_executed") is False, "actual execution flags must remain false")
    check("reward_version", scenario.get("reward_version") == REWARD_VERSION, f"{scenario.get('reward_version')!r}")
    eval_manifest = _read_json(root / "evaluation200" / "task_ids.json")
    fixed_manifest = _read_json(root / "validation32" / "task_ids.json")
    eval_ids = list(eval_manifest["task_ids"]); eval_comps = list(eval_manifest["compositions"])
    fixed_ids = list(fixed_manifest["task_ids"]); fixed_comps = list(fixed_manifest["compositions"])
    check("task_manifests", len(eval_ids) == 200 and len(fixed_ids) == 32 and len(eval_ids) == len(set(eval_ids)) and len(fixed_ids) == len(set(fixed_ids)), "expected 200 and 32 unique tasks")
    all_summaries: dict[str, dict[int, dict[str, Any]]] = {"evaluation200": {}, "validation32": {}}
    all_records: dict[str, dict[int, list[dict[str, Any]]]] = {"evaluation200": {}, "validation32": {}}
    for split, ids, comps in (("evaluation200", eval_ids, eval_comps), ("validation32", fixed_ids, fixed_comps)):
        for step in CHECKPOINTS:
            try:
                summary, records, local_errors = _summary_and_records(root, split, step, ids, comps)
            except Exception as exc:
                summary, records, local_errors = {}, [], [f"{split}/step_{step}: {exc.__class__.__name__}: {exc}"]
            all_summaries[split][step] = summary; all_records[split][step] = records; errors.extend(local_errors)
            check(f"{split}_step_{step}_records", not local_errors, "; ".join(local_errors[:3]))
    eval_metrics: dict[int, dict[str, float]] = {}
    for step in CHECKPOINTS:
        summary = all_summaries["evaluation200"][step]; fixed = summary.get("fixed_denominator", {})
        records = all_records["evaluation200"][step]
        metrics = {"completion": float(fixed.get("completion", -1)), "preference": float(fixed.get("preference_coverage", -1)), "guard": float(fixed.get("guard_rejection_rate", -1)), "phase": float(fixed.get("phase_transition_score", -1)), "search": _search_mean(records) if records else -1, "efficiency": float(fixed.get("efficiency", -1)), "terminal": float(fixed.get("terminal_reward", -1))}
        eval_metrics[step] = metrics
        target = EVAL_TARGETS[step]
        within = all(abs(metrics[key] - target[key]) <= (0.002 if key == "completion" else 0.035 if key in {"preference", "guard", "phase", "search"} else 0.045) for key in target)
        check(f"evaluation200_step_{step}_target_curve", within, json.dumps({"actual": metrics, "target": target}, sort_keys=True))
    for step, expected in OUTCOME_TARGETS.items():
        actual = _category_counts(all_records["evaluation200"][step])
        check(f"evaluation200_step_{step}_outcomes", actual == expected, f"{actual} != {expected}")
    exact_counts = {aspect: sum(1 for record in all_records["evaluation200"][200] if float(record["reward"].get("quality_by_aspect", {}).get(aspect, 0.0)) == 1.0) for aspect in ASPECT_TARGETS}
    check("evaluation200_step_200_aspect_exact_counts", all(abs(exact_counts[a] - ASPECT_TARGETS[a]) <= 2 for a in ASPECT_TARGETS), f"{exact_counts} vs {ASPECT_TARGETS}")
    comp = [eval_metrics[step]["completion"] for step in CHECKPOINTS]; pref = [eval_metrics[step]["preference"] for step in CHECKPOINTS]; search = [eval_metrics[step]["search"] for step in CHECKPOINTS]; phase = [eval_metrics[step]["phase"] for step in CHECKPOINTS]; eff = [eval_metrics[step]["efficiency"] for step in CHECKPOINTS]
    guard = [eval_metrics[step]["guard"] for step in CHECKPOINTS]
    check("nonmonotonic_completion", comp[3] < comp[2] and comp[4] > comp[2], str(comp))
    check("preference_peak_then_regression", pref[2] == max(pref) and pref[4] < pref[2], str(pref))
    check("search_peak_then_regression", search[2] == max(search) and search[4] < search[2], str(search))
    check("phase_peak_at_150", phase[3] == max(phase), str(phase))
    check("efficiency_drop_then_recovery", eff[2] < eff[0] and eff[4] == max(eff), str(eff))
    check("guard_direction_changes", sum((b - a) * (c - b) < 0 for a, b, c in zip(guard, guard[1:], guard[2:])) >= 2, str(guard))
    correct_sets = []
    for step in CHECKPOINTS:
        correct_sets.append({(record["task_id"], aspect) for record in all_records["evaluation200"][step] for aspect, value in record["reward"].get("quality_by_aspect", {}).items() if float(value) > 0.0})
    check("correct_sets_not_monotone", all(not correct_sets[i + 1] >= correct_sets[i] for i in range(4)), "at least one aspect regresses at each adjacent checkpoint")
    fixed_comp = [float(all_summaries["validation32"][step]["fixed_denominator"]["completion"]) for step in CHECKPOINTS]
    check("validation32_ranges", all(FIXED32_RANGES[step][0] <= value <= FIXED32_RANGES[step][1] for step, value in zip(CHECKPOINTS, fixed_comp)), str(fixed_comp))
    check("validation32_step200_above_step0", fixed_comp[-1] > fixed_comp[0], str(fixed_comp))
    training_errors = _check_training(root)
    errors.extend(training_errors); check("training_metrics", not training_errors, "; ".join(training_errors[:3]))
    donor_errors = []
    for step, item in provenance.get("checkpoint_donors", {}).items():
        path = Path(item.get("path", ""))
        if not path.exists() or _sha256(path) != item.get("sha256"):
            donor_errors.append(str(step))
    check("provenance_donors", not donor_errors, f"invalid donors {donor_errors}")
    max_abs_error = 0.0
    for step in CHECKPOINTS:
        for key, target in EVAL_TARGETS[step].items():
            max_abs_error = max(max_abs_error, abs(eval_metrics[step][key] - target))
    return {"overall_status": "passed" if not errors else "failed", "validator": str(Path(__file__).resolve()), "checks": checks, "error_count": len(errors), "errors": errors, "max_abs_error": max_abs_error}


# [项目注释] 功能：`main`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：ArgumentParser, add_argument, parse_args,
# [项目注释]    validate。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `int`；具体值由各分支决定。
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/simulation/grpo-200step-v1")
    args = parser.parse_args()
    report = validate(args.output)
    report_path = args.output / "consistency_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["overall_status"], "errors": report["error_count"], "report": str(report_path)}, ensure_ascii=False))
    return 0 if report["overall_status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
