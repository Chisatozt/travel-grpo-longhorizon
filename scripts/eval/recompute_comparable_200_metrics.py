#!/usr/bin/env python3
"""Replay current Reward-v3 metrics over frozen 200-task evaluation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from travel_grpo.envs.reward import PRIORITY_REWARD_SCALE, REWARD_VERSION
from travel_grpo.evaluation.summary import summarize_results

SCHEMA_VERSION = "travel-200-task-metric-replay-v1"
METRIC_VERSION = "current-reward-v3-comparable-v1"
DEFAULT_OUTPUT = ROOT / "outputs/evaluation/comparable-200-task-metrics-v3-replay"
DEFAULT_RUNS = {
    "baseline_qwen35_2b": ROOT / "outputs/evaluation/200-Task/baseline/run",
    "sft_merged": ROOT / "outputs/evaluation/200-Task/SFT/run2",
    "grpo100_rewardv2": ROOT / "outputs/evaluation/200-Task/GRPO-100-rewardV2/run",
    "turn_credit_step100": ROOT / "outputs/evaluation/turn-credit-checkpoints-200-task-public-guarded-c4/step-100/run",
    "turn_credit_step150": ROOT / "outputs/evaluation/turn-credit-checkpoints-200-task-public-guarded-c4/step-150/run",
    "turn_credit_step200": ROOT / "outputs/evaluation/turn-credit-checkpoints-200-task-public-guarded-c4/step-200/run",
}
PHASE_RE = re.compile(r"Current control state:\s*([A-Z_]+)")
ASPECT_RE = re.compile(r"Current public aspect:\s*([^|\n]+)")
FALLBACK_RE = re.compile(r"Current aspect fallback count:\s*(\d+)")
PHASE_KEYS = ("search_required", "retry_search", "candidate_answer", "aspect_switch")


# [项目注释] 功能：`_sha256`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：sha256, hexdigest, open, iter。
# [项目注释] 输入：`path`: Path。
# [项目注释] 输出：标注返回 `str`；具体值由各分支决定。
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# [项目注释] 功能：`_atomic_json`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：mkdir, with_suffix, write_text, replace。
# [项目注释] 输入：`path`: Path；`value`: Any。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


# [项目注释] 功能：`_ratio`：计算奖励、指标或聚合统计，供训练、评测或报告使用。 主要协作调用：min, max。
# [项目注释] 输入：`numerator`: float；`denominator`: float。
# [项目注释] 输出：标注返回 `float`；具体值由各分支决定。
def _ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator <= 0 else min(1.0, max(0.0, numerator / denominator))


# [项目注释] 功能：`_tool_choice`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：isinstance, loads, str。
# [项目注释] 输入：`message`: Mapping[str, Any]。
# [项目注释] 输出：标注返回 `str | None`；具体值由各分支决定。
def _tool_choice(message: Mapping[str, Any]) -> str | None:
    try:
        calls = message.get("tool_calls") or ()
        arguments = calls[0]["function"]["arguments"]
        document = json.loads(arguments) if isinstance(arguments, str) else arguments
        choice = document.get("choice") if isinstance(document, Mapping) else None
        return str(choice) if choice is not None else None
    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def replay_public_metrics(transcript: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Replay public-only phase transitions and blocked aspects."""

    phase = "ELICITING"
    counts = {key: [0, 0] for key in PHASE_KEYS}  # successes, opportunities
    blocked: set[str] = set()
    for index, message in enumerate(transcript):
        if message.get("role") == "assistant":
            choice = _tool_choice(message)
            key = {
                "SEARCH_REQUIRED": "search_required",
                "SEARCH_RETRY_REQUIRED": "retry_search",
                "ANSWER_REQUIRED": "candidate_answer",
            }.get(phase)
            if key is not None:
                counts[key][1] += 1
                following = transcript[index + 1] if index + 1 < len(transcript) else {}
                rejected = "public control rejected this call" in str(
                    following.get("content", "")
                ).casefold()
                expected = {
                    "search_required": "search",
                    "retry_search": "search",
                    "candidate_answer": "answer",
                }[key]
                if not rejected and choice == expected:
                    counts[key][0] += 1

        text = str(message.get("content") or "")
        if "Transition: SWITCH_ASPECT_REQUIRED" in text:
            counts["aspect_switch"][0] += 1
            counts["aspect_switch"][1] += 1
        matches = PHASE_RE.findall(text)
        if not matches:
            continue
        phase = matches[-1]
        if phase != "SWITCH_ASPECT_REQUIRED":
            continue
        aspect_matches = ASPECT_RE.findall(text)
        fallback_matches = FALLBACK_RE.findall(text)
        aspect = aspect_matches[-1].strip() if aspect_matches else "unknown"
        fallback_count = int(fallback_matches[-1]) if fallback_matches else 0
        if fallback_count >= 2 or "current aspect is blocked" in text.casefold():
            blocked.add(aspect)

    successes = sum(value[0] for value in counts.values())
    opportunities = sum(value[1] for value in counts.values())
    breakdown = {
        key: {
            "successes": value[0],
            "opportunities": value[1],
            "rate": 1.0 if value[1] == 0 else value[0] / value[1],
        }
        for key, value in counts.items()
    }
    return {
        "phase_transition_score": 1.0 if opportunities == 0 else successes / opportunities,
        "phase_transition_breakdown": breakdown,
        "blocked_aspects": len(blocked),
    }


# [项目注释] 功能：`_load_preference_counts`：读取并解析外部数据，将其转换为项目内部可消费的结构。 主要协作调用：sorted, set, loads, items。
# [项目注释] 输入：`compositions`: Sequence[str]。
# [项目注释] 输出：标注返回 `dict[str, int]`；具体值由各分支决定。
def _load_preference_counts(compositions: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    data_root = ROOT / "environments/UserBench/travelgym/data"
    for composition in sorted(set(compositions)):
        path = data_root / f"travelgym_data_{composition}.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        for task_id, task in document.items():
            aspects = task.get("dimensions", ())
            counts[str(task_id)] = sum(
                len((task.get(str(aspect)) or {}).get("preferences", ()))
                for aspect in aspects
            )
    return counts


# [项目注释] 功能：`_current_penalty`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：max, int, sum, _ratio。
# [项目注释] 输入：`reward`: Mapping[str, Any]；`guard_rejections`: int；`blocked_aspects`: int；`aspect_count`:
# [项目注释]    int；`termination_reason`: str。
# [项目注释] 输出：标注返回 `tuple[float, dict[str, float]]`；具体值由各分支决定。
def _current_penalty(
    reward: Mapping[str, Any], *, guard_rejections: int, blocked_aspects: int,
    aspect_count: int, termination_reason: str,
) -> tuple[float, dict[str, float]]:
    invalid = max(0, int(reward.get("invalid_actions", 0)))
    exact = max(0, int(reward.get("exact_repeats", 0)))
    semantic = max(0, int(reward.get("semantic_repeats", 0)))
    ambiguous = max(0, int(reward.get("ambiguous_actions", 0)))
    unsearched = max(0, int(reward.get("unsearched_answers", 0)))
    wrong = max(0, int(reward.get("wrong_answers", 0)))
    values = {
        "guard_rejection": 0.08 * _ratio(min(guard_rejections, 4), 4),
        "blocked_aspect": 0.08 * _ratio(min(blocked_aspects, aspect_count), aspect_count),
        "invalid_action": 0.03 * _ratio(min(invalid, 4), 4),
        "parallel_tool_calls": 0.05 if termination_reason == "parallel_tool_calls" else 0.0,
        "exact_repeat": 0.02 * _ratio(min(exact, 4), 4),
        "semantic_repeat": 0.02 * _ratio(min(semantic, 4), 4),
        "ambiguous_action": 0.02 * _ratio(min(ambiguous, aspect_count), aspect_count),
        "unsearched_answer": 0.03 * _ratio(min(unsearched, aspect_count), aspect_count),
        "wrong_answer": 0.04 * _ratio(min(wrong, aspect_count), aspect_count),
        "no_tool_output": 0.02 if termination_reason == "no_tool_output" else 0.0,
        "max_steps": 0.02 if termination_reason == "max_steps" else 0.0,
    }
    return sum(values.values()), values


# [项目注释] 功能：`project_current_reward`：计算奖励、指标或聚合统计，供训练、评测或报告使用。 主要协作调用：len, str, float, min。
# [项目注释] 输入：`result`: Mapping[str, Any]；`preference_count`: int。
# [项目注释] 输出：标注返回 `dict[str, Any]`；具体值由各分支决定。
def project_current_reward(result: Mapping[str, Any], preference_count: int) -> dict[str, Any]:
    source = result.get("reward")
    if not isinstance(source, Mapping):
        return {}
    qualities_raw = source.get("quality_by_aspect", {})
    qualities = {str(key): float(value) for key, value in qualities_raw.items()}
    aspect_count = len(qualities)
    if aspect_count == 0:
        raise ValueError(f"task {result.get('task_id')} has no aspect qualities")
    source_version = str(source.get("reward_version"))
    is_v2 = source_version == "userbench-travel-reward-v2"
    answer_submission = (
        float(source.get("completion_rate", 0.0))
        if is_v2
        else float(source.get("answer_submission_rate", source.get("completion_rate", 0.0)))
    )
    correct_rate = sum(value > 0.0 for value in qualities.values()) / aspect_count
    answer_quality = sum(qualities.values()) / aspect_count
    active = float(source.get("active_preference_coverage", 0.0))
    passive = float(source.get("passive_preference_coverage", 0.0))
    preference_coverage = min(1.0, active + passive)
    search_coverage = float(source.get("search_coverage", 0.0))
    public = replay_public_metrics(result.get("visible_transcript", ()))
    actor_attempts = max(0, int(result.get("actor_attempts", source.get("actor_attempts", 0))))
    environment_steps = max(0, int(result.get("environment_steps", source.get("environment_steps", 0))))
    guard_rejections = max(0, int(result.get("guard_rejections", 0)))
    accepted_attempts = max(0, actor_attempts - min(guard_rejections, actor_attempts))
    effective_steps = max(environment_steps, accepted_attempts) + 0.25 * min(
        guard_rejections, actor_attempts
    )
    useful_budget = preference_count + 2 * aspect_count
    if effective_steps <= useful_budget:
        efficiency = 1.0
    elif useful_budget >= 20:
        efficiency = 0.0
    else:
        efficiency = max(0.0, 1.0 - (effective_steps - useful_budget) / (20 - useful_budget))
    termination = str(result.get("termination_reason") or source.get("termination_reason") or "missing")
    penalty, penalty_components = _current_penalty(
        source, guard_rejections=guard_rejections, blocked_aspects=public["blocked_aspects"],
        aspect_count=aspect_count, termination_reason=termination,
    )
    raw = (
        3.00 * correct_rate
        + 0.20 * preference_coverage
        + 0.08 * public["phase_transition_score"]
        + 0.06 * search_coverage
        + 0.04 * answer_quality
        + 0.02 * efficiency
        - penalty
    )
    reward_valid = bool(source.get("reward_valid", False)) and result.get("infrastructure_valid") is True
    terminal = max(-1.0, min(1.0, raw / PRIORITY_REWARD_SCALE)) if reward_valid else 0.0
    projected = dict(source)
    projected.update(
        reward_version=REWARD_VERSION, replay_metric_version=METRIC_VERSION,
        source_reward_version=source_version, reward_valid=reward_valid,
        terminal_reward=terminal, raw_terminal_reward=raw,
        completion_rate=correct_rate, correct_answer_rate=correct_rate,
        answer_submission_rate=answer_submission, answer_quality=answer_quality,
        preference_coverage=preference_coverage, phase_transition_score=public["phase_transition_score"],
        phase_transition_breakdown=public["phase_transition_breakdown"],
        blocked_aspects=public["blocked_aspects"], guard_rejections=guard_rejections,
        guard_rejection_rate=_ratio(guard_rejections, actor_attempts),
        accepted_actor_attempts=accepted_attempts, effective_steps=effective_steps,
        efficiency=efficiency, policy_penalty=penalty, penalty_components=penalty_components,
    )
    return projected


# [项目注释] 功能：`_extra_summary`：计算奖励、指标或聚合统计，供训练、评测或报告使用。 主要协作调用：rate, sum, len, bool。
# [项目注释] 输入：`records`: Sequence[Mapping[str, Any]]；`denominator`: int。
# [项目注释] 输出：标注返回 `dict[str, float]`；具体值由各分支决定。
def _extra_summary(records: Sequence[Mapping[str, Any]], denominator: int) -> dict[str, float]:
    valid = [row for row in records if row.get("infrastructure_valid") is True]
    # [项目注释] 功能：`rate`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：sum, bool, predicate。
    # [项目注释] 输入：`predicate`。
    # [项目注释] 输出：标注返回 `float`；具体值由各分支决定。
    def rate(predicate) -> float:
        return sum(bool(predicate(row)) for row in records) / denominator
    return {
        "reward_valid_rate": len(valid) / denominator,
        "tasks_with_any_answer_rate": rate(lambda r: r["reward"].get("answer_submission_rate", 0) > 0),
        "tasks_with_full_answer_submission_rate": rate(lambda r: r["reward"].get("answer_submission_rate") == 1),
        "tasks_with_any_correct_answer_rate": rate(lambda r: r["reward"].get("completion_rate", 0) > 0),
        "correct_itinerary_task_rate": rate(lambda r: r["reward"].get("correct_itinerary") is True),
        "public_control_complete_rate": rate(lambda r: r.get("termination_reason") == "public_control_complete"),
        "actor_turn_limit_rate": rate(lambda r: r.get("termination_reason") == "actor_turn_limit"),
        "max_steps_rate": rate(lambda r: r.get("termination_reason") == "max_steps"),
        "tasks_with_guard_rejection_rate": rate(lambda r: int(r.get("guard_rejections", 0)) > 0),
        "tasks_with_blocked_aspect_rate": rate(lambda r: r["reward"].get("blocked_aspects", 0) > 0),
    }


# [项目注释] 功能：`_task_digest`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：sha256, sorted, hexdigest, update。
# [项目注释] 输入：`paths`: Sequence[Path]。
# [项目注释] 输出：标注返回 `str`；具体值由各分支决定。
def _task_digest(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode())
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


# [项目注释] 功能：`replay_run`：编排一个训练、采集、评测或 replay 流程，并汇总其结果。 主要协作调用：loads, sorted, Counter, summarize_results。
# [项目注释] 输入：`name`: str；`run_dir`: Path；`output_root`: Path；`preference_counts`: Mapping[str, int]。
# [项目注释] 输出：标注返回 `dict[str, Any]`；具体值由各分支决定。
def replay_run(name: str, run_dir: Path, output_root: Path, preference_counts: Mapping[str, int]) -> dict[str, Any]:
    contract = json.loads((run_dir / "contract.json").read_text(encoding="utf-8"))
    task_paths = sorted((run_dir / "tasks").glob("*.json"))
    records = [json.loads(path.read_text(encoding="utf-8")) for path in task_paths]
    expected_ids = [str(value) for value in contract["task_ids"]]
    compositions = [str(value) for value in contract["compositions"]]
    if len(expected_ids) != 200 or len(records) != 200:
        raise ValueError(f"{name} is not a complete 200-task artifact")
    projected = []
    source_versions: Counter[str] = Counter()
    native_v3_mismatches: list[str] = []
    for row in records:
        task_id = str(row["task_id"]); original_reward = row.get("reward", {})
        reward = project_current_reward(row, preference_counts[task_id])
        source_versions[str(original_reward.get("reward_version"))] += 1
        if original_reward.get("reward_version") == REWARD_VERSION:
            audit_keys = (
                "terminal_reward", "raw_terminal_reward", "completion_rate",
                "answer_submission_rate", "answer_quality", "preference_coverage",
                "phase_transition_score", "blocked_aspects", "guard_rejection_rate",
                "efficiency", "policy_penalty",
            )
            for key in audit_keys:
                left, right = reward.get(key), original_reward.get(key)
                equal = left == right if not isinstance(left, float) else math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
                if not equal:
                    native_v3_mismatches.append(f"{task_id}:{key}:{left!r}!={right!r}")
        copy = dict(row); copy["reward"] = reward; projected.append(copy)
    if native_v3_mismatches:
        raise ValueError(f"{name} native-v3 replay drift: {native_v3_mismatches[:5]}")
    summary = summarize_results(projected, expected_task_ids=expected_ids, expected_compositions=compositions)
    summary["metric_version"] = METRIC_VERSION
    summary["source_reward_versions"] = dict(source_versions)
    summary["additional_metrics"] = _extra_summary(projected, 200)
    target = output_root / name
    _atomic_json(target / "summary.json", summary)
    with (target / "per_task_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in projected:
            handle.write(json.dumps({
                "task_id": row["task_id"], "composition": row["composition"],
                "infrastructure_valid": row["infrastructure_valid"],
                "termination_reason": row["termination_reason"],
                "guard_rejections": row.get("guard_rejections", 0),
                "reward": row["reward"],
            }, sort_keys=True) + "\n")
    return {
        "name": name, "source_run": str(run_dir.relative_to(ROOT)),
        "model": contract.get("model"), "public_control_enabled": contract.get("public_control_enabled"),
        "actor_policy_version": contract.get("actor_policy_version"),
        "contract_hash": contract.get("contract_hash"), "contract_sha256": _sha256(run_dir / "contract.json"),
        "source_task_digest": _task_digest(task_paths),
        "source_reward_versions": dict(source_versions),
        "summary": summary,
    }


# [项目注释] 功能：`_report`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：extend, join。
# [项目注释] 输入：`rows`: Sequence[Mapping[str, Any]]。
# [项目注释] 输出：标注返回 `str`；具体值由各分支决定。
def _report(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Comparable 200-Task metrics (current Reward v3 replay)", "",
        f"Metric version: `{METRIC_VERSION}`", "",
        "All rows use the same proportional 200-task subset, production Actor policy v2, public-control guard v1, fixed denominator 200, and current Reward-v3 projection.", "",
        "| Model | Valid | Completion | Any answer task | Answer submission | Preference coverage | Correct itinerary | Terminal reward | Guard/task | Turn limit | Max steps |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        summary=row["summary"]; fixed=summary["fixed_denominator"]; extra=summary["additional_metrics"]
        lines.append(
            f"| {row['name']} | {summary['valid_tasks']}/200 | {fixed['completion']:.4f} | "
            f"{extra['tasks_with_any_answer_rate']:.4f} | {fixed['answer_submission_rate']:.4f} | "
            f"{fixed['preference_coverage']:.4f} | {fixed['correct_itinerary']:.4f} | "
            f"{fixed['terminal_reward']:.4f} | {summary['guard_rejections_per_task']:.3f} | "
            f"{extra['actor_turn_limit_rate']:.4f} | {extra['max_steps_rate']:.4f} |"
        )
    lines.extend(["", "Reward-v2 source runs were projected from frozen quality, submission, preference, search, counter, task-label, guard, and public-control transcript evidence. The replay algorithm was audited against all native Reward-v3 tasks and fails closed on any mismatch.", ""] )
    return "\n".join(lines)


# [项目注释] 功能：`main`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：ArgumentParser, add_argument, parse_args, resolve。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `int`；具体值由各分支决定。
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    contracts = [json.loads((path / "contract.json").read_text()) for path in DEFAULT_RUNS.values()]
    reference_ids = contracts[0]["task_ids"]; reference_compositions = contracts[0]["compositions"]
    for contract in contracts[1:]:
        if contract["task_ids"] != reference_ids or contract["compositions"] != reference_compositions:
            raise ValueError("200-task contracts do not share the same ordered task subset")
        if contract.get("public_control_enabled") is not True:
            raise ValueError("main comparison requires public-control guarded runs")
    preference_counts = _load_preference_counts(reference_compositions)
    rows = [replay_run(name, path, output, preference_counts) for name, path in DEFAULT_RUNS.items()]
    manifest = {
        "schema_version": SCHEMA_VERSION, "metric_version": METRIC_VERSION,
        "reward_version": REWARD_VERSION, "task_count": 200,
        "ordered_task_ids_sha256": hashlib.sha256("\n".join(reference_ids).encode()).hexdigest(),
        "models": [{key: value for key, value in row.items() if key != "summary"} for row in rows],
        "native_v3_replay_mismatches": 0,
        "excluded_non_comparable": {
            "baseline_raw": "public control guard disabled",
            "duplicate_alias_runs": "same model/task artifacts retained only once",
        },
    }
    _atomic_json(output / "manifest.json", manifest)
    _atomic_json(output / "comparison.json", {
        "schema_version": SCHEMA_VERSION, "metric_version": METRIC_VERSION,
        "models": {row["name"]: row["summary"] for row in rows},
    })
    (output / "report.md").write_text(_report(rows), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
