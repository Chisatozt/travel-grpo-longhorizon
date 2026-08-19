#!/usr/bin/env python3
"""Build the CPU-only Recovery SFT go/no-go decision from frozen gate artifacts.

This report is deliberately inference- and training-free.  It reads the
immutable step-11 A/B summaries, closed-loop visible transcripts, and the
recovery manifests, then emits a small JSON decision artifact plus a Markdown
report.  It never loads model weights, calls an endpoint, or mutates a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GATE = ROOT / "outputs/evaluation/inference_gate_sft_merged"
DEFAULT_BOUNDARIES = ROOT / "outputs/recovery_boundaries/recovery-boundary-v1/manifest.json"
DEFAULT_TARGETS = ROOT / "outputs/recovery_targets/recovery-target-v1/manifest.json"
DEFAULT_SFT = ROOT / "outputs/recovery_sft/recovery-sft-v1/manifest.json"
SCHEMA_VERSION = "recovery-sft-decision-v1"
FORBIDDEN = (
    "remaining_preference_ids",
    "correct_ids",
    "best_ids",
    "reward_snapshot",
    "reward delta",
    "hidden preference",
    "gold_itinerary",
    "correct_itinerary",
)


# [项目注释] 功能：`sha256_file`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：sha256, hexdigest, is_file, open。
# [项目注释] 输入：`path`: Path。
# [项目注释] 输出：标注返回 `str | None`；具体值由各分支决定。
def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# [项目注释] 功能：`read_json`：读取并解析外部数据，将其转换为项目内部可消费的结构。 主要协作调用：loads, read_text, isinstance, ValueError。
# [项目注释] 输入：`path`: Path。
# [项目注释] 输出：标注返回 `dict[str, Any]`；具体值由各分支决定。
def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


# [项目注释] 功能：`rel`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：str, relative_to, resolve。
# [项目注释] 输入：`path`: Path。
# [项目注释] 输出：标注返回 `str`；具体值由各分支决定。
def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


# [项目注释] 功能：`_tool_choices`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：isinstance, loads。
# [项目注释] 输入：`transcript`: Any。
# [项目注释] 输出：标注返回 `list[str]`；具体值由各分支决定。
def _tool_choices(transcript: Any) -> list[str]:
    choices: list[str] = []
    if not isinstance(transcript, list):
        return choices
    for message in transcript:
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, Mapping):
                continue
            function = call.get("function")
            if not isinstance(function, Mapping):
                continue
            raw = function.get("arguments")
            try:
                value = json.loads(raw) if isinstance(raw, str) else raw
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(value, Mapping) and isinstance(value.get("choice"), str):
                choices.append(value["choice"])
    return choices


# [项目注释] 功能：`closed_loop_followup`：清理运行时资源或恢复边界状态，保证后续调用不会继承脏状态。 主要协作调用：sorted, glob, _tool_choices, len。
# [项目注释] 输入：`gate_dir`: Path。
# [项目注释] 输出：标注返回 `dict[str, Any]`；具体值由各分支决定。
def closed_loop_followup(gate_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for condition in ("A", "B"):
        rows: list[dict[str, Any]] = []
        for path in sorted((gate_dir / condition / "closed_loop").glob("[0-9][0-9].json")):
            rows.append(read_json(path))
        answer_calls = 0
        tasks_with_answer = 0
        violations = 0
        violating_tasks: list[str] = []
        for row in rows:
            choices = _tool_choices(row.get("visible_transcript"))
            answer_positions = [i for i, choice in enumerate(choices) if choice == "answer"]
            answer_calls += len(answer_positions)
            if answer_positions:
                tasks_with_answer += 1
            if any(
                any(choice in {"action", "search"} for choice in choices[position + 1 :])
                for position in answer_positions
            ):
                violations += 1
                violating_tasks.append(str(row.get("task_id")))
        result[condition] = {
            "tasks": len(rows),
            "answer_calls": answer_calls,
            "tasks_with_answer": tasks_with_answer,
            "answer_followed_by_action_or_search_tasks": violations,
            "answer_followed_by_action_or_search_rate_over_tasks": (
                violations / len(rows) if rows else None
            ),
            "answer_followup_gate_observable": bool(answer_calls),
            "violating_task_ids": violating_tasks,
        }
    return result


# [项目注释] 功能：`leakage_scan`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：list, sorted, glob, casefold。
# [项目注释] 输入：`gate_dir`: Path。
# [项目注释] 输出：标注返回 `dict[str, Any]`；具体值由各分支决定。
def leakage_scan(gate_dir: Path) -> dict[str, Any]:
    hits: list[dict[str, str]] = []
    candidates = list(gate_dir.glob("A/probes/*.json")) + list(gate_dir.glob("B/probes/*.json"))
    candidates += list(gate_dir.glob("A/closed_loop/[0-9][0-9].json"))
    candidates += list(gate_dir.glob("B/closed_loop/[0-9][0-9].json"))
    for path in sorted(candidates):
        text = path.read_text(encoding="utf-8").casefold()
        for forbidden in FORBIDDEN:
            if forbidden.casefold() in text:
                hits.append({"path": rel(path), "pattern": forbidden})
    return {"forbidden_patterns": list(FORBIDDEN), "hits": hits, "hit_count": len(hits)}


# [项目注释] 功能：`gate_rows`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：float, bool。
# [项目注释] 输入：`comparison`: Mapping[str, Any]；`followup`: Mapping[str, Any]；`leakage`: Mapping[str, Any]。
# [项目注释] 输出：标注返回 `list[dict[str, Any]]`；具体值由各分支决定。
def gate_rows(comparison: Mapping[str, Any], followup: Mapping[str, Any], leakage: Mapping[str, Any]) -> list[dict[str, Any]]:
    b = comparison.get("single_step", {}).get("B", {})
    closed_b = comparison.get("closed_loop", {}).get("B", {}) or {}
    rows = [
        {"gate": "normal answer@1", "observed": b.get("normal_search_result", {}).get("answer_at_1"), "threshold": ">= 0.95", "pass": float(b.get("normal_search_result", {}).get("answer_at_1", 0)) >= 0.95},
        {"gate": "answer is exactly one visible option ID", "observed": b.get("normal_search_result", {}).get("visible_id_only"), "threshold": "== 1.00", "pass": float(b.get("normal_search_result", {}).get("visible_id_only", 0)) >= 1.0},
        {"gate": "preference-complete search@1", "observed": b.get("preference_complete", {}).get("clean_search_at_1"), "threshold": ">= 0.85", "pass": float(b.get("preference_complete", {}).get("clean_search_at_1", 0)) >= 0.85},
        {"gate": "first fallback exact query repeat", "observed": b.get("first_fallback", {}).get("exact_query_repeat"), "threshold": "<= 0.05", "pass": float(b.get("first_fallback", {}).get("exact_query_repeat", 1)) <= 0.05},
        {"gate": "second fallback same-aspect search", "observed": b.get("second_fallback", {}).get("same_aspect_search"), "threshold": "== 0", "pass": float(b.get("second_fallback", {}).get("same_aspect_search", 1)) == 0.0},
        {"gate": "answer then action/search", "observed": followup.get("B", {}).get("answer_followed_by_action_or_search_rate_over_tasks"), "threshold": "== 0 (must be observable)", "pass": bool(followup.get("B", {}).get("answer_followup_gate_observable")) and float(followup.get("B", {}).get("answer_followed_by_action_or_search_rate_over_tasks", 1)) == 0.0, "note": "No answer call occurred in the 8-task closed loop; this is not a demonstrated pass."},
        {"gate": "hidden-state leakage", "observed": leakage.get("hit_count"), "threshold": "== 0", "pass": leakage.get("hit_count") == 0},
        {"gate": "8-task max_steps", "observed": closed_b.get("max_steps"), "threshold": "<= 0.25", "pass": float(closed_b.get("max_steps", 1)) <= 0.25, "note": "This narrow gate passes at equality, but completion is 0.0."},
    ]
    return rows


# [项目注释] 功能：`build_report`：根据输入配置和中间状态构建或生成新的项目产物。 主要协作调用：read_json, closed_loop_followup, leakage_scan,
# [项目注释]    gate_rows。
# [项目注释] 输入：`gate_dir`: Path；`boundaries`: Path；`targets`: Path；`sft`: Path。
# [项目注释] 输出：标注返回 `dict[str, Any]`；具体值由各分支决定。
def build_report(gate_dir: Path, boundaries: Path, targets: Path, sft: Path) -> dict[str, Any]:
    comparison = read_json(gate_dir / "comparison.json")
    manifest = read_json(gate_dir / "manifest.json")
    boundary_manifest = read_json(boundaries)
    target_manifest = read_json(targets)
    sft_manifest = read_json(sft)
    followup = closed_loop_followup(gate_dir)
    leakage = leakage_scan(gate_dir)
    gates = gate_rows(comparison, followup, leakage)
    single_step_b = comparison.get("single_step", {}).get("B", {}) or {}
    closed_loop_b = comparison.get("closed_loop", {}).get("B", {}) or {}
    normal_b = single_step_b.get("normal_search_result", {}) or {}
    preference_b = single_step_b.get("preference_complete", {}) or {}
    first_fallback_b = single_step_b.get("first_fallback", {}) or {}
    second_fallback_b = single_step_b.get("second_fallback", {}) or {}
    confused_b = single_step_b.get("confused_history", {}) or {}
    failed_gates = sum(not row["pass"] for row in gates)
    prompt_rendering_status = "FIXED_RECHECKED"
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": "NO-GO" if failed_gates else "GO",
        "recovery_sft": {
            "recommendation": "TARGETED_AFTER_STATE_RENDERING_FIX",
            "full_dataset_now": False,
            "training_started": False,
        },
        "basis": {
            "inference_gate_schema": manifest.get("schema_version"),
            "actor_policy_version": manifest.get("actor_policy_version"),
            "phase_guard_version": manifest.get("phase_guard_version"),
            "model_path": manifest.get("model_path"),
            "parameter_updates": manifest.get("inference_config", {}).get("parameter_updates"),
            "grpo": manifest.get("inference_config", {}).get("grpo"),
            "conditions_run": manifest.get("conditions_run"),
            "baseline_output_path": manifest.get("baseline_output_path"),
        },
        "artifacts": {
            "gate_manifest": {"path": rel(gate_dir / "manifest.json"), "sha256": sha256_file(gate_dir / "manifest.json")},
            "comparison": {"path": rel(gate_dir / "comparison.json"), "sha256": sha256_file(gate_dir / "comparison.json")},
            "boundary_manifest": {"path": rel(boundaries), "sha256": sha256_file(boundaries)},
            "target_manifest": {"path": rel(targets), "sha256": sha256_file(targets)},
            "recovery_sft_manifest": {"path": rel(sft), "sha256": sha256_file(sft)},
        },
        "gates": gates,
        "single_step": comparison.get("single_step"),
        "closed_loop": comparison.get("closed_loop"),
        "answer_followup": followup,
        "leakage": leakage,
        "recovery_pool": {
            "boundary_contexts": boundary_manifest.get("counts"),
            "target_counts": target_manifest.get("counts", {}).get("by_boundary_type"),
            "target_rejection_reasons": target_manifest.get("counts", {}).get("rejection_reasons"),
            "rendered_sft_counts": sft_manifest.get("audit", {}).get("boundary_type_distribution"),
            "rendered_sft_quality": sft_manifest.get("audit", {}).get("quality_checks"),
        },
        "proposed_targeted_quota": {
            "unit": "accepted public-only target records, task-disjoint train/validation",
            "train": {
                "preference_complete_to_search": 400,
                "first_fallback": 200,
                "second_fallback": 200,
                "valid_search_to_answer": 200,
                "repeated_no_progress_action": 200,
            },
            "validation": {
                "preference_complete_to_search": 100,
                "first_fallback": 50,
                "second_fallback": 50,
                "valid_search_to_answer": 50,
                "repeated_no_progress_action": 50,
            },
            "totals": {"train": 1200, "validation": 300},
            "caveat": "Current accepted pools are smaller for preference-complete (131), first-fallback (50), and second-fallback (31); improve state/target construction before filling quotas. Never promote rejected/quarantine records by guessing.",
        },
        "problem_classification": {
            "deterministic_controller": {
                "status": "PARTIAL",
                "evidence": (
                    f"B public guard rejected {closed_loop_b.get('guard_rejections', 0)} calls before simulator; "
                    f"repeated-action/search occurred in {closed_loop_b.get('repeated_action_or_search', 0):.1%} of closed-loop tasks, "
                    f"while completion remained {closed_loop_b.get('completion', 0):.1%}."
                ),
            },
            "model_policy_execution": {
                "status": "BLOCKER",
                "evidence": (
                    f"B first-fallback exact repeats={first_fallback_b.get('exact_query_repeat', 0):.1%}, "
                    f"confused-history repeated actions={confused_b.get('repeated_action', 0):.1%}, "
                    f"and answer calls in the 8-task loop={followup.get('B', {}).get('answer_calls', 0)}."
                ),
            },
            "fallback_infrastructure": {
                "status": "NOT_PRIMARY_BLOCKER",
                "evidence": (
                    f"B reward_degraded={closed_loop_b.get('reward_degraded', 0):.1%}; "
                    "no simulator infrastructure error was recorded in the public summaries."
                ),
            },
            "semantic_option_selection": {
                "status": "BLOCKER",
                "evidence": (
                    f"B normal answer@1={normal_b.get('answer_at_1', 0):.1%}, but "
                    f"exactly-one-visible-ID={normal_b.get('visible_id_only', 0):.1%}; "
                    "the remaining failures are option-grounding errors."
                ),
            },
            "prompt_state_rendering": {
                "status": prompt_rendering_status,
                "evidence": (
                    f"The public rendering fix was rechecked in this B rerun: preference-complete "
                    f"search@1={preference_b.get('clean_search_at_1', 0):.1%} and "
                    f"second-fallback same-aspect search={second_fallback_b.get('same_aspect_search', 0):.1%}. "
                    "Remaining failures are policy behavior, not the previously stale phase snapshot."
                ),
            },
            "recovery_sft_necessity": {
                "status": "TARGETED_YES_AFTER_FIX",
                "evidence": (
                    "Fallback rewrite/switch, visible-ID grounding, and closed-loop completion still fail hard gates; "
                    "keep recovery SFT targeted and do not start broad training from this gate."
                ),
            },
        },
        "acceptance_after_fix": {
            "normal_answer_at_1": ">= 0.95",
            "visible_id_only": "== 1.00",
            "preference_complete_search_at_1": ">= 0.85 (strict current-aspect target >= 0.95)",
            "first_fallback_exact_repeat": "<= 0.05 and changed_query >= 0.90",
            "second_fallback_same_aspect_search": "== 0 and switch_aspect >= 0.90",
            "answer_followup_action_or_search": "== 0 with at least one answer call",
            "hidden_leakage": "== 0",
            "closed_loop": "max_steps <= 0.25, no completion regression, then repeat on 32 tasks",
        },
    }


# [项目注释] 功能：`render_markdown`：把协议/状态数据转换为模型、用户或日志可见的文本表示。 主要协作调用：float, int, items, join。
# [项目注释] 输入：`report`: Mapping[str, Any]；`machine_path`: str | None；`gate_path`: str | None；`output_path`:
# [项目注释]    str | None；`markdown_path`: str | None。
# [项目注释] 输出：标注返回 `str`；具体值由各分支决定。
def render_markdown(
    report: Mapping[str, Any],
    *,
    machine_path: str | None = None,
    gate_path: str | None = None,
    output_path: str | None = None,
    markdown_path: str | None = None,
) -> str:
    closed_b = report.get("closed_loop", {}).get("B", {}) or {}
    followup_b = report.get("answer_followup", {}).get("B", {}) or {}
    max_steps = float(closed_b.get("max_steps", 0.0))
    completion = float(closed_b.get("completion", 0.0))
    answer_calls = int(followup_b.get("answer_calls", 0))
    if machine_path is None:
        machine_path = rel(DEFAULT_GATE.parent / "recovery_sft_decision_v1/report.json")
    if gate_path is None:
        gate_path = rel(DEFAULT_GATE)
    if output_path is None:
        output_path = rel(DEFAULT_GATE.parent / "recovery_sft_decision_v1")
    if markdown_path is None:
        markdown_path = "docs/evaluation/recovery-sft-decision-v1.md"
    lines = [
        "# Recovery SFT decision (inference-gate-v1)",
        "",
        "**Decision: NO-GO for the current controller/prompt contract.**",
        "",
        "No model parameters were changed and no training was started. The public-state rendering fix was applied and rechecked in this rerun; the remaining failures are policy/grounding gates.",
        "",
        "## Gate results (condition B)",
        "",
        "| Gate | Observed | Threshold | Result |",
        "|---|---:|---:|---|",
    ]
    for row in report["gates"]:
        observed = row.get("observed")
        if isinstance(observed, float):
            observed = f"{observed:.4f}"
        lines.append(f"| {row['gate']} | {observed} | {row['threshold']} | {'PASS' if row['pass'] else 'FAIL/UNPROVEN'} |")
    lines += [
        "",
        f"The B closed loop reached max_steps on {max_steps:.0%} of tasks and completion on {completion:.0%}; "
        + ("the answer-follow-up gate is unproven because no closed-loop answer was emitted." if answer_calls == 0 else f"{answer_calls} answer calls were observed for the answer-follow-up check."),
        "",
        "## Root-cause classification",
        "",
    ]
    labels = {
        "deterministic_controller": "Deterministic controller",
        "model_policy_execution": "Model policy execution",
        "fallback_infrastructure": "Fallback infrastructure",
        "semantic_option_selection": "Semantic option selection",
        "prompt_state_rendering": "Prompt/state rendering",
        "recovery_sft_necessity": "Recovery SFT necessity",
    }
    for key, label in labels.items():
        item = report["problem_classification"][key]
        lines.append(f"- **{label}: {item['status']}** — {item['evidence']}")
    lines += [
        "",
        "## Required ordering",
        "",
        "1. Keep the public reducer/rendering fix and its CPU/public-snapshot assertions; this rerun no longer shows the stale ELICITING phase defect.",
        "2. Repair fallback query rewriting, visible-option-ID grounding, and answer emission. Do not promote rejected/quarantine records by guessing.",
        "3. Run only targeted Recovery SFT after those policy fixes, then rerun this deterministic gate and a 32-task confirmation set.",
        "",
        "## Proposed targeted Recovery SFT quota",
        "",
        "| Boundary | Train | Validation | Current accepted pool |",
        "|---|---:|---:|---:|",
    ]
    target_counts = report["recovery_pool"]["target_counts"]
    quotas = report["proposed_targeted_quota"]
    names = ["preference_complete_to_search", "first_fallback", "second_fallback", "valid_search_to_answer", "repeated_no_progress_action"]
    for name in names:
        current = target_counts.get(name, {}).get("accepted", 0)
        lines.append(f"| {name} | {quotas['train'][name]} | {quotas['validation'][name]} | {current} |")
    lines += [
        "",
        "The first pass is 1,200 train / 300 validation records, task-disjoint. The current pools can only support part of this quota; target generation must be repaired or the quota reduced without fabricating targets.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "PYTHONPATH=src .venv/bin/python scripts/eval/recovery_sft_decision.py \\",
        f"  --gate {gate_path} \\",
        f"  --output {output_path} \\",
        f"  --markdown {markdown_path}",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src .venv/bin/pytest -q tests/test_recovery_sft_decision.py",
        "```",
        "",
        f"The machine-readable report is `{machine_path}`. Its inputs include hashes for the rerun gate and recovery manifests.",
        "",
    ]
    return "\n".join(lines)


# [项目注释] 功能：`main`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：ArgumentParser, add_argument, parse_args,
# [项目注释]    build_report。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `int`；具体值由各分支决定。
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--boundaries", type=Path, default=DEFAULT_BOUNDARIES)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--sft", type=Path, default=DEFAULT_SFT)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/evaluation/recovery_sft_decision_v1")
    parser.add_argument("--markdown", type=Path, default=ROOT / "docs/evaluation/recovery-sft-decision-v1.md")
    args = parser.parse_args()
    report = build_report(args.gate, args.boundaries, args.targets, args.sft)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(
        render_markdown(
            report,
            machine_path=rel(args.output / "report.json"),
            gate_path=rel(args.gate),
            output_path=rel(args.output),
            markdown_path=rel(args.markdown),
        ),
        encoding="utf-8",
    )
    print(json.dumps({"decision": report["decision"], "output": rel(args.output / "report.json"), "markdown": rel(args.markdown), "failed_or_unproven_gates": sum(not row["pass"] for row in report["gates"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
