#!/usr/bin/env python3
"""Build the six-object 200-task comparison and immutable raw archive."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from travel_grpo.evaluation.summary import summarize_results

ARCHIVE = ROOT / "outputs/analysis/grpo-200-task-synthetic-v1"
DOC = ROOT / "docs/evaluation/grpo-200-task-synthetic-comparison.md"
SYNTHETIC = ROOT / "outputs/simulation/grpo-200step-v1"
COMPARABLE = ROOT / "outputs/evaluation/comparable-200-task-metrics-v3-replay"

SOURCES = {
    "qwen35_2b_baseline": ROOT / "outputs/evaluation/200-Task/baseline/run",
    "sft_merged": ROOT / "outputs/evaluation/200-Task/SFT/run2",
}
SYNTHETIC_STEPS = (50, 100, 150, 200)
EXCLUDED_MODELS = (
    "historical_grpo_100",
    "historical_grpo_150",
    "historical_grpo_200",
)


# [项目注释] 功能：`sha256`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：sha256, hexdigest, open, iter。
# [项目注释] 输入：`path`: Path。
# [项目注释] 输出：标注返回 `str`；具体值由各分支决定。
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# [项目注释] 功能：`read_json`：读取并解析外部数据，将其转换为项目内部可消费的结构。 主要协作调用：loads, read_text。
# [项目注释] 输入：`path`: Path。
# [项目注释] 输出：标注返回 `Any`；具体值由各分支决定。
def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# [项目注释] 功能：`read_jsonl`：读取并解析外部数据，将其转换为项目内部可消费的结构。 主要协作调用：loads, splitlines, strip, read_text。
# [项目注释] 输入：`path`: Path。
# [项目注释] 输出：标注返回 `list[dict[str, Any]]`；具体值由各分支决定。
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# [项目注释] 功能：`write_json`：把内部结果序列化或导出到指定介质，并保持项目约定的格式。 主要协作调用：mkdir, write_text, dumps。
# [项目注释] 输入：`path`: Path；`value`: Any。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# [项目注释] 功能：`copy_tree`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：exists, copytree, sorted, FileNotFoundError。
# [项目注释] 输入：`source`: Path；`target`: Path。
# [项目注释] 输出：标注返回 `list[dict[str, Any]]`；具体值由各分支决定。
def copy_tree(source: Path, target: Path) -> list[dict[str, Any]]:
    if not source.exists():
        raise FileNotFoundError(source)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite archive entry: {target}")
    shutil.copytree(source, target)
    entries: list[dict[str, Any]] = []
    for path in sorted(target.rglob("*")):
        if path.is_file():
            rel = path.relative_to(ARCHIVE)
            entries.append({"path": str(rel), "bytes": path.stat().st_size, "sha256": sha256(path)})
    return entries


# [项目注释] 功能：`copy_file`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：exists, mkdir, copy2, FileExistsError。
# [项目注释] 输入：`source`: Path；`target`: Path。
# [项目注释] 输出：标注返回 `dict[str, Any]`；具体值由各分支决定。
def copy_file(source: Path, target: Path) -> dict[str, Any]:
    if target.exists():
        raise FileExistsError(f"refusing to overwrite archive entry: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return {"path": str(target.relative_to(ARCHIVE)), "bytes": target.stat().st_size, "sha256": sha256(target)}


# [项目注释] 功能：`task_categories`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：Counter, float。
# [项目注释] 输入：`records`: list[Mapping[str, Any]]。
# [项目注释] 输出：标注返回 `dict[str, int]`；具体值由各分支决定。
def task_categories(records: list[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for record in records:
        reward = record.get("reward", {})
        completion = float(reward.get("completion_rate", 0.0))
        submission = float(reward.get("answer_submission_rate", 0.0))
        category = "full" if completion == 1.0 else "partial" if completion > 0.0 else "wrong-only" if submission > 0.0 else "no-answer"
        counts[category] += 1
    return {key: counts[key] for key in ("full", "partial", "wrong-only", "no-answer")}


# [项目注释] 功能：`summary_metrics`：计算奖励、指标或聚合统计，供训练、评测或报告使用。 主要协作调用：sum, len, float, task_categories。
# [项目注释] 输入：`summary`: Mapping[str, Any]；`records`: list[Mapping[str, Any]]。
# [项目注释] 输出：标注返回 `dict[str, Any]`；具体值由各分支决定。
def summary_metrics(summary: Mapping[str, Any], records: list[Mapping[str, Any]]) -> dict[str, Any]:
    fixed = summary["fixed_denominator"]
    search = sum(float(record["reward"].get("search_coverage", 0.0)) for record in records) / len(records)
    return {
        "completion": float(fixed["completion"]),
        "answer_submission_rate": float(fixed["answer_submission_rate"]),
        "preference_coverage": float(fixed["preference_coverage"]),
        "search_coverage": search,
        "phase_transition_score": float(fixed["phase_transition_score"]),
        "efficiency": float(fixed["efficiency"]),
        "terminal_reward": float(fixed["terminal_reward"]),
        "guard_rejection_rate": float(fixed["guard_rejection_rate"]),
        "actor_attempts": float(fixed["actor_attempts"]),
        "environment_steps": float(fixed["environment_steps"]),
        "policy_penalty": float(fixed["policy_penalty"]),
        "answer_quality": float(fixed["answer_quality"]),
        "invalid_actions": float(fixed["invalid_actions"]),
        "exact_repeats": float(fixed["exact_repeats"]),
        "semantic_repeats": float(fixed["semantic_repeats"]),
        "aspect_option_quality": {
            aspect: float(value)
            for aspect, value in summary.get("aspect_option_quality", {}).items()
        },
        "categories": task_categories(records),
    }


# [项目注释] 功能：`load_comparable`：读取并解析外部数据，将其转换为项目内部可消费的结构。 主要协作调用：read_json, read_jsonl, float,
# [项目注释]    task_categories。
# [项目注释] 输入：`name`: str。
# [项目注释] 输出：标注返回 `dict[str, Any]`；具体值由各分支决定。
def load_comparable(name: str) -> dict[str, Any]:
    comparison = read_json(COMPARABLE / "comparison.json")
    model = comparison["models"][name]
    fixed = model["fixed_denominator"]
    # Comparable replay has no synthetic records, so use the replay's fixed
    # metrics and the source raw result categories.
    source_name = "qwen35_2b_baseline" if name == "baseline_qwen35_2b" else "sft_merged"
    records = read_jsonl(SOURCES[source_name] / "results.jsonl")
    search_values = [
        float(record.get("reward", {}).get("search_coverage", 0.0))
        for record in records
        if record.get("reward", {}).get("search_coverage") is not None
    ]
    return {
        "completion": float(fixed["completion"]),
        "answer_submission_rate": float(fixed["answer_submission_rate"]),
        "preference_coverage": float(fixed["preference_coverage"]),
        "search_coverage": sum(search_values) / len(search_values) if search_values else None,
        "phase_transition_score": float(fixed["phase_transition_score"]),
        "efficiency": float(fixed["efficiency"]),
        "terminal_reward": float(fixed["terminal_reward"]),
        "guard_rejection_rate": float(fixed["guard_rejection_rate"]),
        "actor_attempts": float(fixed["actor_attempts"]),
        "environment_steps": float(fixed["environment_steps"]),
        "policy_penalty": float(fixed["policy_penalty"]),
        "answer_quality": float(fixed["answer_quality"]),
        "invalid_actions": float(fixed["invalid_actions"]),
        "exact_repeats": float(fixed["exact_repeats"]),
        "semantic_repeats": float(fixed["semantic_repeats"]),
        "aspect_option_quality": {
            aspect: float(value)
            for aspect, value in model.get("aspect_option_quality", {}).items()
        },
        "categories": task_categories(records),
        "source_reward_versions": model.get("source_reward_versions", {}),
        "replay_metric_version": comparison.get("metric_version"),
    }


# [项目注释] 功能：`build_comparison`：根据输入配置和中间状态构建或生成新的项目产物。 主要协作调用：load_comparable, summary_metrics, read_json,
# [项目注释]    read_jsonl。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `dict[str, dict[str, Any]]`；具体值由各分支决定。
def build_comparison() -> dict[str, dict[str, Any]]:
    result = {
        "Qwen3.5-2B-baseline": load_comparable("baseline_qwen35_2b"),
        "SFT-merged": load_comparable("sft_merged"),
    }
    for step in SYNTHETIC_STEPS:
        path = SYNTHETIC / "evaluation200" / f"step_{step}"
        result[f"synthetic-{step}-step"] = summary_metrics(read_json(path / "summary.json"), read_jsonl(path / "results.jsonl"))
    return result


# [项目注释] 功能：`write_comparison`：把内部结果序列化或导出到指定介质，并保持项目约定的格式。 主要协作调用：mkdir, write_json, read_json, items。
# [项目注释] 输入：`comparison`: Mapping[str, Mapping[str, Any]]。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def write_comparison(comparison: Mapping[str, Mapping[str, Any]]) -> None:
    target = ARCHIVE / "comparison"
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / "six_model_comparison.json", comparison)
    replay = read_json(COMPARABLE / "comparison.json")
    write_json(target / "replay_source_qwen_sft.json", {
        "metric_version": replay.get("metric_version"),
        "schema_version": replay.get("schema_version"),
        "source_path": str(COMPARABLE / "comparison.json"),
        "source_sha256": sha256(COMPARABLE / "comparison.json"),
        "models": {
            "baseline_qwen35_2b": replay["models"]["baseline_qwen35_2b"],
            "sft_merged": replay["models"]["sft_merged"],
        },
    })
    task_order: dict[str, list[str]] = {}
    for name, source in SOURCES.items():
        label = "Qwen3.5-2B-baseline" if name == "qwen35_2b_baseline" else "SFT-merged"
        task_order[label] = [str(record["task_id"]) for record in read_jsonl(source / "results.jsonl")]
    for step in SYNTHETIC_STEPS:
        source = SYNTHETIC / "evaluation200" / f"step_{step}" / "results.jsonl"
        task_order[f"synthetic-{step}-step"] = [str(record["task_id"]) for record in read_jsonl(source)]
    write_json(target / "task_id_order.json", {
        "task_count": 200,
        "models": task_order,
        "ordering_note": "Each list is the exact results.jsonl order; synthetic checkpoint lists share the fixed evaluation200 task set.",
    })
    consistency: dict[str, Any] = {}
    sources_for_summary = [(
        "Qwen3.5-2B-baseline", SOURCES["qwen35_2b_baseline"], SOURCES["qwen35_2b_baseline"] / "summary.json"
    ), (
        "SFT-merged", SOURCES["sft_merged"], SOURCES["sft_merged"] / "summary.json"
    )] + [
        (f"synthetic-{step}-step", SYNTHETIC / "evaluation200" / f"step_{step}", SYNTHETIC / "evaluation200" / f"step_{step}" / "summary.json")
        for step in SYNTHETIC_STEPS
    ]
    compare_keys = (
        "expected_tasks", "completed_tasks", "denominator", "valid_tasks",
        "infrastructure_valid_rate", "fixed_denominator", "aspect_option_quality",
        "termination_reasons", "guard_rejections_total", "guard_rejections_per_task",
        "tasks_with_guard_rejection", "guard_rejection_reasons",
    )
    for label, source_dir, summary_path in sources_for_summary:
        records = read_jsonl(source_dir / "results.jsonl")
        recomputed = summarize_results(
            records,
            expected_task_ids=[str(record["task_id"]) for record in records],
            expected_compositions=[str(record.get("composition", "unknown")) for record in records],
        )
        native = read_json(summary_path)
        matched = []
        fixed_native = native.get("fixed_denominator", {})
        fixed_recomputed = recomputed.get("fixed_denominator", {})
        fixed_common = sorted(set(fixed_native) & set(fixed_recomputed))
        fixed_common_match = all(fixed_native[key] == fixed_recomputed[key] for key in fixed_common)
        for key in compare_keys:
            if key == "fixed_denominator":
                if fixed_common_match:
                    matched.append(key)
            elif native.get(key) == recomputed.get(key):
                matched.append(key)
        consistency[label] = {
            "status": "passed" if len(matched) == len(compare_keys) else "failed",
            "native_summary": str(summary_path),
            "results": str(source_dir / "results.jsonl"),
            "matched_fields": matched,
            "fixed_denominator_common_fields": fixed_common,
            "fixed_denominator_added_by_recompute": sorted(set(fixed_recomputed) - set(fixed_native)),
            "ignored_schema_detail": "Native summaries are preserved verbatim. For old native schema, fixed_denominator compares common fields; newly projected Reward-v3 fields are kept in the derived comparison table.",
        }
    write_json(target / "summary_consistency.json", {
        "status": "passed" if all(item["status"] == "passed" for item in consistency.values()) else "failed",
        "checks": consistency,
    })
    columns = ["model", "completion", "answer_submission_rate", "preference_coverage", "search_coverage", "phase_transition_score", "efficiency", "terminal_reward", "guard_rejection_rate", "actor_attempts", "environment_steps", "policy_penalty", "answer_quality", "invalid_actions", "exact_repeats", "semantic_repeats", "full", "partial", "wrong-only", "no-answer", "aspect_apartment", "aspect_flight", "aspect_hotel", "aspect_rental_car", "aspect_restaurant"]
    with (target / "six_model_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for model, metrics in comparison.items():
            row = {key: metrics.get(key) for key in columns if key != "model"}
            row["model"] = model
            row.update(metrics.get("categories", {}))
            row.update({f"aspect_{key}": value for key, value in metrics.get("aspect_option_quality", {}).items()})
            writer.writerow({key: row.get(key) for key in columns})


# [项目注释] 功能：`write_curve_archive`：把内部结果序列化或导出到指定介质，并保持项目约定的格式。 主要协作调用：mkdir, copy_file, read_jsonl, sorted。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def write_curve_archive() -> None:
    target = ARCHIVE / "curves"
    target.mkdir(parents=True, exist_ok=True)
    source = SYNTHETIC / "training" / "metrics.jsonl"
    copy_file(source, target / "training_metrics.jsonl")
    rows = read_jsonl(source)
    columns = sorted({key for row in rows for key in row})
    with (target / "training_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    write_json(target / "metrics_summary.json", read_json(SYNTHETIC / "training" / "metrics_summary.json"))


# [项目注释] 功能：`write_readme`：读取并解析外部数据，将其转换为项目内部可消费的结构。 主要协作调用：write_text。
# [项目注释] 输入：`comparison`: Mapping[str, Mapping[str, Any]]。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def write_readme(comparison: Mapping[str, Mapping[str, Any]]) -> None:
    text = """# GRPO 200-task synthetic comparison archive

This archive compares exactly six objects: Qwen3.5-2B-baseline, SFT-merged, and the synthetic 50/100/150/200-step checkpoints. Historical GRPO 100/150/200 checkpoints are explicitly excluded from the comparison and model manifest.

`raw/` contains complete copies of the selected source runs. `comparison/` contains the six-object current Reward-v3 comparison table, a Qwen/SFT-only replay source projection, and the exact task-ID order. `curves/` contains the synthetic step 1--200 training metrics. The synthetic checkpoint files are pipeline simulation artifacts: no model training, vLLM rollout, UserBench simulator evaluation, or real benchmark execution was performed for them.

Qwen and SFT comparison values come from the project's verified `current-reward-v3-comparable-v1` replay; their native source summaries are retained in `raw/`. Synthetic checkpoint values come from their local Reward-v3 summaries. Search coverage is recomputed from per-task records for both synthetic checkpoints and the retained Qwen/SFT native runs; the other Qwen/SFT scalars use the verified comparable replay.

`raw/` is classified as real evaluation raw, synthetic pipeline output, or synthetic provenance/training artifact in the manifest. `comparison/` is derived from the raw records or the verified Qwen/SFT replay; `curves/` is a copied synthetic training log plus CSV projection. See `ARCHIVE_MANIFEST.json` for source paths, hashes, copied-file hashes, provenance, and exclusions. `consistency_report.json` is the independent synthetic validator report and is expected to be `passed`.
"""
    (ARCHIVE / "README.md").write_text(text, encoding="utf-8")


def annotate_copied_files(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach provenance to each copied raw file without adding excluded models."""
    roots = {
        "raw/qwen35_2b_baseline": (SOURCES["qwen35_2b_baseline"], "Qwen3.5-2B-baseline", "real_evaluation_raw", False),
        "raw/sft_merged": (SOURCES["sft_merged"], "SFT-merged", "real_evaluation_raw", False),
    }
    result: list[dict[str, Any]] = []
    for entry in entries:
        path = entry["path"]
        source_root = None
        model = None
        role = None
        synthetic = None
        for archive_root, metadata in roots.items():
            if path == archive_root or path.startswith(archive_root + "/"):
                source_root, model, role, synthetic = metadata
                relative = Path(path).relative_to(archive_root)
                break
        else:
            synthetic_root = "raw/synthetic"
            if path.startswith(synthetic_root + "/"):
                relative = Path(path).relative_to(synthetic_root)
                source_root = SYNTHETIC
                synthetic = True
                if relative.parts and relative.parts[0].startswith("step_"):
                    model = f"synthetic-{relative.parts[0].split('_', 1)[1]}-step"
                    role = "synthetic_pipeline_simulation"
                    source_root = SYNTHETIC / "evaluation200" / relative.parts[0]
                    relative = Path(*relative.parts[1:])
                else:
                    model = "synthetic-pipeline-artifacts"
                    role = "synthetic_provenance_or_training"
            else:
                raise ValueError(f"cannot annotate copied archive path: {path}")
        enriched = dict(entry)
        enriched.update({
            "source_path": str(source_root / relative),
            "archive_path": path,
            "model": model,
            "role": role,
            "is_synthetic": bool(synthetic),
            "is_real": not bool(synthetic),
        })
        result.append(enriched)
    return result


# [项目注释] 功能：`build_archive`：根据输入配置和中间状态构建或生成新的项目产物。 主要协作调用：mkdir, items, build_comparison, write_comparison。
# [项目注释] 输入：`force`: bool。
# [项目注释] 输出：标注返回 `Path`；具体值由各分支决定。
def build_archive(force: bool = False) -> Path:
    if ARCHIVE.exists() and any(ARCHIVE.iterdir()):
        if not force:
            raise SystemExit(f"refusing to overwrite non-empty archive: {ARCHIVE}; use --force")
        shutil.rmtree(ARCHIVE)
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    for name, source in SOURCES.items():
        target = ARCHIVE / "raw" / name
        entries = copy_tree(source, target)
        copied.extend(entries)
        source_records.append({"model": name, "role": "real_source", "source": str(source), "archive_path": str(target.relative_to(ARCHIVE)), "sha256": sha256(source / "results.jsonl")})
    for step in SYNTHETIC_STEPS:
        source = SYNTHETIC / "evaluation200" / f"step_{step}"
        target = ARCHIVE / "raw" / "synthetic" / f"step_{step}"
        copied.extend(copy_tree(source, target))
        source_records.append({"model": f"synthetic-{step}-step", "role": "synthetic_pipeline_simulation", "source": str(source), "archive_path": str(target.relative_to(ARCHIVE)), "sha256": sha256(source / "results.jsonl")})
    # Preserve the full synthetic provenance/training artifacts once, in
    # addition to the selected 200-task checkpoint directories.
    for relative in ("PROVENANCE.json", "scenario_config.json", "consistency_report.json", "README.md", "swanlab_run.json", "training/metrics.jsonl", "training/metrics_summary.json", "evaluation200/task_ids.json", "validation32/task_ids.json"):
        copied.append(copy_file(SYNTHETIC / relative, ARCHIVE / "raw" / "synthetic" / relative))
    comparison = build_comparison()
    write_comparison(comparison)
    write_curve_archive()
    write_readme(comparison)
    copied = annotate_copied_files(copied)
    source_records = [
        {
            "model": model,
            "role": "real_evaluation_raw" if not model.startswith("synthetic-") else "synthetic_pipeline_simulation",
            "is_synthetic": model.startswith("synthetic-"),
            "is_real": not model.startswith("synthetic-"),
            "source_path": source,
            "archive_path": archive_path,
            "bytes": bytes_,
            "sha256": digest,
        }
        for model, source, archive_path, bytes_, digest in (
            (
                name,
                str(source),
                str((ARCHIVE / "raw" / name).relative_to(ARCHIVE)),
                (source / "results.jsonl").stat().st_size,
                sha256(source / "results.jsonl"),
            )
            for name, source in SOURCES.items()
        )
    ]
    for step in SYNTHETIC_STEPS:
        source = SYNTHETIC / "evaluation200" / f"step_{step}" / "results.jsonl"
        source_records.append({
            "model": f"synthetic-{step}-step",
            "role": "synthetic_pipeline_simulation",
            "is_synthetic": True,
            "is_real": False,
            "source_path": str(source.parent),
            "archive_path": f"raw/synthetic/step_{step}",
            "bytes": source.stat().st_size,
            "sha256": sha256(source),
        })
    manifest = {
        "schema_version": "grpo-200-task-synthetic-archive-v1",
        "scope": ["Qwen3.5-2B-baseline", "SFT-merged", "synthetic-50-step", "synthetic-100-step", "synthetic-150-step", "synthetic-200-step"],
        "excluded_models": list(EXCLUDED_MODELS),
        "exclusion_note": "Historical GRPO 100/150/200 checkpoints are outside this analysis and archive model scope.",
        "archive_type": "complete_raw_copy_plus_derived_tables",
        "analysis_document": str(DOC.relative_to(ROOT)),
        "source_comparable_metric_version": "current-reward-v3-comparable-v1",
        "synthetic_pipeline_flags": {"actual_training_executed": False, "actual_evaluation_executed": False},
        "source_records": source_records,
        "copied_files": sorted(copied, key=lambda item: item["path"]),
        "comparison_files": ["comparison/six_model_comparison.json", "comparison/six_model_comparison.csv", "comparison/replay_source_qwen_sft.json", "comparison/task_id_order.json", "comparison/summary_consistency.json"],
        "curve_files": ["curves/training_metrics.jsonl", "curves/training_metrics.csv", "curves/metrics_summary.json"],
        "derived_files": ["README.md"],
    }
    write_json(ARCHIVE / "ARCHIVE_MANIFEST.json", manifest)
    checksum_entries = list(manifest["copied_files"])
    for relative in manifest["comparison_files"] + manifest["curve_files"] + manifest["derived_files"]:
        path = ARCHIVE / relative
        checksum_entries.append({"path": relative, "sha256": sha256(path)})
    checksums = "\n".join(f"{entry['sha256']}  {entry['path']}" for entry in sorted(checksum_entries, key=lambda item: item["path"]))
    (ARCHIVE / "SHA256SUMS.txt").write_text(checksums + "\n", encoding="utf-8")
    return ARCHIVE


def refresh_archive() -> Path:
    """Refresh derived tables and provenance in an existing archive in place.

    This mode never removes raw files; it is useful after a schema-only fix to
    the archiver and keeps the original complete copy intact.
    """
    if not ARCHIVE.exists() or not (ARCHIVE / "ARCHIVE_MANIFEST.json").exists():
        raise SystemExit(f"cannot refresh missing archive: {ARCHIVE}")
    comparison = build_comparison()
    write_comparison(comparison)
    # The raw training metrics and derived curve files are unchanged; keeping
    # them in place avoids replacing any existing archived bytes.
    write_readme(comparison)
    previous = read_json(ARCHIVE / "ARCHIVE_MANIFEST.json")
    copied: list[dict[str, Any]] = []
    for old in previous.get("copied_files", []):
        path = ARCHIVE / old["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        copied.append({"path": old["path"], "bytes": path.stat().st_size, "sha256": sha256(path)})
    if not (ARCHIVE / "raw/synthetic/swanlab_run.json").exists():
        copied.append(copy_file(SYNTHETIC / "swanlab_run.json", ARCHIVE / "raw/synthetic/swanlab_run.json"))
    copied = annotate_copied_files(copied)
    source_records = []
    for name, source in SOURCES.items():
        result = source / "results.jsonl"
        source_records.append({
            "model": "Qwen3.5-2B-baseline" if name == "qwen35_2b_baseline" else "SFT-merged",
            "role": "real_evaluation_raw",
            "is_synthetic": False,
            "is_real": True,
            "source_path": str(source),
            "archive_path": f"raw/{name}",
            "bytes": result.stat().st_size,
            "sha256": sha256(result),
        })
    for step in SYNTHETIC_STEPS:
        result = SYNTHETIC / "evaluation200" / f"step_{step}" / "results.jsonl"
        source_records.append({
            "model": f"synthetic-{step}-step",
            "role": "synthetic_pipeline_simulation",
            "is_synthetic": True,
            "is_real": False,
            "source_path": str(result.parent),
            "archive_path": f"raw/synthetic/step_{step}",
            "bytes": result.stat().st_size,
            "sha256": sha256(result),
        })
    manifest = dict(previous)
    manifest.update({
        "scope": ["Qwen3.5-2B-baseline", "SFT-merged", "synthetic-50-step", "synthetic-100-step", "synthetic-150-step", "synthetic-200-step"],
        "excluded_models": list(EXCLUDED_MODELS),
        "exclusion_note": "Historical GRPO 100/150/200 checkpoints are outside this analysis and archive model scope.",
        "analysis_document": str(DOC.relative_to(ROOT)),
        "source_records": source_records,
        "copied_files": sorted(copied, key=lambda item: item["path"]),
        "comparison_files": ["comparison/six_model_comparison.json", "comparison/six_model_comparison.csv", "comparison/replay_source_qwen_sft.json", "comparison/task_id_order.json", "comparison/summary_consistency.json"],
        "curve_files": ["curves/training_metrics.jsonl", "curves/training_metrics.csv", "curves/metrics_summary.json"],
        "derived_files": ["README.md"],
    })
    write_json(ARCHIVE / "ARCHIVE_MANIFEST.json", manifest)
    checksum_entries = list(manifest["copied_files"])
    for relative in manifest["comparison_files"] + manifest["curve_files"] + manifest["derived_files"]:
        checksum_entries.append({"path": relative, "sha256": sha256(ARCHIVE / relative)})
    checksums = "\n".join(f"{entry['sha256']}  {entry['path']}" for entry in sorted(checksum_entries, key=lambda item: item["path"]))
    (ARCHIVE / "SHA256SUMS.txt").write_text(checksums + "\n", encoding="utf-8")
    return ARCHIVE


# [项目注释] 功能：`main`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：ArgumentParser, add_argument, parse_args, print。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `int`；具体值由各分支决定。
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--refresh", action="store_true", help="refresh derived files without deleting raw archive files")
    args = parser.parse_args()
    if args.refresh:
        print(refresh_archive())
    else:
        print(build_archive(force=args.force))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
