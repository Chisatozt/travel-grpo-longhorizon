"""Audit collected Teacher trajectories for SFT admission.

The default source is the atomic per-task checkpoint directory, which is safer
than the flat JSONL artifacts while collection is still running.  The audit
reuses the same record-level admission and action-only rendering code as the
formal SFT entry point.

Example::

    python scripts/train/sft/check_teacher_trajectories.py

Use ``--no-render`` when only the serialized trajectory contract should be
checked.  Rendering requires the local Qwen tokenizer cache.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from travel_grpo.training.sft.dataset import (  # noqa: E402
    SFTDatasetError,
    build_action_only_dataset,
    load_tool_schema,
    sft_admission_reasons,
)


DEFAULT_RUN_DIR = ROOT / "outputs/teacher_trajectories/runs/sft-train-composition-v3"
DEFAULT_GOLD = ROOT / "outputs/teacher_trajectories/sft_train.accepted.jsonl"
DEFAULT_SILVER = ROOT / "outputs/teacher_trajectories/sft_train.silver.jsonl"
DEFAULT_CONFIG = ROOT / "configs/train/sft/sft_lora.yaml"


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(
                {"line": line_number, "reason": "invalid_json", "detail": str(exc)}
            )
            continue
        if not isinstance(value, Mapping):
            errors.append(
                {"line": line_number, "reason": "record_not_mapping"}
            )
            continue
        records.append(dict(value))
    return records, errors


def _load_checkpoint_records(run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    task_dir = run_dir / "tasks"
    if not task_dir.is_dir():
        raise FileNotFoundError(f"checkpoint task directory does not exist: {task_dir}")
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for path in sorted(task_dir.glob("*.json")):
        try:
            checkpoint = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append({"file": str(path), "reason": "invalid_checkpoint", "detail": str(exc)})
            continue
        if not isinstance(checkpoint, Mapping):
            errors.append({"file": str(path), "reason": "checkpoint_not_mapping"})
            continue
        tier = str(checkpoint.get("quality_tier", ""))
        trajectory = checkpoint.get("trajectory")
        if tier not in {"gold", "silver"} or not isinstance(trajectory, Mapping):
            continue
        record = dict(trajectory)
        record["_checkpoint_quality_tier"] = tier
        record["_checkpoint_file"] = str(path)
        records.append(record)
    return records, errors


def _load_artifact_records(gold_path: Path, silver_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for tier, path in (("gold", gold_path), ("silver", silver_path)):
        if not path.is_file():
            errors.append({"file": str(path), "reason": "missing_artifact"})
            continue
        values, file_errors = _read_jsonl(path)
        for value in values:
            value["_artifact_quality_tier"] = tier
            value["_artifact_file"] = str(path)
        records.extend(values)
        errors.extend({"file": str(path), **item} for item in file_errors)
    return records, errors


def _load_sft_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - project dependency.
        raise RuntimeError("PyYAML is required to read the SFT config") from exc
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ValueError(f"SFT config must be a mapping: {path}")
    return dict(document)


def _render_records(
    records: list[dict[str, Any]],
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    data = config.get("data")
    model = config.get("model")
    if not isinstance(data, Mapping) or not isinstance(model, Mapping):
        return {"status": "skipped", "reason": "SFT config is missing model/data"}
    try:
        from transformers import AutoTokenizer

        cache_dir = ROOT / str(model.get("cache_dir", "outputs/cache/huggingface"))
        tokenizer = AutoTokenizer.from_pretrained(
            str(model["base"]),
            cache_dir=str(cache_dir),
            local_files_only=True,
            trust_remote_code=bool(model.get("trust_remote_code", False)),
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        schema = load_tool_schema(ROOT / str(data["tool_schema_path"]))
    except Exception as exc:
        return {"status": "skipped", "reason": "tokenizer_or_schema_unavailable", "detail": str(exc)}

    max_length = int(data.get("max_sequence_length", 32768))
    tiers = tuple(str(value) for value in data.get("accepted_quality_tiers", ("gold", "silver")))
    rendered = 0
    action_examples = 0
    failures: list[dict[str, Any]] = []
    overlong: list[dict[str, Any]] = []
    for record in records:
        try:
            examples, dropped = build_action_only_dataset(
                [record],
                tokenizer,
                schema,
                max_sequence_length=max_length,
                accepted_quality_tiers=tiers,
            )
        except Exception as exc:
            failures.append({"task_id": record.get("task_id"), "reason": str(exc)})
            continue
        rendered += 1
        action_examples += len(examples)
        overlong.extend(dropped)
    return {
        "status": "checked",
        "rendered_trajectories": rendered,
        "action_examples": action_examples,
        "overlong_trajectories": overlong,
        "render_failures": failures,
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    if args.source == "checkpoints" or (
        args.source == "auto" and (args.run_dir / "tasks").is_dir()
    ):
        records, load_errors = _load_checkpoint_records(args.run_dir)
        source = "checkpoints"
    else:
        records, load_errors = _load_artifact_records(args.gold, args.silver)
        source = "artifacts"

    details: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicate_ids: set[str] = set()
    reasons_counter: Counter[str] = Counter()
    eligible: list[dict[str, Any]] = []
    for record in records:
        task_id = str(record.get("task_id", ""))
        tier = str(record.get("_checkpoint_quality_tier", record.get("_artifact_quality_tier", record.get("quality_tier", "unknown"))))
        reasons = set(sft_admission_reasons(record, accepted_quality_tiers=("gold", "silver")))
        if task_id in seen:
            duplicate_ids.add(task_id)
            reasons.add("duplicate_task_id")
        seen.add(task_id)
        for reason in reasons:
            reasons_counter[reason] += 1
        item = {
            "task_id": task_id,
            "quality_tier": tier,
            "composition": record.get("composition"),
            "eligible_for_sft": not reasons,
            "reasons": sorted(reasons),
        }
        details.append(item)
        if not reasons:
            eligible.append(record)

    config = _load_sft_config(args.config)
    data = config.get("data", {})
    required_compositions = [str(value) for value in data.get("required_compositions", [])]
    observed_compositions = Counter(str(record.get("composition")) for record in eligible)
    missing_compositions = sorted(set(required_compositions) - set(observed_compositions))
    minimum_train = int(data.get("minimum_train_trajectories", 0))
    render = {"status": "disabled"}
    if not args.no_render and eligible:
        render = _render_records(eligible, config=config)

    render_ok = render.get("status") == "checked" and not render.get("render_failures") and not render.get("overlong_trajectories")
    formal_ready = (
        len(eligible) >= minimum_train
        and not missing_compositions
        and render_ok
    )
    report = {
        "source": source,
        "source_path": str(args.run_dir if source == "checkpoints" else args.gold.parent),
        "candidate_trajectories": len(records),
        "eligible_for_sft": len(eligible),
        "ineligible_for_sft": len(records) - len(eligible),
        "quality_tiers": dict(Counter(
            str(value.get("quality_tier", "unknown")) for value in details
        )),
        "composition_distribution": dict(sorted(observed_compositions.items())),
        "minimum_train_trajectories": minimum_train,
        "missing_required_compositions": missing_compositions,
        "formal_sft_ready": formal_ready,
        "render": render,
        "rejection_reasons": dict(sorted(reasons_counter.items())),
        "duplicate_task_ids": sorted(duplicate_ids),
        "load_errors": load_errors,
        "details": details,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("auto", "checkpoints", "artifacts"), default="auto")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--silver", type=Path, default=DEFAULT_SILVER)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/teacher_trajectories/sft_audit.json")
    parser.add_argument("--no-render", action="store_true", help="only run serialized-record admission checks")
    return parser


def main() -> None:
    report = audit(build_parser().parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
