#!/usr/bin/env python3
"""Prepare an SFT model and launch the project GRPO training profile.

The command intentionally does not run SFT.  It is the hand-off point after
Stage 2 SFT has produced a LoRA adapter:

    SFT adapter -> merged model -> hidden-label-free GRPO data -> GRPO

Existing complete merge/data artifacts are verified and reused.  Incomplete
artifacts are never deleted or silently overwritten.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[3]
MERGE_SCRIPT = ROOT / "scripts/train/sft/merge_lora.py"
PREPARE_DATA_SCRIPT = ROOT / "scripts/train/grpo/prepare_data.py"
TRAIN_SCRIPT = ROOT / "scripts/train/grpo/train_grpo.py"

DEFAULT_SFT_ADAPTER = ROOT / "outputs/sft/qwen3.5-2b-lora-stage2"
DEFAULT_MERGED_MODEL = ROOT / "outputs/models/sft-merged"
DEFAULT_GRPO_CONFIG = ROOT / "configs/train/grpo/grpo.yaml"
DEFAULT_TRAIN_SOURCE = ROOT / "data/grpo/train.parquet"
DEFAULT_VALIDATION_SOURCE = ROOT / "data/grpo/validation.parquet"
DEFAULT_DATA_OUTPUT = ROOT / "outputs/grpo/data"


def _project_path(value: str | Path, *, field: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _adapter_complete(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "adapter_config.json").is_file()
        and any(
            (path / filename).is_file()
            for filename in ("adapter_model.safetensors", "adapter_model.bin")
        )
    )


def _merged_model_complete(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "config.json").is_file()
        and (path / "tokenizer_config.json").is_file()
        and (
            any(path.glob("*.safetensors"))
            or (path / "pytorch_model.bin").is_file()
        )
    )


def _merge_manifest_matches(
    path: Path, *, adapter: Path, base_model: str
) -> tuple[bool, str | None]:
    manifest_path = path / "merge_manifest.json"
    if not manifest_path.is_file():
        return False, f"merge manifest is missing: {manifest_path}"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"merge manifest is unreadable: {manifest_path}: {exc}"
    if not isinstance(manifest, dict):
        return False, f"merge manifest must be an object: {manifest_path}"
    observed_adapter = manifest.get("adapter")
    if observed_adapter != str(adapter):
        return (
            False,
            "merge manifest adapter does not match the requested SFT adapter: "
            f"{observed_adapter!r} != {str(adapter)!r}",
        )
    if manifest.get("base_model") != base_model:
        return (
            False,
            "merge manifest base model does not match the requested base model: "
            f"{manifest.get('base_model')!r} != {base_model!r}",
        )
    return True, None


def _data_artifact_state(output: Path) -> str:
    artifacts = (
        output / "train.parquet",
        output / "validation.parquet",
        output / "manifest.json",
    )
    present = [path.exists() for path in artifacts]
    if all(present):
        return "complete"
    if not any(present):
        return "missing"
    return "partial"


def _command_text(command: Sequence[str]) -> str:
    return shlex.join(str(value) for value in command)


def _require_grpo_data_dependency() -> None:
    if importlib.util.find_spec("pyarrow") is not None:
        return
    raise RuntimeError(
        "the selected Python interpreter is missing pyarrow: "
        f"{sys.executable}. Install the project data/GRPO environment with "
        "`bash scripts/setup.sh`, or run `python -m pip install -e '.[data]'` "
        "using the same interpreter. The wrapper uses `.venv/bin/python` when "
        "that environment exists; override it with PYTHON_BIN if needed."
    )


def _run(label: str, command: Sequence[str]) -> None:
    print(f"[grpo-from-sft] {label}: {_command_text(command)}", flush=True)
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode:
        raise RuntimeError(
            f"{label} failed with exit code {completed.returncode}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-adapter", type=Path, default=DEFAULT_SFT_ADAPTER)
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--merged-model", type=Path, default=DEFAULT_MERGED_MODEL)
    parser.add_argument("--config", type=Path, default=DEFAULT_GRPO_CONFIG)
    parser.add_argument(
        "--train-source", type=Path, default=DEFAULT_TRAIN_SOURCE
    )
    parser.add_argument(
        "--validation-source", type=Path, default=DEFAULT_VALIDATION_SOURCE
    )
    parser.add_argument("--data-output", type=Path, default=DEFAULT_DATA_OUTPUT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--logger", choices=("console", "swanlab"), default="console")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force-data",
        action="store_true",
        help="explicitly rebuild existing GRPO data artifacts",
    )
    parser.add_argument(
        "--stall-recovery",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable recovery only for GRPO training rollouts",
    )
    parser.add_argument("--stall-threshold", type=int, default=4)
    parser.add_argument(
        "--turn-credit-mode", choices=("off", "shadow", "train")
    )
    parser.add_argument("--turn-credit-lambda", type=float)
    parser.add_argument("--turn-credit-band", type=float)
    parser.add_argument("overrides", nargs="*", help="extra veRL Hydra overrides")
    return parser


def _merge_command(args: argparse.Namespace, adapter: Path, merged: Path) -> list[str]:
    command = [
        sys.executable,
        str(MERGE_SCRIPT),
        "--base-model",
        str(args.base_model),
        "--adapter",
        str(adapter),
        "--output",
        str(merged),
    ]
    if args.dry_run:
        command.append("--dry-run")
    return command


def _prepare_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(PREPARE_DATA_SCRIPT),
        "--train-source",
        str(_project_path(args.train_source, field="train source")),
        "--validation-source",
        str(_project_path(args.validation_source, field="validation source")),
        "--output-root",
        str(_project_path(args.data_output, field="GRPO data output")),
    ]
    if args.dry_run:
        command.append("--dry-run")
    elif args.force_data:
        command.append("--force")
    return command


def _train_command(
    args: argparse.Namespace, *, merged: Path
) -> list[str]:
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--config",
        str(_project_path(args.config, field="GRPO config")),
        "--model-path",
        str(merged),
        "--data-output",
        str(_project_path(args.data_output, field="GRPO data output")),
    ]
    if args.output is not None:
        command.extend(("--output", str(args.output)))
    if args.resume:
        command.append("--resume")
    if args.logger != "console":
        command.extend(("--logger", args.logger))
    if args.dry_run:
        command.append("--dry-run")
    command.append("--stall-recovery" if args.stall_recovery else "--no-stall-recovery")
    command.extend(("--stall-threshold", str(args.stall_threshold)))
    if args.turn_credit_mode is not None:
        command.extend(("--turn-credit-mode", args.turn_credit_mode))
    if args.turn_credit_lambda is not None:
        command.extend(("--turn-credit-lambda", str(args.turn_credit_lambda)))
    if args.turn_credit_band is not None:
        command.extend(("--turn-credit-band", str(args.turn_credit_band)))
    command.extend(str(value) for value in args.overrides)
    return command


def run(args: argparse.Namespace) -> dict[str, Any]:
    _require_grpo_data_dependency()
    if args.stall_threshold < 1:
        raise ValueError("--stall-threshold must be >= 1")
    adapter = _project_path(args.sft_adapter, field="SFT adapter")
    merged = _project_path(args.merged_model, field="merged model")
    data_output = _project_path(args.data_output, field="GRPO data output")

    if not _adapter_complete(adapter):
        raise RuntimeError(
            "SFT LoRA adapter is incomplete or not generated yet: "
            f"{adapter}. Expected adapter_config.json and adapter_model.safetensors "
            "or adapter_model.bin. Run Stage-2 SFT first."
        )

    actions: list[str] = []
    if _merged_model_complete(merged):
        matches, reason = _merge_manifest_matches(
            merged, adapter=adapter, base_model=str(args.base_model)
        )
        if not matches:
            raise RuntimeError(
                f"existing merged model cannot be safely reused: {reason}. "
                "Choose a new --merged-model or inspect it explicitly."
            )
        actions.append("reuse merged model")
    else:
        if merged.exists():
            if any(merged.iterdir()):
                raise RuntimeError(
                    "merged model directory exists but is incomplete/non-empty; "
                    f"refusing to overwrite: {merged}"
                )
            raise RuntimeError(
                "merged model directory exists but is empty; remove it manually "
                f"before merging: {merged}"
            )
        _run("merge SFT adapter", _merge_command(args, adapter, merged))
        actions.append("merge SFT adapter")

    data_state = _data_artifact_state(data_output)
    if args.dry_run:
        _run("prepare GRPO data (dry-run)", _prepare_command(args))
        actions.append(f"inspect GRPO data ({data_state})")
    elif data_state == "complete" and not args.force_data:
        _run(
            "verify existing GRPO data",
            [
                sys.executable,
                str(PREPARE_DATA_SCRIPT),
                "--verify-only",
                "--output-root",
                str(data_output),
            ],
        )
        actions.append("verify existing GRPO data")
    elif data_state == "partial" and not args.force_data:
        raise RuntimeError(
            "GRPO data output is partial; refusing to overwrite it. "
            f"Use --force-data after inspection: {data_output}"
        )
    else:
        _run("prepare GRPO data", _prepare_command(args))
        actions.append("prepare GRPO data")

    _run("launch GRPO", _train_command(args, merged=merged))
    actions.append("launch GRPO")
    return {
        "status": "dry-run" if args.dry_run else "launched",
        "sft_adapter": str(adapter),
        "merged_model": str(merged),
        "grpo_data": str(data_output),
        "actions": actions,
        "stall_recovery": {
            "enabled": bool(args.stall_recovery),
            "threshold": int(args.stall_threshold),
        },
    }


def main() -> int:
    try:
        report = run(build_parser().parse_args())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"GRPO-from-SFT error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
