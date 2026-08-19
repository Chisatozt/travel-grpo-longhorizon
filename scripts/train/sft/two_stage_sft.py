#!/usr/bin/env python3
"""Audit, render, or train the two-stage Qwen3.5-2B SFT curriculum."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
SFT_TRAIN = ROOT / "scripts/train/sft/sft_train.py"


# [项目注释] 功能：`build_parser`：读取并解析外部数据，将其转换为项目内部可消费的结构。 主要协作调用：ArgumentParser, add_argument,
# [项目注释]    add_mutually_exclusive_group。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `argparse.ArgumentParser`；具体值由各分支决定。
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage1-config",
        type=Path,
        default=ROOT / "configs/train/sft/sft_stage1_lora.yaml",
    )
    parser.add_argument(
        "--stage2-config",
        type=Path,
        default=ROOT / "configs/train/sft/sft_stage2_lora.yaml",
    )
    parser.add_argument("--stage", choices=("all", "1", "2"), default="all")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--render-smoke", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--allow-small-smoke", action="store_true")
    parser.add_argument("--stage1-resume-from-checkpoint", type=Path)
    parser.add_argument("--stage2-resume-from-checkpoint", type=Path)
    return parser


# [项目注释] 功能：`_load_config`：读取并解析外部数据，将其转换为项目内部可消费的结构。 主要协作调用：safe_load, isinstance, ValueError,
# [项目注释]    RuntimeError。
# [项目注释] 输入：`path`: Path。
# [项目注释] 输出：标注返回 `dict[str, Any]`；具体值由各分支决定。
def _load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("two-stage SFT requires PyYAML") from exc
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read SFT stage config: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"SFT stage config must be a mapping: {path}")
    return value


# [项目注释] 功能：`_project_path`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：Path, ValueError, is_absolute, resolve。
# [项目注释] 输入：`value`: Any；`field`: str。
# [项目注释] 输出：标注返回 `Path`；具体值由各分支决定。
def _project_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty path")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


# [项目注释] 功能：`_stage_paths`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：_load_config, _project_path, ValueError,
# [项目注释]    isinstance。
# [项目注释] 输入：`config_path`: Path。
# [项目注释] 输出：标注返回 `tuple[Path, Path | None]`；具体值由各分支决定。
def _stage_paths(config_path: Path) -> tuple[Path, Path | None]:
    config = _load_config(config_path)
    training = config.get("training")
    lora = config.get("lora")
    if not isinstance(training, dict) or not isinstance(lora, dict):
        raise ValueError(f"stage config is missing training/lora mappings: {config_path}")
    output = _project_path(training.get("output_dir"), "training.output_dir")
    init_value = lora.get("init_from")
    init_from = (
        _project_path(init_value, "lora.init_from") if init_value is not None else None
    )
    return output, init_from


# [项目注释] 功能：`_adapter_complete`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：is_file, any。
# [项目注释] 输入：`path`: Path。
# [项目注释] 输出：标注返回 `bool`；具体值由各分支决定。
def _adapter_complete(path: Path) -> bool:
    return (path / "adapter_config.json").is_file() and any(
        (path / name).is_file()
        for name in ("adapter_model.safetensors", "adapter_model.bin")
    )


# [项目注释] 功能：`_command`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：str, extend, resolve。
# [项目注释] 输入：`config`: Path；`args`: argparse.Namespace；`resume`: Path | None。
# [项目注释] 输出：标注返回 `list[str]`；具体值由各分支决定。
def _command(
    config: Path,
    args: argparse.Namespace,
    resume: Path | None,
) -> list[str]:
    command = [sys.executable, str(SFT_TRAIN), "--config", str(config.resolve())]
    if args.dry_run:
        command.append("--dry-run")
    elif args.render_smoke:
        command.extend(("--dry-run", "--render-smoke"))
    if args.limit is not None:
        command.extend(("--limit", str(args.limit)))
    if args.allow_small_smoke:
        command.append("--allow-small-smoke")
    if resume is not None:
        command.extend(("--resume-from-checkpoint", str(resume.resolve())))
    return command


# [项目注释] 功能：`_run`：编排一个训练、采集、评测或 replay 流程，并汇总其结果。 主要协作调用：print, run, RuntimeError。
# [项目注释] 输入：`label`: str；`command`: list[str]。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def _run(label: str, command: list[str]) -> None:
    print(f"[two-stage-sft] {label}", file=sys.stderr, flush=True)
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode:
        raise RuntimeError(f"{label} failed with exit code {completed.returncode}")


# [项目注释] 功能：`run`：编排一个训练、采集、评测或 replay 流程，并汇总其结果。 主要协作调用：resolve, _stage_paths, ValueError, _run。
# [项目注释] 输入：`args`: argparse.Namespace。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def run(args: argparse.Namespace) -> None:
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    stage1_config = args.stage1_config.resolve()
    stage2_config = args.stage2_config.resolve()
    stage1_output, stage1_init = _stage_paths(stage1_config)
    _, stage2_init = _stage_paths(stage2_config)
    if stage1_init is not None:
        raise ValueError("Stage 1 must create a new adapter, not initialize from one")
    if stage2_init != stage1_output:
        raise ValueError(
            "Stage-2 lora.init_from must equal the Stage-1 training.output_dir"
        )

    inspection_only = args.dry_run or args.render_smoke
    if args.stage in {"all", "1"}:
        _run(
            "Stage 1: safe prefix bootstrap",
            _command(stage1_config, args, args.stage1_resume_from_checkpoint),
        )
    if args.stage in {"all", "2"}:
        if not inspection_only and not _adapter_complete(stage1_output):
            raise RuntimeError(
                f"Stage-1 adapter is incomplete; Stage 2 cannot start: {stage1_output}"
            )
        _run(
            "Stage 2: complete Gold/Silver trajectories",
            _command(stage2_config, args, args.stage2_resume_from_checkpoint),
        )


# [项目注释] 功能：`main`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：run, parse_args, SystemExit, build_parser。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def main() -> None:
    try:
        run(build_parser().parse_args())
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Two-stage SFT error: {exc}") from exc


if __name__ == "__main__":
    main()
