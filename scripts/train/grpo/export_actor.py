#!/usr/bin/env python3
"""Export a veRL FSDP actor checkpoint as a Hugging Face model."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


# [项目注释] 功能：`main`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：ArgumentParser, add_argument, parse_args,
# [项目注释]    fullmatch。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `int`；具体值由各分支决定。
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("actor_checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--selection",
        type=Path,
        help="checkpoint_selection.json (defaults to the GRPO run root)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    source, target = args.actor_checkpoint.resolve(), args.output_dir.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"actor checkpoint is missing: {source}")
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {target}")
    match = re.fullmatch(r"global_step_(\d+)", source.parent.name)
    if not match or source.name != "actor":
        raise ValueError("actor checkpoint must end in global_step_<N>/actor")
    selection_path = (
        args.selection.resolve()
        if args.selection
        else source.parents[1] / "checkpoint_selection.json"
    )
    try:
        selection_bytes = selection_path.read_bytes()
        selection = json.loads(selection_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read checkpoint selection: {selection_path}") from exc
    selected_step = int(selection.get("selected_step", -1))
    if selection.get("passed") is not True or selected_step != int(match.group(1)):
        raise ValueError(
            f"checkpoint step {match.group(1)} is not the passed selected step {selected_step}"
        )
    command = [sys.executable, "-m", "verl.model_merger", "merge", "--backend", "fsdp", "--local_dir", str(source), "--target_dir", str(target)]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "valid": True,
                    "selected_step": selected_step,
                    "selection": str(selection_path),
                    "command": command,
                },
                indent=2,
            )
        )
        return 0
    target.mkdir(parents=True, exist_ok=True)
    result = subprocess.call(command)
    if result == 0:
        if (
            not (target / "config.json").is_file()
            or not (target / "tokenizer_config.json").is_file()
            or not (
                any(target.glob("*.safetensors"))
                or (target / "pytorch_model.bin").is_file()
            )
        ):
            raise RuntimeError("veRL model merger returned success but output is incomplete")
        (target / "export_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "travel-grpo-actor-export-v1",
                    "source": str(source),
                    "backend": "fsdp",
                    "selected_step": selected_step,
                    "selection_path": str(selection_path),
                    "selection_sha256": hashlib.sha256(selection_bytes).hexdigest(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
