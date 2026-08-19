#!/usr/bin/env python3
"""Extract recovery-boundary-v1 contexts from existing local artifacts.

This command is CPU-only and read-only with respect to source trajectories.
Derived JSONL and its manifest are written below ``outputs/`` by default.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from travel_grpo.data.recovery.boundaries import (
    extract_recovery_boundaries,
    write_extraction,
)


# [项目注释] 功能：`build_parser`：读取并解析外部数据，将其转换为项目内部可消费的结构。 主要协作调用：ArgumentParser, add_argument。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `argparse.ArgumentParser`；具体值由各分支决定。
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=ROOT,
        help="Project root containing data/ and outputs/ (default: repository root).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Derived output directory (default: outputs/recovery_boundaries/recovery-boundary-v1).",
    )
    return parser


# [项目注释] 功能：`main`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：parse_args, resolve, extract_recovery_boundaries,
# [项目注释]    write_extraction。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `int`；具体值由各分支决定。
def main() -> int:
    args = build_parser().parse_args()
    project_root = args.project_root.resolve()
    output = args.output or project_root / "outputs/recovery_boundaries/recovery-boundary-v1"
    records, manifest = extract_recovery_boundaries(project_root)
    contexts_path, manifest_path = write_extraction(records, manifest, output)
    print(json.dumps(manifest["counts"], ensure_ascii=False, sort_keys=True))
    print(f"contexts: {contexts_path}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
