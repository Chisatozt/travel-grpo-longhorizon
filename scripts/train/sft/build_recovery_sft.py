#!/usr/bin/env python3
"""Build and audit recovery-boundary SFT records without training a model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from travel_grpo.training.sft.recovery import (
    build_recovery_sft_dataset,
    load_cpu_chat_template_tokenizer,
)
from travel_grpo.training.sft.dataset import load_tool_schema


# [项目注释] 功能：`build_parser`：读取并解析外部数据，将其转换为项目内部可消费的结构。 主要协作调用：ArgumentParser, add_argument。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `argparse.ArgumentParser`；具体值由各分支决定。
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--targets-dir",
        type=Path,
        default=None,
        help="recovery-target-v1 directory (default: outputs/recovery_targets/recovery-target-v1)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output directory (default: outputs/recovery_sft/recovery-sft-v1)",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="local tokenizer directory containing tokenizer.json and chat_template.jinja",
    )
    parser.add_argument(
        "--tool-schema",
        type=Path,
        default=None,
        help="UserBench tool schema YAML",
    )
    parser.add_argument("--max-sequence-length", type=int, default=16384)
    return parser


# [项目注释] 功能：`main`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：parse_args, resolve,
# [项目注释]    load_cpu_chat_template_tokenizer, load_tool_schema。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `int`；具体值由各分支决定。
def main() -> int:
    args = build_parser().parse_args()
    root = ROOT.resolve()
    targets = (args.targets_dir or root / "outputs/recovery_targets/recovery-target-v1").resolve()
    output = (args.output or root / "outputs/recovery_sft/recovery-sft-v1").resolve()
    tokenizer_path = (args.tokenizer or root / "outputs/models/sft-merged").resolve()
    schema_path = (args.tool_schema or root / "configs/tool_config/userbench_tools.yaml").resolve()
    tokenizer = load_cpu_chat_template_tokenizer(tokenizer_path)
    tool_schema = load_tool_schema(schema_path)
    paths, manifest = build_recovery_sft_dataset(
        targets,
        output,
        tokenizer,
        tool_schema,
        max_sequence_length=args.max_sequence_length,
    )
    print(json.dumps(manifest["audit"], ensure_ascii=False, sort_keys=True))
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
