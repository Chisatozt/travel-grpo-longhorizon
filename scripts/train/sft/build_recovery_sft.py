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

from travel_grpo.training.recovery_sft import (
    build_recovery_sft_dataset,
    load_cpu_chat_template_tokenizer,
)
from travel_grpo.training.sft_dataset import load_tool_schema


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
