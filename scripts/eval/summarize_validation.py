#!/usr/bin/env python3
"""Convert one veRL validation generation dump into the fixed 132-task summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import pyarrow.parquet as pq
from travel_grpo.evaluation.artifacts import atomic_json
from travel_grpo.evaluation.validation import summarize_validation_file


# [项目注释] 功能：`main`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：ArgumentParser, add_argument, parse_args,
# [项目注释]    to_pylist。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `int`；具体值由各分支决定。
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("generation_jsonl", type=Path)
    parser.add_argument("--tasks", type=Path, default=ROOT / "data/grpo/validation.parquet")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    tasks = pq.read_table(args.tasks).to_pylist()
    summary = summarize_validation_file(args.generation_jsonl, tasks)
    output = args.output or args.generation_jsonl.with_suffix(".summary.json")
    atomic_json(output, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
