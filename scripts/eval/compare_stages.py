#!/usr/bin/env python3
"""Generate the formal paired comparison after all three 471-task runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from travel_grpo.evaluation.artifacts import atomic_json
from travel_grpo.evaluation.comparison import compare_stage_results


# [项目注释] 功能：`read_jsonl`：读取并解析外部数据，将其转换为项目内部可消费的结构。 主要协作调用：loads, splitlines, strip, read_text。
# [项目注释] 输入：`path`: Path。
# [项目注释] 输出：标注返回 `list[dict]`；具体值由各分支决定。
def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# [项目注释] 功能：`main`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：ArgumentParser, add_argument, parse_args,
# [项目注释]    compare_stage_results。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `int`；具体值由各分支决定。
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "outputs/evaluation")
    parser.add_argument(
        "--allow-subset",
        action="store_true",
        help="compare three matching explicit subset contracts instead of formal 471-task runs",
    )
    args = parser.parse_args()
    stages = {}
    for stage in ("baseline", "sft", "grpo"):
        stage_root = args.root / stage
        stages[stage] = {
            "contract": json.loads((stage_root / "contract.json").read_text(encoding="utf-8")),
            "results": read_jsonl(stage_root / "results.jsonl"),
        }
    comparison = compare_stage_results(stages, allow_subset=args.allow_subset)
    atomic_json(args.root / "comparison.json", comparison)
    lines = ["# Baseline -> SFT -> GRPO paired comparison", "", f"Contract: `{comparison['contract_hash']}`", ""]
    for pair, metrics in comparison["paired_deltas"].items():
        lines.extend([f"## {pair}", "", "| Metric | Mean paired delta |", "|---|---:|"])
        lines.extend(f"| {key} | {value:.6f} |" for key, value in sorted(metrics.items()))
        lines.append("")
    (args.root / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(comparison, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
