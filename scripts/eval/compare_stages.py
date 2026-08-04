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


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "outputs/evaluation")
    args = parser.parse_args()
    stages = {}
    for stage in ("baseline", "sft", "grpo"):
        stage_root = args.root / stage
        stages[stage] = {
            "contract": json.loads((stage_root / "contract.json").read_text(encoding="utf-8")),
            "results": read_jsonl(stage_root / "results.jsonl"),
        }
    comparison = compare_stage_results(stages)
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
