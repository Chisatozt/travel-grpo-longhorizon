#!/usr/bin/env python3
"""Compatibility CLI for the frozen UserBench evaluation runner."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from travel_grpo.evaluation.runner import (  # noqa: E402
    MODELS,
    STAGES,
    load_tasks,
    run,
)


# [项目注释] 功能：`main`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：ArgumentParser, add_argument, parse_args, run。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `int`；具体值由各分支决定。
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--model")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/evaluation/tasks.parquet")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--subset-manifest",
        type=Path,
        help=(
            "run an explicit reproducible test subset; its task IDs and "
            "composition counts must match --dataset"
        ),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="maximum number of tasks evaluated concurrently (default: 1)",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--raw-open-loop",
        action="store_true",
        help="disable public guard/feedback for an explicit raw-model ablation",
    )
    parser.add_argument("--retry-infrastructure-invalid", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.model = args.model or MODELS[args.stage]
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.limit is not None and not args.dry_run:
        parser.error("--limit is only supported with --dry-run; use --subset-manifest for actual subsets")
    if args.subset_manifest is not None and args.limit is not None:
        parser.error("--limit cannot be combined with --subset-manifest")
    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
