#!/usr/bin/env python3
"""Build or verify hidden-label-free veRL 0.8 UserBench datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from travel_grpo.training.grpo.data import (  # noqa: E402
    prepare_verl_datasets,
    verify_verl_datasets,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-source", type=Path, default=ROOT / "data/grpo/train.parquet"
    )
    parser.add_argument(
        "--validation-source",
        type=Path,
        default=ROOT / "data/grpo/validation.parquet",
    )
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "outputs/grpo/data"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.verify_only and (args.dry_run or args.force):
        raise SystemExit("--verify-only cannot be combined with --dry-run or --force")
    if args.verify_only:
        result = verify_verl_datasets(args.output_root)
    else:
        result = prepare_verl_datasets(
            train_source=args.train_source,
            validation_source=args.validation_source,
            output_root=args.output_root,
            force=args.force,
            dry_run=args.dry_run,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
