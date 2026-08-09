#!/usr/bin/env python3
"""Construct and validate one-step targets for recovery-boundary contexts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from travel_grpo.data.recovery_targets import build_targets_from_boundary_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=ROOT,
        help="Project root containing data/, outputs/, and source trajectories.",
    )
    parser.add_argument(
        "--contexts",
        type=Path,
        default=None,
        help="recovery-boundary-v1 JSONL (default: outputs/recovery_boundaries/recovery-boundary-v1/contexts.jsonl).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Derived target directory (default: outputs/recovery_targets/recovery-target-v1).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.project_root.resolve()
    contexts = args.contexts or root / "outputs/recovery_boundaries/recovery-boundary-v1/contexts.jsonl"
    output = args.output or root / "outputs/recovery_targets/recovery-target-v1"
    paths, manifest = build_targets_from_boundary_file(contexts, root, output)
    print(json.dumps(manifest["counts"], ensure_ascii=False, sort_keys=True))
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
