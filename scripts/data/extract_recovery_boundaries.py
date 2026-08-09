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

from travel_grpo.data.recovery_boundaries import (
    extract_recovery_boundaries,
    write_extraction,
)


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
