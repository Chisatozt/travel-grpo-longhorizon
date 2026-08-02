"""Build or verify the frozen UserBench project-level task splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from travel_grpo.data import (  # noqa: E402
    build_dataset_splits,
    compute_jsonl_sha256,
    load_split_spec,
    verify_dataset_splits,
    write_dataset_splits,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic, disjoint UserBench SFT/GRPO/evaluation task splits."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "dataset_split.toml",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=REPOSITORY_ROOT / "environments" / "UserBench" / "data",
        help="directory containing travel{composition}_multiturn_onechoice folders",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT / "data",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="validate sources and print the planned counts without writing files",
    )
    mode.add_argument(
        "--verify-only",
        action="store_true",
        help="verify existing files and their manifest without writing files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace existing generated split artifacts",
    )
    args = parser.parse_args(argv)
    if args.force and (args.dry_run or args.verify_only):
        parser.error("--force cannot be combined with --dry-run or --verify-only")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    spec = load_split_spec(args.config)
    if args.verify_only:
        result = verify_dataset_splits(spec, args.source_root, args.output_root)
    else:
        bundle = build_dataset_splits(spec, args.source_root)
        if args.dry_run:
            result = {
                "dry_run": True,
                "split_version": spec.split_version,
                "counts": bundle.manifest_base["counts"],
                "checks": bundle.manifest_base["checks"],
                "planned_jsonl_sha256": {
                    name: compute_jsonl_sha256(records)
                    for name, records in bundle.records.items()
                },
            }
        else:
            write_dataset_splits(bundle, args.output_root, force=args.force)
            result = verify_dataset_splits(spec, args.source_root, args.output_root)
            result["written"] = True
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
