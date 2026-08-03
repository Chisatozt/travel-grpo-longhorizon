# Travel GRPO

A UserBench-based post-training and evaluation project for long-horizon travel-assistant agents.

> Status: **early development**. The pinned UserBench snapshot and reproducible task partitioning are implemented. Teacher collection, SFT, GRPO, model serving, and final rollouts remain scaffolds.

## Intended pipeline

```text
teacher trajectory collection
  -> replay and quality filtering
  -> action-only LoRA SFT
  -> online GRPO in UserBench
  -> frozen-test comparison of Baseline / SFT / GRPO
```

The actor, training-time user simulator, and formal evaluation user simulator are separate runtime boundaries.

## Dataset splits

```bash
pip install -e ".[data]"
python scripts/data/build_dataset_splits.py --dry-run
python scripts/data/build_dataset_splits.py
python scripts/data/build_dataset_splits.py --verify-only
```

The split contract is stored in `configs/data/dataset_split.toml`; counts and hashes are recorded in `data/split_manifest.json`. Generated records follow the five-field contract in `data/example.jsonl`: `task_id`, `composition`, `difficulty`, `source_split`, and `prompt`.

## Layout

Configuration, scripts, and documentation are grouped by data, interaction, SFT, GRPO, serving, and evaluation stages. Reusable Python code remains under the `travel_grpo` package. See `docs/architecture/repository_layout.md` for ownership boundaries.

`environments/UserBench/` is a pinned third-party snapshot and is not modified during normal project work. Provenance and licensing are recorded in `environments/UserBench/EMBEDDED_SOURCE.json`.

The stage-oriented layout is adapted from [qiqihezh/agentic-grpo-longhorizon](https://github.com/qiqihezh/agentic-grpo-longhorizon). τ-bench-specific implementation and reported metrics are not copied.

The root project license has not been selected. The embedded UserBench copyright and Apache-2.0 license remain unchanged. This project currently claims no training or benchmark results.
