# Travel GRPO

A UserBench-based post-training and evaluation project for long-horizon travel-assistant agents.

> Status: **early development**. The pinned UserBench snapshot, reproducible task partitioning, environment wrapper, Reward-v2-gated teacher collection, action-only LoRA/QLoRA SFT pipeline, and veRL 0.6.1 adapter are implemented. Formal SFT runs, GRPO launch completion, and final evaluation rollouts have not been executed.

## Intended pipeline

```text
teacher trajectory collection
  -> replay and quality filtering
  -> action-only LoRA SFT
  -> online GRPO in UserBench
  -> frozen-test comparison of Baseline / SFT / GRPO
```

The actor, training-time user simulator, and formal evaluation user simulator are separate runtime boundaries.

## Environment integration

The core package has no UserBench or veRL dependency. Install the pinned environment and an external veRL 0.6.1 checkout only in rollout/training environments:

```bash
pip install -e environments/UserBench
pip install -e /path/to/verl
```

The actor sees one tool, `interact_with_env(thought, choice, content)`, with `search`, `action`, and `answer` choices. Teacher collection uses separate `TEACHER_*` and `COLLECTION_USER_SIM_*` API settings, both pinned to `deepseek-v4-flash`. GRPO loads `Qwen/Qwen3.5-2B` as the actor and reads its UserBench simulator from `GRPO_USER_SIM_*`. Formal evaluation continues to use `EVAL_USER_SIM_*`. The pinned UserBench reads its OpenAI endpoint through process environment variables, so simulator roles must run in separate processes. See `docs/training/sft.md` and `docs/training/grpo.md`.

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
