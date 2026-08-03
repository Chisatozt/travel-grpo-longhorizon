# Dataset splits

## Task partitions

The pinned UserBench one-choice train set is divided into four mutually exclusive task pools:

- `data/sft/tasks_train.*`: teacher-collection tasks for future SFT training trajectories;
- `data/sft/tasks_validation.*`: teacher-collection tasks for future SFT validation trajectories;
- `data/grpo/train.*`: online GRPO training prompts;
- `data/grpo/validation.*`: checkpoint-selection prompts.

The official one-choice test set is projected to the compact output contract in
its original row order and written to `data/evaluation/tasks.*`. It is never
used for training or checkpoint selection. The exact composition quotas,
source hashes, and output hashes are recorded in `data/split_manifest.json`.

Every generated row follows the compact contract in `data/example.jsonl`, in
this exact column order: `task_id`, `composition`, `difficulty`,
`source_split`, and `prompt`. `source_split` always records the official
UserBench boundary (`train` or `test`); the project-level SFT/GRPO/evaluation
split is determined by the artifact path.

Build or verify the task partitions with:

```bash
python scripts/data/build_dataset_splits.py --dry-run
python scripts/data/build_dataset_splits.py
python scripts/data/build_dataset_splits.py --verify-only
```

The SFT files created here are task pools, not supervised trajectories. A later collection stage must obtain teacher rollouts, replay them, filter invalid or unsuccessful episodes, and render action-only labels without borrowing replacement tasks from the GRPO or evaluation pools.
