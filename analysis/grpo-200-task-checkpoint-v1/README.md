# GRPO 200-task checkpoint comparison archive

This standalone archive compares exactly six objects: Qwen3.5-2B-baseline, SFT-merged, and checkpoint-50/100/150/200. Historical GRPO 100/150/200 checkpoints are excluded from the comparison scope and rankings.

## Execution provenance

The four checkpoint scenario artifacts were produced by a deterministic pipeline for analysis and consistency testing. No actual model training, vLLM rollout, UserBench simulator evaluation, or benchmark execution was performed for those four artifacts.

Qwen and SFT values use the verified `current-reward-v3-comparable-v1` replay while retaining their native raw evaluations. Checkpoint scenario values use the archived Reward-v3 summaries and per-task records.

## Layout

- `raw/qwen35_2b_baseline/` and `raw/sft_merged/`: retained real evaluation records.
- `raw/checkpoints/`: checkpoint scenario records, metrics, task IDs, and provenance.
- `comparison/`: six-object comparison tables, task ordering, replay source, and summary consistency report.
- `curves/`: step 1--200 metrics in JSONL/CSV plus statistics.
- `grpo-200-task-checkpoint-comparison.md`: standalone analysis report.
- `ARCHIVE_MANIFEST.json` and `SHA256SUMS.txt`: provenance and integrity metadata.
