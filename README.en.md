# Travel GRPO

An auditable UserBench post-training pipeline for long-horizon travel agents: frozen task splits, Gold/Silver teacher collection, action-only LoRA SFT, veRL 0.8 online GRPO, checkpoint selection, and paired Baseline/SFT/GRPO evaluation.

Status: the code path and offline validations are implemented. The current working environment contains Baseline, SFT, and GRPO checkpoint artifacts for a fixed 200-Task final test. This 200-Task set is a fixed project subset derived from the full 471-task UserBench test pool; its results must not be extrapolated to the full benchmark. The formal 471-task evaluation contract is retained, but no complete 471-task result is claimed.

The archived comparison separates the real Qwen/SFT evaluation records from the derived or synthetic GRPO checkpoint and curve artifacts. The latter are not evidence that a real GPU training run, vLLM rollout, or UserBench evaluator execution occurred.

## Fixed pipeline

```text
frozen UserBench splits
  → deepseek-v4-flash teacher trajectories (Gold + Silver)
  → Qwen/Qwen3.5-2B action-only LoRA SFT
  → merged SFT model
  → veRL 0.8 + UserBench online GRPO
  → GRPO validation checkpoint selection on 132 tasks
  → current project final test: fixed 200-Task Baseline / SFT / GRPO paired evaluation
  → future formal full evaluation: the complete 471-task test pool
```

The Actor, collection simulator, GRPO simulator, and evaluation simulator are separate runtime boundaries. Although the three simulators use `deepseek-v4-flash`, they must read `COLLECTION_USER_SIM_*`, `GRPO_USER_SIM_*`, and `EVAL_USER_SIM_*` respectively and run in separate processes.

## Local and formal environments

- Windows/RTX 4050: data validation, unit tests, and `--dry-run` only.
- Formal training: Linux, Python 3.12, one visible NVIDIA GPU with at least 80 GiB (96 GiB target), and BF16 support.
- Pinned stack: veRL 0.8.0, vLLM 0.25.1, Torch 2.11.0, and Ray 2.56.1.

Install on Linux:

```bash
bash scripts/setup.sh
```

The setup script creates `.venv`, installs this project and the pinned UserBench snapshot, and verifies the source, patch payload, and result SHA-256 values for the single veRL dynamic-sampling connection patch.

Copy `.env.example` to `.env` and fill in the credentials for each runtime boundary. Before starting each independent process, load it with `set -a; source .env; set +a`; never commit `.env`.

## Execution order

```bash
python scripts/data/build_dataset_splits.py --verify-only
python scripts/train/grpo/prepare_data.py --verify-only

python scripts/train/sft/collect_sft_data.py --dry-run --limit 1
python scripts/train/sft/collect_sft_data.py \
  --input data/sft/tasks_train.jsonl \
  --run-dir outputs/teacher_trajectories/runs/sft-train \
  --output outputs/teacher_trajectories/sft_train.accepted.jsonl
python scripts/train/sft/collect_sft_data.py \
  --input data/sft/tasks_validation.jsonl \
  --run-dir outputs/teacher_trajectories/runs/sft-validation \
  --output outputs/teacher_trajectories/sft_validation.accepted.jsonl

# Composition-proportional adaptive collection (400 accepted Gold+Silver)
python scripts/train/sft/collect_sft_data.py \
  --input data/sft/tasks_train.jsonl \
  --target-accepted 400 \
  --stratify-by composition \
  --stratified-wave-size 32 \
  --sampling-seed sft-train-composition-v4 \
  --run-dir outputs/teacher_trajectories/runs/sft-train-composition-v4 \
  --output outputs/teacher_trajectories/sft_train.accepted.jsonl
python scripts/train/sft/sft_train.py --dry-run
bash scripts/train/sft/run_sft.sh
python scripts/train/sft/merge_lora.py

bash scripts/train/grpo/run_vanilla.sh --dry-run
bash scripts/train/grpo/run_vanilla.sh
bash scripts/train/grpo/run_grpo.sh --dry-run
bash scripts/train/grpo/run_grpo.sh

# Future formal-training template; replace these placeholders with the run directory.
python scripts/eval/select_checkpoint.py \
  --validation-dir outputs/models/grpo/validation_rollouts
bash scripts/train/grpo/export_actor.sh \
  outputs/models/grpo/global_step_<SELECTED>/actor \
  outputs/models/grpo-merged \
  --selection outputs/models/grpo/checkpoint_selection.json
```

For the current project final test, use the explicit 200-task subset commands in [`docs/evaluation/userbench.md`](docs/evaluation/userbench.md), including `tasks_200_proportional_v1.parquet`, its manifest, and `--allow-subset`. Commands without those subset arguments are retained as the future formal 471-task evaluation template.

Each evaluation stage starts its own Actor service and evaluation process. `ACTOR_MODEL` must exactly match the model name served by `scripts/vllm_server/actor.sh`:

```bash
export ACTOR_MODEL=Qwen/Qwen3.5-2B
bash scripts/vllm_server/actor.sh "$ACTOR_MODEL"
bash scripts/eval/run_evaluation.sh baseline

export ACTOR_MODEL=outputs/models/sft-merged
bash scripts/vllm_server/actor.sh "$ACTOR_MODEL"
bash scripts/eval/run_evaluation.sh sft

export ACTOR_MODEL=<SELECTED_GRPO_MODEL>
bash scripts/vllm_server/actor.sh "$ACTOR_MODEL"
bash scripts/eval/run_evaluation.sh grpo

python scripts/eval/compare_stages.py
```

See [`docs/training/sft.md`](docs/training/sft.md), [`docs/training/grpo.md`](docs/training/grpo.md), and [`docs/evaluation/userbench.md`](docs/evaluation/userbench.md) for the full execution contracts.

## Immutable boundary

`environments/UserBench/` is the complete Salesforce AI Research UserBench snapshot at commit `80506d2ab484cab843e60a2401ff3e0290d05b87`. Do not edit it during normal development; source and Apache-2.0 information are recorded in `EMBEDDED_SOURCE.json`. The complete 471-task test set remains the formal source pool, while current project reports use the fixed 200-Task subset derived from that pool. Their metric scopes must not be mixed.
