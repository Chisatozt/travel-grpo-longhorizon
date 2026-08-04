# Travel GRPO

An auditable UserBench post-training pipeline for long-horizon travel agents: frozen splits, Gold/Silver teacher collection, action-only LoRA SFT, veRL 0.8 online GRPO, checkpoint selection, and paired Baseline/SFT/GRPO evaluation.

Status: the code path and offline validations are implemented. Formal SFT, GRPO, and the 471-task evaluation have not been run, so this repository claims no model or benchmark result.

```text
frozen UserBench splits
  → deepseek-v4-flash Gold/Silver teacher trajectories
  → Qwen/Qwen3.5-2B action-only SFT
  → merged SFT model
  → veRL 0.8 online GRPO
  → checkpoint selection on 132 validation tasks
  → paired evaluation on 471 frozen test tasks
```

Collection, GRPO, and evaluation simulators all use `deepseek-v4-flash`, but remain separate processes and environment-variable namespaces. The Actor is Qwen3.5-2B. Windows/6 GiB is limited to tests and dry-runs; formal execution requires Linux, Python 3.12, BF16, and one visible GPU with at least 80 GiB (96 GiB target).

```bash
bash scripts/setup.sh
# Copy .env.example to .env, fill the isolated credentials, then source it
# separately in each collection, GRPO, serving, or evaluation process.
python scripts/data/build_dataset_splits.py --verify-only
python scripts/train/grpo/prepare_data.py --verify-only
python scripts/train/sft/collect_sft_data.py --dry-run --limit 1
python scripts/train/sft/sft_train.py --dry-run
bash scripts/train/grpo/run_vanilla.sh --dry-run
bash scripts/train/grpo/run_grpo.sh --dry-run
python scripts/eval/evaluate_userbench.py --stage baseline --dry-run --limit 2
```

Formal teacher collection runs the train and validation task files in separate run directories. Each evaluation stage must also run in its own process with `ACTOR_MODEL` set to the exact model name served by `scripts/vllm_server/actor.sh`. See `docs/training/sft.md`, `docs/training/grpo.md`, and `docs/evaluation/userbench.md` for the full execution contract. `environments/UserBench/` remains the unmodified Apache-2.0 snapshot at commit `80506d2ab484cab843e60a2401ff3e0290d05b87`.
