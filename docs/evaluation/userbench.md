# Evaluation

Formal evaluation uses all 471 frozen official single-choice test tasks from `data/evaluation/tasks.parquet`. Those task IDs must not be used for teacher calls, SFT, GRPO, reward tuning, or checkpoint selection.

Evaluation uses the same one-tool action contract and raw UserBench reward handling as training, but loads `configs/interaction_config/simulator_eval.yaml` and the `EVAL_USER_SIM_*` environment variables in a separate process. No final rollout launcher or benchmark result is implemented yet.
