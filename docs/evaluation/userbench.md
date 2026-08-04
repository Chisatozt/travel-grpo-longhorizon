# 冻结 UserBench 评测

正式评测只使用 `data/evaluation/tasks.parquet` 的 471 条官方 one-choice test。它不得用于教师调用、SFT、GRPO、Reward 调参或 checkpoint 选择。

Baseline、SFT、GRPO 共用同一 rollout：Actor temperature 0、do_sample false，EVAL deepseek-v4-flash simulator temperature 0、seed 42、最多 20 step、唯一工具 `interact_with_env`、qwen3_coder parser 和 32768 context。三阶段 `contract_hash` 和 task ID 必须相同。

```bash
python scripts/eval/evaluate_userbench.py --stage baseline --dry-run --limit 2
export ACTOR_MODEL=Qwen/Qwen3.5-2B
bash scripts/vllm_server/actor.sh "$ACTOR_MODEL"
bash scripts/eval/run_evaluation.sh baseline --resume

export ACTOR_MODEL=outputs/models/sft-merged
bash scripts/vllm_server/actor.sh "$ACTOR_MODEL"
bash scripts/eval/run_evaluation.sh sft --resume

export ACTOR_MODEL=outputs/models/grpo-merged
bash scripts/vllm_server/actor.sh "$ACTOR_MODEL"
bash scripts/eval/run_evaluation.sh grpo --resume
python scripts/eval/compare_stages.py
```

每个阶段必须在独立进程中运行，`ACTOR_MODEL` 必须与该阶段冻结契约中的模型标识及 vLLM `--served-model-name` 完全相同；不一致时在写入任务产物前失败。

每个任务原子写入 `outputs/evaluation/{stage}/tasks/`。`--resume` 不重复有效任务；`--retry-infrastructure-invalid` 只重试基础设施无效任务。结果不会保存 API key、best ID、隐藏偏好或内部 snapshot。

主汇总固定分母 471：缺失和 infrastructure-invalid 任务按 0 计入，同时报告 valid-only diagnostics。指标包括 UserBench `micro_avg`、`micro_max`、`avg_number_of_1`、`avg_number_of_08`，各 aspect option quality，Travel Reward v2、成功率、覆盖、效率、policy penalty、调用/重复/终止诊断及 composition 分项。

只有三阶段都完整覆盖 471 条且 contract hash 一致，`compare_stages.py` 才生成 `comparison.json` 和 `comparison.md`。否则失败，不允许称为正式结果。
