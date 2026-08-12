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

评测默认逐题运行（`--concurrency 1`）。如果 Actor 服务和 EVAL 用户模拟器允许并发，可以用受控任务并发重叠不同任务；每个任务内部仍保持最多 20 轮串行，且每题继续原子写盘：

```bash
bash scripts/eval/run_evaluation.sh baseline \
  --concurrency 4 \
  --resume \
  --retry-infrastructure-invalid
```

建议先从 `--concurrency 2` 开始观察 API 限流、超时和 GPU 显存，再提高到 4。并发评测时必须继续使用独立的 Actor 和 EVAL simulator 进程；不要把 `GRPO_USER_SIM_*` 变量用于评测。`--concurrency` 只控制任务之间的并发，不改变 Actor temperature、simulator temperature、seed 或单题最大轮数。

### Composition 分层 200-task diagnostic 子集

如果只想实际运行 200 条任务，可从正式 471-task test pool 生成一个固定的 composition 分层子集。该模式是 diagnostic validation，不改变正式 471-task 契约，也不用于替代最终正式结果。

```bash
.venv/bin/python scripts/eval/create_task_subset.py

SUBSET_DATASET=outputs/evaluation/subsets/tasks_200_proportional_v1.parquet
SUBSET_MANIFEST=outputs/evaluation/subsets/tasks_200_proportional_v1.json
SUBSET_ROOT=outputs/evaluation/subset-200-proportional-v1

.venv/bin/python scripts/eval/evaluate_userbench.py \
  --stage baseline \
  --model Qwen/Qwen3.5-2B \
  --dataset "$SUBSET_DATASET" \
  --subset-manifest "$SUBSET_MANIFEST" \
  --output "$SUBSET_ROOT/baseline" \
  --concurrency 2 \
  --resume

.venv/bin/python scripts/eval/evaluate_userbench.py \
  --stage sft \
  --model outputs/models/sft-merged \
  --dataset "$SUBSET_DATASET" \
  --subset-manifest "$SUBSET_MANIFEST" \
  --output "$SUBSET_ROOT/sft" \
  --concurrency 2 \
  --resume

.venv/bin/python scripts/eval/evaluate_userbench.py \
  --stage grpo \
  --model outputs/models/grpo-merged \
  --dataset "$SUBSET_DATASET" \
  --subset-manifest "$SUBSET_MANIFEST" \
  --output "$SUBSET_ROOT/grpo" \
  --concurrency 2 \
  --resume

.venv/bin/python scripts/eval/compare_stages.py \
  --root "$SUBSET_ROOT" \
  --allow-subset
```

运行前仍需分别启动与当前 `--model` 完全一致的 Actor 服务。生成器固定 seed 为 `47120042`，当前配额为 `22:43`、`33:37`、`44:28`、`233:26`、`333:23`、`334:20`、`444:16`、`2222:7`。不要把正式评测中的 `--limit 200` 当作子集运行；`--limit` 现在只允许用于 dry-run，实际子集必须同时提供 Parquet 和 manifest。

主汇总固定分母 471：缺失和 infrastructure-invalid 任务按 0 计入，同时报告 valid-only diagnostics。指标包括 UserBench `micro_avg`、`micro_max`、`avg_number_of_1`、`avg_number_of_08`，各 aspect option quality，Travel Reward v3、成功率、覆盖、效率、policy penalty、调用/重复/终止诊断及 composition 分项。

只有三阶段都完整覆盖 471 条且 contract hash 一致，`compare_stages.py` 才生成 `comparison.json` 和 `comparison.md`。否则失败，不允许称为正式结果。
