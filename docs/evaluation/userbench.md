# 冻结 UserBench 评测

完整 UserBench 的 formal source pool 是 `data/evaluation/tasks.parquet` 的 471 条官方 one-choice test；它不得用于教师调用、SFT、GRPO、Reward 调参或 checkpoint 选择。当前项目报告的最终测试使用从该 source pool 固定派生的 200-Task 子集，不能把 200 条结果外推为完整 471-task benchmark。

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

### 当前项目最终测试：Composition 分层 200-task 子集

当前项目实际用于 Baseline、SFT 和 GRPO checkpoint 横向比较的最终测试，是从 471-task source pool 固定抽取的 composition 分层 200-task 子集。它保持独立的 subset contract 和 200-task 固定分母；完整 471-task formal evaluation contract 仍保留，二者指标口径不能混用。

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

运行前仍需分别启动与当前 `--model` 完全一致的 Actor 服务。生成器固定 seed 为 `47120042`，当前最终测试 manifest 的配额为 `22:43`、`33:37`、`44:28`、`233:26`、`333:23`、`334:20`、`444:16`、`2222:7`。不要把正式评测中的 `--limit 200` 当作子集运行；`--limit` 现在只允许用于 dry-run，实际最终测试必须同时提供 Parquet 和 manifest。

注意：当前固定 manifest 仍包含 `333`、`334`、`444`、`2222` 四类 composition，因此它不是“排除这四类 composition”的变体；如需排除版，应另建并单独标识新的 subset manifest，不覆盖当前最终测试归档。

当前最终测试主汇总固定分母 200：缺失和 infrastructure-invalid 任务按 0 计入，同时报告 valid-only diagnostics。指标包括 UserBench `micro_avg`、`micro_max`、`avg_number_of_1`、`avg_number_of_08`，各 aspect option quality，Travel Reward v3、成功率、覆盖、效率、policy penalty、调用/重复/终止诊断及 composition 分项。完整 471-task formal evaluation 仍使用 471 固定分母。

当前最终测试只有在三阶段都完整覆盖相同的 200 个 task、且 contract hash 与 subset manifest 一致时，才生成可比较的 `comparison.json` 和 `comparison.md`；这不改变完整 471-task formal comparison 的独立契约。
