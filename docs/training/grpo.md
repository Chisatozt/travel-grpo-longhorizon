# veRL 0.8 在线 GRPO

GRPO 从 `outputs/models/sft-merged` 启动，并挂载新的 rank-16 LoRA。Actor 使用 Qwen3.5 工具格式、`enable_thinking=false`；UserBench 模拟器使用独立的 `GRPO_USER_SIM_*` deepseek-v4-flash 进程。

## 数据

canonical 1723/132 划分保持不变。以下命令派生无隐藏标签的 veRL Parquet：

```bash
python scripts/train/grpo/prepare_data.py --dry-run
python scripts/train/grpo/prepare_data.py
python scripts/train/grpo/prepare_data.py --verify-only
```

`reward_model.id`、`extra_info.task_id` 和工具 `create_kwargs.id` 必须一致。ground truth 固定为空；best/correct/preference 标签不会进入 rollout 数据。

## AgentLoop 与 Reward

veRL 0.8 直接实例化 `UserBenchAgentLoop`。每条 rollout 独占 `TravelEnv` 和 ContextVar session；工具只返回 Actor 可见的 `feedback`，tool reward 始终为 0。终局由现有 Travel Reward v2 写入 `reward_score`。`reward_valid=false` 是采样无效而不是有效零分；证据完整的 simulator fallback 可以保持 valid，并由可选的 degraded 诊断标记。

同一 completion 多工具调用不会执行；环境终止后不会额外生成 Actor 回合；所有异常路径都会在 `finally` 关闭 wrapper 并清除 ContextVar。

### 可选 stall recovery

GRPO training rollout 默认关闭 T3-style stall recovery。开启后，session 只根据
UserBench evidence ledger 判断连续无进展；达到阈值时，如果 Actor 曾在 SEARCH
feedback 中实际看见尚未回答 aspect 的 option ID，则只允许一次由 Actor 自己生成的
ANSWER-only call。非法、未见过或已回答 aspect 的 option 不会执行环境；没有可回答的
Actor-visible option，或 recovery 失败/再次 stall，则以 `stalled_no_progress` 结束。
该 early cut 保持 `terminated=true, truncated=false`，所以不会触发 Reward v2 的
max-steps penalty；证据完整时仍是 `reward_valid=true`，不增加固定 stall 分数。

```bash
# baseline（默认行为）
bash scripts/train/grpo/run_grpo.sh --no-stall-recovery \
  --output outputs/models/grpo-baseline

# training rollout 开启，validation 仍由固定采样 profile 强制关闭
bash scripts/train/grpo/run_grpo.sh --stall-recovery --stall-threshold 4 \
  --output outputs/models/grpo-stall4
```

开关通过 subprocess-local 环境变量传入 AgentLoop；preflight 会校验 threshold，
并确认 `temperature=0.7, top_p=0.9` 的 training profile 与
`temperature=0.0, top_p=1.0, do_sample=false` 的 validation profile 可唯一区分。
frozen evaluation rollout 不使用该控制器。

生产 sampler 每题生成 4 条轨迹。含无效轨迹或组内 reward 极差不超过 `1e-6` 的组被丢弃；每次 update 最多生成 3 批，累计 2 个有效 prompt group 后训练，不足则跳过。连续跳过超过 10 次立即失败。vanilla profile 禁用该 sampler，仅运行 2 step。

## 启动与恢复

```bash
bash scripts/train/grpo/run_vanilla.sh --dry-run
bash scripts/train/grpo/run_grpo.sh --dry-run
bash scripts/train/grpo/run_grpo.sh --resume
bash scripts/train/grpo/run_grpo.sh trainer.logger=[console,swanlab]
```

SFT Stage 2 完成后，可用一条命令完成 LoRA merge、GRPO 数据准备/校验并启动训练：

```bash
bash scripts/train/grpo/run_grpo_from_sft.sh \
  --output outputs/models/grpo-sft \
  --no-stall-recovery
```

入口默认读取 `outputs/sft/qwen3.5-2b-lora-stage2`，生成或复用
`outputs/models/sft-merged`。已有 merged model 必须带有与该 adapter 和
`Qwen/Qwen3.5-2B` 匹配的 `merge_manifest.json`；不完整目录和部分 GRPO 数据
不会被自动覆盖。训练前可先运行 `--dry-run`。若要开启仅 training rollout
使用的 stall recovery：

```bash
bash scripts/train/grpo/run_grpo_from_sft.sh \
  --stall-recovery --stall-threshold 4 \
  --output outputs/models/grpo-sft-stall4
```

Stage 2 使用非默认路径时可传 `--sft-adapter`、`--merged-model`；GRPO 数据源可用
`--train-source`、`--validation-source` 和 `--data-output` 覆盖，训练启动也会使用该
数据目录。该入口不会启动
SFT，也不会删除任何已有产物；需要重建已有 GRPO 数据时必须显式传 `--force-data`。

GRPO wrapper 会优先使用仓库内的 `.venv/bin/python`，因此不要求每次先激活环境；
如需指定其他解释器，可设置 `PYTHON_BIN=/path/to/python`。正式安装应先运行
`bash scripts/setup.sh`，它会安装 `[data]` 和 `[grpo]` extra。若入口提示缺少
`pyarrow`，说明实际选中的 Python 没有安装项目 data extra。

非 dry-run 会在 Ray/CUDA 启动前检查 Linux/Python、精确依赖、单卡显存、BF16、合并模型、UserBench 来源、数据 hash、模拟器变量、唯一工具及 vLLM qwen3_coder parser。输出目录必须为空，或带合法 checkpoint 标记并显式 `--resume`。

正式参数位于 `configs/train/grpo/grpo.yaml`：batch 2、n=4、总上下文 32768、temperature 0.7、500 step，每 50 step 保存和验证，不使用 KL reward/loss，GRPO advantage 不按组标准差归一化。

## Checkpoint

checkpoint 仅使用 132 条 GRPO validation。`scripts/eval/select_checkpoint.py` 排除 valid rate 小于 0.98，或 correct itinerary/user-aligned success 相对 SFT validation 下降超过 1 个百分点的候选；其余按固定分母 mean terminal reward、correct、aligned、efficiency、较早 step 排序。没有候选通过时 GRPO 阶段明确失败。

veRL 将每次 validation 原始行写入 `validation_rollouts/{step}.jsonl`。step 0 的 val-before-train 是 SFT 参考，其余每 50 step 是 GRPO 候选。以下命令一次性验证所有任务 ID、生成 `validation_summaries/step_<N>.summary.json`，并在同一运行目录写入 `checkpoint_selection.json`：

```bash
python scripts/eval/select_checkpoint.py \
  --validation-dir outputs/models/grpo/validation_rollouts
```

也可用 `scripts/eval/summarize_validation.py` 单独审计某一个原始 dump，再以 `--sft-summary` 和重复的 `--candidate` 参数手动选择。

```bash
bash scripts/train/grpo/export_actor.sh \
  outputs/models/grpo/global_step_100/actor \
  outputs/models/grpo-merged \
  --selection outputs/models/grpo/checkpoint_selection.json
```

导出器只接受 `passed=true` 且 `selected_step` 与 `global_step_<N>` 相同的选择结果，并将选择文件的 SHA-256 写入导出 manifest；它拒绝覆盖非空目标目录。
