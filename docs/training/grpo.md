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

veRL 0.8 直接实例化 `UserBenchAgentLoop`。每条 rollout 独占 `TravelEnv` 和 ContextVar session；工具只返回 Actor 可见的 `feedback`，tool reward 始终为 0。终局由现有 Travel Reward v2 写入 `reward_score`。`reward_valid=false` 是采样无效而不是有效零分。

同一 completion 多工具调用不会执行；环境终止后不会额外生成 Actor 回合；所有异常路径都会在 `finally` 关闭 wrapper 并清除 ContextVar。

生产 sampler 每题生成 4 条轨迹。含无效轨迹或组内 reward 极差不超过 `1e-6` 的组被丢弃；每次 update 最多生成 3 批，累计 2 个有效 prompt group 后训练，不足则跳过。连续跳过超过 10 次立即失败。vanilla profile 禁用该 sampler，仅运行 2 step。

## 启动与恢复

```bash
bash scripts/train/grpo/run_vanilla.sh --dry-run
bash scripts/train/grpo/run_grpo.sh --dry-run
bash scripts/train/grpo/run_grpo.sh --resume
bash scripts/train/grpo/run_grpo.sh trainer.logger=[console,swanlab]
```

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
