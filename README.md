# Travel GRPO

面向 UserBench 旅游助手的可审计 Agentic 后训练项目，覆盖冻结数据划分、教师轨迹 Gold/Silver 验收、action-only LoRA SFT、veRL 0.8 在线 GRPO，以及 Baseline/SFT/GRPO 冻结测试集配对评测。

当前状态：代码链路与离线验证已实现；尚未在正式 Linux GPU 环境执行 SFT、GRPO 或 471 题评测，因此本仓库不声明任何模型或 benchmark 指标。

## 固定流水线

```text
UserBench 固定划分
  → deepseek-v4-flash 教师轨迹（Gold + Silver）
  → Qwen/Qwen3.5-2B action-only LoRA SFT
  → 合并 SFT 模型
  → veRL 0.8 + UserBench 在线 GRPO
  → 132 题 validation 选择 checkpoint
  → 471 题 Baseline / SFT / GRPO 配对评测
```

Actor、采集模拟器、GRPO 模拟器和评测模拟器是独立运行边界。后三者虽都使用 `deepseek-v4-flash`，仍必须分别读取 `COLLECTION_USER_SIM_*`、`GRPO_USER_SIM_*`、`EVAL_USER_SIM_*` 并运行在不同进程。

## 本机与正式环境

- Windows/RTX 4050：仅用于数据验证、单元测试和 `--dry-run`。
- 正式训练：Linux、Python 3.12、单张可见 NVIDIA GPU，至少 80 GiB（目标 96 GiB），支持 BF16。
- 固定运行栈：veRL 0.8.0、vLLM 0.25.1、Torch 2.11.0、Ray 2.56.1。

Linux 安装：

```bash
bash scripts/setup.sh
```

该脚本创建 `.venv`，editable 安装本项目和固定 UserBench 快照，并对 veRL 唯一的动态采样连接补丁执行源文件、补丁载荷和结果 SHA-256 校验。

复制 `.env.example` 为 `.env`，填写各运行边界的凭据；每个独立进程启动前使用 `set -a; source .env; set +a` 加载，但不要把 `.env` 提交到仓库。

## 执行顺序

```bash
python scripts/data/build_dataset_splits.py --verify-only
python scripts/train/grpo/prepare_data.py

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
python scripts/eval/select_checkpoint.py \
  --validation-dir outputs/models/grpo/validation_rollouts
bash scripts/train/grpo/export_actor.sh \
  outputs/models/grpo/global_step_100/actor outputs/models/grpo-merged \
  --selection outputs/models/grpo/checkpoint_selection.json

# 配置和 checkpoint 完全冻结后，每个阶段分别启动 Actor 服务和评测进程
export ACTOR_MODEL=Qwen/Qwen3.5-2B
bash scripts/vllm_server/actor.sh "$ACTOR_MODEL"
bash scripts/eval/run_evaluation.sh baseline
export ACTOR_MODEL=outputs/models/sft-merged
bash scripts/vllm_server/actor.sh "$ACTOR_MODEL"
bash scripts/eval/run_evaluation.sh sft
export ACTOR_MODEL=outputs/models/grpo-merged
bash scripts/vllm_server/actor.sh "$ACTOR_MODEL"
bash scripts/eval/run_evaluation.sh grpo
python scripts/eval/compare_stages.py
```

完整契约见 [SFT](docs/training/sft.md)、[GRPO](docs/training/grpo.md) 和 [评测](docs/evaluation/userbench.md)。

## 不可变边界

`environments/UserBench/` 是 Salesforce AI Research UserBench 提交 `80506d2ab484cab843e60a2401ff3e0290d05b87` 的完整快照。日常开发不得修改；来源和 Apache-2.0 信息记录在 `EMBEDDED_SOURCE.json`。正式 471 条 test 只允许在训练配方和 checkpoint 完全冻结后使用。
