# Travel GRPO

面向长程旅游助手 Agent 的 UserBench 后训练与评测项目。

> 当前状态：**早期开发**。固定版本的 UserBench 快照、可复现任务划分、环境包装、Reward v2 严格准入的教师轨迹采集、action-only LoRA/QLoRA SFT 流水线和 veRL 0.6.1 适配层已经实现；尚未执行正式 SFT、完整 GRPO 启动和最终评测 rollout。

## 目标流水线

```text
教师模型采集多轮轨迹
  -> 轨迹回放与质量过滤
  -> action-only LoRA SFT
  -> UserBench 在线 GRPO
  -> 冻结测试集上的 Baseline / SFT / GRPO 对比
```

Actor、训练用户模拟器和正式评测用户模拟器是三个独立运行边界，禁止混用端点、模型配置或采样参数。

## 环境包装

核心包不依赖 UserBench 或 veRL。需要运行真实环境或 GRPO 时，分别从固定快照和外部 veRL 0.6.1 checkout 进行 editable install：

```bash
pip install -e environments/UserBench
pip install -e /path/to/verl
```

Actor 只使用单一工具 `interact_with_env(thought, choice, content)`，其中 `choice` 为 `search`、`action` 或 `answer`。教师轨迹采集使用 `TEACHER_*` 和 `COLLECTION_USER_SIM_*` 两套独立的 `deepseek-v4-flash` API 配置；GRPO Actor 使用 `Qwen/Qwen3.5-2B`，其 UserBench 模拟器读取 `GRPO_USER_SIM_*`。正式评测继续使用独立的 `EVAL_USER_SIM_*`。由于固定 UserBench 通过进程环境读取 OpenAI endpoint，不同模拟器角色必须运行在不同进程。详见 `docs/training/sft.md` 和 `docs/training/grpo.md`。

## 数据划分

```bash
pip install -e ".[data]"
python scripts/data/build_dataset_splits.py --dry-run
python scripts/data/build_dataset_splits.py
python scripts/data/build_dataset_splits.py --verify-only
```

划分规则位于 `configs/data/dataset_split.toml`，产物数量与哈希记录在 `data/split_manifest.json`。派生记录遵循 `data/example.jsonl` 的五字段契约：`task_id`、`composition`、`difficulty`、`source_split`、`prompt`。

## 项目结构

```text
configs/
├── data/                 # 冻结数据划分配置
├── interaction_config/   # UserBench、AgentLoop 与隔离的模拟器配置
├── tool_config/          # Actor 可见工具协议
├── train/{sft,grpo}/     # 训练阶段配置
└── eval/                 # 冻结评测配置
scripts/
├── data/                 # 数据构建与验证入口
├── train/{sft,grpo}/     # 分阶段训练入口
├── eval/                 # 独立评测入口
└── vllm_server/          # Actor 与训练模拟器服务入口
src/travel_grpo/
├── data/                 # 已实现的 UserBench 数据划分
├── envs/                 # UserBench 包装、交互、工具与奖励边界
├── models/               # Actor 推理客户端
├── training/             # SFT 与 GRPO 核心逻辑
├── evaluation/           # 冻结评测逻辑
└── utils/                # 通用基础设施
```

完整职责说明见 `docs/architecture/repository_layout.md`。

## 第三方环境

`environments/UserBench/` 是 Salesforce AI Research
[UserBench](https://github.com/SalesforceAIResearch/UserBench) 的固定快照。来源提交与许可信息见
`environments/UserBench/EMBEDDED_SOURCE.json`，日常开发不得直接修改该目录。

本项目结构参考
[qiqihezh/agentic-grpo-longhorizon](https://github.com/qiqihezh/agentic-grpo-longhorizon)，但保留 UserBench 数据契约、`travel_grpo` 包命名空间和三类运行时隔离边界，不复制 τ-bench 专用实现。

项目根许可证尚未指定；内嵌 UserBench 的版权与 Apache-2.0 许可保持不变。当前没有训练或 benchmark 结果声明。
