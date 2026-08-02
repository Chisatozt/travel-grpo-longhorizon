# Travel GRPO

面向长程旅游助手 Agent 的后训练与评测项目骨架。

> 当前状态：**早期开发**。固定版本的 UserBench 快照和可复现任务划分已经实现；教师轨迹采集、SFT、GRPO、模型服务和最终 rollout 尚未实现。

## 目标流水线

```text
教师模型采集多轮轨迹
  -> 轨迹回放与质量过滤
  -> action-only LoRA SFT
  -> UserBench 在线 GRPO
  -> 冻结测试集上的 Baseline / SFT / GRPO 对比
```

训练 Actor、训练阶段用户模拟器和正式评测用户模拟器将作为三个独立运行边界，以避免模型端点、采样配置和凭据混用。

## UserBench

仓库在 `environments/UserBench/` 内嵌了 Salesforce AI Research 的 [UserBench](https://github.com/SalesforceAIResearch/UserBench) 固定快照。UserBench 是一个面向旅游规划的 Gymnasium 环境，包含多轮偏好澄清、模拟搜索和推荐选择。来源提交及许可信息见 `environments/UserBench/EMBEDDED_SOURCE.json`。

## 任务划分

安装数据工具依赖后，可以构建或验证互不重叠的 SFT 教师采集、GRPO 和冻结评测任务：

```bash
pip install -e ".[data]"
python scripts/build_dataset_splits.py --dry-run
python scripts/build_dataset_splits.py
python scripts/build_dataset_splits.py --verify-only
```

划分规则固定在 `configs/dataset_split.toml`，产物哈希与数量记录在 `data/split_manifest.json`。每条记录遵循 `data/example.jsonl` 的五字段契约：`task_id`、`composition`、`difficulty`、`source_split`、`prompt`；项目 split 由文件路径表达。这里的 SFT 产物只是教师轨迹采集任务，不是可直接训练的 SFT 对话。

## 目录

- `configs/`：环境、模拟器、AgentLoop、GRPO 与工具配置占位。
- `data/`：未来生成的 SFT、GRPO 和冻结评测数据。
- `docs/`：数据采集、训练、奖励与评测设计文档占位。
- `src/travel_grpo/`：项目自有 Python 包骨架。
- `scripts/`：未来面向用户的薄入口脚本。
- `tests/`：核心契约与入口测试占位。

## 参考

- [YYHDBL/shopping-grpo-longhorizon](https://github.com/YYHDBL/shopping-grpo-longhorizon)：仓库分层和后训练流水线参考。
- [SalesforceAIResearch/UserBench](https://github.com/SalesforceAIResearch/UserBench)：旅游环境、数据和官方评测实现。
- [UserBench 论文](https://arxiv.org/abs/2507.22034)

本项目根许可证尚未指定。内嵌 UserBench 的版权与 Apache-2.0 许可保持不变。
