# recovery-sft-v1 渲染与审计

`recovery-sft-v1` 是从 `recovery-target-v1` 派生的、只包含公开状态边界样本的 SFT 格式。它不修改原始 target，不启动训练，也不读取 reward/correctness 数据。

每条记录包含：

- 第一个 `system` 消息中的 production `ACTOR_RUNTIME_POLICY`（当前版本 `actor-runtime-v1`）；
- 截断到 boundary 前的 actor-visible history；
- 用 `render_actor_control_info` 生成的同格式 public control note；
- 最后一个 assistant 的单个 `interact_with_env` tool call。

有 tool observation 的历史将 note 追加在最后一个 `tool` 消息（9322 条）；只有初始 user 历史的边界将 note 追加在最后一个 `user` 消息（10 条），不伪造 simulator observation。

历史 assistant calls 保留在上下文中并标记 `loss_mask: true`。最后一个 target 不标记 mask；渲染统一调用 `sft_dataset.build_action_only_examples(..., record_format="recovery")`，因此 action-only loss masking 不存在第二套实现。

## 构建命令

```bash
PYTHONPATH=src .venv/bin/python scripts/train/sft/build_recovery_sft.py \
  --targets-dir outputs/recovery_targets/recovery-target-v1 \
  --output outputs/recovery_sft/recovery-sft-v1 \
  --tokenizer outputs/models/sft-merged \
  --tool-schema configs/tool_config/userbench_tools.yaml \
  --max-sequence-length 16384
```

CLI 使用本地 `tokenizer.json` 和 `chat_template.jinja`，不加载模型权重。默认输出：

- `train.jsonl`；
- `validation.jsonl`；
- `rejected.jsonl`；
- `manifest.json`；
- `audit.json`。

## 审计契约

manifest/audit 记录 policy parity、Teacher-only 指令缺失、control-note parity、最后一个 assistant-only loss mask、token 长度分位数、超长样本、task split 归属、boundary/composition 分布、answer ID 可见性、hidden-state leakage、完全重复样本和输入/输出 SHA-256。

Task 允许产生多个 boundary 样本，但同一个 task 不得跨 train/validation。完全重复的 rendered sample 会保留第一条，后续记录进入 `rejected.jsonl`。正式 evaluation split 不得进入输出训练文件。
