# Completion-priority Travel Reward v3

当前运行时奖励版本为 `userbench-travel-reward-v3-priority`。它只在 rollout 终局写入 GRPO，UserBench 原始 step reward 仍仅作诊断；`reward_valid=false` 仍表示基础设施无效样本，不是可学习的负例。

## 目标顺序

1. 完成公开 aspect（`completion_rate`）
2. 覆盖已 elicited 的偏好（active 与 passive 的并集）
3. 遵守公开控制状态机的阶段转移
4. 搜索覆盖、答案质量和效率
5. guard/重复/错误行为只施加有上限的小扣分

## 计算

令 `C` 为已提交答案的公开 aspect 比例，`P` 为 active/passive preference ID 并集覆盖率，`T` 为公开阶段转移成功率，`S` 为已搜索 aspect 比例，`Q` 为答案质量（best=1、其他正确 option=0.8），`E` 为有效步效率，则：

```text
raw = 3.00*C + 0.20*P + 0.08*T + 0.06*S + 0.04*Q + 0.02*E - bounded_penalty
terminal_reward = clip(raw / 3.4, -1, 1)
```

没有公开阶段机会的 legacy session 将 `T` 视为 1；一旦阶段有机会，`T` 是成功数/机会数。这样不会因为旧数据没有 guard ledger 被隐式扣分。

扣分组件均有固定上限：guard rejection 0.08、blocked aspect 0.08、invalid action 0.03、parallel tool call 0.05、exact/semantic repeat 各 0.02、ambiguous 0.02、unsearched answer 0.03、wrong answer 0.04、no-tool/max-step 各 0.02。`BLOCKED` 只计入 blocked，不计入 completion。

## 运行时和兼容性

- 新 GRPO/Teacher collection 配置必须使用 v3。
- SFT loader 接受 v3 和历史 `userbench-travel-reward-v2` 记录；历史 JSONL 不被原地改写。
- phase/guard 计数只读取 public control state、Actor action 和公开 simulator feedback；不读取 reward snapshot、correct/best ID 或 hidden preference value。
- guard rejection 在调用 simulator 前计数；effective steps 将其作为低权重的无效尝试，避免重复 guard 错误主导训练。

离线 replay 与报告中的构成、版本和 hash 应与 `reward_version` 一并保存。
