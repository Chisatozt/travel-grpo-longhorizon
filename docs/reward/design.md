# Travel Reward v2（历史基线）

> 当前运行时已切换到 [Completion-priority Travel Reward v3](design-v3-priority.md)。本页保留 v2 公式，供历史轨迹审计和兼容性对照。

GRPO 使用版本化的确定性终局奖励 `userbench-travel-reward-v2`。UserBench 原始逐步奖励只用于诊断；工具奖励恒为 `0.0`，每条 rollout 结束时只写入一次终局分，避免重复计奖。

## 主分数

对每个旅游维度 `a`：最佳选项质量 `q_a=1.0`，其他正确选项为 `0.8`，错误或缺失为 `0.0`。成功搜索该维度后，主动获取偏好的覆盖率为 `c_a`，证据门控为：

```text
g_a = 0                              未成功搜索
g_a = 0.25 + 0.75 * c_a              已成功搜索
Q_g = mean(q_a * g_a)
```

令 `A` 为全局主动偏好覆盖率、`P` 为被动偏好覆盖率、`C` 为已回答维度比例。有效步数预算 `B = 隐藏偏好数 + 2 * 维度数`；在 `B` 步内效率 `E=1`，之后线性衰减，到第 20 步为 0。

```text
raw = 0.65 * (2 * Q_g - 1)
    + 0.15 * A
    + 0.10 * search_coverage
    + 0.10 * E
    - 0.10 * P
    - 0.30 * (1 - C)
    - policy_penalty

terminal_reward = raw if raw >= 0 else -tanh((-raw) / 1.5)
```

`policy_penalty` is the uncapped sum of the per-event components. Negative
terminal rewards use the deterministic temperature `1.5` squash, so finite
negative raw rewards remain ordered instead of collapsing to `-1.0`.

完整搜索、主动澄清全部相关偏好并选择所有最佳选项时恰好得到 `1.0`。没有搜索和偏好证据的猜测即使碰巧命中，也会得到负分。

## 策略违规

精确配置位于 `configs/interaction_config/userbench.yaml`。当前覆盖非法参数、同轮多工具、精确/语义重复、模糊或捆绑问题、未搜索即回答、错误回答、无工具输出和步数耗尽；总策略罚分封顶 `0.75`。同轮多工具是独立事件，不与非法参数重复计罚。

## 有效性与诊断

包装层从冻结的任务标签和 TravelEnv 内部状态建立证据账本；这些字段不会出现在 Actor observation 中。任务/状态 schema 不匹配、快照缺失、证据账本无法更新或 reset/step/API 异常会令 `reward_valid=false`，终局分固定为 `0.0`，并在 `reward_breakdown.infrastructure_errors` 中说明原因。若模拟器判断、响应或搜索后端出现 fallback，但前后快照和账本仍完整可验证，则保留 `reward_valid=true`，并通过可选的 `reward_degraded` 与 `simulator_fallback_counts` 记录退化诊断；这类轨迹不是严格 Gold。抛出的 reset/step/API 异常保持 fail-loud，由 rollout 调度层重试，不转换成可训练失败样本。

评测和训练同时记录原始 `step_rewards`、`raw_cumulative_reward`、各维度质量/证据、覆盖率、效率和逐项罚分，以便复算和审计。
