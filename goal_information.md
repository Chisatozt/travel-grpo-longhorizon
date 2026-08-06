# Travel Reward v2 逻辑优化与 GRPO 动态采样重构规范

## 1. 任务背景

当前项目使用 UserBench 旅行环境生成多轮 Agent trajectory，并使用 terminal Reward 进行在线 GRPO。

当前存在两个主要问题：

1. 惩罚分项、总惩罚和最终 Reward 过早封顶，不同失败程度大量变成相同分数。
2. rollout 中经常出现组内全部为 `-1`，或某条 trajectory 的 `reward_valid=false` 导致整个 group 被丢弃。

本次修改只优化实际逻辑，不修改 Reward 版本标识。

---

## 2. 固定版本与兼容策略

本次修改必须始终保持：

```text
userbench-travel-reward-v2
```

禁止将其修改为 v3 或其他版本。

必须保持：

* `REWARD_VERSION` 的值不变；
* `configs/interaction_config/userbench.yaml` 中的 Reward version 不变；
* 教师采集配置中的 Reward version 不变；
* 教师轨迹顶层 `reward_version` 不变；
* `reward_breakdown.reward_version` 不变；
* 教师轨迹 `schema_version` 不变；
* GRPO 数据集版本和 Parquet schema 不变。

禁止新增以下或类似字段：

```text
reward_profile
reward_formula_revision
reward_revision
reward_implementation_version
legacy_reward_version
```

本次修改接受“修改前后的实现均称为 Reward v2”这一约束。

不得为区分新旧逻辑而修改持久化数据结构。

---

## 3. 兼容性目标

修改后必须保证：

1. 已有 Reward v2 教师轨迹仍可加载。
2. 已有教师轨迹不会因 Reward 版本检查被拒绝。
3. SFT Gold/Silver audit 仍能处理旧轨迹。
4. SFT action-only 渲染结构不变。
5. 已有 GRPO 输入 Parquet 无需重新生成。
6. GRPO rollout 输出仍符合 veRL trainer 的 DataProto 契约。
7. 评测模块原先读取的字段继续存在且类型不变。
8. 新增诊断字段必须是可选字段。
9. 旧产物缺少新增字段时必须使用安全默认值。

---

## 4. 不可修改的边界

禁止修改：

```text
environments/UserBench/
```

禁止改变：

* UserBench 官方任务标签；
* 正确答案集合；
* 最优答案集合；
* 工具协议；
* Actor 可见 prompt；
* Actor 可见工具观察；
* 教师轨迹 messages；
* assistant/tool message 配对规则；
* SFT tokenization 与 action-only label 结构；
* GRPO canonical dataset 列结构。

禁止向 Actor 暴露：

```text
best_ids
correct_ids
preference_ids_by_aspect
remaining_preference_ids
reward_snapshot
hidden environment state
```

---

## 5. 优先检查的文件

开始修改前检查：

```text
src/travel_grpo/envs/reward.py
src/travel_grpo/envs/userbench_context.py
src/travel_grpo/envs/userbench_wrapper.py

src/travel_grpo/training/sft_collection.py
src/travel_grpo/training/sft_dataset.py

src/travel_grpo/training/grpo/adapter/agent_loop.py
src/travel_grpo/training/grpo/adapter/session.py
src/travel_grpo/training/grpo/dynamic_sampling.py
src/travel_grpo/training/grpo/data.py

src/travel_grpo/evaluation/validation.py
src/travel_grpo/evaluation/checkpoint_selection.py

configs/interaction_config/userbench.yaml
configs/train/grpo/grpo.yaml

tests/test_reward.py
tests/test_userbench_context.py
tests/test_grpo_adapter.py
tests/test_grpo_pipeline.py
```

重点搜索：

```text
REWARD_VERSION
compute_travel_reward
terminal_reward
raw_terminal_reward
policy_penalty
reward_valid
infrastructure_errors
actor_attempts
num_tool_calls
select_reward_varying_groups
install_verl_bounded_sampler
trajectory_rejection_reasons
sft_admission_reasons
```

---

## 6. 修改前兼容性基线

在修改 Reward 前，先运行并记录：

```bash
python -m pytest \
  tests/test_reward.py \
  tests/test_userbench_context.py \
  tests/test_grpo_adapter.py \
  tests/test_grpo_pipeline.py \
  -q
```

如项目中已有 SFT audit fixture 或已采集教师轨迹，运行：

```bash
python scripts/train/sft/sft_train.py \
  --audit-only <existing-v2-teacher-trajectory-file>
```

记录：

* 输入轨迹数；
* 接受数；
* 拒绝数；
* 拒绝原因；
* Reward version；
* schema version。

修改后必须再次运行相同命令进行对比。

不得通过修改已有教师轨迹文件使其通过测试。

---

# 7. Reward v2 逻辑优化

## 7.1 取消分项惩罚 cap

现有逻辑可能包含：

```python
min(0.30, invalid_actions * 0.10)
min(0.15, exact_repeats * 0.05)
min(0.15, semantic_repeats * 0.05)
```

修改为按实际次数累计：

```python
penalty_components = {
    "invalid_action": (
        0.0
        if parallel_tool_calls
        else max(0, invalid_actions) * 0.10
    ),
    "parallel_tool_calls": 0.25 if parallel_tool_calls else 0.0,
    "exact_repeat": max(0, exact_repeats) * 0.05,
    "semantic_repeat": max(0, semantic_repeats) * 0.05,
    "ambiguous_action": max(0, ambiguous_actions) * 0.05,
    "unsearched_answer": max(0, unsearched_answers) * 0.10,
    "wrong_answer": max(0, wrong_answers) * 0.15,
    "no_tool_output": 0.15 if no_tool_output else 0.0,
    "max_steps": 0.15 if max_steps_reached else 0.0,
}
```

如果 parallel tool call 已作为独立协议事件计罚，不重复计算同一次 invalid action。

## 7.2 删除总惩罚 cap

修改为：

```python
policy_penalty = sum(penalty_components.values())
```

不得继续使用低值 `total_cap`。

必须继续输出：

```text
penalty_components
policy_penalty
```

---

## 8. 增加搜索进度信号

新增：

```python
search_coverage = _ratio(
    len(searched_aspects),
    len(task.aspects),
)
```

该指标必须来自 evidence ledger，而不是 Actor 自述。

推荐公式：

```python
raw_reward = (
    0.65 * (2.0 * grounded_quality - 1.0)
    + 0.15 * active_coverage
    + 0.10 * search_coverage
    + 0.10 * efficiency
    - 0.10 * passive_coverage
    - 0.30 * (1.0 - completion_rate)
    - policy_penalty
)
```

要求：

* 完整最优轨迹仍为 `1.0`；
* 完整正确备选轨迹约为 `0.74`；
* 搜索但未完成答案的轨迹优于完全无进展；
* 未搜索直接猜答案仍为负分；
* 犯错次数增加时 raw reward 严格下降。

`search_coverage` 可以作为 Reward breakdown 中的可选附加诊断字段。

不得让 SFT loader 强制要求旧轨迹包含该字段。

---

## 9. 平滑负分压缩

禁止继续使用会把所有小于 `-1` 的值变成同一个结果的硬裁剪：

```python
terminal_reward = min(1.0, max(-1.0, raw_reward))
```

推荐：

```python
NEGATIVE_REWARD_TEMPERATURE = 1.5


def squash_terminal_reward(
    raw_reward: float,
    *,
    negative_temperature: float = NEGATIVE_REWARD_TEMPERATURE,
) -> float:
    if not math.isfinite(raw_reward):
        raise UserBenchRewardError("raw reward must be finite")

    if (
        not math.isfinite(negative_temperature)
        or negative_temperature <= 0.0
    ):
        raise UserBenchRewardError(
            "negative reward temperature must be positive and finite"
        )

    if raw_reward >= 0.0:
        return min(1.0, raw_reward)

    return -math.tanh(
        (-raw_reward) / negative_temperature
    )
```

计算：

```python
terminal_reward = (
    squash_terminal_reward(raw_reward)
    if reward_valid
    else 0.0
)
```

必须满足：

```text
raw=-1.0
raw=-1.1
raw=-1.4
```

映射到三个不同的 terminal Reward。

不得使用随机噪声。

---

## 10. Efficiency 计入 Actor attempts

非法工具调用可能增加：

```text
actor_attempts
invalid_actions
```

但不增加：

```text
num_tool_calls
```

因此 efficiency 不能只基于环境 step。

给 Reward 计算增加可选参数：

```python
actor_attempts: int | None = None
```

计算：

```python
environment_steps = max(0, steps)

normalized_actor_attempts = (
    environment_steps
    if actor_attempts is None
    else max(0, actor_attempts)
)

effective_steps = max(
    environment_steps,
    normalized_actor_attempts,
)
```

efficiency 使用 `effective_steps`。

可以新增以下可选诊断字段：

```text
environment_steps
actor_attempts
effective_steps
```

旧轨迹缺少这些字段时，SFT audit 不得失败。

---

# 11. hard invalid 与可恢复 fallback

## 11.1 hard invalid

只有 Reward evidence 无法可信计算时才使用：

```text
reward_valid=false
terminal_reward=0.0
```

典型情况：

* 缺少 reward task；
* 缺少必要 reward snapshot；
* evidence ledger 无法建立；
* 必要隐藏状态损坏；
* 非有限 Reward 输入；
* task ID 严重不一致；
* 环境状态无法读取。

## 11.2 可恢复 simulator fallback

以下 fallback 不应无条件导致 hard invalid：

```text
userbench_judgment_fallbacks
userbench_response_fallbacks
userbench_search_fallbacks
```

但只有在以下条件同时满足时，轨迹才可以继续保持 `reward_valid=true`：

* reward task 完整；
* 前后 reward snapshot 完整；
* evidence ledger 可确定更新；
* answer/search/preference 状态仍可验证；
* Reward 输入有限；
* 没有证据缺失。

允许新增可选 runtime 诊断字段：

```text
reward_degraded
simulator_fallback_counts
```

这些字段是功能诊断字段，不是 Reward 版本区分字段。

要求：

* 旧教师轨迹缺少这些字段时继续可用；
* SFT loader 不得将它们设为必填项；
* 教师轨迹 schema version 不变；
* 不因新增字段要求重新采集数据。

如果 fallback 导致 snapshot 或证据缺失，仍然必须 hard invalid。

---

# 12. 动态采样的 task 与 batch 语义

当前一次 optimizer update 使用固定输入 task batch。

例如：

```text
Task A × 4 rollout
Task B × 4 rollout
```

当动态采样再次调用：

```python
original(batch)
```

必须仍然对相同 Task A 和 Task B 重新采样。

不同 generation batch 之间：

* task UID 相同；
* prompt 相同；
  -环境初始任务相同；
* Actor 随机采样结果可以不同。

禁止把不同 task 的 trajectory 组成同一个 GRPO group。

正确候选池：

```python
candidates_by_uid = {
    task_a_uid: [...],
    task_b_uid: [...],
}
```

禁止：

```text
[A1, A2, B1, B2]
```

---

# 13. 动态采样跨 batch 补齐

## 13.1 候选保留

当前行为可能是某一组出现一条 invalid 就丢弃完整 group。

修改为：

1. 按 UID 跨 generation batch 保存单条 valid candidate；
2. hard invalid trajectory 不进入候选池；
3. 同 UID 的其他 valid trajectory 继续保留；
4. 最多采样 `max_generation_batches` 次；
5. 每个 UID 最终选择恰好 `group_size` 条 trajectory。

例如：

```text
Batch 1:
Task A = valid, valid, invalid, valid

Batch 2:
Task A = valid, invalid, valid, valid
```

应能从两个 batch 中选择四条独立 valid trajectory。

## 13.2 DataProto 保存方式

候选池必须保存完整单行 DataProto slice，例如：

```python
row = output.slice(index, index + 1)
```

不要只保存手工构造的 reward 和 metadata 字典。

不得遗漏：

* token IDs；
* attention mask；
* response mask；
* rollout token 字段；
* rm_scores；
* non_tensor_batch；
* UID；
* multi-turn metadata；
* tool metadata。

## 13.3 最终结构

最终返回 trainer 的结构必须与修改前一致：

```text
required_groups × group_size
```

在当前固定配置下通常是：

```text
2 × 4 = 8 行
```

必须保证：

* task UID 顺序与原输入一致；
* 每个 UID 恰好四行；
* 同一 UID 的四行连续；
* 不同 UID 不混组；
* 最终 batch 字段集合不变；
* tensor 与 non-tensor 行数严格对齐。

---

# 14. 候选选择规则

当同一 UID 有超过四条 valid candidate 时，使用确定性规则选择：

1. 优先选择 evidence 完整、无 fallback 的轨迹；
2. clean 不足时才使用可恢复 fallback 轨迹；
3. 最大化：
   `max(reward) - min(reward)`；
4. 范围相同时，优先 Reward 唯一值更多的组合；
5. 再相同时按 generation batch 和原始 row 顺序选择；
6. 不使用随机选择；
7. 不复制同一条 trajectory。

只有最终四条满足：

```python
max_reward - min_reward <= tolerance
```

才判定 constant group。

真正 constant 的 group 仍应丢弃。

---

# 15. 不修改持久化数据结构

本次任务不得修改：

## 教师轨迹

不得改变：

```text
schema_version
messages
step_rewards
reward_version
reward_breakdown 的既有字段类型
```

可以让新的 `reward_breakdown` 包含额外可选诊断字段，但：

* 旧文件没有这些字段也必须可加载；
* 不得将其加入必填校验；
* 不得修改 trajectory schema version；
* 不得重写旧文件。

## SFT trainer examples

必须保持：

```text
input_ids
attention_mask
labels
task_id
trajectory_id
assistant_turn_index
```

结构和类型不变。

## GRPO Parquet

必须保持：

```text
data_source
prompt
ability
agent_name
reward_model
extra_info
```

及其原有 schema 不变。

本次 Reward 修改不应要求重新运行 GRPO 数据准备。

---

# 16. 评测兼容性

必须继续输出评测模块已读取的字段：

```text
reward_valid
terminal_reward
quality_by_aspect
correct_itinerary
gold_itinerary
user_aligned_success
completion_rate
active_preference_coverage
passive_preference_coverage
efficiency
policy_penalty
invalid_actions
exact_repeats
semantic_repeats
```

不得删除、重命名或改变类型。

因为修改前后的实现均称为 v2，禁止新增版本辨识字段。

运行目录必须避免混合旧、新 validation summary。

本次代码修改后进行 checkpoint 选择时，应重新生成同一实验中的：

```text
SFT step-0 validation
GRPO candidate validation
```

不要把旧逻辑生成的 summary 与新逻辑生成的 summary 混合比较。

---

# 17. 必须增加的 Reward 测试

## 17.1 版本保持不变

```python
assert REWARD_VERSION == "userbench-travel-reward-v2"
```

配置校验也必须继续接受该版本。

## 17.2 满分轨迹

```python
assert gold["terminal_reward"] == pytest.approx(1.0)
```

## 17.3 错误次数单调

```python
r1 = score(invalid_actions=1)["terminal_reward"]
r4 = score(invalid_actions=4)["terminal_reward"]
r8 = score(invalid_actions=8)["terminal_reward"]

assert r1 > r4 > r8
```

## 17.4 负分不再全部为 -1

```python
a = squash_terminal_reward(-1.0)
b = squash_terminal_reward(-1.1)
c = squash_terminal_reward(-1.4)

assert a > b > c
assert len({a, b, c}) == 3
assert all(-1.0 < value < 0.0 for value in (a, b, c))
```

## 17.5 进度区分

验证：

```text
完成询问和搜索但未回答
>
完全无进展
```

同时：

```text
未搜索直接猜答案 < 0
```

## 17.6 极端边界

所有输入均满足：

```python
assert math.isfinite(terminal_reward)
assert -1.0 <= terminal_reward <= 1.0
```

---

# 18. 必须增加的动态采样测试

至少包括：

1. 同一 UID 的 `3 valid + 1 invalid` 保留三条 valid。
2. 两个 generation batch 可以为同一 UID 补齐四条。
3. 不同 UID 不会混组。
4. 最终 UID 顺序与原输入一致。
5. 每个 UID 的四行连续。
6. final DataProto 行数和字段集合不变。
7. hard invalid 不用于补齐。
8. clean 足够时不选择 degraded。
9. clean 不足时允许可恢复 fallback 轨迹补齐。
10. constant group 仍被丢弃。
11. 不复制 trajectory。
12. 选择过程确定性。

---

# 19. 必须增加的 SFT 兼容测试

增加一个已有 Reward v2 教师轨迹 fixture。

验证：

```python
assert record["reward_version"] == "userbench-travel-reward-v2"
assert record["reward_breakdown"]["reward_version"] == (
    "userbench-travel-reward-v2"
)
```

并验证：

```python
assert sft_admission_reasons(
    record,
    accepted_quality_tiers=("gold", "silver"),
) == ()
```

还应验证：

* 旧轨迹没有 `search_coverage` 时不报错；
* 旧轨迹没有 `effective_steps` 时不报错；
* 旧轨迹没有 `reward_degraded` 时不报错；
* action-only rendering 结果字段不变；
* 教师轨迹 schema version 不变。

如果测试 fixture 中本身不是可接受轨迹，应验证其拒绝原因在修改前后保持一致，而不是强行断言接受。

---

# 20. 推荐执行顺序

## 阶段一：建立兼容性基线

运行：

```bash
python -m pytest \
  tests/test_reward.py \
  tests/test_userbench_context.py \
  tests/test_grpo_adapter.py \
  tests/test_grpo_pipeline.py \
  -q
```

审计已有教师轨迹。

## 阶段二：Reward 逻辑

修改：

```text
reward.py
userbench_context.py
相关配置校验
tests/test_reward.py
```

运行：

```bash
python -m pytest tests/test_reward.py -q
```

## 阶段三：validity 逻辑

修改：

```text
userbench_context.py
agent_loop.py
相关测试
```

运行：

```bash
python -m pytest \
  tests/test_userbench_context.py \
  tests/test_grpo_adapter.py \
  -q
```

## 阶段四：动态采样

修改：

```text
dynamic_sampling.py
tests/test_grpo_pipeline.py
```

运行：

```bash
python -m pytest tests/test_grpo_pipeline.py -q
```

## 阶段五：SFT 兼容性

运行已有 SFT audit 测试和：

```bash
python scripts/train/sft/sft_train.py \
  --audit-only <existing-v2-teacher-trajectory-file>
```

修改前后接受数和拒绝原因应一致，除非明确发现原代码 bug；不得静默改变 admission。

## 阶段六：完整回归

运行：

```bash
python -m pytest \
  tests/test_reward.py \
  tests/test_userbench_context.py \
  tests/test_grpo_adapter.py \
  tests/test_grpo_pipeline.py \
  -q
```

以及项目中已有的 SFT dataset、collection 和 evaluation 相关测试。

最后运行：

```bash
python -m compileall src/travel_grpo
```

---

# 21. 完成标准

任务完成必须满足：

* 所有 Reward version 仍为 `userbench-travel-reward-v2`；
* 未新增用于区分新旧 Reward 实现的字段；
* 惩罚不再过早封顶；
* 不同负向 raw reward 不再全部变成 `-1`；
* 完整最优轨迹仍为 `1.0`；
* 不同失败进度可以获得不同 Reward；
* hard invalid 仍不作为普通负样本；
* 可恢复 fallback 不再无条件令整个 group 报废；
* 同一 task UID 可以跨 generation batch 补齐；
* 不同 task 不会混组；
* final DataProto schema、行数和顺序符合原契约；
* 旧教师轨迹继续可用于 SFT；
* SFT action-only 数据结构不变；
* GRPO 输入 Parquet 无需重新生成；
* 所有指定测试通过；
* `compileall` 通过。

---

# 22. 最终报告格式

完成后输出：

## 修改文件

列出每个修改文件及用途。

## Reward v2 新逻辑

给出最终公式、压缩函数和典型数值。

## 兼容性

说明：

* Reward version 是否保持不变；
* 教师轨迹 schema 是否保持不变；
* 旧教师轨迹 audit 是否通过；
* SFT 数据结构是否变化；
* GRPO Parquet 是否需要重新生成；
* DataProto 最终结构是否变化。

## 动态采样

说明：

* generation batch 是否为同一批 task；
* 如何按 UID 保存候选；
* 如何跨 batch 补齐；
* 如何避免不同 task 混组；
* 如何保持最终顺序和行数。

## 测试结果

列出实际测试命令、通过数量和失败数量。

## 尚存风险

说明当前尚存的风险
