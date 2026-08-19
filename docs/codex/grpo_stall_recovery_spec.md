# GRPO T3-Style Stall Recovery 实现规范

> **归档 / Historical design.** 本文件记录旧版 stall/recovery 方案及其推演，不代表当前训练已启用该分支；当前 GRPO 行为应以代码和 [GRPO 训练契约](../training/grpo.md) 为准。

## 1. 背景

当前项目使用 UserBench 多轮环境进行在线 GRPO。

Actor 在弱能力阶段可能出现：

* 连续重复询问；
* 重复搜索；
* 合法但没有新增 evidence 的动作；
* malformed tool call；
* 已经获得可回答搜索结果，却迟迟不进入 ANSWER；
* 长时间没有任务进展，最终耗尽 20 个 assistant turns。

这类 trajectory 的长尾会：

* 增加 rollout token；
* 增加 UserBench simulator 调用；
* 增加 actor attempts；
* 降低 efficiency；
* 增加 max-steps trajectory；
* 可能降低有效 GRPO group 产生概率。

本方案参考 T3 的“无信息尾部处理”思想，但不做完全等价的 T3 复现。

本项目采用：

```text
stall detection
    ↓
有 Actor-visible answer evidence
    → 一次 ANSWER-only recovery

无 Actor-visible answer evidence
    → early hard cut
```

---

# 2. 设计目标

需要满足：

```text
正常 rollout
    │
    │ 连续无 progress
    ▼
达到 threshold
    │
    ├─ 有可回答的 Actor-visible option
    │      ↓
    │   一次 ANSWER-only generation
    │      │
    │      ├─ 成功登记 answer
    │      │      ↓
    │      │   恢复正常 rollout
    │      │
    │      └─ 失败
    │             ↓
    │        stalled_no_progress
    │
    └─ 无可回答 option
           ↓
      stalled_no_progress
```

每条 trajectory 最多使用一次 recovery。

---

# 3. 固定边界

以下保持不变：

```text
REWARD_VERSION = "userbench-travel-reward-v2"
```

禁止修改：

* `src/travel_grpo/envs/reward.py` 的 Reward 公式；
* teacher trajectory schema；
* SFT action-only schema；
* SFT Gold/Silver admission；
* GRPO canonical Parquet schema；
* dynamic sampling 的 task-UID group 语义；
* `environments/UserBench/`；
* frozen evaluation rollout；
* external veRL 代码。

除非回归测试发现已有 bug，否则上述模块只允许增加必要测试，不修改业务行为。

---

# 4. 建议修改范围

优先检查和修改：

```text
src/travel_grpo/envs/userbench_tools.py
src/travel_grpo/envs/userbench_context.py

src/travel_grpo/training/grpo/adapter/tools.py
src/travel_grpo/training/grpo/adapter/agent_loop.py

configs/interaction_config/agent_loop.yaml
scripts/train/grpo/train_grpo.py
src/travel_grpo/training/grpo/preflight.py

tests/test_userbench_context.py
tests/test_grpo_adapter.py
tests/test_grpo_pipeline.py

docs/training/grpo.md
```

重点复用当前已有：

```text
UserBenchSessionState
UserBenchRewardSnapshot
searched_aspects
answers
active_preference_ids
passive_preference_ids
actor_attempts
OPTION_ID
aspect_from_option_id
UserBenchAgentLoop
execute_userbench_action
```

---

# 5. CLI 设计

`run_grpo.sh` 当前已经：

```bash
exec python scripts/train/grpo/train_grpo.py \
  --config configs/train/grpo/grpo.yaml \
  "$@"
```

因此不要在 shell 中重复解析参数。

在 `train_grpo.py` 中增加：

```python
parser.add_argument(
    "--stall-recovery",
    action=argparse.BooleanOptionalAction,
    default=False,
)

parser.add_argument(
    "--stall-threshold",
    type=int,
    default=4,
)
```

约束：

```text
stall_threshold >= 1
```

Baseline：

```bash
bash scripts/train/grpo/run_grpo.sh \
  --no-stall-recovery \
  --output outputs/models/grpo-baseline
```

开启：

```bash
bash scripts/train/grpo/run_grpo.sh \
  --stall-recovery \
  --stall-threshold 4 \
  --output outputs/models/grpo-stall4
```

未显式指定时：

```text
stall_recovery = false
stall_threshold = 4
```

因此默认行为必须等价于当前项目。

---

# 6. 配置传播

AgentLoop 当前由：

```text
configs/interaction_config/agent_loop.yaml
```

实例化。

建议通过 subprocess-local environment variable 传递：

```text
TRAVEL_GRPO_STALL_RECOVERY
TRAVEL_GRPO_STALL_THRESHOLD
```

`train_grpo.py`：

```python
launch_env = {
    **os.environ,
    "PYTHONPATH": str(SRC),
    "TRAVEL_GRPO_STALL_RECOVERY": (
        "true" if args.stall_recovery else "false"
    ),
    "TRAVEL_GRPO_STALL_THRESHOLD": str(
        args.stall_threshold
    ),
}
```

然后：

```python
subprocess.call(
    command,
    cwd=ROOT,
    env=launch_env,
)
```

`agent_loop.yaml`：

```yaml
- name: userbench_tool_agent
  _target_: travel_grpo.training.grpo.adapter.agent_loop.UserBenchAgentLoop

  environment_config_path: configs/interaction_config/userbench.yaml
  simulator_config_path: configs/interaction_config/simulator_train.yaml

  max_steps: 20

  stall_recovery_enabled: ${oc.env:TRAVEL_GRPO_STALL_RECOVERY,false}
  stall_no_progress_threshold: ${oc.env:TRAVEL_GRPO_STALL_THRESHOLD,4}
```

AgentLoop 中必须显式规范化：

* boolean；
* integer；
* threshold 范围。

不要使用：

```python
bool("false")
```

这种错误解析方式。

---

# 7. Training 与 Validation 隔离

目标：

```text
GRPO training rollout
→ 可启用 stall recovery

GRPO validation
→ 始终关闭

frozen evaluation
→ 不修改
```

当前 profile 已固定：

```text
training:
temperature = 0.7
top_p = 0.9

validation:
temperature = 0.0
top_p = 1.0
do_sample = false
```

在不修改 pinned veRL 的前提下，可以增加集中 helper：

```python
def is_validation_sampling(
    sampling_params: Mapping[str, Any],
) -> bool:
    ...
```

基于当前固定 sampling profile 判断是否为 validation。

有效开关：

```text
effective_stall_recovery
=
configured_stall_recovery
AND
NOT validation_sampling
```

必须在 `preflight.py` 中检查：

1. 当前 training sampling 不会被 classifier 识别为 validation；
2. 当前 validation sampling 可以被识别；
3. threshold 合法。

如果未来配置漂移导致判断不再唯一，应 fail loudly。

不要静默把 stall recovery 应用到 validation。

---

# 8. Progress 定义

Progress 必须来自可验证环境 evidence。

不得根据：

```text
Actor thought
Actor self-report
LLM Judge
```

判定。

## 8.1 Preference progress

`record_step()` 已能比较：

```text
before.remaining_preference_ids
snapshot.remaining_preference_ids
```

以及：

```text
active_delta
passive_delta
```

如果环境记录了新的 preference evidence：

```python
preference_progress = (
    active_delta > 0
    or passive_delta > 0
)
```

则视为 progress。

Passive preference 虽然 Reward 中价值与 active 不同，但它仍是新信息，因此 stall detector 中应视为 progress。

---

## 8.2 Search progress

当前：

```python
before.remaining_search_aspects
-
snapshot.remaining_search_aspects
```

用于更新：

```python
session.searched_aspects
```

如果首次成功搜索了新的 aspect：

```python
search_progress = True
```

重复搜索已经完成的 aspect：

```python
search_progress = False
```

---

## 8.3 Answer progress

只有环境真正登记新的 answer 才算：

```text
new choice_initial
+
session.answers 新增 aspect
```

不能因为 Actor 发出了：

```text
choice="answer"
```

就自动视为 progress。

被环境拒绝的 answer 不算 progress。

---

## 8.4 综合

```python
made_progress = (
    preference_progress
    or search_progress
    or answer_progress
)
```

有 progress：

```python
consecutive_no_progress = 0
```

否则：

```python
consecutive_no_progress += 1
```

同时维护：

```python
max_consecutive_no_progress
```

---

# 9. Infrastructure invalid 不进入 stall 语义

如果：

* reward snapshot 缺失；
* evidence transition 无法验证；
* simulator infrastructure failure；
* task/reward evidence 损坏；

继续走当前 infrastructure-invalid 路径。

不得把这些错误重新分类成：

```text
stalled_no_progress
```

stall detector 只处理：

> evidence 本身可信，但是 Actor 没产生实质进展。

---

# 10. 不经过环境的无效动作

当前以下错误可能不会进入 `record_step()`：

```text
unknown tool
malformed JSON arguments
invalid UserBenchAction
```

因此需要提供集中入口，例如：

```python
session.record_non_progress(reason)
```

该方法负责：

```text
consecutive_no_progress += 1
max_consecutive_no_progress 更新
检查是否达到 stall threshold
```

原有：

```text
invalid_actions
protocol error
```

逻辑继续保留。

Parallel tool call 当前直接终止 trajectory，不必先经过 stall threshold。

---

# 11. Actor-visible Search Evidence

这是本方案最重要的安全边界之一。

不能只使用：

```python
session.searched_aspects
```

决定是否可以强制 ANSWER。

因为 stall controller 需要确保：

> Actor 的上下文里真的出现过可以提交的 option ID。

---

## 11.1 Option ID

项目已有：

```text
F\d+ → flight
H\d+ → hotel
A\d+ → apartment
C\d+ → rental_car
R\d+ → restaurant
```

应复用 `userbench_tools.py` 中已有 option ID 规则和 aspect 映射。

增加辅助函数，例如：

```python
def extract_visible_option_ids(
    feedback: str,
) -> set[str]:
    ...
```

要求：

* 使用边界匹配；
* 只接受官方 option ID 格式；
* 不从 hidden reward state 构造。

---

## 11.2 只从 SEARCH feedback 记录

仅当：

```python
action.choice is ActionChoice.SEARCH
```

时，从：

```python
result.observation.feedback
```

中提取。

保存：

```python
visible_option_ids_by_aspect: dict[str, set[str]]
```

例如：

```python
{
    "hotel": {"H1", "H2", "H3"},
    "rental_car": {"C1", "C2"},
}
```

不要从：

* task best IDs；
* correct IDs；
* reward snapshot；
* SFT 标签；

中填充这个集合。

---

# 12. Answerable Evidence

定义当前 Actor 可以被要求提交的 option：

```python
visible_answer_options = {
    option_id
    for aspect, option_ids
    in visible_option_ids_by_aspect.items()
    if aspect not in answers
    for option_id in option_ids
}
```

只有：

```python
bool(visible_answer_options)
```

时，stall 才进入 ANSWER-only recovery。

否则：

```text
hard_stop_stalled()
```

不能要求 Actor 在未看到候选项时猜答案。

---

# 13. Session Runtime State

建议增加：

```python
stall_recovery_enabled: bool = False
stall_no_progress_threshold: int = 4

consecutive_no_progress: int = 0
max_consecutive_no_progress: int = 0

stall_recovery_triggered: bool = False
stall_recovery_used: bool = False

answer_only_pending: bool = False
answer_only_generation_started: bool = False

stall_hard_truncated: bool = False

visible_option_ids_by_aspect: dict[str, set[str]]
```

这些字段属于 runtime control/diagnostics。

不得：

* 改 teacher schema；
* 变成 SFT 必填项；
* 改 Reward version。

---

# 14. Stall Trigger

当：

```text
stall_recovery_enabled
AND
consecutive_no_progress >= threshold
```

触发判断。

---

## 14.1 已经用过 recovery

如果：

```python
stall_recovery_used is True
```

直接：

```python
hard_stop_stalled()
```

每条 trajectory 不提供第二次 recovery。

---

## 14.2 第一次 stall，有 answerable evidence

```python
stall_recovery_triggered = True
answer_only_pending = True
```

trajectory 暂时不终止。

下一次 generation 是 recovery generation。

---

## 14.3 第一次 stall，没有 answerable evidence

直接：

```python
hard_stop_stalled()
```

---

# 15. Hard Stop 语义

必须实现为：

```python
session.terminated = True
session.truncated = False

session.termination_reason = (
    "stalled_no_progress"
)

session.stall_hard_truncated = True
```

关键：

```text
truncated=false
```

因为当前 Reward v2 使用：

```python
max_steps_reached=self.truncated
```

判断 max-steps penalty。

T3-style early cut 并不是：

```text
Actor 实际跑满 20 步
```

所以不应额外受到 `max_steps` penalty。

---

# 16. Reward 语义

Stall trajectory：

```text
不是 infrastructure invalid
不是人为固定失败分
不是 max_steps
```

如果 reward evidence 完整：

```text
reward_valid=true
```

继续使用已有：

* preference coverage；
* search coverage；
* completion；
* grounded quality；
* efficiency；
* policy penalty；

正常计算 Reward v2。

因此可以自然形成：

```text
完全无进展
<
部分 preference progress
<
preference + search
<
部分正确 answer
<
完整成功
```

不要增加：

```text
stall 固定 penalty
```

除非未来通过独立实验明确决定。

本次任务不做该修改。

---

# 17. ANSWER-only Prompt

当：

```python
answer_only_pending is True
```

时，将安全恢复指令追加到 Actor-visible tool feedback。

推荐内容：

```text
Recovery instruction:
You have made no verifiable progress for several consecutive turns.
Your next interact_with_env call must use choice="answer".
Only submit option IDs that were explicitly shown in previous successful
search results for unanswered travel aspects.
Do not search or ask another question.
```

该提示：

* 不列出正确答案；
* 不指定具体 option；
* 不暴露 hidden preference；
* 不暴露 Reward；
* 不告诉 Actor best/correct 信息。

---

# 18. ANSWER-only Enforcement

仅靠提示不够，controller 必须真正 enforce。

下一次 generation 只接受：

```text
exactly one interact_with_env call
choice == "answer"
```

并验证：

```text
submitted option IDs
⊆
visible_answer_options
```

---

## 18.1 禁止行为

以下行为不得执行环境：

```text
choice=search
choice=action
unknown tool
malformed call
parallel tool calls
unseen option ID
already answered aspect 的 option
没有有效 tool call
```

这些情况立即：

```text
hard_stop_stalled()
```

不给第二次 generation。

---

# 19. 可见但错误的 Answer 必须允许

假设 Actor 搜索结果中真实看到：

```text
H1
H2
H3
```

隐藏标签实际为：

```text
best = H3
correct = H2/H3
```

Controller 必须允许 Actor 提交：

```text
H1
H2
H3
```

三者任意一个。

不能因为：

```text
H1 实际错误
```

就阻止执行。

正确性由：

```text
Reward v2
```

评估，而不是 stall controller。

---

# 20. Recovery Generation 只能一次

在 AgentLoop generation path 中记录：

```python
answer_only_generation_started
```

第一次进入：

```text
answer_only_pending=true
```

时允许生成一次。

不能进入：

```text
重新提示
重新生成
重新修复
```

循环。

如果该 generation 没有形成可执行 answer：

```text
hard_stop_stalled()
```

---

# 21. Recovery Success

如果 Actor 提交合法可见 option，且环境真正新增：

```text
session.answers
```

则：

```python
consecutive_no_progress = 0

answer_only_pending = False
answer_only_generation_started = False

stall_recovery_used = True
```

如果 episode 尚未完成：

```text
恢复正常 rollout
```

例如：

```text
hotel + rental_car
```

recovery 成功回答 hotel 后：

```text
继续正常处理 rental_car
```

---

# 22. 第二次 Stall

第一次 recovery 成功后：

```text
stall_recovery_used=true
```

如果 later rollout 再次达到：

```text
no_progress threshold
```

直接：

```text
stalled_no_progress
```

不再救第二次。

这样降低训练策略对 controller 的依赖。

---

# 23. Diagnostics

建议加入：

```text
stall_recovery_enabled
stall_recovery_triggered
stall_recovery_used
stall_hard_truncated

consecutive_no_progress
max_consecutive_no_progress

answer_only_generation_started
visible_answer_option_count
```

写入现有：

```text
session.metrics()
extra_fields["reward_extra_info"]
```

这些字段只用于实验分析。

不得记录：

```text
best_ids
correct_ids
hidden preference IDs
reward snapshot
```

---

# 24. Dynamic Sampling

不要修改现有 dynamic sampling 选择规则。

对于：

```text
termination_reason=stalled_no_progress
reward_valid=true
```

trajectory 是合法 candidate。

它仍由现有逻辑根据：

```text
task UID
reward_valid
reward variance
clean/degraded priority
```

决定是否保留。

不得仅因 `stalled_no_progress` 将其标为 sampling invalid。

---

# 25. Baseline Compatibility

当：

```text
stall_recovery=false
```

时：

* 不追踪 stall 不应改变原有 control flow；
* 不追加 ANSWER-only feedback；
* 不提前终止；
* 不修改 tool call；
* 不改变 Reward；
* 不改变 dynamic sampling；
* 不改变 validation；
* 不改变 output schema 的既有字段。

新增 diagnostics 可以为空/默认值，但不得影响消费者。

---

# 26. 必须新增的单元测试

## 26.1 Feature default

验证：

```text
默认关闭
```

---

## 26.2 Progress reset

分别构造：

```text
new preference
new search
new answer
```

确保：

```text
consecutive_no_progress → 0
```

---

## 26.3 No-progress increments

构造：

```text
重复动作
重复搜索
无新 evidence
invalid tool
```

确保 streak 增加。

---

## 26.4 无可回答 evidence

达到 threshold：

```text
visible_answer_options = empty
```

结果：

```text
terminated=true
truncated=false
termination_reason=stalled_no_progress
```

---

## 26.5 Reward validity

evidence 完整时：

```text
reward_valid=true
```

并且：

```text
max_steps penalty = 0
```

---

## 26.6 有 visible option

达到 threshold 且：

```text
visible option = H1,H2,H3
```

结果：

```text
answer_only_pending=true
```

而不是立即终止。

---

## 26.7 Recovery SEARCH

Actor 下一步输出：

```text
choice=search
```

要求：

```text
环境不执行
stalled_no_progress
```

---

## 26.8 Recovery ACTION

同上。

---

## 26.9 Recovery unseen answer

Actor：

```text
H99
```

但从未在 SEARCH feedback 出现。

要求：

```text
环境不执行
stalled_no_progress
```

---

## 26.10 Visible wrong answer

Actor 已见过：

```text
H1
```

但 H1 隐藏标签实际错误。

要求：

```text
环境仍然执行
```

不能利用 hidden correctness 过滤。

---

## 26.11 Recovery success

Answer 真正被登记后：

```text
streak=0
answer_only_pending=false
stall_recovery_used=true
```

且未完成多-aspect task 时继续 rollout。

---

## 26.12 Second stall

再次达到 threshold：

```text
直接 hard cut
```

---

## 26.13 Validation off

validation sampling 下：

```text
effective stall recovery=false
```

即使 CLI 全局启用。

---

## 26.14 Baseline off

关闭 feature 后，旧测试行为保持。

---

# 27. 回归测试

至少运行：

```bash
python -m pytest tests/test_userbench_context.py -q
```

```bash
python -m pytest tests/test_grpo_adapter.py -q
```

```bash
python -m pytest tests/test_grpo_pipeline.py -q
```

然后：

```bash
python -m pytest \
  tests/test_reward.py \
  tests/test_userbench_context.py \
  tests/test_grpo_adapter.py \
  tests/test_grpo_pipeline.py \
  tests/test_sft_collection.py \
  tests/test_sft_dataset.py \
  -q
```

如项目还有相关 evaluation/preflight tests，一并运行。

最后：

```bash
python -m compileall src/travel_grpo
```

---

# 28. Dry-run

Baseline：

```bash
bash scripts/train/grpo/run_grpo.sh \
  --dry-run \
  --no-stall-recovery \
  --output outputs/models/grpo-dry-baseline
```

Experiment：

```bash
bash scripts/train/grpo/run_grpo.sh \
  --dry-run \
  --stall-recovery \
  --stall-threshold 4 \
  --output outputs/models/grpo-dry-stall4
```

检查：

1. 两次 preflight 均通过；
2. CLI 配置真正传入 AgentLoop；
3. baseline 未启用；
4. experiment 启用且 threshold=4；
5. 除 stall 配置外没有非预期训练参数变化。

---

# 29. 小规模真实 A/B

功能验证通过后，不要立即运行完整 500-step。

先比较：

```text
A: baseline
B: stall recovery, threshold=4
```

保持一致：

```text
SFT checkpoint
GRPO data
random seed
temperature/top_p
n
group size
dynamic sampling
Reward v2
learning rate
training steps
```

建议先使用：

```text
20～50 optimizer steps
```

观察 rollout 行为。

---

# 30. A/B 核心指标

至少统计：

```text
stall trigger rate
stall hard cut rate
answer recovery attempt rate
answer recovery success rate

mean actor_attempts
mean environment_steps

max_steps rate
completion rate
search coverage
active preference coverage

wrong answers
unsearched answers

terminal reward
correct itinerary
user aligned success

constant reward group rate
accepted GRPO group rate
skipped update rate

rollout token usage
UserBench simulator request count
```

---

# 31. 最重要的最终指标

不要只根据：

```text
training terminal reward
```

判断是否成功。

因为 stall controller 本身会改变 training rollout 分布。

优先观察：

```text
无 stall controller 的 validation
```

是否改善：

```text
correct itinerary
user aligned success
completion
grounding
```

同时确认：

```text
token / simulator cost
```

确实下降。

理想结果是：

```text
训练 rollout 更短
+
有效 GRPO group 更多
+
最终无干预 validation 不下降或提升
```

而不是：

```text
训练 reward 看起来更高
但 validation Actor 仍不会自主结束
```

---

# 32. 完成报告要求

完成后报告：

## 修改文件

每个文件的实际用途。

## Progress Detector

说明：

* preference；
* search；
* answer；

各自如何判断 progress。

## Actor-visible Evidence

说明 option ID 如何从 SEARCH feedback 获得，以及如何保证不使用隐藏标签。

## Stall State Machine

说明：

```text
normal
stall
answer-only
success
hard cut
second stall
```

完整路径。

## Reward Semantics

说明：

```text
stalled_no_progress
terminated=true
truncated=false
reward_valid=true（若 evidence 完整）
```

以及为什么不会触发 max_steps penalty。

## Training / Validation

说明隔离机制和 preflight 保护。

## CLI

给出 baseline 与实验组命令。

## Tests

列出实际执行命令和结果。

## Compatibility

明确：

* Reward 是否修改；
* dynamic sampling 是否修改；
* SFT 是否修改；
* evaluation 是否修改；
* parquet 是否需要重建。

## Risks

至少分析：

* controller dependency；
* false-positive stall；
* early answer；
* answer diversity collapse；
* train/validation distribution shift。
