# Agent 状态机与行为契约

> **归档 / Historical design.** 本文件是早期状态机设计与测试规划，包含 Reward v2 时代的约束；它不是当前运行时的唯一权威契约，使用前应以当前代码和评测契约为准。

> 状态：设计与测试规划；本文件不实现状态机、不启动训练或 GPU 推理，也不修改
> `environments/UserBench/`。  
> 适用边界：Actor、UserBench session、`interact_with_env` 工具适配层和 GRPO
> rollout 控制器。

## 0. 目标、非目标与术语

本契约把一个 task 的 Agent 行为约束为可审计的有限状态机（FSM）。它解决的是
“下一次工具调用是否合法、什么时候必须搜索/回答、什么时候必须换 aspect，以及
何时安全结束”，不是奖励公式、提示词优化或模型训练策略。

本步骤不做以下事情：

- 不实现新的 Python 控制器或修改现有运行代码；
- 不改变 `Travel Reward v2`、teacher/SFT/GRPO 数据 schema 或 veRL；
- 不改变 pinned `environments/UserBench/`；
- 不运行模型、模拟器、GPU 推理或正式 benchmark。

术语约定：

- **aspect**：一个需要独立搜索和选择的维度，例如 `flight`、`hotel`；顺序使用
  task 的公开 `dimensions` 顺序，不通过隐藏 best/correct 标签排序。
- **field**：某 aspect 的一个偏好字段。字段值可能是隐藏的，但 Actor 只有在用户
  明确说出后才可使用它。
- **normal candidate list**：最近一次当前 aspect 的 `search` 结果中，至少有一个
  正式 option ID，且没有 search fallback 标记的 Actor 可见候选列表。带 fallback
  标记的文本即使碰巧含有 ID，也不算 normal list。
- **fallback**：search 后端明确报告不可用/退化。当前 wrapper 可从
  `UserBenchStepResult.diagnostics["userbench_search_fallbacks"]` 或可见文本
  `searching backend is experiencing some issues` 识别。judgment/response fallback
  是独立的基础设施事件，不消耗 search retry 预算。
- **progress**：用户反馈产生新的、可见且可归因的信息，得到 normal candidate list，
  或提交一个来自当前候选列表的 answer。Actor 的 thought、它自己的“已完成”声明、
  LLM judge 结论都不是 progress。
- **terminal aspect**：该 aspect 已经 `ANSWERED` 或 `BLOCKED`。两者都不能再次
  action/search/answer，但奖励与 completion 语义不同。

## 1. 现有实现审查

### 1.1 Agent loop

`src/travel_grpo/training/grpo/adapter/agent_loop.py` 当前直接使用 veRL 0.8
`ToolAgentLoop`。现有职责包括：

- `session_requests_termination()` 只检查 `session.done`；
- 并行 tool call 在环境执行前被拒绝并终止（约第 70--85 行）；
- malformed/unknown tool 返回稳定的 Actor 可见错误，不执行环境（约第 162--201
  行）；
- generation 阶段按 validation sampling 关闭 stall recovery，并累计
  `actor_attempts`（约第 142--160 行）；
- rollout 结束后写入 Reward v2 和不含隐藏 ID 的 metrics（约第 203--278 行）。

这里没有一个按 aspect 管理 `phase`、search retry 次数、normal candidate 证据和
blocked 原因的 canonical FSM。下一步实现应在 session 与 tool dispatch 之间放置
一个单一的 guard/transition controller，而不是继续增加互相独立的布尔标志。

### 1.2 `UserBenchSessionState`

`src/travel_grpo/envs/userbench_context.py` 已有可复用的运行账本：

- `searched_aspects`、`answers`、`exact_repeats`、`semantic_repeats`、
  `invalid_actions`；
- `active_preference_ids`、`passive_preference_ids` 和
  `UserBenchRewardSnapshot` 用于 Reward v2 evidence ledger；
- `visible_option_ids_by_aspect` 只从 search 的 Actor 可见反馈提取 option ID；
- `consecutive_no_progress`、`stall_no_progress_threshold`、
  `answer_only_pending` 支持当前一次性 stall recovery。

这些字段不能直接等同于本契约的 phase：`active_preference_ids`、
`remaining_preference_ids`、`remaining_search_aspects`、`correct_ids` 等是 reward
侧信息；它们可用于审计和判定 `reward_valid`，不能通过控制行为告诉 Actor 隐藏状态。
未来建议新增一个独立的 per-aspect controller record；不要把新的 phase 继续
编码成更多散落的布尔值。

### 1.3 工具调用适配层

`src/travel_grpo/envs/userbench_tools.py` 固定唯一工具
`interact_with_env(thought, choice, content)`，`choice` 为 `action`、`search`、
`answer`。`UserBenchAction.from_parameters()` 已检查字段、choice、内容和前缀；
`normalized_action_signature()`、`semantic_action_signature()`、
`extract_visible_option_ids()` 可复用于重复和可见候选判断。

`src/travel_grpo/training/grpo/adapter/tools.py` 在进入 wrapper 前处理 malformed
调用、answer-only recovery 约束和环境异常。新的 FSM guard 必须位于环境执行前：被
禁止的 action/search/answer 要返回稳定、通用的错误并且不调用 UserBench。

### 1.4 现有 stall recovery

`UserBenchSessionState._maybe_trigger_stall()` 的现行行为是：连续无 progress 达到
阈值后，若有 Actor-visible option，则只给一次 answer-only generation；否则直接以
`stalled_no_progress` hard stop；再次 stall 或 recovery 失败也 hard stop。training
可配置而 validation sampling 会关闭它。

这套机制是 trajectory-level 兜底，不是本契约的 per-aspect search retry。实现 FSM
时必须保留以下边界：

1. no-progress threshold 到达后，正常 FSM 必须转入 `SEARCH_REQUIRED`，而不是继续
   action；现有 answer-only recovery 只能作为兼容的最后安全兜底，不能绕过“normal
   candidate 必须 answer”的规则。
2. fallback retry 以 aspect 为单位，最多一次实质改写；不能被 trajectory-level
   stall recovery 重置。
3. fallback、正确性和 preference completion 不得通过 recovery instruction 泄漏。
4. validation 的 effective stall recovery 继续为关闭；任何 future FSM 的
   training-only recovery 也必须显式遵守这一隔离。

## 2. Canonical runtime model

### 2.1 Episode 与 aspect record

每个 session 维护一个 controller（以下为设计字段，不是当前代码变更）：

```text
episode:
  current_aspect: aspect | None
  phase: one of the seven states below
  aspect_order: task.dimensions in public/canonical order
  no_progress_actions: int
  done: bool

aspect[aspect]:
  terminal: OPEN | ANSWERED | BLOCKED
  elicited_fields: set[field]              # 来自 Actor-visible user feedback
  no_preference_fields: set[field]
  normal_candidate_ids: set[option_id]     # 只来自公开 search feedback
  search_attempts: 0 | 1 | 2
  search_fallbacks: 0 | 1 | 2
  last_search_signature: str | None
  answer_id: option_id | None
  forced_search: bool
```

`elicited_fields` 的“完成”只表示 Actor-visible 会话已经得到具体值，或用户明确
表示“无偏好/都可以/不在意”。它不读取隐藏的 `remaining_preference_ids` 来决定
下一动作。若 reward evidence 与此 ledger 不一致，标记 infrastructure/reward
invalid 并 fail closed，而不是用一次隐蔽的控制分支修正 Actor。

### 2.2 七个状态的不变量

| 状态 | 进入条件 | 允许的下一工具调用 | 明确禁止 | 退出/终止条件 |
|---|---|---|---|---|
| `ELICITING` | 当前 aspect 仍有 Actor-visible 未完成 field | `action`，且只能问一个具体、未完成 field；用户明确无偏好也算完成 | `search`、`answer`；已明确无偏好 field 不得再次询问 | field 全部完成 → `SEARCH_REQUIRED`；连续无进展达到阈值 → 强制 `SEARCH_REQUIRED` |
| `SEARCH_REQUIRED` | 当前 aspect 可搜索，或 stall guard 强制搜索 | 仅一次当前 aspect 的 `search` | `action`、`answer`、相同 query 重复 | normal list → `ANSWER_REQUIRED`；第一次 fallback/空结果 → `SEARCH_RETRY_REQUIRED`；第二次 fallback/空结果 → `BLOCKED` → `SWITCH_ASPECT_REQUIRED` |
| `SEARCH_RETRY_REQUIRED` | 第一次 search fallback/空结果，且 retry 尚未消耗 | 仅一次对同一 aspect 的实质改写 `search` | `action`、`answer`、完全相同或仅改标点/顺序的 query | normal list → `ANSWER_REQUIRED`；再次 fallback/空结果 → `BLOCKED` → `SWITCH_ASPECT_REQUIRED` |
| `ANSWER_REQUIRED` | 当前 aspect 已有 normal candidate list | 仅一个 `answer`，且 ID 必须来自当前 normal list | `action`、任何 `search`、多 ID、不可见 ID、其他 aspect ID | 合法可见 ID → `ANSWERED` → `SWITCH_ASPECT_REQUIRED`；非法调用留在本状态 |
| `SWITCH_ASPECT_REQUIRED` | 当前 aspect 已 answered 或 blocked | 无；这是 controller 内部过渡，不产生 tool call | 对旧 aspect 的一切 `action/search/answer` | 按 canonical 顺序选下一个 OPEN aspect：缺 field → `ELICITING`，可搜索 → `SEARCH_REQUIRED`；没有 OPEN aspect → episode terminate |
| `ANSWERED` | 已接受一个来自 normal list 的可见 ID | 无 | 对该 aspect 的所有 action/search/answer；不得修改已登记 answer | 立即转 `SWITCH_ASPECT_REQUIRED`；不能因 hidden correctness 为 false 而回退 |
| `BLOCKED` | 当前 aspect 的 search fallback/空结果达到上限，或明确不可恢复的协议/基础设施策略 | 无 | 伪造 option、answer、再次 search、重开 preference | 立即转 `SWITCH_ASPECT_REQUIRED`；不能伪造 completion 或 `ANSWERED` |

`action` 在本表专指偏好 elicitation，不是任意自然语言。任何不满足 guard 的
调用都在 wrapper 之前拒绝、记为 invalid/non-progress，并保留原状态；不得让被拒
绝的调用改变 hidden reward ledger。

## 3. 事件与状态转换

### 3.1 统一事件分类

controller 只接收下列事件；事件中的文本和候选来自 Actor-visible transcript，
fallback 可辅以 wrapper 的基础设施诊断计数：

```text
ACTION_PROGRESS(field, user_value | NO_PREFERENCE)
ACTION_NO_PROGRESS(field | UNKNOWN, reason)
SEARCH_NORMAL(aspect, visible_option_ids)
SEARCH_FALLBACK(aspect, attempt_number)
SEARCH_EMPTY(aspect, attempt_number)
ANSWER_VISIBLE(aspect, option_id)
ANSWER_INVALID(aspect, reason)
REPEAT_QUERY(aspect, exact_or_semantic)
MALFORMED_OR_PARALLEL(reason)
ENVIRONMENT_TERMINATED / ENVIRONMENT_TRUNCATED
INFRASTRUCTURE_FAILURE(reason)
```

`SEARCH_EMPTY` 是“没有 fallback 标记且没有正式 option ID”的失败搜索；为了避免
无结果死循环，按 fallback 的 retry 预算处理。不是 normal candidate 的普通文本
不能触发 `ANSWER_REQUIRED`。

### 3.2 主转换表

| 当前状态 | 事件/guard | 新状态 | 必须副作用 |
|---|---|---|---|
| `ELICITING` | `ACTION_PROGRESS`，仍有未完成 field | `ELICITING` | 记录 field/value；清零 no-progress streak |
| `ELICITING` | `ACTION_PROGRESS(..., NO_PREFERENCE)` | `ELICITING` 或 `SEARCH_REQUIRED` | 将 field 标为完成；以后禁止重新措辞询问同一 field |
| `ELICITING` | `ACTION_NO_PROGRESS` 且 streak < threshold | `ELICITING` | 递增 streak；不得重复相同/语义相同问题 |
| `ELICITING` | streak ≥ threshold | `SEARCH_REQUIRED` | `forced_search=true`；查询只能使用已知公开约束，不得编造缺失偏好 |
| `ELICITING` | action 试图询问已 `NO_PREFERENCE` field | `ELICITING` | 不调用环境；稳定错误；计 invalid/non-progress |
| `SEARCH_REQUIRED` | 一次合法 search，normal list | `ANSWER_REQUIRED` | 保存候选 IDs；下一次调用只能 answer |
| `SEARCH_REQUIRED` | 第一次 fallback 或空结果 | `SEARCH_RETRY_REQUIRED` | `search_attempts=1`、`search_fallbacks=1`；要求实质改写 |
| `SEARCH_REQUIRED` | 第二次 fallback 或空结果（不应存在第三次） | `BLOCKED` | 屏蔽当前 aspect；不创建 answer；转 switch |
| `SEARCH_RETRY_REQUIRED` | 合法实质改写 search，normal list | `ANSWER_REQUIRED` | 保存候选 IDs；禁止再 search |
| `SEARCH_RETRY_REQUIRED` | fallback/空结果 | `BLOCKED` | `search_attempts=2`；屏蔽当前 aspect；转 switch |
| `SEARCH_RETRY_REQUIRED` | query 非实质改写 | `SEARCH_RETRY_REQUIRED` | 不调用环境；不消耗真实 retry；按 invalid/non-progress 处理，达到 guard 上限时 block |
| `ANSWER_REQUIRED` | 一个 ID，且 ID 在当前 normal list | `ANSWERED` | 登记可见 answer；hidden correctness 只进 reward，不影响状态 |
| `ANSWER_REQUIRED` | 多 ID、不可见 ID、其他 aspect ID | `ANSWER_REQUIRED` | 不调用环境；稳定错误；不得改候选或 answer |
| `ANSWER_REQUIRED` | action/search | `ANSWER_REQUIRED` | 不调用环境；“已有候选，下一步只能 answer” |
| `ANSWERED`/`BLOCKED` | 任意 tool call | 原状态 | 不调用环境；旧 aspect 永久屏蔽 |
| `SWITCH_ASPECT_REQUIRED` | 有下一个 OPEN aspect | `ELICITING` 或 `SEARCH_REQUIRED` | 按公开 canonical 顺序设置 current aspect |
| `SWITCH_ASPECT_REQUIRED` | 没有 OPEN aspect | episode done | 终止 rollout；不伪造 answer/completion |

`ENVIRONMENT_TERMINATED` 只能在 FSM 已确认所有 aspect `ANSWERED` 时作为正常成功
终止。若环境提前 terminated，或环境结束时仍有 OPEN/BLOCKED aspect，必须记录
protocol/infrastructure mismatch；不得把它改写成 `ANSWERED`。

### 3.3 fallback 的硬上限与实质改写

对每个 aspect 固定：

```text
正常 search       → normal list → ANSWER_REQUIRED
第一次 fallback   → 只允许一次重写 search
第二次 fallback   → BLOCKED，立即切换
```

“实质改写”至少同时满足：

1. `normalized_action_signature` 与上一次 query 不同；
2. 不能只是大小写、空白、标点或词序变化；
3. 保持同一 aspect，并改变/补充一个公开可验证的 query slot（地点、日期、已知
   用户约束或合法的 fallback 修复词）；
4. 不引入 Actor 没有从用户/公开 prompt 得到的偏好值；
5. 不能复用已失败的 exact 或 semantic query signature。

实现可使用 token overlap/edit-ratio 加结构化 slot 检查；阈值必须配置化并用单元
测试固定边界。无法证明是实质改写时，拒绝而不调用环境；绝不能靠无限次近似
query 绕过两次 fallback 上限。

### 3.4 正常候选列表后的强制 answer

一旦 `SEARCH_NORMAL` 发生，controller 立即进入 `ANSWER_REQUIRED`。后续允许的
唯一调用是：

```json
{"choice": "answer", "content": "<one visible option ID>"}
```

ID 必须在当前 search 反馈中出现、属于 current aspect、且该 aspect 尚未回答。
模型选错一个“可见但 hidden-in-correct=false”的 ID 仍是合法 `ANSWERED`；错误性
只能由 reward 统计，不能用错误反馈诱导二次搜索或泄漏 correct ID。

候选列表中有多个 aspect 的 ID 时，只接受 current aspect 的一个 ID；其余 ID 不能
作为“顺手一起回答”的理由。Actor 停止、输出 malformed call 或重新 search 都是
不完整/协议错误，不得隐式替它 answer。

### 3.5 连续无进展与无偏好

**无进展阈值**：只统计连续被接受但没有新 Actor-visible 信息、normal list 或
answer 的 action，以及 malformed/unknown/被 guard 拒绝的协议事件（沿用现有
`record_non_progress` 账本）。达到 `stall_no_progress_threshold` 时必须转
`SEARCH_REQUIRED`，不再允许继续 action。强制 search 使用已知信息；未取得的 field
保持未完成，不由 controller 填值。

**明确无偏好**：只有用户反馈包含明确的“无偏好/都可以/不在意/whatever is fine”
语义时才生成 `NO_PREFERENCE`。沉默、模糊、答非所问不能当作无偏好。确认后：

1. field 记为完成；若没有其他缺失 field，进入 `SEARCH_REQUIRED`；
2. 后续任何针对该 field 的 action（同义、换说法、打包在另一个问题中）都在适配
   层拒绝，不发送到 UserBench；
3. 只有用户主动明确改口时，未来版本才允许产生显式 `PREFERENCE_OVERRIDE`；本版
   不允许 Actor 通过重复提问自行重开 field。

## 4. 终止、completion 与 BLOCKED 语义

### 4.1 所有 aspect 的终止

`SWITCH_ASPECT_REQUIRED` 检查 canonical aspect order 中是否还有 `OPEN`：

- 有：转到下一个 aspect 的 `ELICITING` 或 `SEARCH_REQUIRED`；
- 无：设置 episode done，终止 controller/rollout；不额外生成 tool call；
- 环境仍未同步结束：由 adapter 记录 controlled termination；不得伪造环境 answer。

公开状态机的 `ANSWERED` 仍表示 Actor 提交了一个当前可见 ID；它不是 Reward 的 correctness 判断。

当前 Reward 的 completion 定义为正确答案比例：

```text
answer_submission_rate = count(aspect has submitted answer) / count(all aspects)
completion_rate = correct_answer_rate = count(submitted ID ∈ correct_ids) / count(all aspects)
```

`BLOCKED` 只表示该 aspect 的可行动路径被安全封闭，不计入 completion numerator；正确性仍只用于隐藏 Reward/offline analysis，不反馈给 Actor。
若所有 aspect 都 `BLOCKED`，episode 可以正常结束，但 completion 必须为 0（或相应
的部分值），不能把“已切换/已结束”写成“已回答”。禁止构造不存在的 option ID、
伪造 `answers`、`correct_itinerary` 或 success completion。

### 4.2 协议/基础设施终止

parallel call、无法解析的 tool、wrapper/simulator 异常、缺失 reward evidence
是 protocol/infrastructure 结果，不是 `BLOCKED` 的业务 fallback。它们应沿用现有
fail-loud / `reward_valid=false` 语义；如果当前 aspect 因 search fallback 被 block，
则仍保留 `BLOCKED` 记录和非完整 completion。两类终止的 `termination_reason` 必须
可区分，便于审计。

## 5. Actor 可见信息、隐藏奖励信息与防泄漏

### 5.1 可以进入 Actor transcript 的信息

- system/task prompt 中明确提供的公开任务约束和 aspect 名称；
- UserBench reset/step 返回的 `observation.feedback`；
- 用户明确说出的偏好值、明确无偏好反馈和公开的对话历史；
- normal search 反馈中逐字出现的 option ID/候选描述；
- search fallback 的通用文本（例如后端暂不可用）；
- 工具 schema、Actor 自己过去的调用以及不含隐藏标签的稳定协议错误；
- 通用 recovery 指令，例如“已有候选时下一步只能提交一个可见 option ID”。

### 5.2 只能留在 controller/reward/logger 的信息

- `TravelRewardTask.best_ids`、`correct_ids`、偏好 ID/偏好值；
- `UserBenchRewardSnapshot` 的 `remaining_preference_ids`、
  `remaining_search_aspects`、active/passive 计数和 `choice_initials`；
- `active_preference_ids`、`passive_preference_ids`、wrong/best/correct 判定、
  raw/terminal reward、quality breakdown；
- hidden evidence ledger、内部 fallback 计数和用于审计的 simulator diagnostics；
- 任何“当前 hidden preference 已完整”“这个 ID 是正确答案”的标签。

`TravelRewardTask.preference_fields_by_aspect` 只能作为本地 phase controller 的
字段目录；字段值和隐藏 ID 永不进入 Actor prompt。除非某字段名称已经在公开 system
prompt 中出现，否则不要通过询问顺序或错误消息让 Actor反推出它。

### 5.3 行为侧信道规则

1. phase guard 只能依赖 Actor-visible ledger、公开 query/result 和非标签基础设施
   事件；不得用 hidden correct ID 或 hidden preference completion 选择下一动作。
2. answer guard 只检查格式、current aspect 和“是否曾公开出现”，绝不检查正确性。
3. fallback 的 generic 文案可说明“搜索不可用/换一个查询/切换 aspect”，但不得
   说明 hidden preference、best/correct、reward 或剩余 hidden IDs。
4. 所有拒绝使用固定模板和相同的字段结构；不因 hidden correct/wrong 改变文本长度、
   token 形状、延迟、工具列表或终止方式。
5. reward metrics 与 Actor tool response 分离；当前 agent loop 的
   `extra_fields` 只进入 rollout logger，不回灌 transcript。
6. blocked 的原因只保留可见的 fallback/protocol 类别；禁止通过“有时继续询问、
   有时立即成功”反推隐藏状态。

## 6. 实现伪代码（非实现）

```text
initialize session:
    build public aspect order
    build empty per-aspect visible ledger
    current = first OPEN aspect
    phase = ELICITING if missing_visible_field(current) else SEARCH_REQUIRED

on_actor_call(call):
    if episode_done:
        reject_without_environment("episode already ended")

    guard = contract_guard(phase, current, call, visible_ledger)
    if guard.reject:
        record_invalid_and_no_progress(guard.reason)
        if no_progress_reached_threshold_in_eliciting():
            phase = SEARCH_REQUIRED
            current.forced_search = true
        return stable_actor_visible_error(guard.public_message)

    result = execute_userbench_only_after_guard(call)
    event = classify_visible_result(call, result)
    phase = transition(phase, current, event)
    update_visible_ledger(event)

    if event == SEARCH_FALLBACK or event == SEARCH_EMPTY:
        if current.search_fallbacks == 0:
            current.search_fallbacks = 1
            phase = SEARCH_RETRY_REQUIRED
        else:
            current.terminal = BLOCKED
            phase = SWITCH_ASPECT_REQUIRED

    if event == SEARCH_NORMAL:
        phase = ANSWER_REQUIRED

    if event == ANSWER_VISIBLE:
        current.answer_id = event.option_id
        current.terminal = ANSWERED
        phase = SWITCH_ASPECT_REQUIRED

    if phase == SWITCH_ASPECT_REQUIRED:
        current = next_open_aspect_in_canonical_order()
        if current is None:
            episode_done = true
        else:
            phase = ELICITING if missing_visible_field(current) else SEARCH_REQUIRED

    assert_invariants()
```

`assert_invariants()` 至少检查：

```text
normal_candidate_seen  => next legal call is answer only
search_fallbacks       <= 2 per aspect
search_attempts        <= 2 per aspect
ANSWERED/BLOCKED       => no further call for that aspect
NO_PREFERENCE field    => no repeated elicitation of that field
BLOCKED                => no answer/completion fabrication
all aspects terminal   => no further actor generation
```

## 7. 建议配置（只规划，不修改配置文件）

建议将以下字段归入 `configs/interaction_config/agent_loop.yaml` 下的独立
`state_machine` 节；已有字段保持兼容映射：

| 配置项 | 建议默认值 | 约束/用途 |
|---|---:|---|
| `state_machine.enabled` | `true`（未来正式启用时） | 关闭只能用于离线兼容/诊断；不能绕过工具 schema |
| `state_machine.max_steps` | `20` | 必须与当前 UserBench rollout contract 一致 |
| `state_machine.stall_no_progress_threshold` | `4` | 复用现有 `stall_no_progress_threshold`，整数 ≥ 1 |
| `state_machine.force_search_on_stall` | `true` | 达阈值后不得继续 action |
| `state_machine.search_retry_max` | `1` | 每 aspect 第一次 fallback 后只有一次 retry |
| `state_machine.search_fallback_block_after` | `2` | 第二次 fallback/空结果立即 `BLOCKED` |
| `state_machine.search_retry_requires_rewrite` | `true` | 禁止 exact/semantic/表面改写 |
| `state_machine.search_retry_min_edit_ratio` | `0.20`（待校准） | 与 slot 变化联合判定；需 fixture 固定边界 |
| `state_machine.search_empty_counts_as_fallback` | `true` | 无 ID 且无 fallback 标记也不能无限搜索 |
| `state_machine.answer_requires_normal_candidates` | `true` | 正常候选出现后强制 answer |
| `state_machine.max_answer_ids_per_aspect` | `1` | one-choice contract；逗号多 ID 拒绝 |
| `state_machine.no_preference_marks_field_complete` | `true` | 明确无偏好后不可重问 |
| `state_machine.repeat_no_preference_policy` | `reject` | 在环境执行前拒绝同 field 的换说法 |
| `state_machine.terminate_when_all_aspects_terminal` | `true` | ANSWERED 或 BLOCKED 都是 aspect terminal |
| `state_machine.completion_denominator` | `all_aspects` | BLOCKED 不计入答案完成 |
| `state_machine.expose_internal_state` | `false` | 不把 phase、hidden snapshot、correctness 回灌 Actor |
| `state_machine.fallback_classifier` | `structured_or_marker` | 结构化 diagnostics 优先，可见 marker 兜底 |

training/validation 的 effective 开关仍遵守现有约定：训练可开，validation sampling
默认关闭；不能因为 FSM 启用就把隐藏 recovery 注入 frozen evaluation。

## 8. 测试规划（本步骤不执行）

### 8.1 状态与 guard 单元测试

- 初始缺 field → `ELICITING`；已完成公开 field → `SEARCH_REQUIRED`。
- `ELICITING` 允许一个具体未完成 field 的 action；bundled/vague/已完成 field
  action 在环境执行前拒绝。
- 明确 `NO_PREFERENCE` 完成 field；同 field 的原问、同义问、换说法均拒绝。
- 连续无进展为 `threshold - 1` 时仍可按规则 action；达到阈值后下一状态必须为
  `SEARCH_REQUIRED`，并标记 forced search。
- `SEARCH_REQUIRED` 只接受 search；action/answer 不得调用 wrapper。
- 第一次 search fallback/empty → `SEARCH_RETRY_REQUIRED`；retry 只能一次。
- retry 的 exact、只改标点、只换词序、不同 aspect query 均不是实质改写。
- retry normal list → `ANSWER_REQUIRED`；第二次 fallback/empty → `BLOCKED`。
- normal list 出现后 action/search/multiple ID/invisible ID 均被拒绝。
- 可见但 hidden-wrong 的一个 ID 仍进入 `ANSWERED`；测试输出中不能出现 correct ID。
- `ANSWERED` 与 `BLOCKED` 都禁止再次调用；`BLOCKED` 不写 answer、不增加
  completion numerator。
- 全部 `ANSWERED`、混合 `ANSWERED+BLOCKED`、全部 `BLOCKED` 三种终止组合。
- parallel call、malformed call、unknown tool 与 simulator failure 的
  `termination_reason` 不与业务 `BLOCKED` 混淆。

### 8.2 事件序列/性质测试

用 fake wrapper 和纯事件驱动 controller 生成任意序列，断言以下不变量永远成立：

- 每 aspect `search_attempts <= 2`、`search_fallbacks <= 2`；
- `SEARCH_NORMAL` 后没有合法 action/search 分支；
- `ANSWERED/BLOCKED` 不会回到 OPEN；
- no-preference field 不会再次产生 elicitation；
- episode done 后没有新的 Actor generation/tool call；
- reward hidden labels 改变时，Actor-visible transcript 和 guard 结果不改变（除
  非公开候选/用户反馈本身改变）。

### 8.3 wrapper/adapter 集成测试

- 以现有 `tests/test_userbench_context.py`、`tests/test_grpo_adapter.py` 的 fake
  wrapper 模式补充：normal list、search fallback 1/2、empty result、visible wrong
  answer、controlled all-blocked termination。
- 断言被拒绝调用的 `wrapper.calls == 0`，且恢复文本没有 hidden IDs/snapshot。
- 断言 `extract_visible_option_ids()` 只提取 Actor-visible 正式 ID；fallback 文本
  不会误变成 normal list。
- 断言 `UserBenchSessionState` 的 reward snapshot 缺失仍是 infrastructure invalid，
  不借 FSM 伪造进度。
- 断言 parallel/malformed 行为保留现有协议错误与 no-progress 计数。
- 训练 sampling 可启用 stall recovery，validation sampling 不启用；两者都不会
  改变 FSM 的 hidden-label 可见性。

### 8.4 泄漏与回归测试

- 对所有 tool response、Actor transcript 和 prompt 做字符串/结构扫描，禁止出现
  `correct_ids`、`best_ids`、preference ID/value、reward snapshot、wrong/best 标签。
- 改变 hidden correct ID、hidden preference ID 或 reward 数值，公开候选与用户反馈不
  变时，phase、错误模板和下一合法 action 必须保持一致。
- 正常候选出现后故意提交错误可见 ID，必须 `ANSWERED` 而非再次 search；这验证
  “正确性是 hidden reward，不是 Actor guard”。
- 运行现有轻量 Python 测试与文档/配置静态检查即可；本设计阶段不调用真实
  UserBench、外部模拟器、GPU 或训练脚本。

## 9. 尚未解决的设计风险

1. **normal list/fallback 的信号脆弱。** 当前 wrapper 同时依赖可见 marker 和可选
   diagnostics；上游文本变化可能导致误判。优先增加结构化结果类型，但不把 hidden
   snapshot 暴露给 Actor。
2. **公开 phase 与 reward snapshot 可能不一致。** 当前 `record_step()` 用隐藏
   snapshot 计算 evidence ledger，而 FSM 需要公开 ledger；实现必须定义 mismatch 的
   fail-closed 行为，不能让隐藏状态悄悄改写 Actor 行为。
3. **强制 search 的质量折衷。** no-progress 达阈值时可能仍缺 field；强制 search
   会减少死循环，却可能得到 vague/fallback。要用离线 fixture 校准 threshold 和
   empty-result 策略，不能通过编造偏好解决。
4. **“实质改写”需要稳定判定。** token edit ratio、slot 变化和 semantic signature
   的组合尚未在现有工具层定义；过松会重复搜索，过严会错误 block。
5. **无偏好自然语言识别。** “随便”“都行”“你决定”等语义依赖 simulator 文本；
   需明确词表/分类器边界，并把模糊答复与明确无偏好区分开。
6. **环境终止与 BLOCKED 的接口差异。** pinned one-choice TravelEnv 主要以所有
   choice 终止，没有业务上的 blocked action；future adapter 需要 controlled
   termination，而不应修改第三方环境来伪造 blocked answer。
7. **候选与 aspect 归属。** 一段反馈可能含多个 aspect 的 option ID；需要按当前
   search 请求和 option 前缀做可审计归属，不能把别的 aspect 的 ID 当作当前答案。
8. **现有一次性 stall recovery 的兼容。** 新 FSM 的 per-aspect retry 与旧的
   answer-only recovery 可能竞争；实现时要定义优先级，避免 recovery 绕过
   `ANSWER_REQUIRED` 或重置 retry budget。
9. **配置漂移风险。** validation sampling 判定和 `max_steps=20` 是现有硬约束；若
   将来 profile 改动，必须 fail loudly，而不是静默应用训练 recovery。

## 10. 实施入口（后续步骤）

本设计完成后，后续实现应按以下顺序进行，但本次不执行：

1. 新增纯 Python controller（建议置于 `src/travel_grpo/agent/` 或
   `src/travel_grpo/training/grpo/adapter/`），使 transition/guard 可离线单测；
2. 在 `UserBenchSessionState` 保存 controller 的公开 ledger，保留 reward ledger
   只供 scorer/audit；
3. 在 `execute_userbench_action()` 调用 wrapper 前接入 guard，在返回后分类事件并
   更新 phase；
4. 让 `UserBenchAgentLoop` 的 termination 只读取 controller 的 episode-done 与现有
   environment/protocol 终止；
5. 补齐第 8 节测试后，才考虑把配置接入训练入口；validation/evaluation 需单独回归。

## 8. 步骤 5 实现备注

有限恢复 guard 已在项目自有 adapter 中落地：

- public_control.py 的 validate_public_action 在环境执行前执行公开 phase guard；
  reducer 只接收公开初始文本、Actor action 和 Actor 可见反馈。
- SEARCH_REQUIRED、ANSWER_REQUIRED、一次性 SEARCH_RETRY_REQUIRED 和
  SWITCH_ASPECT_REQUIRED 均 fail-closed；第二次 fallback 将 aspect 标为
  BLOCKED，但不会构造答案或成功 completion。
- NONE/ELICITING、BLOCKED/SWITCH_ASPECT_REQUIRED 保留身份兼容别名；
  PublicControlState.phase 是新 guard 的 canonical phase，旧
  recovery_mode 字段继续可读。
- UserBenchSessionState 从 reset 的公开反馈初始化 ledger；缺少公开初始文本的
  旧/offline Session 继续使用原 answer-only/stall 兼容路径。
