# Teacher 轨迹采集优化：会话交接总结

> 更新时间：2026-08-04（Asia/Shanghai）  
> 项目目录：`D:\Code\Python\vscodeProjects\travel-grpo-longhorizon`  
> 当前结论：开发批次已经达到 2/3，但独立 confirmation 批次尚未达到 2/3；目标暂未完成。

## 1. 我们正在做什么

这是一个基于 UserBench 的旅游助手 Agentic 后训练项目。当前阶段不是训练 Actor，而是采集可用于 action-only SFT 的 Teacher 轨迹：

- Teacher 模型：通过 OpenAI-compatible API 调用 `deepseek-v4-flash`。
- UserBench 用户模拟器：同样调用 `deepseek-v4-flash`。
- 后续 GRPO Actor：计划使用 Qwen3.5-2B，但不属于当前采集阶段。
- 数据范围：只使用 UserBench 官方 `SFT train` 任务；官方 test 集冻结，不参与教师调用、SFT、GRPO 或调参。
- 当前成功标准：在一批从未用于调参的 3 个任务中，至少 2 条轨迹满足严格 gold 接纳条件。

目标循环是：

1. 固定同一批 development 任务定位问题。
2. 每轮只修复最主要的一类失败原因。
3. 先跑离线测试，再真实采集 3 条轨迹。
4. development 达到 2/3 后，使用全新的 confirmation 任务验证泛化。
5. 只有独立 confirmation 批次达到 2/3，才可宣称本阶段完成。

## 2. 已完成的基础工作

### 数据和第三方快照

- UserBench 完整内嵌于 `environments/UserBench/`，固定提交：
  `80506d2ab484cab843e60a2401ff3e0290d05b87`。
- 保留上游 Apache-2.0 许可证和 `EMBEDDED_SOURCE.json`。
- UserBench one-choice 数据共 3,122 条，已完成固定互斥划分：
  - SFT train：716
  - SFT validation：80
  - GRPO train：1,723
  - GRPO validation：132
  - Final evaluation：471
- 快照目录没有嵌套 `.git`，数据 verifier 和 split 交集检查已通过。

### 环境和工具包装

`src/travel_grpo/envs/` 已提供 UserBench 包装、action contract、ContextVar session 隔离、原始奖励 ledger 和 veRL 单工具适配。模型只能看到公开 `feedback`，不能看到：

- 隐藏偏好值或隐藏偏好 ID；
- `ground_truth`、`best_id`、reward model；
- 未通过真实搜索公开的候选 ID。

工具协议固定为：

```text
interact_with_env(thought, choice, content)
choice ∈ {search, action, answer}
```

每轮只允许一个工具调用；`search`、`action`、`answer` 的字段和前缀必须一致。工具 reward 固定为 0，最终 interaction score 使用 UserBench 原始逐步奖励之和，避免重复计奖。

### Teacher 状态机和轨迹过滤

当前 Teacher 策略版本是 `teacher-state-machine-v4`：

- 阶段固定为 `ELICIT -> SEARCH -> ANSWER`；
- 本地控制器只把当前 aspect、field、允许的 choice 和公开对话状态传给 Teacher；
- answer 只能从此前搜索反馈中实际出现的 option ID 中选择；
- 采用 natural → strict → canonical 的请求级重试；
- 检查重复 action、语义重复、bundled action、错误 field/phase 和多工具调用；
- 普通文本必须为空，只保留工具调用，适配 action-only SFT；
- 支持整轨迹重试、诊断分流、单次 `search_not_recorded` 修复；被修复的 search turn 标记 `loss_mask=true`，不会作为 SFT 监督标签，但仍保留在上下文中。

## 3. 真实采集结果

所有结果均为真实 API 调用，使用 `concurrency=1`、每个任务最多 3 次整轨迹尝试。旧产物不能与当前策略版本混用。

| 批次 | 任务数 | 接纳 | 拒绝 | 主要结论 |
|---|---:|---:|---:|---|
| `smoke10_v4_deepseek_v4_flash`（旧版） | 10 | 0 | 10 | 旧策略大量协议错误、重复和 vague action，不能作为当前版本基线 |
| `state-machine-v2-smoke-20260804-1` | 1 | 0 | 1 | 状态机已能走到搜索/回答，但出现模拟器 fallback、vague 和 search 未记录 |
| `goal-dev-round1` | 3 | 0 | 3 | 主要是字段覆盖和模拟器 fallback |
| `goal-dev-round2` | 3 | 0 | 3 | 仍有 vague、search 未记录和 coverage unreachable |
| `goal-dev-round3` | 3 | 0 | 3 | coverage 与 simulator search fallback 仍明显 |
| `goal-dev-round4` | 3 | 1 | 2 | 首次达到 1/3；出现 search 未记录和少数 coverage 问题 |
| `goal-dev-round5` | 3 | **2** | 1 | development 达到 2/3；说明字段感知、搜索修复和 loss mask 有效 |
| `goal-confirmation1` | 3 | 0 | 3 | 新任务泛化失败，主要为 vague、coverage 和 simulator judgment fallback |
| `goal-confirmation2` | 3 | 1 | 2 | rental_car + restaurant 任务成功 1 条 |
| `goal-confirmation3` | 3 | 1 | 2 | apartment + rental_car 任务成功 1 条；另有 duplicate action exhaustion |
| `goal-confirmation4` | 3 | 0 | 3 | 出现一次明确 wrong answer，另有 flight/restaurant coverage 和 fallback |

### `goal-dev-round5` 的两个成功样本

1. `apartment:2-38|rental_car:2-7`
   - 9 步，terminal reward 约 0.9917，`correct=true`，policy penalty 为 0。
   - 包含 1 个被 `loss_mask` 的 search repair turn。
2. `apartment:2-10|rental_car:2-23`
   - 8 步，terminal reward 1.0，`correct=true`，policy penalty 为 0。

这两个样本证明当前实现可以生成合格轨迹，但不能证明对未参与调参的任务稳定泛化。

### `goal-confirmation4` 的关键失败

- `flight:2-14|apartment:2-43`：某次尝试已经得到有效 apartment 搜索结果，却选择了错误的 `A8`；这是硬性 wrong-answer，不能通过放宽阈值解决。其他尝试在开始 flight company 偏好时触发 judgment fallback。
- `flight:2-97|rental_car:2-19`：隐藏字段包含 rental model、insurance damage waiver、flight carry-on allowance、business class；现有询问模板对 carry-on 等字段覆盖不足。
- `flight:2-39|restaurant:2-49`：隐藏字段包含 restaurant vegetarian cuisine、delivery tag、flight shortest travel time、specific airline；restaurant tags 模板目前缺少 delivery。

## 4. 当前代码改动和未提交状态

当前工作树有未提交修改，且没有生成提交：

```text
M scripts/train/sft/collect_sft_data.py
M src/travel_grpo/envs/reward.py
M src/travel_grpo/envs/userbench_tools.py
M src/travel_grpo/training/sft_collection.py
M src/travel_grpo/training/sft_dataset.py
M src/travel_grpo/training/teacher_policy.py
M tests/test_sft_collection.py
M tests/test_sft_dataset.py
M tests/test_userbench_tools.py
?? configs/train/sft/teacher_smoke_batches.json
?? docs/training/teacher-trajectory-optimization-handoff.md
```

`environments/UserBench/` 没有改动。不要使用 `git reset --hard`、`git checkout --` 或覆盖这些修改。

固定批次文件为 `configs/train/sft/teacher_smoke_batches.json`，当前包含 `development`、`confirmation_1` 到 `confirmation_4`。下一轮需要增加新的、从未使用过的 `confirmation_5` 任务，不能复用已经跑过的 task ID。

## 5. 当前卡点

### 5.1 独立 confirmation 尚未达到 2/3

目前最好结果是 `confirmation_2` 和 `confirmation_3` 各 1/3，`confirmation_4` 为 0/3。因此最终目标仍未完成。不能用 development 的 2/3 代替 confirmation 的 2/3。

### 5.2 字段询问模板仍不完整

已经通过 field-aware `preference_fields_by_aspect` 改善了多数 coverage，但 UserBench 的具体字段很多，当前仍有：

- flight amenities 没有明确覆盖 `carry-on baggage allowance`；
- restaurant tags 没有明确覆盖 `delivery`；
- 某些 service 字段会被 Teacher 说成泛化的“你偏好哪些服务”，触发 vague action。

### 5.3 answer 选择仍可能错误

Teacher 有时看到多个公开候选后，未完整比较所有公开偏好，直接按顺序或局部匹配选 ID。应加强“满足全部已披露条件；预算有限时在满足条件的候选中选最便宜者；不得依据 ID 或候选顺序”的可见信息指令，但绝不能把 ground truth 或隐藏 best ID 传给 Teacher。

### 5.4 UserBench 模拟器 fallback

`simulator.judgment_fallback`、`simulator.search_fallback` 和 response fallback 不是同一种问题：

- Teacher 协议错误应修 Teacher；
- 模拟器 fallback 应记录为基础设施诊断；
- fallback 不能直接伪装为 gold 成功；
- 若后续允许 recoverable/silver 轨迹，必须单独统计，不能混入严格 gold。

### 5.5 成本和长上下文

一条任务的多次整轨迹重试会重复发送长搜索反馈，token 成本很高。优化上下文前必须确认不会删掉 answer 所需的公开候选信息，也不能把隐藏信息压缩进 prompt。

## 6. 下一步执行顺序

### 第一步：完成一轮小范围模板修复

建议只修改以下内容，然后先跑离线测试：

1. `flight.amenities` canonical 问题加入 `carry-on baggage allowance`。
2. `FIELD_QUERY_HINTS.flight.amenities` 加入同一短语，并避免与 `flight.service` 的 `carry-on` 发生错误匹配。
3. `restaurant.tags` canonical 问题加入 `delivery`，字段 hint 也加入 `delivery`。
4. generic vague action 检查把 `service` 也纳入，拒绝泛化的服务问题。
5. answer instruction 加入“检查所有公开偏好、只从满足全部偏好的可见候选中选择、预算有限选最便宜、不得依据 ID/顺序”。

这些改动仍然不允许传入隐藏偏好值、正确答案或 reward evidence。

### 第二步：离线验证

```powershell
conda run -n travel_grpo python -m pytest tests/test_sft_collection.py tests/test_sft_dataset.py tests/test_reward.py tests/test_userbench_tools.py -q
conda run -n travel_grpo python -m compileall -q src scripts tests
conda run -n travel_grpo python scripts/train/sft/collect_sft_data.py --batch confirmation_5 --dry-run
```

### 第三步：新建 confirmation_5 并真实采集 3 条

`confirmation_5` 必须从 `data/sft/tasks_train.jsonl` 选择 3 个从未在 development 或 confirmation_1~4 使用过的 task ID。使用新 run 目录，不能覆盖旧产物：

```powershell
conda run -n travel_grpo python scripts/train/sft/collect_sft_data.py `
  --batch confirmation_5 `
  --concurrency 1 `
  --attempts 3 `
  --run-dir outputs/teacher_trajectories/runs/goal-confirmation5 `
  --output outputs/teacher_trajectories/goal-confirmation5.accepted.jsonl `
  --rejected-output outputs/teacher_trajectories/goal-confirmation5.rejected.jsonl `
  --diagnostics-output outputs/teacher_trajectories/goal-confirmation5.diagnostics.jsonl
```

### 第四步：按结果决策

- confirmation_5 达到 2/3：停止继续放宽，做完整回归、数据审计和人工检查。
- confirmation_5 低于 2/3：只根据诊断中占比最高且可泛化的一类原因再改一轮，并使用另一个全新 3-task confirmation 批次。
- wrong answer、unsearched answer、不可见 option ID、多工具调用、未完成 itinerary、truncated、隐藏信息泄漏都不能通过降低阈值解决。
- simulator fallback 如需放宽，只能引入单独的 silver/infra 分类，不能把它们算作严格 gold。

## 7. 完成前验收清单

```powershell
conda run -n travel_grpo python -m pytest -q
conda run -n travel_grpo python -m compileall -q src scripts tests
conda run -n travel_grpo python scripts/data/build_dataset_splits.py --verify-only
git diff --check
git status --short -- environments/UserBench
```

还需人工审计最终 accepted 轨迹：

- 每个 assistant turn 只有一个合法工具调用，无普通文本；
- 每个 action 只处理一个 aspect/field，问题具体而非 vague；
- 每个 search 明确且被环境记录；
- 每个 answer ID 来自此前对应 search 的公开候选；
- 所有 aspect 都完成，环境正常 `terminated` 而非 `truncated`；
- `correct_itinerary=true`、无 wrong/unsearched answer、policy penalty 为 0；
- reward breakdown 与原始逐步 reward 一致；
- Teacher 消息和结构化状态没有隐藏偏好、ground truth、best ID 或 reward evidence；
- accepted、rejected、diagnostics 三路产物可追溯，旧策略版本不混入新统计。

## 8. 整个对话中最容易踩的坑

1. **把 development 成功当成最终成功。** development 只用于调参；必须有全新 confirmation 2/3。
2. **为了提高接纳率放宽 wrong answer。** 这是数据污染，不能接受。
3. **把 UserBench fallback 当成 Teacher 质量问题。** 必须区分协议、模拟器、网络和工具执行层诊断。
4. **把隐藏状态传给 Teacher。** 本地 controller 可以用字段名控制流程，但不能发送偏好值、ID、ground truth 或 best option。
5. **用静态模糊模板询问所有字段。** UserBench 的搜索判定要求 query 覆盖公开地点、日期和具体偏好；泛化 query 会被判为 vague 或 search fallback。
6. **修改失败任务或重复使用任务。** 固定 batch 用于可比实验；失败任务不能被替换，confirmation 必须使用全新 task ID。
7. **修复 search 后删除历史 turn。** 环境已经消费了错误 turn，应该保留并 `loss_mask`，不能伪造连续轨迹或重写 reward。
8. **混用旧策略产物。** 检查 `schema_version`、`policy_version`、reward 版本和运行目录；旧版 accepted 不能当作当前证据。
9. **在 Python 3.10 使用 `tomllib`。** `tomllib` 是 Python 3.11 标准库；当前脚本配置采用 JSON，除非显式增加兼容依赖，不要重新引入。
10. **从仓库根目录运行脚本导致 `ModuleNotFoundError: travel_grpo`。** 使用 conda 环境并确保 `src` 可导入；不要为了临时运行破坏正式安装边界。
11. **覆盖已有 run 或用 `--force`。** 中断后优先使用完全相同参数加 `--resume`；新实验使用新目录。
12. **一次性扩大到 10 条真实采集。** 先 fake/offline，再 1 条 smoke，再固定 3 条 development，再做独立 confirmation，可显著降低 API 成本和诊断噪声。
13. **为了压缩 token 丢失公开候选。** 可压缩重复上下文，但不能丢掉 answer 所需的公开搜索结果。
14. **把 loss-masked repair turn 当成正常 gold action。** 它可以保留在轨迹上下文，但不能作为 SFT 监督标签；gold/silver/infra 统计要分开。
15. **误改 `environments/UserBench/`。** 该目录是固定第三方快照；升级只能替换完整快照并同步更新来源元数据。

## 9. 当前交接结论

下一位会话不需要重新设计项目。应从“模板修复 → 离线测试 → 新鲜 confirmation_5 真实 3-task 采集”继续。当前最重要的未决问题是：提高不同 aspect/field 的具体询问覆盖率，并降低公开候选上的错误 answer；同时保持严格 gold 过滤和隐藏信息隔离。只有独立 confirmation 批次达到至少 2/3，并完成上述回归与人工审计，才能把目标标记为完成。
