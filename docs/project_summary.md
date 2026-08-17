# Travel-GRPO Long-Horizon 项目总结

> 本文是当前仓库的阶段性技术总结，结构参考了[yyhdbl.github.io 的长程 Agent 项目复盘](https://yyhdbl.github.io/)，但内容、指标和结论均以本项目的代码、配置和本地实验产物为准。
>
> 状态记录：2026-08-14。当前 `20→200` 的 turn-credit GRPO continuation 仍在运行；当时观测到约 `66/200` step，已有 step-50 checkpoint，因此本文不把该 continuation 当作最终结果。

## 摘要

本项目研究一个面向 Travel UserBench 的长程旅行规划 Agent，目标不是只让模型生成更长的对话，而是让它在多轮偏好收集、搜索、候选筛选和最终回答之间保持可验证的协议行为：

```text
用户任务
  → 偏好收集（action）
  → 参数完整后搜索（search）
  → 候选列表中的单个选项回答（answer）
  → 公开状态终止或有限恢复
```

项目的核心经验是：长程 Agent 的瓶颈同时存在于环境契约、控制状态、训练数据和 reward 四个层面。单纯增加 GRPO step 并不能替代这些基础设施；必须先把“模型能看见什么、何时允许什么动作、如何判定完成”固定下来，再解释训练曲线。

目前已完成的主要工程包括：

- 固定的 task split 和可审计的 Travel UserBench 数据管线；
- Teacher 轨迹的 Gold/Silver 质量分层与 action-only SFT；
- 独立的 Public Control State，隔离 hidden reward state；
- 统一的 `ACTOR_RUNTIME_POLICY`，供 SFT、GRPO 和评测复用；
- 有限恢复状态机、工具调用前 guard 和公开 control feedback；
- completion 优先的 Reward v3，以及可选的 trainer-only turn-level credit；
- baseline、SFT-merged 和 GRPO checkpoint 的 200-task 探索性评测。

当前最重要的未决问题是：SFT 已明显改善协议完成率，但模型仍会在候选结果后的 answer、fallback 重试和 aspect 切换上犯错；现有 GRPO 结果还受到 reward 版本和 harness 版本混杂的影响，尚不足以宣称 turn-credit 或某个 checkpoint 已经带来稳定的最终收益。

## 1. 问题背景与目标

### 1.1 为什么是长程旅行 Agent

旅行任务天然具有多阶段结构：用户先给出目的地、日期或预算等任务参数，再逐步表达交通、住宿、活动等偏好；系统需要在合适的时机搜索，读取候选列表，最后从候选中选出一个可验证的 option ID。任意一步出错，都可能使后续状态不可恢复。

因此本项目优化的不是单一 token-level likelihood，而是以下组合目标：

1. **completion 增加**：流程能在有限步数内正常终止；
2. **偏好覆盖率增加**：用户已经明确或明确拒绝的偏好被正确处理；
3. **guard 错误减少**：少做违反公开状态契约的 action/search/answer；
4. **答案质量和效率改善**：答案来自可见候选，重复和无效动作更少。

### 1.2 项目边界

项目把运行时拆成三个边界：

- **Actor**：只接收公开对话、工具反馈和公开控制提示；
- **训练 User Simulator**：为 SFT/GRPO 提供交互反馈；
- **评测 User Simulator**：用于冻结任务集上的独立评测。

`environments/UserBench/` 是 pinned 第三方快照；本项目自己的 adapter、状态机和评测 harness 位于 `src/` 与 `scripts/`，避免通过读取 hidden reward 直接替 Actor 决策。

## 2. 数据、环境与任务切分

### 2.1 Travel UserBench 任务

每个任务包含公开的旅行组成（composition）和任务参数。运行时重点观察以下公开事件：

- 用户明确给出或拒绝偏好；
- Actor 的 `action`、`search`、`answer` 调用；
- simulator 返回的正常候选列表或 fallback 文本；
- 候选列表中实际出现的 option ID；
- Actor 已提交的答案。

hidden preference IDs/values、`correct_ids`、`best_ids`、reward snapshot 等只用于 reward 和离线分析，不得进入 Actor 的控制决策或 prompt。

### 2.2 冻结 split

split manifest 为 `travel-dataset-split-manifest-v2`，共 3,122 个任务。任务级切分如下：

| split | 数量 | 用途 |
|---|---:|---|
| SFT train | 716 | Teacher 轨迹和 SFT |
| SFT validation | 80 | SFT 选择/诊断 |
| GRPO train | 1,723 | 在线 rollout 和 policy optimization |
| GRPO validation | 132 | GRPO 过程验证 |
| evaluation | 471 | 冻结测试，不进入训练 |

所有项目 split 交集均为 0，上游 train/test 交集也为 0。完整 manifest 位于 [`data/split_manifest.json`](../data/split_manifest.json)。

### 2.3 环境契约的关键修正

早期轨迹暴露出一个重要的契约不匹配：当前 UserBench 更稳定地支持“基础任务参数 + 已收集的相关偏好”进行 `search`，而不是把完整交互内容混合成另一种 query 形态。这种差异会被 simulator 表现为 fallback，随后又被误判为模型搜索能力不足。

因此，项目将搜索契约抽象为 `base_args_plus_preferences_v1`，并在 replay/evaluation 路径中单独检查 query 是否包含基础任务参数和当前公开偏好。该诊断路径不读取 hidden correct answer，也不改变 pinned UserBench 快照。

## 3. 端到端技术路线

```text
冻结 task split
      │
      ├── Teacher / User Simulator 采集 accepted 轨迹
      │       └── Gold / Silver 质量筛选
      │
      ├── action-only LoRA SFT
      │       └── merge → outputs/models/sft-merged
      │
      ├── Public Control State + Actor Runtime Policy
      │       └── Agent loop / tools / eval 共用
      │
      ├── veRL 0.8 online GRPO
      │       └── Reward v3，turn-credit 可选
      │
      └── 固定任务评测
              ├── completion / correct itinerary
              ├── preference coverage
              ├── guard / invalid / repeat
              └── 轨迹级错误分析
```

推荐阅读入口：

- 状态机契约：[`docs/codex/agent_state_machine_behavior_contract.md`](codex/agent_state_machine_behavior_contract.md)
- SFT 方案：[`docs/training/sft.md`](training/sft.md)
- GRPO 方案：[`docs/training/grpo.md`](training/grpo.md)
- Reward v3：[`docs/reward/design-v3-priority.md`](reward/design-v3-priority.md)
- Recovery/turn-credit：[`docs/reward/turn-credit-v1.md`](reward/turn-credit-v1.md)

## 4. Eval-first：先固定可观测指标

### 4.1 指标定义

本项目同时报告两种容易混淆的“完成”：

- **public completion**：所有公开 aspect 已 `ANSWERED` 或 `BLOCKED`，并且状态机正常终止；
- **correct itinerary**：答案满足 hidden reward 中的正确性判定。

前者是运行协议指标，后者才是任务答案质量。`BLOCKED` 可以让公开状态终止，但不能伪造成功答案，因此 public completion 不等于 correct itinerary。

常用诊断指标：

| 类别 | 指标 |
|---|---|
| 过程 | public completion、环境步数、actor turn、max-step/turn-limit 终止 |
| 偏好 | active/passive preference coverage |
| 工具 | search、answer、guard rejection、invalid action |
| 质量 | correct itinerary、gold itinerary、answer ID 可见性 |
| 循环 | exact/semantic repeat、错误 aspect、query 重复 |
| 训练 | reward validity、有效 group、turn evidence 分布 |

### 4.2 Public Control State

Actor-facing 状态机包含：

`ELICITING` → `SEARCH_REQUIRED` → `ANSWER_REQUIRED` → `ANSWERED`，以及有限恢复分支 `SEARCH_RETRY_REQUIRED`、`SWITCH_ASPECT_REQUIRED`、`BLOCKED`。

关键契约：

- 偏好已经回答或明确无偏好后，不得换一种说法重复询问；
- 连续无进展 action 达阈值后强制进入 `SEARCH_REQUIRED`，禁止继续 action；
- 正常候选列表出现后进入 `ANSWER_REQUIRED`，只接受一个当前可见 option ID；
- 第一次 fallback 只允许一次实质改写后的 search；
- 第二次 fallback 屏蔽当前 aspect，进入 `SWITCH_ASPECT_REQUIRED`；
- `BLOCKED` 不计入 `ANSWERED`，但在所有 aspect 已 answered/blocked 后允许公开终止；
- 非法动作在调用 simulator 前被拒绝，并返回可恢复的公开反馈。

状态 reducer 只读取 Actor 可见信息，见 [`src/travel_grpo/envs/public_control.py`](../src/travel_grpo/envs/public_control.py)。

### 4.3 Prompt parity

SFT、GRPO train/validation 和离线 eval 共用：

- `ACTOR_RUNTIME_POLICY`（当前版本 `actor-runtime-v2`）；
- 同一套公开 control feedback；
- 同一套 guard 语义。

Teacher 专用的 `TEACHER_GENERATION_INSTRUCTION` 不写入 Actor 训练上下文。Agent loop 会深拷贝 raw prompt、幂等注入 policy，并只记录 policy version，不记录 hidden state。

## 5. Teacher 轨迹与 SFT

### 5.1 Teacher 采集

Teacher 使用独立 instruction 和 User Simulator 生成完整轨迹，再按公开状态机和质量规则筛选。轨迹分为：

- **Gold**：协议、偏好覆盖、搜索和答案均满足要求；
- **Silver**：可用于恢复/边界 curriculum，但存在可解释的局部缺陷；
- **Rejected/quarantine**：无法安全确定下一工具调用，不能猜测 target。

当前已落地的阶段性产物：

| 文件 | 记录数 | 说明 |
|---|---:|---|
| `sft_train.accepted.jsonl` | 46 | accepted 训练轨迹 |
| `sft_train.silver.jsonl` | 41 | silver 训练轨迹 |
| `sft_validation.from_train.accepted.jsonl` | 5 | 训练来源的 validation accepted |
| `sft_validation.from_train.silver.jsonl` | 5 | 训练来源的 validation silver |

`outputs/teacher_trajectories/sft_audit.json` 显示当前候选 97 条、eligible 97 条，Gold 51、Silver 46；正式 400 条目标仍未完成，`formal_sft_ready=false`。因此这些数字代表当前 curriculum/artifact 状态，不代表完整数据集已构建完毕。

### 5.2 SFT 渲染

Recovery-boundary-v1 的每条样本由以下部分构成：

1. production system policy；
2. 截断的公开历史；
3. 与运行时一致的 public control note；
4. 一个目标 assistant tool call。

loss 只作用于最后一个 assistant tool call，避免模型通过复述历史或隐藏字段获得训练捷径。样本级 split 先按 `task_id` 决定，再抽取边界上下文，防止 task 泄漏。

### 5.3 SFT 模型

当前已有 LoRA stage1/stage2 checkpoint，并生成 merged 模型：

```text
outputs/models/sft-merged
```

SFT 的主要作用是学习工具协议、阶段边界和“候选后立即 answer”的格式约束；它不等价于已经学会所有隐藏正确答案判定。

## 6. Reward 设计

### 6.1 Reward v3：completion 优先

当前配置为 `userbench-travel-reward-v3-priority`，terminal-only，输出范围 `[-1, 1]`：

```text
raw = 3.00*C + 0.20*P + 0.08*T + 0.06*S + 0.04*Q + 0.02*E
      - bounded_penalty

terminal_reward = clip(raw / 3.4, -1, 1)
```

其中：

- `C`：completion/correctness 主项；
- `P`：active/passive preference coverage；
- `T`：公开阶段转换质量；
- `S`：搜索覆盖；
- `Q`：answer quality；
- `E`：效率。

guard rejection、blocked、invalid action、repeat、wrong answer 等惩罚都有上限，避免长轨迹中的同一种错误淹没 completion 信号。`BLOCKED` 只表示受控失败，不给 completion 成功加分。

### 6.2 Turn-level credit

turn-credit 是 trainer-only 的可选 advantage 重整，不改变环境 reward：

- `off`：标准 GRPO；
- `shadow`：计算并记录 turn/span evidence，不改变训练；
- `train`：在标准 GRPO advantage 后施加有界 `0.90x–1.10x` 的乘数。

证据主要来自公开事件链：偏好收集链、成功 search、正确 answer，以及 fallback、重复和非法工具调用。当前设计刻意限制 credit 幅度，避免 answer 这一末端动作获得全部 credit，同时保留真正决定后续可行性的 preference→search 链。

### 6.3 隐藏状态隔离

Reward 可以使用 `correct_ids`、`best_ids`、reward snapshot 等隐藏字段，但这些字段不能被 `PublicControlState`、guard、prompt 或 Actor feedback 读取。所有 leakage assertions 检查以下字符串不会出现在 Actor 可见输出中：

```text
remaining_preference_ids / correct_ids / best_ids
reward_snapshot / reward delta / hidden preference values
```

## 7. GRPO 训练配置

GRPO 基于 veRL 0.8 的 `UserBenchAgentLoop`，核心设置为：

- rollout group size `n=4`；
- vLLM async rollout，temperature `0.7`，top-p `0.9`；
- LoRA rank `16`、alpha `32`、学习率 `1e-6`；
- `max_assistant_turns=20`，最大上下文/response 受配置限制；
- dynamic group filtering，要求足够的有效 group 才更新；
- `no-stall-recovery` 可用于测量模型本身的恢复能力；
- turn-credit 默认关闭，可显式设置为 shadow/train。

当前配置关闭了 KL reward 和 KL loss：

```yaml
algorithm.kl_in_reward: false
actor.use_kl_loss: false
```

因此策略偏移主要由 SFT 初始化、较小学习率、LoRA 参数子空间、group-relative advantage、动态有效样本过滤和评测 guard 间接控制。关闭 KL 使 completion 信号不被 reference penalty 稀释，但也减少了对策略漂移的直接约束；如需加 KL，应使用匹配的 reference、系数扫描和同一 reward/harness 版本做 A/B，不能只看单次 reward 均值。

## 8. 已有实验结果

### 8.1 200-task public-guarded 探索性评测

下表的 `completion` 是 public completion，不是正确答案率。GRPO step-100 产物来自历史 Reward v2 条件，不能与当前 Reward v3 直接做因果比较；表格仅用于定位行为问题。

| 模型/条件 | public completion | correct itinerary | guard rejects/task | exact repeat | valid ratio | terminal reward |
|---|---:|---:|---:|---:|---:|---:|
| raw baseline + public guard | 0.1967 | 0.000 | 11.04 | 2.28 | 0.980 | -0.850 |
| SFT-merged + public guard | **0.5158** | 0.070 | **1.62** | 4.00 | **0.995** | -0.595 |
| GRPO step-100 + public guard | 0.4479 | 0.050 | 1.42 | 4.61 | 1.000 | -0.651 |

可观察到：

1. SFT 显著减少 guard rejection，并将终止从大量 actor-turn-limit/max-step 转向 public-control completion；
2. SFT 的正确 itinerary 仍只有 14/200，说明“遵守协议”和“选对答案”是两个不同问题；
3. GRPO step-100 的 guard 数进一步下降，但 completion、correct itinerary 和 repeat 并未显示出相对 SFT 的稳定优势；
4. 由于 GRPO 这次评测使用旧 Reward v2/历史运行条件，不能据此判定 Reward v3 或 turn-credit 的独立效果。

旧的无 public guard raw baseline 还出现过 `completion≈0.049`、`max_steps=184/200`、`exact repeat≈14.665`；它只能说明 harness/guard 会显著改变可观测行为，不能当作同条件基线。

### 8.2 8-task 边界探针

在 SFT-merged 上直接截断上下文、只生成下一工具调用的结果：

| 场景 | 结果 |
|---|---:|
| 正常候选列表后 answer@1 | 1.000 |
| answer 只有一个可见 option ID | 1.000 |
| ID 存在于候选列表 | 1.000 |
| ID 为 correct option | 0.875 |
| 偏好收集完成后 search@1（Teacher policy prompt） | 0.875 |
| 第一次 fallback 后 query 实质改写 | 0.667 |
| 第一次 fallback 原 query 重复 | 0.333 |
| 第二次 fallback 后仍搜旧 aspect | 0.0417 |
| 第二次 fallback 后切换 aspect | 0.583 |

这组探针说明：正常 search→answer 协议已经相对清晰，偏好完整识别也有改善；主要缺口集中在闭环 recovery 和模型是否真正执行控制提示。

### 8.3 20-step turn-credit 诊断

no-credit 和 turn-credit 的 20-step validation 都没有产生 answer completion。turn-credit 版本的公开事件/span 对齐检查为 `8/8`，但 evidence 以负向事件为主，不能把该结果解释为 credit 设计已经有效。两次运行使用不同 reward 版本和统计字段，reward 均值不可直接比较。

正在继续的 `20→200` run 应在 step 50/100/150/200 使用相同 validation 配置重新评测后，才能回答“turn-credit 是否改善 completion、偏好覆盖和 guard”的问题。

## 9. 主要结论与问题定位

### 9.1 已经验证的部分

- **共享 policy 和公开状态机是必要的**：SFT、GRPO、validation 使用同一 runtime policy 后，prompt parity 和 leakage 检查可以自动化；
- **guard 不能代替模型能力，但能提供可解释边界**：它将错误拆成“只允许 search/answer”“不可见 ID”“错误 aspect”“重复 query”等具体原因；
- **SFT 对协议完成有明显帮助**：相对 raw baseline，SFT-merged 的 public completion 和 valid ratio 更高，guard rejection 大幅下降；
- **正常候选后的 answer 不是当前最主要瓶颈**：离线 boundary probe 的 answer@1 很高，真正困难是闭环里何时 search、fallback 如何恢复、何时切换 aspect；
- **环境契约必须先固定**：如果 query 形态不符合 simulator 的基础参数+偏好契约，模型错误和 fallback 错误会被混为一谈。

### 9.2 当前最可能的瓶颈

1. 模型在真实多轮历史中未稳定执行 public control note；
2. fallback 后的 query 改写缺少足够的训练覆盖；
3. second fallback 的 aspect 切换属于稀有边界，SFT 轨迹数量不足；
4. completion 主项与正确答案之间仍存在 credit assignment 间隙；
5. GRPO 结果尚未在固定 Reward v3、固定 harness、固定 checkpoint 序列下完成 matched ablation；
6. 当前 Teacher 轨迹候选只有 97 条，正式 400 条目标尚未达到，数据规模和 composition 覆盖仍是限制。

## 10. 尚未解决的风险

- **实验可比性风险**：Reward v2/v3、旧/新 harness、是否启用 guard 的结果不能直接横向排名；
- **评测规模风险**：现有 200-task 多为诊断集，正式 evaluation split 为 471 task，尚需冻结配置后完整运行；
- **数据规模风险**：Teacher accepted/silver 当前仍是阶段性产物，不能代表完整 SFT 数据质量；
- **reward 欺骗风险**：过度奖励 public completion 可能诱导模型频繁 BLOCKED，因此必须同时看 correct itinerary、偏好覆盖和 blocked 比例；
- **KL 缺失风险**：无 KL 时策略可能偏离 SFT，需用 matched KL A/B 和行为指标监控，而不是凭 reward 单点判断；
- **turn-credit 风险**：span 对齐正确不代表 evidence 具有因果性；当前 bounded multiplier 只能降低风险，不能替代消融；
- **simulator 侧信道风险**：fallback 文本、错误码和时延可能间接暴露环境状态，公开反馈应保持稳定、最小化；
- **方差风险**：GRPO `n=4` 和单次 validation 不足以证明小幅收益，需要固定任务、多 seed 或置信区间。

## 11. 推荐复现流程

以下命令均为项目内可复现入口；正式运行前应确认模型、数据和 UserBench 服务路径。

### 11.1 CPU 测试

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=src \
.venv/bin/pytest -q
```

当前结果：`325 passed, 2 skipped, 1 warning`。

### 11.2 检查 split 和公开状态模块

```bash
python scripts/data/build_dataset_splits.py \
  --verify-only

python -m compileall -q src scripts
```

### 11.3 SFT/merged 模型诊断评测

```bash
bash scripts/eval/run_evaluation.sh sft \
  --model outputs/models/sft-merged \
  --dataset outputs/evaluation/subsets/tasks_200_proportional_v1.parquet \
  --subset-manifest outputs/evaluation/subsets/tasks_200_proportional_v1.json \
  --output outputs/evaluation/repro-sft-200 \
  --concurrency 4
```

实际仓库中的历史运行命令和完整参数以对应 `outputs/evaluation/*/manifest.json` 为准；评测结果目录应保持在 ignored `outputs/` 下。

### 11.4 GRPO

```bash
bash scripts/train/grpo/run_grpo_from_sft.sh \
  --output outputs/models/grpo-sft-merged-turn-credit-20 \
  --no-stall-recovery \
  trainer.total_training_steps=20 \
  trainer.save_freq=20 \
  trainer.test_freq=20 \
  data.val_max_samples=32 \
  algorithm.turn_credit.mode=train
```

长任务建议放入 tmux，并为每次运行保存命令、git revision、配置快照和日志路径。若 DataLoader worker 被系统杀死，应先降低 worker/并发或关闭持久 worker，再重新启动，而不是把已经打印到 100% 的进度直接当作完整成功。

## 12. 下一阶段计划

1. 完成当前 `20→200` continuation，并在 50/100/150/200 checkpoint 上用完全相同的 32-task validation 评测；
2. 对 Reward v3 做 matched ablation：无 turn-credit、shadow、train，以及不同 KL 系数；
3. 补齐 recovery-boundary-v1 的 accepted target，优先覆盖 first fallback、second fallback 和 repeated/no-progress；
4. 扩大 Teacher Gold/Silver 到正式 quota，并审计 composition、task 去重和 answer ID 可见性；
5. 在配置冻结后运行完整 471-task evaluation，报告 completion、correct itinerary、blocked、guard、重复和置信区间；
6. 将“正常 search→answer”“偏好完成→search”“fallback→rewrite/switch”分别作为独立 gate，避免单一总 reward 掩盖协议错误；
7. 如果加入 KL，必须在同一初始化 checkpoint、同一任务顺序、同一 UserBench 契约下比较策略漂移和任务指标。

## 附录：关键文件与产物

| 主题 | 路径 |
|---|---|
| 项目入口 | [`README.md`](../README.md) |
| split manifest | [`data/split_manifest.json`](../data/split_manifest.json) |
| UserBench 配置 | [`configs/interaction_config/userbench.yaml`](../configs/interaction_config/userbench.yaml) |
| GRPO 配置 | [`configs/train/grpo/grpo.yaml`](../configs/train/grpo/grpo.yaml) |
| 公开状态/guard | [`src/travel_grpo/envs/public_control.py`](../src/travel_grpo/envs/public_control.py) |
| Agent loop policy 注入 | [`src/travel_grpo/training/grpo/adapter/agent_loop.py`](../src/travel_grpo/training/grpo/adapter/agent_loop.py) |
| 工具适配与反馈 | [`src/travel_grpo/training/grpo/adapter/tools.py`](../src/travel_grpo/training/grpo/adapter/tools.py) |
| Actor policy | [`src/travel_grpo/prompts/actor_policy.py`](../src/travel_grpo/prompts/actor_policy.py) |
| Reward v3 | [`docs/reward/design-v3-priority.md`](reward/design-v3-priority.md) |
| Turn-credit | [`docs/reward/turn-credit-v1.md`](reward/turn-credit-v1.md) |
| SFT 训练说明 | [`docs/training/sft.md`](training/sft.md) |
| GRPO 训练说明 | [`docs/training/grpo.md`](training/grpo.md) |
| 200-task 评测产物 | `outputs/evaluation/` |
| SFT/Teacher 产物 | `outputs/teacher_trajectories/`、`outputs/models/sft-merged/` |

> 结论边界：本文总结的是当前仓库中已经存在的设计和探索性产物；任何正式 benchmark 声明都应附带对应 manifest、配置快照、模型 checkpoint、日志和可复现命令。
