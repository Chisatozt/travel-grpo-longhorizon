# Travel GRPO 项目完整会话交接

> 更新时间：2026-08-12（UTC）
> 仓库：`D:\Code\Python\vscodeProjects\travel-grpo-longhorizon`  
> 当前运行时 Reward：`userbench-travel-reward-v3-priority`；v2 仅作为历史 SFT 输入兼容。
> 分支与提交：`main`，`b5eaa45 add GRPO`  
> 远程：`origin = https://github.com/Chisatozt/travel-grpo-longhorizon.git`  
> 当前工作树：干净  
> 一句话状态：**Baseline → SFT → GRPO → Evaluation 的代码与离线验证已完成，但正式教师数据、模型训练和 471 题评测均未执行。**

本文是整个项目会话的总交接，供一个完全没有上下文的新会话直接接手。历史文档
[`training/teacher-trajectory-optimization-handoff.md`](training/teacher-trajectory-optimization-handoff.md)
只记录了教师轨迹优化过程中的一个中间状态，不应替代本文判断当前项目状态。

## 1. 项目在做什么

目标是构建一个面向旅游助手的 Agentic 后训练项目，整体结构参考：

- `YYHDBL/shopping-grpo-longhorizon`；
- 本地只读参考项目 `D:\Code\Python\vscodeProjects\shopping-grpo-longhorizon`；
- `qiqihezh/agentic-grpo-longhorizon`；
- veRL 0.8 Agentic RL / ToolAgentLoop 的官方接口。

购物环境被替换为 Salesforce UserBench 的 TravelGym，完整目标链路为：

```text
UserBench 固定互斥划分
  → deepseek-v4-flash 教师轨迹采集
  → Gold + Silver 重新验收
  → Qwen/Qwen3.5-2B action-only LoRA SFT
  → 合并 SFT 模型
  → veRL 0.8 + UserBench 在线 GRPO
  → 132 题 validation 选择 checkpoint
  → 471 题 Baseline / SFT / GRPO 配对评测
```

这个项目的核心不是简单训练一个聊天模型，而是让 Actor 在多轮会话中使用唯一工具：

```text
interact_with_env(thought, choice, content)
choice ∈ {search, action, answer}
```

Actor 需要主动询问用户偏好、搜索候选并提交推荐。Actor 只能看到 UserBench 返回的公开 `feedback`，不能看到隐藏偏好、best ID、ground truth 或 Reward 内部快照。

## 2. 已锁定且不要随意修改的决策

| 项目 | 固定决策 |
|---|---|
| Actor 基座 | `Qwen/Qwen3.5-2B` |
| Teacher | `deepseek-v4-flash`，OpenAI-compatible API |
| 采集模拟器 | `deepseek-v4-flash`，独立 `COLLECTION_USER_SIM_*` |
| GRPO 模拟器 | `deepseek-v4-flash`，独立 `GRPO_USER_SIM_*` |
| 评测模拟器 | `deepseek-v4-flash`，独立 `EVAL_USER_SIM_*` |
| UserBench | 提交 `80506d2ab484cab843e60a2401ff3e0290d05b87` |
| GRPO 框架 | veRL `0.8.0` |
| 正式训练系统 | Linux、Python 3.12、单张至少 80 GiB，目标 96 GiB NVIDIA GPU |
| 本地 Windows | 仅离线测试、数据验证和 dry-run，不运行 vLLM/Ray/正式训练 |
| SFT | Gold + Silver、action-only LoRA、`enable_thinking=false` |
| GRPO Reward | completion-priority Travel Reward v3；SFT loader 兼容历史 v2 |
| 最终评测 | 官方 test 471 条，训练配置和 checkpoint 冻结后才可使用 |

必须保持 Actor、Teacher、三类模拟器和正式评测为不同运行边界。即使它们调用同一个供应商，也不能复用角色变量或在同一进程切换 UserBench 模拟器绑定。

## 3. 会话中完成了哪些工作

### 3.1 仓库骨架与 UserBench

- 按 SFT → GRPO → Evaluation 分层建立项目结构。
- 将 UserBench 固定快照完整内嵌到 `environments/UserBench/`。
- 保留 Apache-2.0 许可证、来源提交和 `EMBEDDED_SOURCE.json`。
- UserBench 是第三方冻结目录；日常开发没有修改它。

### 3.2 冻结数据划分

从八种 `travel{composition}_multiturn_onechoice/{train,test}.parquet` 派生互斥任务：

| 项目 split | 数量 | 用途 |
|---|---:|---|
| SFT train | 716 | 教师轨迹采集与 SFT |
| SFT validation | 80 | 轨迹质量及 SFT loss 验证 |
| GRPO train | 1,723 | 在线 GRPO |
| GRPO validation | 132 | checkpoint 选择 |
| Final evaluation | 471 | 三阶段最终配对评测 |

总计 3,122 条，所有项目 split 两两无交集。划分规则、canonical 数据和 `data/split_manifest.json` 已固定。

### 3.3 UserBench 包装、工具与并发隔离

`src/travel_grpo/envs/` 已实现：

- action 格式、choice 和前缀验证；
- UserBench 延迟导入和固定来源检查；
- 同步/异步 reset、step 和 close；
- `ContextVar` trajectory 隔离；
- UserBench 进程级模拟器绑定；
- Actor 可见 observation 防泄漏；
- 原始逐步 Reward、累计 Reward 和 Travel Reward v2 审计状态。

参数格式错误会返回稳定的 `Error:` observation，且不会调用环境。配置漂移、task ID 不一致和无法解释的程序异常会 fail-loud。

### 3.4 教师轨迹采集优化

会话中对教师轨迹采集做过多轮真实 API 迭代，主要增加了：

- 请求内单工具协议重试；
- Teacher 本地状态机与阶段约束；
- 多工具调用拒绝；
- exact/semantic repeat 检查；
- bundled/vague/wrong-field action 诊断；
- answer 必须来自可见搜索结果；
- 整轨迹重试与原子 task checkpoint；
- accepted/silver/rejected/diagnostics 分流；
- Gold 严格 gate 与有限 Silver gate；
- Silver 修复 assistant turn 的 `loss_mask=true`。

当前代码不信任 JSON 里的 `quality_tier` 字符串，而是用轨迹证据重新推断 Gold 或 Silver。`reward_valid=false` 永远不能伪装成 Gold。

历史 smoke 文件位于被忽略的 `outputs/teacher_trajectories/`。当前盘点结果为：

- accepted 文件内共 7 行；
- 仅 5 个唯一 task ID，存在跨批次重复；
- 6 行是 v4，1 行是旧 v3；
- 没有正式配置所需的 `sft_train.accepted.jsonl`、`sft_train.silver.jsonl`、`sft_validation.accepted.jsonl` 和 `sft_validation.silver.jsonl`。

因此这些历史 smoke **只能用于诊断，不能直接宣称正式 SFT 数据已经准备好**。

### 3.5 Travel Reward v2

当前终局 Reward 为：

```text
raw =
    0.75 × (2 × grounded_quality - 1)
  + 0.15 × active_preference_coverage
  + 0.10 × efficiency
  - 0.10 × passive_preference_coverage
  - 0.40 × incomplete_rate
  - policy_penalties

terminal_reward = clip(raw, -1, 1)
```

策略惩罚覆盖 invalid action、parallel calls、exact/semantic repeat、ambiguous action、未搜索直接回答和错误回答。GRPO 工具 Reward 恒为 `0.0`，只在 rollout 终局写一次 `terminal_reward`，避免重复计奖。

### 3.6 Gold + Silver action-only SFT

SFT 已实现：

- train/validation 多 Gold/Silver 文件输入；
- 重新执行 Gold/Silver gate；
- task ID 去重、split 边界和 train/validation 隔离；
- 一个未 masked assistant tool call 生成一个 action-only 样本；
- Qwen3.5 tool-call chat template，`enable_thinking=false`；
- 32,768 token 上限；任一 turn 超长时整条轨迹拒绝，不截断；
- 超长审计写入 `overlong_rejections`；
- 排除超长后再次检查 readiness；
- LoRA r=16、alpha=32、dropout=0.05、BF16、3 epochs；
- Qwen3.5 模型类识别、LoRA 合并和非空目录保护。

正式 readiness 要求：

- train 至少 50 条重新验收后的轨迹；
- validation 至少 5 条；
- 八种 composition 在两个 split 中都至少有一条；
- 不允许从 GRPO 或 final evaluation 回填。

当前 `python scripts/train/sft/sft_train.py --dry-run` 会因为正式轨迹文件不存在而停止，这是预期行为，不是代码缺陷。

### 3.7 veRL 0.8 GRPO

已经完成从旧 veRL 0.6.1 BaseInteraction 路径到 veRL 0.8 直接 AgentLoop 的迁移：

- 每条 rollout 创建独立 `UserBenchWrapper`；
- `finally` 无条件关闭环境并清理 ContextVar；
- tool reward 为 0；
- Travel Reward v2 写入 `output.reward_score`；
- terminated/truncated 后不再多生成一轮 Actor；
- 同一 completion 多工具调用全部不执行并终止；
- malformed/unknown tool 返回稳定错误；
- 无工具输出计为被惩罚的协议错误；
- simulator/API 最终失败标为 invalid rollout；
- `reward_valid=false` 不作为有效零分样本训练。

veRL 专用派生数据位于本机忽略目录 `outputs/grpo/data/`：

- train：1,723；
- validation：132；
- `ground_truth` 为空；
- `reward_model.id == extra_info.task_id == create_kwargs.id`；
- `extra_info.index` 是稳定唯一行号，避免 veRL 把不同 prompt 都当成 index 0；
- manifest 固定 veRL 0.8.0、PyArrow 25.0.0、生成器版本和所有 SHA-256；
- 当前 manifest SHA-256：`8da3e830353e36e12d96af4701bf5644a6983243310f023e6f6b88a3b5db8c86`。

生产动态采样器固定：

- 每个 prompt `n=4`；
- 任一 rollout invalid 时丢弃整组；
- reward 极差 `<=1e-6` 的同分组丢弃；
- 每次 update 最多 3 个新 batch；
- 累计 2 个有效 prompt group 才更新；
- 连续跳过 10 次后 fail-loud；
- 跨 batch 接受的 group 会恢复到原 prompt 顺序。

veRL 只允许一个带版本、源文件 SHA、payload SHA 和结果 SHA 的最小动态采样连接补丁；未知版本或源码哈希立即失败。

### 3.8 Checkpoint 选择、导出和评测

已实现：

- 2-step vanilla smoke profile；
- 正式 500-step GRPO profile；
- Linux/GPU/版本/API/模型/数据/UserBench/tool parser preflight；
- resume、输出目录保护、console/SwanLab；
- validation 原始 `0.jsonl/50.jsonl/...` 自动汇总；
- 固定 132 分母 checkpoint 选择；
- 只导出 `passed=true` 且 step 一致的 checkpoint；
- OpenAI-compatible Actor rollout；
- Baseline/SFT/GRPO 共用冻结评测逻辑；
- 每 task 原子 checkpoint 和 `--resume`；
- infrastructure-invalid 显式重试历史；
- 主指标固定 471 分母，缺失/无效任务按 0；
- valid-only、composition 和逐 task 配对 delta；
- 仅三个阶段完整且 contract hash 一致时生成 comparison。

当前不存在以下正式产物：

```text
outputs/sft/qwen3.5-2b-lora
outputs/models/sft-merged
outputs/models/grpo
outputs/models/grpo-merged
outputs/evaluation/baseline
outputs/evaluation/sft
outputs/evaluation/grpo
```

所以目前没有任何 SFT、GRPO 或 UserBench benchmark 指标可以声明。

## 4. 当前验证证据

最后一次完整离线验收：

| 检查 | 结果 |
|---|---|
| `pytest -q` | 156 passed，2 skipped |
| `python -m compileall -q src scripts` | 通过 |
| canonical split verify | 3,122 条，`716/80/1723/132/471`，交集全 0 |
| veRL data verify | 1,723/132，manifest 和 hash 通过 |
| vanilla GRPO dry-run | 通过，2 step，动态采样关闭 |
| formal GRPO dry-run | 通过，500 step，动态采样开启 |
| baseline eval dry-run | 471 frozen tasks，limit 2 |
| evaluation contract hash | `266139275b97b3093a44823645b9a1fb1b061603e1ed380b7dedcb79caede67c` |
| `git diff --check` | 通过，仅 Windows LF/CRLF 提示 |
| `git diff --name-only -- data environments/UserBench` | 空 |

两个 skipped 测试是明确 opt-in 的真实 UserBench/veRL 0.8 外部集成 smoke，不是默认离线失败。

## 5. 当前真正的卡点

### 5.1 正式教师轨迹不足

这是当前第一个业务卡点。代码已准备好，但尚未获得：

- 至少 50 条 train Gold+Silver；
- 至少 5 条 validation Gold+Silver；
- 两个 split 的八 composition 完整覆盖；
- 当前 v4 schema、无重复、通过重新 gate 的正式文件。

历史 smoke 重复、含 v3 且命名不符合正式配置，不能直接凑数。

### 5.2 当前机器不能验证正式训练栈

Windows RTX 4050 6GB 不适合也不应运行：

- vLLM 0.25.1 正式 rollout；
- Ray + veRL GRPO；
- Qwen3.5-2B 32K 多 rollout；
- 500-step GRPO；
- 471 题正式评测。

需要 Linux、Python 3.12、单张至少 80 GiB、目标 96 GiB GPU。

### 5.3 仍需要外部资源

- Teacher 和三类 UserBench simulator 的 API 凭据及预算；
- Qwen/Qwen3.5-2B 权重；
- 正式 Linux GPU；
- veRL/vLLM 真实 opt-in smoke；
- 最终三个模型的独立 Actor 服务。

## 6. 下一步应按什么顺序做

### 阶段 A：冻结当前代码并准备正式机器

1. 从提交 `b5eaa45` 开始，不修改 `environments/UserBench/`。
2. 在 Linux 96GB GPU 机器运行：

   ```bash
   bash scripts/setup.sh
   cp .env.example .env
   # 填写真实配置，绝不提交 .env
   set -a; source .env; set +a
   ```

3. 先重跑默认离线测试和所有 dry-run。
4. 显式运行 veRL 0.8/UserBench opt-in smoke；失败时先修兼容，不开始付费大批量采集。

### 阶段 B：正式采集 SFT 数据

1. train 和 validation 使用独立 run-dir：

   ```bash
   python scripts/train/sft/collect_sft_data.py \
     --input data/sft/tasks_train.jsonl \
     --run-dir outputs/teacher_trajectories/runs/sft-train \
     --output outputs/teacher_trajectories/sft_train.accepted.jsonl

   python scripts/train/sft/collect_sft_data.py \
     --input data/sft/tasks_validation.jsonl \
     --run-dir outputs/teacher_trajectories/runs/sft-validation \
     --output outputs/teacher_trajectories/sft_validation.accepted.jsonl
   ```

2. 小批量逐级扩大，不要一开始采满 716/80。
3. 汇总 Gold/Silver、拒绝原因、API 失败、token/成本和 composition 覆盖。
4. 使用正式 loader 重新 gate，不以文件名或序列化 tier 为准。
5. 达到 400/40 readiness 后冻结数据 manifest，不再为了训练数量降低 gate。

### 阶段 C：SFT

```bash
python scripts/train/sft/sft_train.py --dry-run
bash scripts/train/sft/run_sft.sh
python scripts/train/sft/merge_lora.py
```

确认 `outputs/models/sft-merged` 包含完整模型、tokenizer/processor 和 `merge_manifest.json`。

### 阶段 D：GRPO

```bash
python scripts/train/grpo/prepare_data.py --verify-only
bash scripts/train/grpo/run_vanilla.sh --dry-run
bash scripts/train/grpo/run_vanilla.sh
bash scripts/train/grpo/run_grpo.sh --dry-run
bash scripts/train/grpo/run_grpo.sh
```

训练后：

```bash
python scripts/eval/select_checkpoint.py \
  --validation-dir outputs/models/grpo/validation_rollouts

bash scripts/train/grpo/export_actor.sh \
  outputs/models/grpo/global_step_<SELECTED>/actor \
  outputs/models/grpo-merged \
  --selection outputs/models/grpo/checkpoint_selection.json
```

若没有 checkpoint 通过 gate，GRPO 阶段应判失败，不能导出最后一步冒充正式模型。

### 阶段 E：冻结 471 题评测

只有模型、配置、checkpoint 和 contract 全部冻结后才执行。三个阶段分别启动 Actor 服务和独立评测进程，并确保 `ACTOR_MODEL` 与 vLLM served model name 完全相同：

```bash
export ACTOR_MODEL=Qwen/Qwen3.5-2B
bash scripts/vllm_server/actor.sh "$ACTOR_MODEL"
bash scripts/eval/run_evaluation.sh baseline --resume

export ACTOR_MODEL=outputs/models/sft-merged
bash scripts/vllm_server/actor.sh "$ACTOR_MODEL"
bash scripts/eval/run_evaluation.sh sft --resume

export ACTOR_MODEL=outputs/models/grpo-merged
bash scripts/vllm_server/actor.sh "$ACTOR_MODEL"
bash scripts/eval/run_evaluation.sh grpo --resume

python scripts/eval/compare_stages.py
```

## 7. 最重要的文件地图

| 位置 | 职责 |
|---|---|
| `README.md` | 当前状态和正式执行顺序 |
| `AGENTS.md` | 仓库不可变边界 |
| `.env.example` | Actor/Teacher/三类模拟器变量契约 |
| `data/split_manifest.json` | canonical 3,122 条划分证据 |
| `src/travel_grpo/data/` | canonical 数据加载、划分和验证 |
| `src/travel_grpo/envs/` | UserBench wrapper、action、session、Reward v2 |
| `src/travel_grpo/training/sft_dataset.py` | Gold/Silver 重验和 action-only 渲染 |
| `scripts/train/sft/sft_train.py` | SFT audit/dry-run/train 入口 |
| `scripts/train/sft/merge_lora.py` | SFT LoRA 合并 |
| `src/travel_grpo/training/grpo/data.py` | canonical → veRL 0.8 数据 |
| `src/travel_grpo/training/grpo/adapter/` | veRL Tool/AgentLoop/session 生命周期 |
| `src/travel_grpo/training/grpo/dynamic_sampling.py` | 有界 reward-varying sampler |
| `src/travel_grpo/training/grpo/preflight.py` | 正式运行时预检 |
| `scripts/train/grpo/apply_verl_patch.py` | hash 校验的最小 veRL 补丁 |
| `scripts/train/grpo/train_grpo.py` | GRPO launcher 与 Hydra overrides |
| `scripts/eval/select_checkpoint.py` | 132 题汇总和 checkpoint 选择 |
| `scripts/eval/evaluate_userbench.py` | 三阶段冻结评测入口 |
| `src/travel_grpo/evaluation/` | artifact、summary、metrics、comparison |
| `docs/training/sft.md` | SFT 与采集契约 |
| `docs/training/grpo.md` | veRL/GRPO 契约 |
| `docs/evaluation/userbench.md` | 471 题评测契约 |

## 8. 这段会话中可以避免的坑

### 8.1 把代码完成误认为实验完成

当前只是全链路实现和离线验证完成。没有 SFT 模型、GRPO checkpoint 或 471 题结果。今后汇报必须区分：

- implemented；
- offline verified；
- opt-in integration verified；
- formally trained；
- formally evaluated。

### 8.2 用被忽略的历史输出作为稳定测试 fixture

曾有测试直接读取 `outputs/teacher_trajectories/` 的真实 smoke 文件，并假设其中包含某个固定错误。真实产物会变化，导致测试不稳定。该依赖已移除。以后默认测试只能使用 `tmp_path`、fake runtime 或仓库内固定 fixture。

### 8.3 信任 `quality_tier` 字符串

文件声明的 Gold/Silver 不是事实来源。必须重新执行 gate，否则旧采集器或人工改动可以错误升级轨迹。

### 8.4 把 infrastructure-invalid 当作零分样本

`reward_valid=false` 对外可显示 0，但训练中必须丢弃。否则 API 故障会污染 GRPO advantage。

### 8.5 重复计算 Reward

UserBench step reward 只用于 metadata；tool reward 固定为 0；Travel Reward v2 只在终局进入 `reward_score`。不能同时把 step/tool/final reward 都计入优化。

### 8.6 忘记 veRL 的 `extra_info.index`

veRL 默认缺失 index 时会把所有 prompt 视为 index 0。派生数据现在写入唯一稳定行号，后续重构不能删除。

### 8.7 动态采样跨 batch 改变 prompt 顺序

两个有效 group 可能分别在不同采样 batch 中出现。拼接前必须恢复原 prompt UID 顺序，否则 veRL 按行 union 时会把 rollout 配错 prompt。

### 8.8 在 Windows 上强行验证 Linux 训练栈

当前 Windows 只有 WSL shim，没有可用 Bash；shell launcher 的验收使用了等价 Python dry-run。不要把 WSL 权限错误误判成项目逻辑错误，也不要在 6GB GPU 上尝试 vLLM/Ray 正式运行。

### 8.9 UserBench 进程级 endpoint 绑定

固定 UserBench 会通过进程环境读取 OpenAI endpoint。同一进程不能从 collection 切换到 GRPO 或 eval 角色。不同角色必须重启独立进程。

### 8.10 最终 test 泄漏

471 条 test 不能用于：

- Teacher 调用；
- SFT；
- GRPO；
- Reward 参数调整；
- checkpoint 选择；
- smoke test。

### 8.11 修改第三方快照

`environments/UserBench/` 不可做日常补丁。如果升级，必须替换完整快照并同步更新来源文件、提交 SHA 和许可证。

### 8.12 截断 32K 轨迹

不能截断 tool call、Observation 或搜索证据。任一 assistant decision 超长时整条轨迹拒绝，并在排除后重新检查 readiness。

### 8.13 配置/文档与真实运行不一致

特别注意：

- vLLM `--served-model-name` 必须等于评测的 `ACTOR_MODEL`；
- checkpoint export 必须匹配通过选择的 step；
- `.env` 不会自动安全地提交或共享；
- Windows 的 LF/CRLF warning 不是 `git diff --check` 失败；
- 历史教师优化 handoff 是中间快照，结论可能已经过时。

## 9. 新会话开始时的建议检查清单

```bash
git status --short
git log -1 --oneline
python -m pytest -q
python -m compileall -q src scripts
python scripts/data/build_dataset_splits.py --verify-only
python scripts/train/grpo/prepare_data.py --verify-only
python scripts/train/grpo/train_grpo.py \
  --config configs/train/grpo/vanilla_grpo.yaml --dry-run
python scripts/train/grpo/train_grpo.py \
  --config configs/train/grpo/grpo.yaml --dry-run
python scripts/eval/evaluate_userbench.py --stage baseline --dry-run --limit 2
git diff --name-only -- data environments/UserBench
git diff --check
```

Windows 上建议使用：

```powershell
conda run -n travel_grpo python -m pytest -q
```

本机 Conda 环境位于 `D:\Environment\anaconda3\envs\travel_grpo`。

## 10. 下一阶段的完成定义

下一阶段不是继续增加 scaffold，而是形成第一个可审计的正式实验闭环。至少应满足：

1. 正式 train/validation Gold+Silver 达到 400/40 readiness；
2. 数据 manifest、成本和拒绝原因完整；
3. SFT 模型成功训练并合并；
4. Linux 96GB 环境通过 vanilla 2-step smoke；
5. 正式 GRPO 完成并从 132 题 validation 选择通过 gate 的 checkpoint；
6. 三阶段 471 题都完整，contract hash 一致；
7. 生成 comparison，同时明确 infrastructure-invalid 固定分母处理；
8. 只有到这一步才报告真实模型或 benchmark 指标。

接手者应优先推进正式 SFT 数据采集和 Linux 环境集成，不应再重写已通过离线测试的核心架构，除非真实 opt-in smoke 暴露了明确的 veRL/UserBench 兼容问题。
