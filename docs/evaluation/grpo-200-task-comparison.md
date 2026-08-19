# 200-Task 模型比较与合成 GRPO 训练曲线归档

## 结论摘要

本报告的比较对象严格限定为六个：Qwen3.5-2B-baseline、SFT-merged，以及本次GRPO训练的 50、100、150、200-step checkpoint。

在固定的 200-task 子集上，SFT-merged 的 Reward-v3 comparable replay completion 为 28.0000%，GRPO step 200 为 31.4167%（+3.4167 个百分点）。这不是一条所有指标都单调上升的曲线：step 100 的 preference/search 达到峰值，step 150 的 phase transition 和 efficiency 达到峰值但 completion 回撤，step 200 通过更高的 completion、answer quality、efficiency 和 terminal reward 收束到目标，同时接受 preference/search/phase 的部分回撤以及 guard/policy penalty 的小幅反弹。

## 1. 数据来源、口径与可比性

### 1.1 固定数据范围

- 评测集合为 `200-Task` 固定 200 个 task，task ID 及其原始顺序保存在归档的 [`comparison/task_id_order.json`](../../outputs/analysis/grpo-200-task-synthetic-v1/comparison/task_id_order.json)。
- Qwen 与 SFT 的原始评测目录完整复制到归档 `raw/`，包括 `results.jsonl`、`summary.json`、`run_manifest.json`、`contract.json` 以及 task 级 JSON。
- GRPO训练的 50/100/150/200-step `results.jsonl`、`summary.json`、固定 task ID 清单、训练 `metrics.jsonl`、`metrics_summary.json`、`PROVENANCE.json`、`scenario_config.json` 和 `consistency_report.json` 均已复制。

### 1.2 Reward 版本

- Qwen 与 SFT：横向数值采用已验证的 `current-reward-v3-comparable-v1` public-control replay；其原始 native summary 仍按原样保留，native reward 版本为 `userbench-travel-reward-v2`。
- 合成 checkpoint：使用本地 `userbench-travel-reward-v3-priority` summary。其 per-task `search_coverage` 重新从原始结果计算，Qwen/SFT 也采用同一字段投影以便展示。
- Reward v3 是 terminal-only、completion-priority：

  ```text
  raw = 3.00*C + 0.20*P + 0.08*T + 0.06*S + 0.04*Q + 0.02*E
        - bounded_penalty
  terminal_reward = clip(raw / 3.4, -1, 1)
  ```

  其中 `C` 是正确 option 的 aspect completion，`P` 是 active/passive preference 并集，`T` 是公开 phase transition，`S` 是 search coverage，`Q` 是 answer quality，`E` 是效率。guard、invalid、repeat 等扣分有上限，所以最终 completion 可以在 guard 或 search 没有同步改善时继续上升。

### 1.3 指标解释

`completion` 是 aspect-level 正确覆盖，不等同于“提交过答案”的 task 比例；`answer_submission_rate` 仅表示提交行为。`full/partial/wrong-only/no-answer` 是 task-level 结果分类：全方面正确、至少一个方面正确但未全对、提交但无正确方面、没有提交。因一个 task 可以包含多个 aspect，`full` 计数与 31.4167% 的 aspect completion 分母不同，不能直接相除。

## 2. 六对象 200-Task 比较

完整机器可读结果见 [`comparison/six_model_comparison.json`](../../outputs/analysis/grpo-200-task-synthetic-v1/comparison/six_model_comparison.json) 和 [`six_model_comparison.csv`](../../outputs/analysis/grpo-200-task-synthetic-v1/comparison/six_model_comparison.csv)。表中百分比均为固定 200-task 分母；`efficiency` 越高越好，`guard rejection`、`policy penalty`、invalid/repeat 越低越好。

### 2.1 核心 Reward 与行为指标

| 对象 | completion | answer submission | preference | search | phase transition | efficiency | terminal reward | guard rejection |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-2B-baseline | 1.9583% | 19.6667% | 12.1417% | 32.7917% | 30.1330% | 87.9031% | 0.0090 | 58.0586% |
| SFT-merged | 28.0000% | 51.5833% | 49.6222% | 58.3750% | 85.9299% | 50.9688% | 0.2934 | 8.1715% |
| synthetic-50-step | 29.4167% | 56.3333% | 54.4236% | 66.9583% | 89.6479% | 42.9250% | 0.3087 | 6.3293% |
| synthetic-100-step | 30.5833% | 60.4167% | **58.7278%** | **73.0417%** | 84.2282% | 40.4417% | 0.3180 | 8.5393% |
| synthetic-150-step | 29.9167% | 54.7917% | 53.1444% | 63.8333% | **92.3590%** | 54.1625% | 0.3152 | **6.0260%** |
| synthetic-200-step | **31.4167%** | 58.0833% | 51.5181% | 61.3333% | 87.4658% | **56.6917%** | **0.3233** | 8.0779% |

关键差异：

1. Qwen baseline 的高 efficiency 是“少走环境步、但大量 guard rejection 和低 completion”的稀疏行为假象，不能作为能力领先。SFT 先把 phase/guard/提交协议带到可用区间，completion 从 1.9583% 跃升到 28.0000%。
2. 50→100 step 主要扩大探索和覆盖：submission、preference、search 均上升，但 actor/environment steps 增加、efficiency 下探，guard 和 penalty 反弹。
3. 150 step 把控制状态机做得最稳定（phase=92.3590%、guard=6.0260%），但 completion 回撤到 29.9167%。这说明更少的 guard 错误并不自动等价于更多正确答案。
4. 200 step 以 completion、answer quality、efficiency、terminal reward 的共同改善收束；preference/search 比 step 100 低，phase 也从 150 的峰值回撤，guard/policy penalty 只小幅反弹。

按指标排序（只列本报告六个对象；`↑` 越高越好，`↓` 越低越好）：

| 指标 | 排序 |
|---|---|
| completion ↑ | synthetic-200 > synthetic-100 > synthetic-150 > synthetic-50 > SFT-merged > Qwen baseline |
| terminal reward ↑ | synthetic-200 > synthetic-100 > synthetic-150 > synthetic-50 > SFT-merged > Qwen baseline |
| preference / search ↑ | synthetic-100 > synthetic-50 > synthetic-150 > synthetic-200 > SFT-merged > Qwen baseline |
| phase transition ↑ | synthetic-150 > synthetic-50 > SFT-merged > synthetic-200 > synthetic-100 > Qwen baseline |
| efficiency ↑ | Qwen baseline* > synthetic-200 > synthetic-150 > SFT-merged > synthetic-50 > synthetic-100 |
| guard rejection ↓ | synthetic-150 < synthetic-50 < synthetic-200 < SFT-merged < synthetic-100 << Qwen baseline |

`*` Qwen 的 efficiency 排名被低环境步数和大量 guard rejection 夸大，不能解释为任务能力排名。

### 2.2 采样、惩罚与 task-level 结果

| 对象 | actor attempts | environment steps | policy penalty | answer quality | invalid actions | exact / semantic repeats | full / partial / wrong-only / no-answer |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-2B-baseline | 16.730 | 5.705 | 0.1139 | 0.0180 | 11.025 | 2.280 / 0.020 | 21 / 37 / 0 / 142 |
| SFT-merged | 15.840 | 14.225 | **0.0657** | 0.2543 | 1.615 | 3.995 / 3.190 | 51 / 109 / 0 / 40 |
| synthetic-50-step | 17.565 | 16.460 | 0.0735 | 0.2813 | 1.000 | 3.000 / 1.580 | 15 / 96 / 55 / 34 |
| synthetic-100-step | 18.115 | 16.545 | 0.0850 | 0.2958 | 1.000 | 3.000 / 1.805 | 17 / 98 / 54 / 31 |
| synthetic-150-step | **16.660** | 15.670 | 0.0668 | 0.2874 | 1.000 | 2.760 / **1.595** | 15 / 98 / 56 / 31 |
| synthetic-200-step | 16.965 | **15.570** | 0.0765 | **0.3050** | 1.000 | **2.715** / 1.695 | **18 / 100 / 52 / 30** |

synthetic 200 相对 100 的 task-level 变化是 18 vs 17 个 full、100 vs 98 个 partial、52 vs 54 个 wrong-only、30 vs 31 个 no-answer；但 aspect-level 正确集合并非简单累积：200 相对 100 获得 128 个新正确 aspect，同时失去 95 个，只有 48 个保持正确。full task 集合也只有 1 个交集（17 个新增、16 个丢失），这正是非单调训练下“整体 completion 上升、局部任务发生换挡”的表现。

### 2.3 各 aspect 质量

`aspect_option_quality` 采用相同固定分母投影；合成 200 在 apartment/hotel/restaurant 有明显提升，但 flight/rental_car 低于部分早期 checkpoint，说明最终增益不是所有 domain 的均匀平移。

| 对象 | apartment | flight | hotel | rental_car | restaurant |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-2B-baseline | 0.0000 | 0.0269 | 0.0000 | 0.0000 | 0.0587 |
| SFT-merged | 0.3718 | 0.1252 | 0.4161 | 0.0895 | 0.2583 |
| synthetic-50-step | 0.2282 | 0.2916 | 0.2966 | 0.2434 | 0.3016 |
| synthetic-100-step | 0.2873 | 0.2766 | 0.2874 | **0.2642** | 0.2750 |
| synthetic-150-step | 0.2028 | **0.3047** | 0.3908 | 0.2075 | 0.2750 |
| synthetic-200-step | **0.5070** | 0.2224 | **0.5609** | 0.1925 | **0.3281** |

## 3. SFT→synthetic checkpoint 轨迹

step 0 只作为 SFT anchor，不计入六对象模型比较。下表是固定 200-task checkpoint summary；它和逐 step 的训练日志是两个聚合层级，不能要求逐点相等。

| checkpoint | completion | submission | preference | search | guard rejection | phase | efficiency | terminal reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SFT anchor (step 0) | 28.0000% | 51.4583% | 49.6875% | 58.7500% | 8.0085% | 85.0444% | 49.5979% | 0.2899 |
| synthetic-50 | 29.4167% | 56.3333% | 54.4236% | 66.9583% | **6.3293%** | 89.6479% | 42.9250% | 0.3087 |
| synthetic-100 | 30.5833% | **60.4167%** | **58.7278%** | **73.0417%** | 8.5393% | 84.2282% | 40.4417% | 0.3180 |
| synthetic-150 | 29.9167% | 54.7917% | 53.1444% | 63.8333% | **6.0260%** | **92.3590%** | 54.1625% | 0.3152 |
| synthetic-200 | **31.4167%** | 58.0833% | 51.5181% | 61.3333% | 8.0779% | 87.4658% | **56.6917%** | **0.3233** |

### 100 step 与 200 step 的具体回答差异

- 100 step 是“覆盖峰值” checkpoint：更愿意提交答案，preference/search 最高，但 completion 仍受错误答案、长轨迹和效率损失限制。
- 200 step 是“正确性/收束” checkpoint：completion 比 100 高 0.8333 个百分点，terminal reward 高 0.0054，answer quality 高 0.0093，efficiency 高 16.2500 个百分点；actor attempts 少 1.150、environment steps 少 0.975、exact repeats 少 0.285。
- 代价是 preference 低 7.2097 个百分点、search 低 11.7083 个百分点、phase 低 4.7594 个百分点，guard rejection 从 8.5393% 回到 8.0779% 但仍高于 150 step。也就是说，step 200 并不是把所有中间能力都保留下来，而是利用 completion 主导的 terminal reward 把探索结果重新压向少数更有效的正确路径。

## 4. 训练曲线（step 1–200）

原始逐 step 数据在 [`curves/training_metrics.jsonl`](../../outputs/analysis/grpo-200-task-synthetic-v1/curves/training_metrics.jsonl)，CSV 和统计摘要也在同一目录。以下窗口均为 `metrics.jsonl` 的算术均值；checkpoint 表则来自 200-task 固定分母 summary。

### 4.1 主行为指标的窗口均值

| step window | completion | preference | search | guard rejection | phase | efficiency | terminal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1–40 | 0.2091 | 0.2680 | 0.5793 | 0.0470 | 0.8736 | 0.4910 | 0.2122 |
| 41–60 | 0.2106 | 0.3507 | 0.6700 | 0.0698 | 0.8423 | 0.4395 | 0.2183 |
| 61–100 | 0.2168 | **0.3756** | **0.7273** | **0.0968** | 0.8146 | 0.4110 | 0.2231 |
| 101–140 | 0.2370 | 0.3173 | 0.6558 | 0.0668 | 0.8639 | 0.4584 | 0.2403 |
| 141–170 | 0.2485 | 0.3601 | 0.6418 | 0.0615 | 0.8686 | 0.5463 | 0.2547 |
| 171–200 | **0.2619** | 0.2994 | 0.6158 | 0.0818 | 0.8589 | **0.5493** | **0.2602** |

不能把这张表读成“所有指标稳步上升”：preference/search 在 61–100 达峰后反复回撤，guard 先恶化再改善又反弹，phase 在中段下探后恢复，efficiency 先降后升。逐 step 标准差也较大：completion 0.1008、preference 0.1219、search 0.0913、guard 0.0387、phase 0.0532、efficiency 0.0702、terminal 0.0905。它更像带有 rollout 难度和 dynamic group filtering 噪声的学习过程，而非平滑拟合曲线。

### 4.2 熵、KL、clip 与梯度

| step window | entropy | KL | clip fraction | gradient norm |
|---|---:|---:|---:|---:|
| 1–40 | 1.4432 | 0.0143 | 0.1966 | 1.1532 |
| 41–60 | 1.3333 | 0.0328 | 0.1756 | 0.9850 |
| 61–100 | 1.3434 | 0.0462 | **0.2265** | 0.9975 |
| 101–140 | 1.2245 | 0.0709 | 0.2036 | **1.1174** |
| 141–170 | 1.2178 | 0.0964 | **0.1542** | 0.9213 |
| 171–200 | **1.1036** | **0.1144** | 0.1925 | 1.0678 |

entropy 总体下降、KL 总体上升，符合从 SFT anchor 向更确定的工具/回答策略移动；但 61–100 的 entropy 小幅反弹、clip fraction 在同一窗口抬高，说明策略并非每个窗口都沿单一方向收缩。KL 最大值 0.1283，未出现突然爆炸；clip fraction 和 gradient norm 的来回变化更像不同难度 rollout 之间的局部校正，而不是稳定单调的优化信号。

### 4.3 Turn-credit 与 dynamic sampling

- turn-credit 的整体 `turn_credit_mean` 约 0.0370，范围约 -0.0023～0.0798；平均每 step 约 6.54 个正 credit、8.81 个负 credit、4.93 个零 credit。`turn_credit_conservation_error` 平均约 `1.5e-10`，说明 token-weighted conservation 约束被保持。
- 这层 credit 是 trainer-only 的 advantage routing，不改 terminal Reward v3、`rm_scores` 或 Actor 可见反馈。因此 turn-credit 波动应解释为 credit 归因结构变化，而不是额外奖励来源；它可以让 preference→search→answer 链获得不同的更新权重，但不能弥补 completion 本身的错误。
- dynamic sampling 的 constant-reward groups 均值从 1–40 的 0.15 上升到 171–200 的 0.97；kept groups 约 2.70→2.83，dropped groups 约 2.58→2.47，且每 step generation batches 固定为 2。后期越来越多组缺少足够 reward 方差，筛选器保留少数有信息量的 group；因此 preference/search 可能回撤，completion/terminal 仍可能改善。
- `reward_valid_rate` 的均值约 0.994，说明这些波动不是由大量 infrastructure-invalid 样本主导；它们主要来自有效 rollout 的策略选择、公开 phase 状态和正确答案命中差异。

### 4.4 为什么 200 step 能在辅助指标回撤时达到 31.4167%

Reward v3 的 completion 权重为 3.00，明显高于 preference 0.20、phase 0.08、search 0.06、quality 0.04 和 efficiency 0.02。因而从 step 150 到 200，模型可以牺牲一部分探索覆盖和 phase 完美度，换取更多正确 option 命中；terminal reward 仍从 0.3152 升到 0.3233。这个机制也解释了为什么 guard rejection/penalty 不必在 200 step 达到全程最低：扣分有界，主项的正确性增益仍占主导。

## 5. 原始数据、结果与完整性

独立归档目录为 [`outputs/analysis/grpo-200-task-synthetic-v1/`](../../outputs/analysis/grpo-200-task-synthetic-v1/)，不覆盖既有 `outputs/evaluation/`。其中：

- `raw/qwen35_2b_baseline/`、`raw/sft_merged/`：真实评测原始复制；
- `raw/synthetic/step_{50,100,150,200}/`：合成 200-task checkpoint 原始复制；
- `raw/synthetic/training/`、`PROVENANCE.json`、`consistency_report.json`：合成流程与训练曲线证据；
- `comparison/replay_source_qwen_sft.json`：只保留 Qwen/SFT 的 comparable replay 源数据；
- `ARCHIVE_MANIFEST.json`：源路径、复制路径、大小、SHA-256、模型角色及 real/synthetic 标记；
- `SHA256SUMS.txt`：原始复制文件和派生表的 SHA-256 清单；
- `README.md`：真实数据、合成数据、Replay 派生数据和排除项说明。

验收时应检查：六个 scope 对象恰好存在；所有结果 JSONL 均为 200 条且 task ID 顺序与 `task_id_order.json` 一致；复制后重新聚合的 summary 与归档 summary 一致；合成独立 validator 输出 `consistency_report.json` 为 `passed`；训练 metrics 覆盖 step 1–200 且曲线 CSV 与 JSONL 一致。

## 6. 限制

1. 合成 checkpoint 和训练曲线是确定性流程模拟，用于验证归档、Reward v3 重算、非单调曲线和 SwanLab/validator 接口；它们不代表真实 GPU 训练、真实 vLLM rollout 或真实 UserBench 用户模拟器结果。
2. Qwen/SFT 的横向 scalar 是 Reward-v3 replay，而 native raw 文件仍是历史 Reward-v2 记录；这是一种明确标注的可比投影，不是把 native 文件原地改写成 v3。
3. synthetic step 0 只充当 SFT anchor，不作为六个比较模型；step 100/150/200 的曲线只用于本次 synthetic 流程内部推演。
4. 固定 200-task 子集上的结果不能外推到完整 UserBench，也不能替代正式 matched ablation 或真实 checkpoint selection。
