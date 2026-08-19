# 200-Task 模型比较与场景 GRPO 训练曲线归档

## 范围与结论

本归档严格比较六个对象：Qwen3.5-2B-baseline、SFT-merged，以及本次 checkpoint 场景流水线的 50、100、150、200-step checkpoint。历史 GRPO checkpoint 不进入比较表、曲线结论或模型排序，仅在 manifest 的排除清单中声明。

checkpoint step 200 在固定 200-task 子集上的 completion 为 **31.4167%**，相对 SFT-merged 的 28.0000% 提升 3.4167 个百分点。该结果来自确定性checkpoint 场景推演，并非真实 GPU 训练、vLLM rollout 或 UserBench 用户模拟器观测。

## 六对象统一比较

Qwen 与 SFT 的横向指标采用 `current-reward-v3-comparable-v1` replay；其 native Reward-v2 原始文件仍完整保留。checkpoint 使用本地 Reward-v3 summary。

| 对象 | completion | submission | preference | search | phase | efficiency | terminal | guard rejection |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-2B-baseline | 1.9583% | 19.6667% | 12.1417% | 32.7917% | 30.1330% | 87.9031% | 0.0090 | 58.0586% |
| SFT-merged | 28.0000% | 51.5833% | 49.6222% | 58.3750% | 85.9299% | 50.9688% | 0.2934 | 8.1715% |
| checkpoint-50-step | 29.4167% | 56.3333% | 54.4236% | 66.9583% | 89.6479% | 42.9250% | 0.3087 | 6.3293% |
| checkpoint-100-step | 30.5833% | 60.4167% | **58.7278%** | **73.0417%** | 84.2282% | 40.4417% | 0.3180 | 8.5393% |
| checkpoint-150-step | 29.9167% | 54.7917% | 53.1444% | 63.8333% | **92.3590%** | 54.1625% | 0.3152 | **6.0260%** |
| checkpoint-200-step | **31.4167%** | 58.0833% | 51.5181% | 61.3333% | 87.4658% | **56.6917%** | **0.3233** | 8.0779% |

| 对象 | actor attempts | environment steps | penalty | answer quality | invalid | exact / semantic repeats | full / partial / wrong / no-answer |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-2B-baseline | 16.730 | 5.705 | 0.1139 | 0.0180 | 11.025 | 2.280 / 0.020 | 21 / 37 / 0 / 142 |
| SFT-merged | 15.840 | 14.225 | 0.0657 | 0.2543 | 1.615 | 3.995 / 3.190 | 51 / 109 / 0 / 40 |
| checkpoint-50-step | 17.565 | 16.460 | 0.0735 | 0.2813 | 1.000 | 3.000 / 1.580 | 15 / 96 / 55 / 34 |
| checkpoint-100-step | 18.115 | 16.545 | 0.0850 | 0.2958 | 1.000 | 3.000 / 1.805 | 17 / 98 / 54 / 31 |
| checkpoint-150-step | 16.660 | 15.670 | 0.0668 | 0.2874 | 1.000 | 2.760 / 1.595 | 15 / 98 / 56 / 31 |
| checkpoint-200-step | 16.965 | 15.570 | 0.0765 | **0.3050** | 1.000 | 2.715 / 1.695 | 18 / 100 / 52 / 30 |

## 非单调 checkpoint 变化

- step 100 是 preference/search 峰值：覆盖和提交最积极，但轨迹更长、效率最低，guard 与 penalty 上升。
- step 150 是 phase/guard 峰值：控制状态机最稳定，completion 却回撤到 29.9167%，说明协议正确并不自动产生更多正确答案。
- step 200 通过更高 completion、answer quality 和 efficiency 达到 31.4167%；preference/search/phase 没有保持峰值，guard 和 penalty 也不是全程最低。
- 100→200 期间获得 128 个新正确 aspect、失去 95 个，只有 48 个保持正确；full task 集合只有 1 个交集。这反映的是任务级成功集合换挡，而非旧能力集合的单调累积。

## step 1–200 训练曲线

| 窗口 | completion | preference | search | guard | phase | efficiency | terminal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1–40 | 0.2091 | 0.2680 | 0.5793 | 0.0470 | 0.8736 | 0.4910 | 0.2122 |
| 41–60 | 0.2106 | 0.3507 | 0.6700 | 0.0698 | 0.8423 | 0.4395 | 0.2183 |
| 61–100 | 0.2168 | 0.3756 | 0.7273 | 0.0968 | 0.8146 | 0.4110 | 0.2231 |
| 101–140 | 0.2370 | 0.3173 | 0.6558 | 0.0668 | 0.8639 | 0.4584 | 0.2403 |
| 141–170 | 0.2485 | 0.3601 | 0.6418 | 0.0615 | 0.8686 | 0.5463 | 0.2547 |
| 171–200 | 0.2619 | 0.2994 | 0.6158 | 0.0818 | 0.8589 | 0.5493 | 0.2602 |

逐 step 标准差较大：completion 0.1008、preference 0.1219、search 0.0913、guard 0.0387、phase 0.0532、efficiency 0.0702、terminal 0.0905。因此窗口均值上升不代表单步曲线平滑。

entropy 总体从早期窗口的 1.4432 降至后期 1.1036，KL 从 0.0143 升至 0.1144；clip fraction 在 0.1542–0.2265 的窗口均值间震荡，gradient norm 也有阶段性反弹。turn-credit 的 conservation error 约为 `1.5e-10`，保持 token-weighted 守恒；它只重分配 advantage，不改变 terminal Reward。dynamic sampling 的 constant-reward groups 均值由 0.15 增至 0.97，说明后期越来越多 group 缺少有效相对信号。

Reward v3 中 completion 权重为 3.00，显著高于 preference 0.20、phase 0.08、search 0.06、quality 0.04 和 efficiency 0.02；这解释了为何辅助指标回撤时，step 200 的 terminal reward 仍能升至 0.3233。

## 归档结构

- `raw/qwen35_2b_baseline/`、`raw/sft_merged/`：真实评测原始复制与 task 级结果。
- `raw/checkpoints/step_{50,100,150,200}/`：checkpoint 原始结果。
- `raw/checkpoints/training/`：step 1–200 训练指标。
- `comparison/`：六对象统一表、Qwen/SFT replay 投影、task ID 顺序及 summary 重算报告。
- `curves/`：训练曲线 JSONL、CSV 与统计摘要。
- `ARCHIVE_MANIFEST.json`：源路径、归档路径、文件大小、SHA-256、模型角色与 real/scenario 标记。
- `SHA256SUMS.txt`：复制文件和派生结果的校验清单。

独立 scenario validator、summary 重算、200-step 曲线连续性和原始 SHA-256 均已通过。checkpoint 场景结果只用于流程与逻辑推演，不应表述为实际训练观测。
