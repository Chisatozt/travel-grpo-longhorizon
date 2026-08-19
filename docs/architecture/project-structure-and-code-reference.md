# 项目文件结构、运行框架与代码参考

> 本文覆盖项目自有的 `src/`、`scripts/`、`configs/` 和 `tests/`。`environments/UserBench/` 是 pinned third-party snapshot，按仓库规范不修改，也不将其内部文件复制进本索引。

## 1. 项目定位与执行主链

本项目是一个以 UserBench 旅游助手为对象的可审计 agentic post-training 工程。代码分成四类边界：

1. **数据与契约**：固定 task split、teacher trajectory、SFT admission、GRPO parquet 和评测 artifact。
2. **运行时环境**：Actor 可见的工具/phase/guard 与 UserBench session；隐藏的 correct/best/preference 标签只进入 terminal reward。
3. **训练与模型**：teacher collection、action-only LoRA SFT、merge、veRL 0.8 multi-turn GRPO、dynamic sampling 和可选 turn-credit。
4. **评测与报告**：frozen rollout、fixed-denominator summary、checkpoint selection、阶段比较和可审计归档。

```text
pinned UserBench + fixed split
        │
        ├── scripts/data/ + src/travel_grpo/data/
        │       └── train/validation/test task manifests
        │
        ├── teacher simulator ──> src/travel_grpo/training/sft/collection.py
        │       └── Gold/Silver admitted trajectories
        │               └── action-only LoRA SFT ──> merged SFT model
        │
        ├── merged SFT model + GRPO parquet
        │       └── src/travel_grpo/training/grpo/adapter/
        │              ├── project env/session/tools
        │              ├── Travel Reward v3 terminal score
        │              ├── dynamic group sampling
        │              └── optional turn-credit advantage routing
        │
        └── frozen actor + independent eval simulator
                └── rollout artifacts → metrics/summary → checkpoint selection/comparison
```

## 2. 运行边界与所有权

| 边界 | 负责代码 | 输入 | 输出/约束 |
|---|---|---|---|
| pinned UserBench | `environments/UserBench/` | task/session API | 第三方快照；日常不得编辑 |
| 数据 | `src/travel_grpo/data/`, `scripts/data/` | snapshot、JSONL、TOML | 固定顺序的 task manifests、recovery targets |
| Actor | `src/travel_grpo/models/`, `prompts/`, `protocols/` | visible messages/tool schema | assistant tool call；不暴露 hidden labels |
| training simulator | `configs/interaction_config/simulator_train.yaml` + 外部进程 | GRPO rollout requests | 独立 simulator process |
| collection simulator | `simulator_collection.yaml` + teacher scripts | teacher requests | SFT trajectory |
| evaluation simulator | `simulator_eval.yaml` + eval scripts | frozen model/task | 200/471-task result artifacts |
| environment/session | `src/travel_grpo/envs/` | Actor action + visible observation | public state、ledger、terminal reward |
| SFT | `src/travel_grpo/training/sft/`, `scripts/train/sft/` | admitted trajectories | LoRA adapter、merged model |
| GRPO | `src/travel_grpo/training/grpo/`, `scripts/train/grpo/` | parquet、SFT model、veRL | checkpoints、validation rollouts |
| evaluation | `src/travel_grpo/evaluation/`, `scripts/eval/` | actor + tasks | JSONL、summary、selection、comparison |

## 3. 顶层目录

```text
configs/       可复现配置：data、interaction、tool、SFT、GRPO、eval
src/           travel_grpo 可安装 Python package
scripts/       按 data/eval/train/simulate 分组的 CLI 与薄封装
 tests/        单元、契约、导入、smoke/integration 测试
data/          固定 task、teacher trajectory、GRPO parquet 等输入派生物
environments/  pinned UserBench third-party snapshot（不修改）
docs/          架构、Reward、训练、评测与实验归档说明
outputs/       被忽略的模型、rollout、日志、cache、simulation 和 analysis 产物
pyproject.toml 包元数据、可选依赖、pytest 和 setuptools 配置
```

## 4. Python 包分层

### `src/travel_grpo/data/`

任务划分和 recovery 数据构建。`userbench.py` 是固定 task composition、hash、顺序和 split 的核心；`recovery/` 子包负责从失败轨迹提取可恢复边界并生成 recovery SFT target；flat `recovery_boundaries.py` / `recovery_targets.py` 保留兼容导入路径。

### `src/travel_grpo/envs/`

环境集成层。`public_control.py` 只能读取 Actor 可见输入，负责有限状态机和 guard；`userbench_context.py` 维护 session ledger；`reward.py` 计算 terminal-only Reward v3；`userbench_wrapper.py` 把 pinned snapshot 包装成稳定生命周期；`userbench_tools.py` 定义 tool schema；`observation.py` 和 `userbench_interaction.py` 负责 observation/调用适配。

### `src/travel_grpo/evaluation/`

评测数据平面和控制平面。`rollout.py`/`runner.py` 执行 frozen rollout，`metrics.py`/`summary.py` 做固定分母聚合，`validation.py` 检查输入，`checkpoint_selection.py` 做 SFT guard 后的候选选择，`comparison.py`/`artifacts.py` 负责横向报告和持久化。

### `src/travel_grpo/training/`

SFT 和 GRPO 训练逻辑。SFT 子包负责 contracts、planning、collection、action-only dataset 和 recovery；GRPO 子包负责 veRL adapter、preflight、launcher、dynamic sampling、GRPO data；`trajectory/turn_credit.py` 负责 trainer-only credit routing。

### `src/travel_grpo/models/`, `prompts/`, `protocols/`, `utils/`

模型 API client、vLLM policy、Actor prompt、消息协议、I/O/hash/logging 复用工具。它们不直接拥有 UserBench hidden labels。

## 5. 配置与入口文件

| 文件 | 功能 |
|---|---|

| `configs/data/dataset_split.toml` | canonical 数据划分、seed、比例和输出路径。 |
| `configs/eval/eval_userbench.yaml` | frozen UserBench evaluation 的模型、simulator、采样和输出参数。 |
| `configs/interaction_config/agent_loop.yaml` | agent loop 公共默认和 tool-call 行为。 |
| `configs/interaction_config/simulator_collection.yaml` | teacher collection simulator 边界。 |
| `configs/interaction_config/simulator_eval.yaml` | evaluation simulator 边界。 |
| `configs/interaction_config/simulator_train.yaml` | GRPO training simulator 边界。 |
| `configs/interaction_config/userbench.yaml` | UserBench task/session/environment 基础参数。 |
| `configs/interaction_config/userbench_interaction.yaml` | 对话交互和 context 参数。 |
| `configs/tool_config/userbench_tools.yaml` | Actor 可见工具 schema、参数和返回格式。 |
| `configs/train/grpo/grpo.yaml` | veRL 0.8 GRPO profile、LoRA、rollout、dynamic sampling、turn-credit。 |
| `configs/train/grpo/uv-overrides.txt` | 项目自有实现文件，负责 `configs/train/grpo/uv-overrides.txt` 对应阶段的逻辑。 |
| `configs/train/grpo/vanilla_grpo.yaml` | 最小 vanilla GRPO profile，用于兼容性/冒烟。 |
| `configs/train/sft/sft_lora.yaml` | 通用 LoRA SFT trainer 参数。 |
| `configs/train/sft/sft_stage1_lora.yaml` | stage-1 prefix/action-only SFT 参数。 |
| `configs/train/sft/sft_stage2_lora.yaml` | stage-2 trajectory SFT 参数。 |
| `configs/train/sft/teacher_collection.yaml` | teacher collection 模型、采样、admission 和 simulator 参数。 |
| `configs/train/sft/teacher_smoke_batches.json` | teacher collection smoke test 的 batch 配置。 |
| `scripts/eval/run_baseline.sh` | Shell/配置入口：scripts/eval/run_baseline.sh |
| `scripts/eval/run_evaluation.sh` | Shell/配置入口：scripts/eval/run_evaluation.sh |
| `scripts/setup.sh` | Shell/配置入口：scripts/setup.sh |
| `scripts/train/grpo/export_actor.sh` | Shell/配置入口：scripts/train/grpo/export_actor.sh |
| `scripts/train/grpo/run_grpo.sh` | Shell/配置入口：scripts/train/grpo/run_grpo.sh |
| `scripts/train/grpo/run_grpo_from_sft.sh` | Shell/配置入口：scripts/train/grpo/run_grpo_from_sft.sh |
| `scripts/train/grpo/run_vanilla.sh` | Shell/配置入口：scripts/train/grpo/run_vanilla.sh |
| `scripts/train/sft/launch_two_stage_sft.sh` | Shell/配置入口：scripts/train/sft/launch_two_stage_sft.sh |
| `scripts/train/sft/run_sft.sh` | Shell/配置入口：scripts/train/sft/run_sft.sh |
| `scripts/vllm_server/actor.sh` | Shell/配置入口：scripts/vllm_server/actor.sh |
| `scripts/vllm_server/train_user_simulator.sh` | Shell/配置入口：scripts/vllm_server/train_user_simulator.sh |

## 6. 测试布局

测试按契约边界分组，而不是按实现文件简单镜像：

- `tests/test_reward.py`、`test_public_control.py`、`test_public_phase_guard.py`、`test_public_rendering_fix.py`：Reward v3、public phase、guard 和 hidden-state isolation。
- `tests/test_userbench_context.py`、`test_userbench_tools.py`、`test_userbench_wrapper.py`：环境 session、工具 schema、wrapper 生命周期。
- `tests/test_sft_collection.py`、`test_sft_dataset.py`、`test_teacher_api.py`、`test_recovery_*`：teacher trajectory、SFT admission、dataset 和 recovery target。
- `tests/test_grpo_adapter.py`、`test_grpo_pipeline.py`、`test_public_entrypoints.py`：veRL adapter、launcher/preflight 和公共入口。
- `tests/test_evaluation_*`、`test_inference_gate.py`：rollout、summary、validation、inference gate。
- `tests/test_dataset_split.py`、`test_repository_layout.py`、`test_imports.py`：可复现划分、仓库边界和安装导入。
- `tests/smoke/`：明确 opt-in 的真实 UserBench/veRL 环境冒烟检查。

## 7. 函数级参考索引

下面按文件列出类、函数和方法的输入、输出以及职责。源文件中新增的 `[项目注释]` 行提供就地速查；本节提供跨文件导航。类型注解是源码中的 declared interface，未标注的返回值以实际分支为准。

### `src/travel_grpo/__init__.py`

职责：UserBench-based long-horizon travel-agent post-training project.

代码行数：3。

本文件主要提供常量、类型或配置，没有可调用函数。

### `src/travel_grpo/data/__init__.py`

职责：UserBench task loading and reproducible project-level dataset splits.

代码行数：79。

本文件主要提供常量、类型或配置，没有可调用函数。

### `src/travel_grpo/data/recovery/__init__.py`

职责：Recovery-stage data extraction and target construction.

代码行数：1。

本文件主要提供常量、类型或配置，没有可调用函数。

### `src/travel_grpo/data/recovery/boundaries.py`

职责：Extract public recovery-boundary contexts from existing trajectories.

代码行数：1450。

类型：`SourceSpec`、`_Event`、`_Candidate`、`_PublicReplay`。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `_normalise_text` (L157) | `value`: str | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_sha256_file` (L164) | `path`: Path | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_relative_path` (L175) | `path`: Path；`root`: Path | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_jsonl_rows` (L182) | `path`: Path | 标注返回 `Iterable[tuple[int, dict[str, Any]]]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `load_task_split_map` (L194) | `project_root`: str \| Path | 标注返回 `dict[str, dict[str, Any]]`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `discover_sources` (L233) | `project_root`: str \| Path | 标注返回 `list[SourceSpec]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_is_failure_record` (L287) | `record`: Mapping[str, Any] | 标注返回 `bool`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_safe_json` (L306) | `value`: Any | 标注返回 `Any`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_action_from_message` (L320) | `message`: Mapping[str, Any] | 标注返回 `UserBenchAction \| None`；具体值由分支决定。 | 把协议/状态数据渲染成可见文本或消息。 |
| `events_from_messages` (L340) | `messages`: Sequence[Mapping[str, Any]] | 标注返回 `list[_Event]`；具体值由分支决定。 | 把协议/状态数据渲染成可见文本或消息。 |
| `parse_grpo_transcript` (L371) | `input_text`: str；`output_text`: str | 标注返回 `list[dict[str, Any]]`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `_initial_user_message` (L430) | `messages`: Sequence[Mapping[str, Any]] | 标注返回 `str`；具体值由分支决定。 | 把协议/状态数据渲染成可见文本或消息。 |
| `_advance_if_terminal` (L440) | `state`: PublicControlState | 标注返回 `PublicControlState`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_PublicReplay.__init__` (L471) | `initial_user_message`: str | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_PublicReplay._target_aspect` (L479) | `action`: UserBenchAction | 标注返回 `str \| None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_PublicReplay.prepare` (L488) | `action`: UserBenchAction | 标注返回 `PublicControlState`；具体值由分支决定。 | 根据配置和中间状态构建项目产物。 |
| `_PublicReplay.before_event` (L515) | 无显式业务参数 | 标注返回 `PublicControlState`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_PublicReplay.apply` (L522) | `action`: UserBenchAction；`feedback`: str \| None | 标注返回 `PublicControlState`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_phase_label` (L537) | `state`: PublicControlState | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `public_state_payload` (L544) | `state`: PublicControlState | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `_is_no_preference` (L573) | `text`: str \| None | 标注返回 `bool`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_observation_kind` (L583) | `action`: UserBenchAction；`feedback`: str \| None | 标注返回 `PublicObservationKind \| None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_target_aspect_from_action` (L595) | `action`: UserBenchAction；`aspects`: Sequence[str] | 标注返回 `str \| None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_candidate` (L608) | `task_id`: str；`boundary_type`: str；`policy_version`: str；`messages`: Sequence[Mapping[str, Any]]；`state`: PublicControlState；`provenance`: Mapping[str, Any]；`composition`: str；`project_split`: str；`replay_ok`: bool | 标注返回 `_Candidate`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `extract_message_boundaries` (L640) | `task_id`: str；`messages`: Sequence[Mapping[str, Any]]；`policy_version`: str；`provenance`: Mapping[str, Any]；`composition`: str；`project_split`: str | 标注返回 `list[_Candidate]`；具体值由分支决定。 | 把协议/状态数据渲染成可见文本或消息。 |
| `_split_for_task` (L808) | `task_id`: str；`assignments`: Mapping[str, Mapping[str, Any]]；`source`: SourceSpec | 标注返回 `tuple[str, str, bool]`；具体值由分支决定。 | 按固定约束拆分、采样或选择数据。 |
| `_policy_version` (L837) | `record`: Mapping[str, Any]；`source`: SourceSpec | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_provenance` (L854) | `source`: SourceSpec；`root`: Path；`line`: int \| None；`record_index`: int \| None；`extra`: Mapping[str, Any] \| None | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_record_to_context` (L882) | `candidate`: _Candidate；`dedupe_key`: str | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_dedupe_candidates` (L916) | `candidates`: Sequence[_Candidate] | 标注返回 `tuple[list[dict[str, Any]], int]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_load_record_at` (L948) | `path`: Path；`line`: int | 标注返回 `Mapping[str, Any] \| None`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `_probe_metadata` (L961) | `root`: Path | 标注返回 `dict[tuple[str, str, int], dict[str, Any]]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_load_ab_source_messages` (L991) | `root`: Path；`metadata`: Mapping[str, Any] | 标注返回 `tuple[list[dict[str, Any]], Mapping[str, Any]] \| None`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `_find_event` (L1006) | `events`: Sequence[_Event]；`metadata`: Mapping[str, Any] | 标注返回 `_Event \| None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_synthetic_fallback_messages` (L1025) | `base_messages`: Sequence[Mapping[str, Any]]；`original_search`: Mapping[str, Any]；`retry_search`: Mapping[str, Any] \| None；`fallback_text`: str | 标注返回 `list[dict[str, Any]]`；具体值由分支决定。 | 把协议/状态数据渲染成可见文本或消息。 |
| `_synthetic_fallback_messages.append_call` (L1036) | `parameters`: Mapping[str, Any]；`index`: int | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_extract_ab_file` (L1078) | `source`: SourceSpec；`root`: Path；`record`: Mapping[str, Any]；`line`: int；`assignments`: Mapping[str, Mapping[str, Any]]；`metadata_index`: Mapping[tuple[str, str, int], Mapping[str, Any]] | 标注返回 `list[_Candidate]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_source_inventory_entry` (L1229) | `source`: SourceSpec；`root`: Path | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `extract_recovery_boundaries` (L1243) | `project_root`: str \| Path；`sources`: Sequence[SourceSpec] \| None | 标注返回 `tuple[list[dict[str, Any]], dict[str, Any]]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `write_extraction` (L1407) | `records`: Sequence[Mapping[str, Any]]；`manifest`: Mapping[str, Any]；`output_dir`: str \| Path | 标注返回 `tuple[Path, Path]`；具体值由分支决定。 | 序列化并持久化内部结果。 |

### `src/travel_grpo/data/recovery/targets.py`

职责：Construct and validate one-step recovery targets.

代码行数：1014。

类型：`_SourceAction`、`TargetDecision`。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `_normalise` (L130) | `value`: str | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_relative` (L137) | `root`: Path；`path`: Path | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_jsonl_row` (L147) | `path`: Path；`line`: int | 标注返回 `Mapping[str, Any] \| None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_load_messages` (L160) | `path`: Path；`root`: Path；`line`: int \| None | 标注返回 `list[dict[str, Any]] \| None`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `_source_for_provenance` (L197) | `root`: Path；`provenance`: Mapping[str, Any] | 标注返回 `tuple[list[dict[str, Any]], dict[str, Any]] \| None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_message_equal` (L220) | `left`: Mapping[str, Any]；`right`: Mapping[str, Any] | 标注返回 `bool`；具体值由分支决定。 | 把协议/状态数据渲染成可见文本或消息。 |
| `_prefix_matches` (L252) | `context_messages`: Sequence[Mapping[str, Any]]；`source_messages`: Sequence[Mapping[str, Any]]；`end`: int | 标注返回 `bool`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_next_source_action` (L266) | `context`: Mapping[str, Any]；`root`: Path | 标注返回 `_SourceAction \| None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_last_action` (L290) | `context`: Mapping[str, Any]；`choice`: ActionChoice \| None | 标注返回 `UserBenchAction \| None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_public_aspects` (L304) | `context`: Mapping[str, Any] | 标注返回 `tuple[str, ...]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_state` (L316) | `context`: Mapping[str, Any] | 标注返回 `Mapping[str, Any]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_current_aspect` (L324) | `context`: Mapping[str, Any] | 标注返回 `str \| None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_human_aspect` (L332) | `aspect`: str | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_field_prompt` (L339) | `aspect`: str；`field`: str | 标注返回 `str`；具体值由分支决定。 | 把协议/状态数据渲染成可见文本或消息。 |
| `_search_prompt` (L348) | `aspect`: str | 标注返回 `str`；具体值由分支决定。 | 把协议/状态数据渲染成可见文本或消息。 |
| `_target_message` (L355) | `action`: UserBenchAction；`target_id`: str | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 把协议/状态数据渲染成可见文本或消息。 |
| `_generated_action` (L384) | `choice`: str；`content`: str | 标注返回 `UserBenchAction`；具体值由分支决定。 | 根据配置和中间状态构建项目产物。 |
| `_asked_fields` (L394) | `context`: Mapping[str, Any]；`aspect`: str | 标注返回 `set[str]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_next_open_aspect` (L410) | `context`: Mapping[str, Any] | 标注返回 `str \| None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_target_for_boundary` (L425) | `context`: Mapping[str, Any]；`root`: Path | 标注返回 `tuple[UserBenchAction \| None, dict[str, Any], list[str]]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_target_hidden_key_hits` (L548) | `value`: Any；`path`: str | 标注返回 `list[str]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `validate_target` (L561) | `context`: Mapping[str, Any]；`action`: UserBenchAction | 标注返回 `tuple[bool, list[str], dict[str, Any]]`；具体值由分支决定。 | 校验输入、状态或产物，输出诊断或抛出异常。 |
| `construct_target` (L633) | `context`: Mapping[str, Any]；`project_root`: str \| Path | 标注返回 `TargetDecision`；具体值由分支决定。 | 根据配置和中间状态构建项目产物。 |
| `_target_record` (L689) | `context`: Mapping[str, Any]；`decision`: TargetDecision | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_read_contexts` (L704) | `path`: Path | 标注返回 `list[dict[str, Any]]`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `_write_jsonl` (L718) | `path`: Path；`records`: Sequence[Mapping[str, Any]] | 标注返回 `None`；具体值由分支决定。 | 序列化并持久化内部结果。 |
| `build_target_dataset` (L724) | `contexts`: Sequence[Mapping[str, Any]]；`project_root`: str \| Path | 标注返回 `tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]`；具体值由分支决定。 | 根据配置和中间状态构建项目产物。 |
| `build_target_dataset.finalize` (L761) | `value`: Counter[str] | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `write_target_dataset` (L834) | `train`: Sequence[Mapping[str, Any]]；`validation`: Sequence[Mapping[str, Any]]；`rejected`: Sequence[Mapping[str, Any]]；`manifest`: Mapping[str, Any]；`output_dir`: str \| Path | 标注返回 `dict[str, Path]`；具体值由分支决定。 | 序列化并持久化内部结果。 |
| `build_targets_from_boundary_file` (L863) | `context_path`: str \| Path；`project_root`: str \| Path；`output_dir`: str \| Path | 标注返回 `tuple[dict[str, Path], dict[str, Any]]`；具体值由分支决定。 | 根据配置和中间状态构建项目产物。 |
| `build_targets_from_boundary_file.finalize` (L941) | `value`: Counter[str] | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |

### `src/travel_grpo/data/recovery_boundaries.py`

职责：Compatibility facade for :mod:`travel_grpo.data.recovery.boundaries`.

代码行数：11。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `__getattr__` (L10) | `name`: str | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |

### `src/travel_grpo/data/recovery_targets.py`

职责：Compatibility facade for :mod:`travel_grpo.data.recovery.targets`.

代码行数：5。

本文件主要提供常量、类型或配置，没有可调用函数。

### `src/travel_grpo/data/userbench.py`

职责：Load pinned UserBench Parquet tasks and build disjoint project splits.

代码行数：870。

类型：`DatasetSplitError`、`CompositionSpec`、`SplitSpec`、`LoadedTaskSet`、`SplitBundle`。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `_require_pyarrow` (L113) | 无显式业务参数 | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_require_string` (L128) | `mapping`: Mapping[str, Any]；`key`: str | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_require_non_negative_int` (L141) | `mapping`: Mapping[str, Any]；`key`: str | 标注返回 `int`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `load_split_spec` (L150) | `path`: str \| Path | 标注返回 `SplitSpec`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `_sha256_file` (L225) | `path`: Path | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_canonical_json` (L236) | `record`: Mapping[str, Any] | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `compute_jsonl_sha256` (L243) | `records`: Sequence[Mapping[str, Any]] | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_nested` (L256) | `mapping`: Mapping[str, Any]；`path`: Sequence[str]；`context`: str | 标注返回 `Any`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_validate_source_row` (L271) | `row`: Mapping[str, Any]；`task_data`: Mapping[str, Any]；`composition`: str；`upstream_split`: str；`source_path`: str；`source_row_index`: int | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 校验输入、状态或产物，输出诊断或抛出异常。 |
| `load_onechoice_tasks` (L382) | `source_root`: str \| Path；`composition`: str；`upstream_split`: str；`expected_count`: int \| None；`source_label`: str | 标注返回 `LoadedTaskSet`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `_stable_task_key` (L460) | `task_id`: str；`seed`: str | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_as_output_record` (L467) | `record`: Mapping[str, Any] | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_pairwise_intersections` (L474) | `records`: Mapping[str, Sequence[Mapping[str, Any]]] | 标注返回 `dict[str, int]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_read_embedded_source` (L490) | `source_root`: Path；`expected_commit`: str | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `build_dataset_splits` (L503) | `spec`: SplitSpec；`source_root`: str \| Path | 标注返回 `SplitBundle`；具体值由分支决定。 | 根据配置和中间状态构建项目产物。 |
| `_write_jsonl` (L639) | `path`: Path；`records`: Sequence[Mapping[str, Any]] | 标注返回 `None`；具体值由分支决定。 | 序列化并持久化内部结果。 |
| `_temporary_path` (L649) | `target`: Path | 标注返回 `Path`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `write_dataset_splits` (L657) | `bundle`: SplitBundle；`output_root`: str \| Path；`force`: bool | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 序列化并持久化内部结果。 |
| `_read_jsonl` (L745) | `path`: Path | 标注返回 `list[dict[str, Any]]`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `_safe_artifact_path` (L769) | `output_root`: Path；`relative`: str | 标注返回 `Path`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `verify_dataset_splits` (L780) | `spec`: SplitSpec；`source_root`: str \| Path；`output_root`: str \| Path | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 校验输入、状态或产物，输出诊断或抛出异常。 |

### `src/travel_grpo/envs/__init__.py`

职责：Project-owned UserBench environment integration boundary.

代码行数：93。

本文件主要提供常量、类型或配置，没有可调用函数。

### `src/travel_grpo/envs/observation.py`

职责：Actor-safe projections of TravelGym observations and step results.

代码行数：86。

类型：`UserBenchObservationError`、`UserBenchObservation`、`UserBenchStepResult`。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `UserBenchObservation.from_upstream` (L30) | `observation`: Mapping[str, Any] | 标注返回 `UserBenchObservation`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `UserBenchObservation.to_tool_text` (L66) | 无显式业务参数 | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `UserBenchStepResult.done` (L85) | 无显式业务参数 | 标注返回 `bool`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |

### `src/travel_grpo/envs/public_control.py`

职责：Public-only control state for a UserBench trajectory.

代码行数：1298。

类型：`PublicControlError`、`RecoveryMode`、`PublicAspectStatus`、`PublicObservationKind`、`PublicObservation`、`PublicAspectState`、`PublicControlState`、`PublicControlEvent`。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `_normalise_public_text` (L94) | `value`: str | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_hint_pattern` (L101) | `hint`: str | 标注返回 `re.Pattern[str]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `extract_public_aspects` (L109) | `initial_user_message`: str | 标注返回 `tuple[str, ...]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `PublicObservation.__post_init__` (L147) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `PublicObservation.is_fallback` (L164) | 无显式业务参数 | 标注返回 `bool`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `PublicObservation.is_normal_search` (L174) | 无显式业务参数 | 标注返回 `bool`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `PublicObservation.signature` (L178) | 无显式业务参数 | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_coerce_choice` (L189) | `choice`: ActionChoice \| str \| None | 标注返回 `ActionChoice \| None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `classify_public_observation` (L202) | `feedback`: str；`choice`: ActionChoice \| str \| None | 标注返回 `PublicObservation`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `public_action_signature` (L238) | `action`: UserBenchAction | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `public_search_signature` (L246) | `action`: UserBenchAction | 标注返回 `str \| None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `public_semantic_signature` (L256) | `action`: UserBenchAction；`public_aspects`: Sequence[str] | 标注返回 `tuple[str, str] \| None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_validate_public_aspects` (L275) | `aspects`: Sequence[str] | 标注返回 `tuple[str, ...]`；具体值由分支决定。 | 校验输入、状态或产物，输出诊断或抛出异常。 |
| `_aspect_mentioned_in_content` (L296) | `content`: str；`public_aspects`: Sequence[str] | 标注返回 `str \| None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_option_aspect` (L310) | `option_id`: str；`public_aspects`: Sequence[str] | 标注返回 `str \| None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `PublicAspectState.__post_init__` (L336) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `PublicControlState.__post_init__` (L396) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `PublicControlState.public_aspects` (L436) | 无显式业务参数 | 标注返回 `tuple[str, ...]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `PublicControlState.open_aspects` (L443) | 无显式业务参数 | 标注返回 `tuple[str, ...]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `PublicControlState.current` (L454) | 无显式业务参数 | 标注返回 `PublicAspectState \| None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `PublicControlState.phase` (L464) | 无显式业务参数 | 标注返回 `RecoveryMode`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `PublicControlState.answered_aspects` (L493) | 无显式业务参数 | 标注返回 `tuple[str, ...]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `PublicControlState.blocked_aspects` (L503) | 无显式业务参数 | 标注返回 `tuple[str, ...]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `PublicControlState.answered_count` (L516) | 无显式业务参数 | 标注返回 `int`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `PublicControlState.blocked_count` (L523) | 无显式业务参数 | 标注返回 `int`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `PublicControlState.all_aspects_terminal` (L530) | 无显式业务参数 | 标注返回 `bool`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `PublicControlEvent.__post_init__` (L544) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `new_public_control_state` (L553) | `initial_user_message`: str；`no_progress_threshold`: int | 标注返回 `PublicControlState`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_validate_threshold` (L572) | `value`: Any | 标注返回 `int`；具体值由分支决定。 | 校验输入、状态或产物，输出诊断或抛出异常。 |
| `_replace_aspect` (L581) | `state`: PublicControlState；`updated`: PublicAspectState | 标注返回 `PublicControlState`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `mark_public_preference_complete` (L593) | `state`: PublicControlState；`aspect`: str \| None | 标注返回 `PublicControlState`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_progress_reset` (L625) | `state`: PublicControlState | 标注返回 `PublicControlState`；具体值由分支决定。 | 清理资源或恢复状态边界。 |
| `_progress_increment` (L632) | `state`: PublicControlState | 标注返回 `PublicControlState`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_target_aspect` (L658) | `state`: PublicControlState；`action`: UserBenchAction | 标注返回 `str \| None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `normalize_public_query` (L679) | `value`: str | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `public_query_signature` (L698) | `action`: UserBenchAction | 标注返回 `str \| None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `is_substantive_query_change` (L708) | `previous_query`: str；`candidate_query`: str | 标注返回 `bool`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_answer_option_ids` (L726) | `action`: UserBenchAction | 标注返回 `list[str]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `validate_public_action` (L730) | `state`: PublicControlState；`action`: UserBenchAction | 标注返回 `str \| None`；具体值由分支决定。 | 校验输入、状态或产物，输出诊断或抛出异常。 |
| `note_public_non_progress` (L817) | `state`: PublicControlState | 标注返回 `PublicControlState`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_record_submitted_answers` (L835) | `state`: PublicControlState；`action`: UserBenchAction | 标注返回 `PublicControlState`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_all_public_aspects_terminal` (L855) | `state`: PublicControlState | 标注返回 `bool`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_apply_search_action` (L863) | `state`: PublicControlState；`action`: UserBenchAction；`observation`: PublicObservation \| None | 标注返回 `PublicControlState`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_apply_answer_action` (L974) | `state`: PublicControlState；`action`: UserBenchAction | 标注返回 `PublicControlState`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `reduce_public_control_state` (L1008) | `state`: PublicControlState；`event`: PublicControlEvent | 标注返回 `PublicControlState`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `reduce_public_feedback` (L1094) | `state`: PublicControlState；`action`: UserBenchAction；`feedback`: str | 标注返回 `PublicControlState`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `advance_public_aspect` (L1110) | `state`: PublicControlState | 标注返回 `PublicControlState`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_render_phase_label` (L1167) | `state`: PublicControlState | 标注返回 `str`；具体值由分支决定。 | 把协议/状态数据渲染成可见文本或消息。 |
| `_render_allowed_tool_calls` (L1179) | `state`: PublicControlState | 标注返回 `str`；具体值由分支决定。 | 把协议/状态数据渲染成可见文本或消息。 |
| `_render_constraint` (L1203) | `state`: PublicControlState | 标注返回 `str \| None`；具体值由分支决定。 | 编排训练、采集、评测或 replay 流程。 |
| `render_actor_control_info` (L1230) | `state`: PublicControlState | 标注返回 `str`；具体值由分支决定。 | 把协议/状态数据渲染成可见文本或消息。 |

### `src/travel_grpo/envs/reward.py`

职责：Deterministic terminal reward for UserBench travel trajectories.

代码行数：452。

类型：`UserBenchRewardError`、`RawRewardTrace`、`TravelRewardTask`、`UserBenchRewardSnapshot`。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `RawRewardTrace.append` (L36) | `value`: float | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `RawRewardTrace.values` (L48) | 无显式业务参数 | 标注返回 `tuple[float, ...]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `RawRewardTrace.total` (L55) | 无显式业务参数 | 标注返回 `float`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `TravelRewardTask.from_upstream` (L79) | `task`: Mapping[str, Any] | 标注返回 `'TravelRewardTask'`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_ratio` (L164) | `numerator`: int；`denominator`: int | 标注返回 `float`；具体值由分支决定。 | 计算奖励、指标或聚合统计。 |
| `squash_terminal_reward` (L168) | `raw_reward`: float；`negative_temperature`: float | 标注返回 `float`；具体值由分支决定。 | 计算奖励、指标或聚合统计。 |
| `scale_priority_reward` (L200) | `raw_reward`: float；`scale`: float | 标注返回 `float`；具体值由分支决定。 | 计算奖励、指标或聚合统计。 |
| `_count` (L222) | `value`: int；`name`: str | 标注返回 `int`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `compute_travel_reward` (L228) | `task`: TravelRewardTask；`answers`: Mapping[str, str]；`active_preference_ids`: AbstractSet[str]；`passive_preference_ids`: AbstractSet[str]；`searched_aspects`: AbstractSet[str]；`steps`: int；`actor_attempts`: int \| None；`max_steps`: int；`invalid_actions`: int；`exact_repeats`: int；`semantic_repeats`: int；`ambiguous_actions`: int；`unsearched_answers`: int；`wrong_answers`: int；`parallel_tool_calls`: bool；`no_tool_output`: bool；`max_steps_reached`: bool；`guard_rejections`: int；`blocked_aspects`: int；`valid_search_required_transitions`: int；`search_required_opportunities`: int；`valid_candidate_answer_transitions`: int；`candidate_answer_opportunities`: int；`valid_retry_search_transitions`: int；`retry_search_opportunities`: int；`valid_aspect_switch_transitions`: int；`aspect_switch_opportunities`: int；`reward_valid`: bool；`termination_reason`: str \| None | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 计算奖励、指标或聚合统计。 |

### `src/travel_grpo/envs/userbench_context.py`

职责：Pinned-source validation and per-trajectory UserBench context.

代码行数：1398。

类型：`UserBenchSourceError`、`UserBenchSessionError`、`_TurnLedgerSnapshot`、`EmbeddedUserBench`、`UserBenchSessionState`。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `_contains_explicit_no_preference` (L112) | `text`: str | 标注返回 `bool`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_is_complete_reward_task` (L120) | `task`: TravelRewardTask \| None | 标注返回 `bool`；具体值由分支决定。 | 计算奖励、指标或聚合统计。 |
| `_is_complete_reward_snapshot` (L138) | `snapshot`: UserBenchRewardSnapshot \| None | 标注返回 `bool`；具体值由分支决定。 | 计算奖励、指标或聚合统计。 |
| `_snapshot_matches_task` (L166) | `task`: TravelRewardTask；`snapshot`: UserBenchRewardSnapshot | 标注返回 `bool`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_evidence_transition_is_valid` (L185) | `task`: TravelRewardTask；`before`: UserBenchRewardSnapshot；`after`: UserBenchRewardSnapshot | 标注返回 `bool`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_fallback_counts` (L216) | `diagnostics`: object | 标注返回 `dict[str, int]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `validate_embedded_userbench` (L237) | `root`: str \| Path \| None | 标注返回 `EmbeddedUserBench`；具体值由分支决定。 | 校验输入、状态或产物，输出诊断或抛出异常。 |
| `_validate_embedded_userbench_cached` (L250) | `source_root`: Path | 标注返回 `EmbeddedUserBench`；具体值由分支决定。 | 校验输入、状态或产物，输出诊断或抛出异常。 |
| `UserBenchSessionState.__post_init__` (L373) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `UserBenchSessionState.done` (L416) | 无显式业务参数 | 标注返回 `bool`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `UserBenchSessionState.configure_turn_credit` (L419) | `mode`: str；`config`: TurnCreditConfig \| None | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `UserBenchSessionState._capture_turn_ledger_snapshot` (L439) | 无显式业务参数 | 标注返回 `_TurnLedgerSnapshot`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `UserBenchSessionState.begin_actor_turn` (L463) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `UserBenchSessionState._ensure_actor_turn` (L483) | 无显式业务参数 | 标注返回 `tuple[TurnEvent, _TurnLedgerSnapshot] \| None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `UserBenchSessionState._finish_actor_turn` (L497) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `UserBenchSessionState._turn_aspect_for_action` (L510) | `action`: UserBenchAction；`fallback`: str \| None | 标注返回 `str \| None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `UserBenchSessionState.complete_actor_turn_from_step` (L526) | `action`: UserBenchAction；`result`: UserBenchStepResult | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `UserBenchSessionState.reject_actor_turn` (L613) | `reason`: str；`action`: UserBenchAction \| None；`category`: str | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `UserBenchSessionState.record_turn_infrastructure_failure` (L681) | `action`: UserBenchAction \| None | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `UserBenchSessionState.finalize_pending_actor_turn` (L698) | `reason`: str | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `UserBenchSessionState.finalize_turn_credit` (L703) | `reward_report`: ABCMapping[str, Any] | 标注返回 `TurnCreditTrace`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `UserBenchSessionState.record_public_guard_rejection` (L722) | `reason`: str | 标注返回 `None`；具体值由分支决定。 | 校验输入、状态或产物，输出诊断或抛出异常。 |
| `UserBenchSessionState._record_public_phase_attempt` (L733) | `action`: UserBenchAction；`reason`: str \| None | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `UserBenchSessionState._sync_public_control_metrics` (L758) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 计算奖励、指标或聚合统计。 |
| `UserBenchSessionState._advance_public_control_if_needed` (L775) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `UserBenchSessionState.prepare_public_action` (L794) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 根据配置和中间状态构建项目产物。 |
| `UserBenchSessionState.validate_public_action` (L799) | `action`: UserBenchAction | 标注返回 `str \| None`；具体值由分支决定。 | 校验输入、状态或产物，输出诊断或抛出异常。 |
| `UserBenchSessionState.record_public_non_progress` (L813) | `reason`: str \| None | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `UserBenchSessionState._record_public_step` (L826) | `result`: UserBenchStepResult；`action`: UserBenchAction \| None | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `UserBenchSessionState.configure_stall_recovery` (L844) | `enabled`: bool；`threshold`: int | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `UserBenchSessionState.visible_answer_options` (L860) | 无显式业务参数 | 标注返回 `set[str]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `UserBenchSessionState.render_actor_feedback` (L877) | `observation_text`: str | 标注返回 `str`；具体值由分支决定。 | 把协议/状态数据渲染成可见文本或消息。 |
| `UserBenchSessionState.append_recovery_instruction` (L904) | `feedback`: str | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `UserBenchSessionState.validate_answer_only_action` (L910) | `action`: UserBenchAction | 标注返回 `str \| None`；具体值由分支决定。 | 校验输入、状态或产物，输出诊断或抛出异常。 |
| `UserBenchSessionState.recovery_instruction` (L931) | 无显式业务参数 | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `UserBenchSessionState.begin_answer_only_generation` (L941) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 计算奖励、指标或聚合统计。 |
| `UserBenchSessionState.hard_stop_stalled` (L949) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `UserBenchSessionState._stall_evidence_is_valid` (L962) | 无显式业务参数 | 标注返回 `bool`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `UserBenchSessionState._maybe_trigger_stall` (L973) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `UserBenchSessionState._record_no_progress` (L1002) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `UserBenchSessionState.record_non_progress` (L1019) | `reason`: str \| None | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `UserBenchSessionState._complete_answer_only_recovery` (L1041) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `UserBenchSessionState.record_step` (L1052) | `result`: UserBenchStepResult；`action`: UserBenchAction \| None；`snapshot`: UserBenchRewardSnapshot \| None；`count_action_repetition`: bool | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `UserBenchSessionState.reward_report` (L1215) | 无显式业务参数 | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 计算奖励、指标或聚合统计。 |
| `UserBenchSessionState.metrics` (L1301) | 无显式业务参数 | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 计算奖励、指标或聚合统计。 |
| `set_current_session` (L1367) | `session`: UserBenchSessionState | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `get_current_session` (L1374) | 无显式业务参数 | 标注返回 `UserBenchSessionState \| None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `require_current_session` (L1382) | 无显式业务参数 | 标注返回 `UserBenchSessionState`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `clear_current_session` (L1394) | `close`: bool | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |

### `src/travel_grpo/envs/userbench_interaction.py`

职责：Process-isolated configuration for UserBench user simulators.

代码行数：130。

类型：`SimulatorBoundaryError`、`SimulatorRole`、`UserSimulatorRuntime`。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `UserSimulatorRuntime.__post_init__` (L48) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `UserSimulatorRuntime.from_environment` (L65) | `role`: SimulatorRole \| str；`environ`: Mapping[str, str] \| None | 标注返回 `UserSimulatorRuntime`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `UserSimulatorRuntime.from_environment.require` (L77) | `name`: str | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `bind_user_simulator_process` (L105) | `runtime`: UserSimulatorRuntime；`environ`: MutableMapping[str, str] \| None | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_reset_user_simulator_binding_for_tests` (L127) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 清理资源或恢复状态边界。 |

### `src/travel_grpo/envs/userbench_tools.py`

职责：Provider-neutral contract for UserBench's single interaction tool.

代码行数：461。

类型：`UserBenchActionError`、`ActionChoice`、`UserBenchAction`。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `UserBenchAction.from_parameters` (L197) | `parameters`: Mapping[str, Any] | 标注返回 `UserBenchAction`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `UserBenchAction.to_environment_action` (L251) | 无显式业务参数 | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `get_interact_with_env_schema` (L306) | 无显式业务参数 | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `normalized_action_signature` (L312) | `action`: UserBenchAction | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_contains_hint` (L322) | `query`: str；`hint`: str | 标注返回 `bool`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_hint_spans` (L329) | `query`: str；`hint`: str | 标注返回 `tuple[tuple[int, int], ...]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_normalize_query_words` (L343) | `value`: str | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `action_field_matches` (L360) | `content`: str；`task_dimensions`: Sequence[str] | 标注返回 `set[tuple[str, str]]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `action_mentions_aspect` (L403) | `content`: str；`aspect`: str | 标注返回 `bool`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `semantic_action_signature` (L413) | `action`: UserBenchAction；`task_dimensions`: Sequence[str] | 标注返回 `tuple[str, str] \| None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `action_query_issue` (L424) | `action`: UserBenchAction；`task_dimensions`: Sequence[str] | 标注返回 `str \| None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `aspect_from_option_id` (L439) | `option_id`: object | 标注返回 `str \| None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `extract_visible_option_ids` (L447) | `feedback`: str | 标注返回 `set[str]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |

### `src/travel_grpo/envs/userbench_wrapper.py`

职责：Project-owned lifecycle wrapper around the pinned ``travelgym.TravelEnv``.

代码行数：500。

类型：`UserBenchEnvironmentError`、`UserBenchLifecycleError`、`UserBenchEnvironmentConfig`、`TravelEnvProtocol`、`UserBenchWrapper`。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `_classify_upstream_stdout` (L41) | `output`: str | 标注返回 `dict[str, int]`；具体值由分支决定。 | 计算奖励、指标或聚合统计。 |
| `UserBenchEnvironmentConfig.__post_init__` (L75) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `TravelEnvProtocol.reset` (L103) | `seed`: int \| None；`options`: Any | 标注返回 `tuple[Any, Any]`；具体值由分支决定。 | 清理资源或恢复状态边界。 |
| `TravelEnvProtocol.step` (L110) | `action_input`: str | 标注返回 `tuple[Any, float, bool, bool, Any]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `TravelEnvProtocol.step_async` (L115) | `action_input`: str | 标注返回 `Awaitable[tuple[Any, float, bool, bool, Any]]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `TravelEnvProtocol.close` (L122) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 清理资源或恢复状态边界。 |
| `_fallback_diagnostics` (L125) | `observation`: Any；`info`: Mapping[str, Any] | 标注返回 `dict[str, int]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_default_environment_factory` (L162) | `task_id`: str；`config`: UserBenchEnvironmentConfig；`runtime`: UserSimulatorRuntime；`source`: EmbeddedUserBench | 标注返回 `TravelEnvProtocol`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `UserBenchWrapper.__init__` (L218) | `task_id`: str；`runtime`: UserSimulatorRuntime；`config`: UserBenchEnvironmentConfig \| None；`source_root`: str \| Path \| None；`environment_factory`: EnvironmentFactory \| None | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `UserBenchWrapper.reset` (L251) | 无显式业务参数 | 标注返回 `UserBenchObservation`；具体值由分支决定。 | 清理资源或恢复状态边界。 |
| `UserBenchWrapper.areset` (L265) | 无显式业务参数 | 标注返回 `UserBenchObservation`；具体值由分支决定。 | 清理资源或恢复状态边界。 |
| `UserBenchWrapper.reward_task` (L275) | 无显式业务参数 | 标注返回 `TravelRewardTask`；具体值由分支决定。 | 计算奖励、指标或聚合统计。 |
| `UserBenchWrapper.reward_snapshot` (L291) | 无显式业务参数 | 标注返回 `UserBenchRewardSnapshot`；具体值由分支决定。 | 计算奖励、指标或聚合统计。 |
| `UserBenchWrapper.step` (L325) | `action`: UserBenchAction \| Mapping[str, Any] | 标注返回 `UserBenchStepResult`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `UserBenchWrapper.astep` (L339) | `action`: UserBenchAction \| Mapping[str, Any] | 标注返回 `UserBenchStepResult`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `UserBenchWrapper.close` (L379) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 清理资源或恢复状态边界。 |
| `UserBenchWrapper._prepare_step` (L394) | `action`: UserBenchAction \| Mapping[str, Any] | 标注返回 `UserBenchAction`；具体值由分支决定。 | 根据配置和中间状态构建项目产物。 |
| `UserBenchWrapper._project_transition` (L410) | `transition`: Any | 标注返回 `UserBenchStepResult`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `UserBenchWrapper._async_one_choice_termination` (L447) | `terminated`: bool | 标注返回 `bool`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `UserBenchWrapper._validate_info_task_id` (L468) | `info`: Any | 标注返回 `None`；具体值由分支决定。 | 校验输入、状态或产物，输出诊断或抛出异常。 |
| `UserBenchWrapper._require_open` (L479) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `UserBenchWrapper.__enter__` (L486) | 无显式业务参数 | 标注返回 `UserBenchWrapper`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `UserBenchWrapper.__exit__` (L494) | `exc_type`: type[BaseException] \| None；`exc`: BaseException \| None；`traceback`: TracebackType \| None | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |

### `src/travel_grpo/evaluation/__init__.py`

职责：Frozen, fixed-denominator UserBench evaluation.

代码行数：8。

本文件主要提供常量、类型或配置，没有可调用函数。

### `src/travel_grpo/evaluation/artifacts.py`

职责：Atomic, resumable evaluation artifact storage.

代码行数：98。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `atomic_json` (L16) | `path`: Path；`value`: Any | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `task_path` (L32) | `root`: Path；`task_id`: str | 标注返回 `Path`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `load_completed` (L40) | `root`: Path | 标注返回 `dict[str, dict[str, Any]]`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `write_results_jsonl` (L55) | `path`: Path；`records`: Sequence[Mapping[str, Any]] | 标注返回 `None`；具体值由分支决定。 | 序列化并持久化内部结果。 |
| `attach_attempt_history` (L68) | `result`: dict[str, Any]；`previous`: Mapping[str, Any] \| None | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |

### `src/travel_grpo/evaluation/checkpoint_selection.py`

职责：Deterministic GRPO checkpoint selection over the frozen 132-task validation.

代码行数：42。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `_fixed` (L12) | `summary`: Mapping[str, Any] | 标注返回 `Mapping[str, Any]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `select_checkpoint` (L22) | `candidates`: Sequence[Mapping[str, Any]]；`sft_summary`: Mapping[str, Any] | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 校验输入、状态或产物，输出诊断或抛出异常。 |

### `src/travel_grpo/evaluation/comparison.py`

职责：Paired three-stage comparison over one frozen contract.

代码行数：56。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `compare_stage_results` (L14) | `stages`: Mapping[str, Mapping[str, Any]]；`allow_subset`: bool | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |

### `src/travel_grpo/evaluation/contracts.py`

职责：Frozen evaluation contracts shared by Baseline, SFT, and GRPO.

代码行数：103。

类型：`EvaluationContract`。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `canonical_hash` (L19) | `value`: Any | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `EvaluationContract.contract_hash` (L44) | 无显式业务参数 | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `EvaluationContract.to_dict` (L50) | `stage`: str；`model`: str | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_build_contract` (L59) | `records`: Sequence[Mapping[str, Any]]；`simulator_endpoint`: str \| None | 标注返回 `EvaluationContract`；具体值由分支决定。 | 根据配置和中间状态构建项目产物。 |
| `build_contract` (L81) | `records`: Sequence[Mapping[str, Any]]；`simulator_endpoint`: str \| None | 标注返回 `EvaluationContract`；具体值由分支决定。 | 根据配置和中间状态构建项目产物。 |
| `build_subset_contract` (L94) | `records`: Sequence[Mapping[str, Any]]；`simulator_endpoint`: str \| None | 标注返回 `EvaluationContract`；具体值由分支决定。 | 根据配置和中间状态构建项目产物。 |

### `src/travel_grpo/evaluation/metrics.py`

职责：Per-task metric projection with no hidden-label artifact fields.

代码行数：125。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `sanitize_reward` (L40) | `report`: Mapping[str, Any] | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 计算奖励、指标或聚合统计。 |
| `result_metrics` (L62) | `result`: Mapping[str, Any] | 标注返回 `dict[str, float]`；具体值由分支决定。 | 计算奖励、指标或聚合统计。 |

### `src/travel_grpo/evaluation/rollout.py`

职责：One deterministic Actor/UserBench rollout for frozen evaluation.

代码行数：316。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `_initial_user_content` (L27) | `messages`: Sequence[Mapping[str, Any]] | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `rollout_task` (L36) | `task`: Mapping[str, Any]；`actor`: Any；`simulator`: UserSimulatorRuntime；`source_root`: str \| Path \| None；`wrapper_factory`: Any；`apply_actor_policy`: bool；`public_control_enabled`: bool | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_guarded_rollout_task` (L146) | `task`: Mapping[str, Any]；`actor`: Any；`simulator`: UserSimulatorRuntime；`source_root`: str \| Path \| None；`wrapper_factory`: Any；`apply_actor_policy`: bool | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 校验输入、状态或产物，输出诊断或抛出异常。 |

### `src/travel_grpo/evaluation/runner.py`

职责：Runtime orchestration for the frozen UserBench evaluation stage.

代码行数：327。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `load_tasks` (L55) | `path`: Path | 标注返回 `list[dict]`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `_sha256` (L66) | `path`: Path | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_load_subset_manifest` (L77) | `path`: Path；`records`: Sequence[Mapping[str, object]] | 标注返回 `dict`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `_run_pending_tasks` (L116) | `pending`: Sequence[dict]；`selected_count`: int；`completed`: dict[str, dict]；`actor`: OpenAICompatibleActorClient；`simulator`: UserSimulatorRuntime；`output`: Path；`concurrency`: int；`retry_infrastructure_invalid`: bool；`public_control_enabled`: bool | 标注返回 `None`；具体值由分支决定。 | 编排训练、采集、评测或 replay 流程。 |
| `_run_pending_tasks.run_one` (L143) | `task`: dict | 标注返回 `None`；具体值由分支决定。 | 编排训练、采集、评测或 replay 流程。 |
| `run` (L184) | `args`: argparse.Namespace | 标注返回 `int`；具体值由分支决定。 | 编排训练、采集、评测或 replay 流程。 |

### `src/travel_grpo/evaluation/summary.py`

职责：Fixed-denominator evaluation aggregation.

代码行数：109。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `_aggregate` (L15) | `records`: Sequence[Mapping[str, Any]]；`denominator`: int | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 计算奖励、指标或聚合统计。 |
| `_aggregate.averages` (L44) | `divisor`: int | 标注返回 `dict[str, float]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `summarize_results` (L72) | `records`: Sequence[Mapping[str, Any]]；`expected_task_ids`: Sequence[str]；`expected_compositions`: Sequence[str] \| None | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 计算奖励、指标或聚合统计。 |

### `src/travel_grpo/evaluation/validation.py`

职责：Normalize veRL validation dumps into the fixed UserBench summary contract.

代码行数：90。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `summarize_validation_rows` (L17) | `rows`: Sequence[Mapping[str, Any]]；`tasks`: Sequence[Mapping[str, Any]] | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 计算奖励、指标或聚合统计。 |
| `summarize_validation_file` (L80) | `path`: Path；`tasks`: Sequence[Mapping[str, Any]] | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 计算奖励、指标或聚合统计。 |

### `src/travel_grpo/models/__init__.py`

职责：Actor and external teacher model runtime boundaries.

代码行数：17。

本文件主要提供常量、类型或配置，没有可调用函数。

### `src/travel_grpo/models/openai_compatible.py`

职责：OpenAI-compatible teacher API boundary for UserBench trajectory collection.

代码行数：499。

类型：`TeacherApiError`、`TeacherProtocolError`、`TeacherRuntime`、`TeacherRequestConstraint`、`TeacherToolCall`、`TeacherClientProtocol`、`OpenAICompatibleTeacherClient`。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `TeacherRuntime.__post_init__` (L53) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `TeacherRuntime.from_environment` (L84) | `environ`: Mapping[str, str] \| None | 标注返回 `TeacherRuntime`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `TeacherRuntime.from_environment.require` (L92) | `name`: str | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `TeacherRequestConstraint.__post_init__` (L124) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `TeacherToolCall.parameters` (L145) | 无显式业务参数 | 标注返回 `dict[str, str]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `TeacherToolCall.to_assistant_message` (L155) | 无显式业务参数 | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 把协议/状态数据渲染成可见文本或消息。 |
| `TeacherClientProtocol.generate_action` (L182) | `messages`: Sequence[Mapping[str, Any]]；`force_answer`: bool；`constraint`: TeacherRequestConstraint \| None | 标注返回 `TeacherToolCall`；具体值由分支决定。 | 根据配置和中间状态构建项目产物。 |
| `TeacherClientProtocol.close` (L193) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 清理资源或恢复状态边界。 |
| `_attribute` (L199) | `value`: Any；`name`: str | 标注返回 `Any`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_usage_record` (L206) | `response`: Any | 标注返回 `dict[str, int]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `OpenAICompatibleTeacherClient.__init__` (L222) | `runtime`: TeacherRuntime；`client`: Any \| None | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `OpenAICompatibleTeacherClient.generate_action` (L248) | `messages`: Sequence[Mapping[str, Any]]；`force_answer`: bool；`constraint`: TeacherRequestConstraint \| None | 标注返回 `TeacherToolCall`；具体值由分支决定。 | 根据配置和中间状态构建项目产物。 |
| `OpenAICompatibleTeacherClient._protocol_messages` (L371) | `messages`: Sequence[Mapping[str, Any]]；`attempt`: int；`correction`: str \| None | 标注返回 `list[dict[str, Any]]`；具体值由分支决定。 | 把协议/状态数据渲染成可见文本或消息。 |
| `OpenAICompatibleTeacherClient._retry_correction` (L402) | `error`: TeacherProtocolError；`locked_action`: tuple[ActionChoice, str] \| None | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `OpenAICompatibleTeacherClient._parse_response` (L440) | `response`: Any | 标注返回 `TeacherToolCall`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `OpenAICompatibleTeacherClient.close` (L492) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 清理资源或恢复状态边界。 |

### `src/travel_grpo/models/vllm_policy.py`

职责：Strict OpenAI-compatible Actor boundary used by frozen evaluation.

代码行数：118。

类型：`ActorRuntime`、`OpenAICompatibleActorClient`。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `ActorRuntime.__post_init__` (L33) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `ActorRuntime.from_environment` (L43) | `environ`: Mapping[str, str] \| None | 标注返回 `'ActorRuntime'`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `ActorRuntime.from_environment.required` (L48) | `name`: str | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `ActorRuntime.require_model` (L62) | `expected_model`: str | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `OpenAICompatibleActorClient.__init__` (L77) | `runtime`: ActorRuntime；`client`: Any \| None | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `OpenAICompatibleActorClient.generate_action` (L92) | `messages`: Sequence[Mapping[str, Any]] | 标注返回 `TeacherToolCall`；具体值由分支决定。 | 根据配置和中间状态构建项目产物。 |
| `OpenAICompatibleActorClient.close` (L111) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 清理资源或恢复状态边界。 |

### `src/travel_grpo/prompts/__init__.py`

职责：Prompt contracts shared across Actor training and inference.

代码行数：25。

本文件主要提供常量、类型或配置，没有可调用函数。

### `src/travel_grpo/prompts/actor_policy.py`

职责：Versioned production Actor policy shared by SFT, GRPO, and evaluation.

代码行数：170。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `_copy_messages` (L77) | `messages`: Sequence[Mapping[str, Any]] | 标注返回 `list[dict[str, Any]]`；具体值由分支决定。 | 把协议/状态数据渲染成可见文本或消息。 |
| `_remove_blocks` (L89) | `content`: str；`blocks`: Sequence[str] | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `strip_teacher_generation_instruction` (L97) | `messages`: Sequence[Mapping[str, Any]] | 标注返回 `list[dict[str, Any]]`；具体值由分支决定。 | 计算奖励、指标或聚合统计。 |
| `strip_actor_runtime_policy` (L109) | `messages`: Sequence[Mapping[str, Any]] | 标注返回 `list[dict[str, Any]]`；具体值由分支决定。 | 编排训练、采集、评测或 replay 流程。 |
| `ensure_actor_runtime_policy` (L132) | `messages`: Sequence[Mapping[str, Any]] | 标注返回 `list[dict[str, Any]]`；具体值由分支决定。 | 编排训练、采集、评测或 replay 流程。 |
| `ensure_teacher_generation_messages` (L143) | `messages`: Sequence[Mapping[str, Any]] | 标注返回 `list[dict[str, Any]]`；具体值由分支决定。 | 计算奖励、指标或聚合统计。 |

### `src/travel_grpo/protocols/__init__.py`

职责：Shared external protocol normalization and schema helpers.

代码行数：3。

本文件主要提供常量、类型或配置，没有可调用函数。

### `src/travel_grpo/protocols/actor_messages.py`

职责：Normalize messages to the actor-visible UserBench protocol.

代码行数：83。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `_clean_tool_call` (L13) | `call`: Mapping[str, Any] | 标注返回 `dict[str, Any] \| None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `normalize_actor_messages` (L46) | `messages`: Sequence[Mapping[str, Any]] | 标注返回 `list[dict[str, Any]]`；具体值由分支决定。 | 把协议/状态数据渲染成可见文本或消息。 |

### `src/travel_grpo/training/__init__.py`

职责：Teacher collection, action-only SFT, and online GRPO.

代码行数：1。

本文件主要提供常量、类型或配置，没有可调用函数。

### `src/travel_grpo/training/grpo/__init__.py`

职责：veRL 0.8 data, sampling, environment, and runtime integration.

代码行数：1。

本文件主要提供常量、类型或配置，没有可调用函数。

### `src/travel_grpo/training/grpo/adapter/__init__.py`

职责：Direct veRL 0.8 ToolAgentLoop-to-UserBench adapter.

代码行数：5。

本文件主要提供常量、类型或配置，没有可调用函数。

### `src/travel_grpo/training/grpo/adapter/agent_loop.py`

职责：veRL 0.8 ToolAgentLoop with one direct UserBench session per rollout.

代码行数：519。

类型：`UserBenchAgentLoop`。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `session_requests_termination` (L47) | 无显式业务参数 | 标注返回 `bool`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_parse_bool` (L55) | `value`: Any；`name`: str | 标注返回 `bool`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `_parse_threshold` (L70) | `value`: Any | 标注返回 `int`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `select_post_tool_state` (L87) | `default_state`: Any；`terminated_state`: Any | 标注返回 `Any`；具体值由分支决定。 | 按固定约束拆分、采样或选择数据。 |
| `reject_parallel_tool_calls` (L95) | `state`: Any | 标注返回 `bool`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `finalize_actor_stop` (L116) | `session`: Any | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_parse_actor_policy_version` (L149) | `value`: Any；`enabled`: bool | 标注返回 `str`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `prepare_actor_prompt` (L164) | `raw_prompt`: Any；`actor_policy_enabled`: bool \| str；`actor_policy_version`: str \| None | 标注返回 `list[dict[str, Any]]`；具体值由分支决定。 | 根据配置和中间状态构建项目产物。 |
| `actor_policy_metadata` (L199) | `actor_policy_enabled`: bool；`actor_policy_version`: str | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `UserBenchAgentLoop.__init__` (L223) | `environment_config_path`: str \| Path；`simulator_config_path`: str \| Path；`max_steps`: int；`stall_recovery_enabled`: bool \| str；`stall_no_progress_threshold`: int \| str；`actor_policy_enabled`: bool \| str；`actor_policy_version`: str \| None；`turn_credit_mode`: str；`turn_credit_config`: Mapping[str, Any] \| None；`turn_credit_config_json`: str \| None；*`args`；**`kwargs` | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `UserBenchAgentLoop._handle_processing_tools_state` (L274) | `state`: Any | 标注返回 `Any`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `UserBenchAgentLoop._handle_generating_state` (L284) | `state`: Any；`sampling_params`: Any；`ignore_termination`: bool | 标注返回 `Any`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `UserBenchAgentLoop._call_tool` (L313) | `tool_call`: Any；`tools_kwargs`: dict[str, Any]；`agent_data`: Any | 标注返回 `tuple[Any, float, dict[str, Any]]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `UserBenchAgentLoop.run` (L364) | `sampling_params`: Any；**`kwargs` | 标注返回 `Any`；具体值由分支决定。 | 编排训练、采集、评测或 replay 流程。 |

### `src/travel_grpo/training/grpo/adapter/session.py`

职责：veRL 0.8 rollout metadata and direct UserBench session construction.

代码行数：266。

类型：`UserBenchRolloutRuntime`、`UserBenchInteraction`。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `_non_empty` (L33) | `value`: Any；`name`: str | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `build_rollout_extra_info` (L39) | `task_id`: str | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 根据配置和中间状态构建项目产物。 |
| `validate_rollout_extra_info` (L61) | `extra_info`: Mapping[str, Any] | 标注返回 `str`；具体值由分支决定。 | 校验输入、状态或产物，输出诊断或抛出异常。 |
| `task_id_from_run_kwargs` (L84) | `kwargs`: Mapping[str, Any] | 标注返回 `str`；具体值由分支决定。 | 编排训练、采集、评测或 replay 流程。 |
| `calculate_current_session_score` (L97) | 无显式业务参数 | 标注返回 `float`；具体值由分支决定。 | 计算奖励、指标或聚合统计。 |
| `_project_path` (L104) | `value`: str \| Path | 标注返回 `Path`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_load_yaml` (L115) | `path`: str \| Path | 标注返回 `Mapping[str, Any]`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `UserBenchRolloutRuntime.from_config_files` (L139) | `environment_path`: str \| Path；`simulator_path`: str \| Path | 标注返回 `'UserBenchRolloutRuntime'`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `UserBenchRolloutRuntime.start_session` (L182) | `task_id`: str；`request_id`: str \| None；`wrapper_factory`: Any | 标注返回 `UserBenchSessionState`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `UserBenchRolloutRuntime.astart_session` (L214) | `task_id`: str；`request_id`: str \| None；`wrapper_factory`: Any | 标注返回 `UserBenchSessionState`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `UserBenchInteraction.__init__` (L265) | *`_`；**`__` | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |

### `src/travel_grpo/training/grpo/adapter/tools.py`

职责：Expose the official UserBench action as one veRL native tool.

代码行数：242。

类型：`UserBenchToolExecution`、`UserBenchTool`。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `_rejected_tool_execution` (L38) | `session`: Any；`message`: str；`reason`: str；`action`: UserBenchAction \| None | 标注返回 `UserBenchToolExecution`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `execute_userbench_action` (L65) | `parameters`: Mapping[str, Any] | 标注返回 `UserBenchToolExecution`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `UserBenchTool.__init__` (L182) | `config`: dict[str, Any]；`tool_schema`: OpenAIFunctionToolSchema | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `UserBenchTool.create` (L192) | `instance_id`: str \| None；**`kwargs` | 标注返回 `tuple[str, Any]`；具体值由分支决定。 | 根据配置和中间状态构建项目产物。 |
| `UserBenchTool.execute` (L214) | `instance_id`: str；`parameters`: dict[str, Any]；**`kwargs` | 标注返回 `tuple[Any, float, dict[str, Any]]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `UserBenchTool.calc_reward` (L228) | `instance_id`: str；**`kwargs` | 标注返回 `float`；具体值由分支决定。 | 计算奖励、指标或聚合统计。 |
| `UserBenchTool.release` (L234) | `instance_id`: str；**`kwargs` | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `UserBenchTool.schema_dict` (L241) | 无显式业务参数 | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |

### `src/travel_grpo/training/grpo/compat.py`

职责：Narrow compatibility boundary for the pinned external veRL runtime.

代码行数：104。

类型：`VerlCompatibilityError`。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `require_verl_080` (L29) | 无显式业务参数 | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `require_verl_dynamic_sampling_patch` (L46) | 无显式业务参数 | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `install_torch_padding_fallback` (L62) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |

### `src/travel_grpo/training/grpo/data.py`

职责：Deterministic conversion from canonical UserBench splits to veRL rows.

代码行数：394。

类型：`GRPODataError`。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `_pyarrow` (L42) | 无显式业务参数 | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `sha256_file` (L54) | `path`: str \| Path | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_non_empty` (L65) | `value`: Any；`name`: str | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_validate_prompt` (L75) | `prompt`: Any；`task_id`: str | 标注返回 `list[dict[str, str]]`；具体值由分支决定。 | 校验输入、状态或产物，输出诊断或抛出异常。 |
| `build_verl_records` (L95) | `source_path`: str \| Path；`project_split`: str | 标注返回 `tuple[dict[str, Any], ...]`；具体值由分支决定。 | 根据配置和中间状态构建项目产物。 |
| `_validate_derived_records` (L175) | `records`: Sequence[Mapping[str, Any]]；`project_split`: str | 标注返回 `None`；具体值由分支决定。 | 校验输入、状态或产物，输出诊断或抛出异常。 |
| `_atomic_write_parquet` (L234) | `records`: Sequence[Mapping[str, Any]]；`destination`: Path | 标注返回 `None`；具体值由分支决定。 | 序列化并持久化内部结果。 |
| `_atomic_write_json` (L254) | `document`: Mapping[str, Any]；`destination`: Path | 标注返回 `None`；具体值由分支决定。 | 序列化并持久化内部结果。 |
| `prepare_verl_datasets` (L272) | `train_source`: str \| Path；`validation_source`: str \| Path；`output_root`: str \| Path；`force`: bool；`dry_run`: bool | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 根据配置和中间状态构建项目产物。 |
| `verify_verl_datasets` (L347) | `output_root`: str \| Path | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 校验输入、状态或产物，输出诊断或抛出异常。 |

### `src/travel_grpo/training/grpo/dynamic_sampling.py`

职责：Bounded group selection and the project-owned veRL rollout adapter.

代码行数：613。

类型：`BoundedSamplingState`、`DynamicSamplingExhausted`、`_RolloutCandidate`、`_CandidateSelection`。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `extract_userbench_group_signals` (L13) | `infos`: Sequence[object] | 标注返回 `tuple[list[float], list[bool], list[tuple[str, ...]]]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `select_reward_varying_groups` (L52) | `uids`: Sequence[Hashable]；`rewards`: Sequence[float]；`sampling_invalid`: Sequence[bool] \| None；`sampling_invalid_reasons`: Sequence[Sequence[str]] \| None；`expected_group_size`: int；`tolerance`: float | 标注返回 `tuple[list[int], dict[str, Any]]`；具体值由分支决定。 | 计算奖励、指标或聚合统计。 |
| `BoundedSamplingState.record_batch` (L176) | `stats`: Mapping[str, Any] | 标注返回 `bool`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `BoundedSamplingState.may_generate` (L189) | 无显式业务参数 | 标注返回 `bool`；具体值由分支决定。 | 根据配置和中间状态构建项目产物。 |
| `BoundedSamplingState.finish_update` (L195) | 无显式业务参数 | 标注返回 `bool`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_CandidateSelection.reward_range` (L241) | 无显式业务参数 | 标注返回 `float`；具体值由分支决定。 | 计算奖励、指标或聚合统计。 |
| `_CandidateSelection.uses_degraded` (L248) | 无显式业务参数 | 标注返回 `bool`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_python_value` (L255) | `value`: Any | 标注返回 `Any`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_ordered_unique` (L259) | `values`: Sequence[Any] | 标注返回 `list[Any]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_select_candidate_group` (L272) | `candidates`: Sequence[_RolloutCandidate]；`expected_group_size`: int | 标注返回 `_CandidateSelection \| None`；具体值由分支决定。 | 按固定约束拆分、采样或选择数据。 |
| `_verl_sampling_signals` (L323) | `output`: Any | 标注返回 `tuple[list[Any], list[float], list[bool], list[tuple[str, ...]]]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_verl_candidate_signals` (L343) | `output`: Any | 标注返回 `tuple[list[Any], list[float], list[bool], list[tuple[str, ...]], list[bool]]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `install_verl_bounded_sampler` (L366) | `manager`: Any；`config`: Mapping[str, Any] | 标注返回 `None`；具体值由分支决定。 | 按固定约束拆分、采样或选择数据。 |
| `install_verl_bounded_sampler.generate_sequences` (L399) | `batch`: Any | 标注返回 `Any`；具体值由分支决定。 | 根据配置和中间状态构建项目产物。 |

### `src/travel_grpo/training/grpo/launcher.py`

职责：GRPO profile validation and veRL launch orchestration.

代码行数：264。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `load_profile` (L29) | `path`: Path | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `hydra_overrides` (L39) | `profile`: dict[str, Any]；`output`: Path；`resume`: bool；`logger`: str | 标注返回 `list[str]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `main` (L134) | 无显式业务参数 | 标注返回 `int`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |

### `src/travel_grpo/training/grpo/preflight.py`

职责：Static and production runtime checks performed before Ray or CUDA starts.

代码行数：271。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `_check` (L41) | `condition`: bool；`message`: str | 标注返回 `None`；具体值由分支决定。 | 校验输入、状态或产物，输出诊断或抛出异常。 |
| `_sampling_value` (L49) | `sampling_params`: Mapping[str, Any]；`name`: str | 标注返回 `Any`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_sampling_float` (L59) | `sampling_params`: Mapping[str, Any]；`name`: str | 标注返回 `float`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_sampling_profile` (L70) | `sampling_params`: Mapping[str, Any] | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `is_validation_sampling` (L91) | `sampling_params`: Mapping[str, Any] | 标注返回 `bool`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `validate_sampling_profiles` (L97) | `training_sampling`: Mapping[str, Any]；`validation_sampling`: Mapping[str, Any] | 标注返回 `None`；具体值由分支决定。 | 校验输入、状态或产物，输出诊断或抛出异常。 |
| `_complete_model` (L118) | `path`: Path | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_simulator_environment` (L131) | `environ`: Mapping[str, str] | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `run_preflight` (L141) | `profile`: Mapping[str, Any]；`project_root`: Path；`output_dir`: Path；`resume`: bool；`strict_runtime`: bool；`environ`: Mapping[str, str] \| None；`stall_threshold`: int；`data_output_dir`: Path \| None | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 编排训练、采集、评测或 replay 流程。 |

### `src/travel_grpo/training/grpo/turn_credit.py`

职责：Compatibility facade for :mod:`travel_grpo.trajectory.turn_credit`.

代码行数：4。

本文件主要提供常量、类型或配置，没有可调用函数。

### `src/travel_grpo/training/recovery_sft.py`

职责：Compatibility facade for :mod:`travel_grpo.training.sft.recovery`.

代码行数：4。

本文件主要提供常量、类型或配置，没有可调用函数。

### `src/travel_grpo/training/sft/__init__.py`

职责：SFT-stage contracts, collection, rendering, and recovery workflows.

代码行数：5。

本文件主要提供常量、类型或配置，没有可调用函数。

### `src/travel_grpo/training/sft/collection.py`

职责：Teacher trajectory collection against an isolated UserBench simulator API.

代码行数：1408。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `validate_teacher_collection_config` (L83) | `path`: str \| Path | 标注返回 `Mapping[str, Any]`；具体值由分支决定。 | 校验输入、状态或产物，输出诊断或抛出异常。 |
| `write_stratified_selection_manifest` (L179) | `path`: str \| Path；`document`: Mapping[str, Any] | 标注返回 `Path`；具体值由分支决定。 | 序列化并持久化内部结果。 |
| `task_dimensions` (L192) | `task_id`: str | 标注返回 `tuple[str, ...]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_prepare_teacher_messages` (L208) | `prompt`: Sequence[Mapping[str, Any]] | 标注返回 `list[dict[str, Any]]`；具体值由分支决定。 | 根据配置和中间状态构建项目产物。 |
| `_simulator_fallback` (L222) | `result`: Any | 标注返回 `bool`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_simulator_diagnostic_count` (L235) | `result`: Any；`name`: str | 标注返回 `int`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_partial_trajectory_record` (L240) | `task_id`: str；`trajectory_attempt`: int；`messages`: Sequence[Mapping[str, Any]]；`rewards`: Sequence[float]；`dimensions`: Sequence[str]；`answered_aspects`: set[str]；`committed_actions`: Sequence[Mapping[str, Any]]；`generation_diagnostics`: Sequence[Mapping[str, Any]]；`simulator_fallbacks`: int；`simulator_judgment_fallbacks`: int；`simulator_search_fallbacks`: int；`terminated`: bool；`truncated`: bool；`teacher_request_count`: int；`teacher_usage`: Mapping[str, int] | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_message_contract_errors` (L288) | `messages`: Sequence[Mapping[str, Any]] | 标注返回 `tuple[str, ...]`；具体值由分支决定。 | 把协议/状态数据渲染成可见文本或消息。 |
| `_feedback_policy_errors` (L314) | `messages`: Sequence[Mapping[str, Any]] | 标注返回 `tuple[str, ...]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `trajectory_rejection_reasons` (L333) | `trajectory`: TeacherTrajectory | 标注返回 `tuple[str, ...]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_silver_markers` (L403) | `trajectory`: TeacherTrajectory | 标注返回 `set[str]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `quality_tier_for_trajectory` (L418) | `trajectory`: TeacherTrajectory；`strict_reasons`: Sequence[str] \| None | 标注返回 `str \| None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `collect_teacher_trajectory` (L484) | `task`: Mapping[str, Any]；`teacher`: TeacherClientProtocol；`simulator`: UserSimulatorRuntime；`wrapper_factory`: WrapperFactory；`source_root`: str \| Path \| None；`trajectory_attempt`: int | 标注返回 `TeacherTrajectory`；具体值由分支决定。 | 编排训练、采集、评测或 replay 流程。 |
| `collect_teacher_trajectories` (L950) | `tasks`: Sequence[Mapping[str, Any]]；`teacher`: TeacherClientProtocol；`simulator`: UserSimulatorRuntime；`concurrency`: int；`wrapper_factory`: WrapperFactory；`source_root`: str \| Path \| None | 标注返回 `tuple[TeacherTrajectory, ...]`；具体值由分支决定。 | 编排训练、采集、评测或 replay 流程。 |
| `collect_teacher_trajectories.collect` (L968) | `task`: Mapping[str, Any] | 标注返回 `TeacherTrajectory`；具体值由分支决定。 | 编排训练、采集、评测或 replay 流程。 |
| `collect_teacher_task_with_retries` (L981) | `task`: Mapping[str, Any]；`teacher`: TeacherClientProtocol；`simulator`: UserSimulatorRuntime；`max_attempts`: int；`wrapper_factory`: WrapperFactory；`source_root`: str \| Path \| None | 标注返回 `TeacherTaskOutcome`；具体值由分支决定。 | 编排训练、采集、评测或 replay 流程。 |
| `collect_teacher_outcomes` (L1066) | `tasks`: Sequence[Mapping[str, Any]]；`teacher`: TeacherClientProtocol；`simulator`: UserSimulatorRuntime；`concurrency`: int；`max_attempts`: int；`wrapper_factory`: WrapperFactory；`source_root`: str \| Path \| None；`on_outcome`: Callable[[TeacherTaskOutcome], Any] \| None | 标注返回 `tuple[TeacherTaskOutcome, ...]`；具体值由分支决定。 | 编排训练、采集、评测或 replay 流程。 |
| `collect_teacher_outcomes.collect` (L1087) | `task`: Mapping[str, Any] | 标注返回 `TeacherTaskOutcome`；具体值由分支决定。 | 编排训练、采集、评测或 replay 流程。 |
| `_atomic_json_record` (L1109) | `record`: Mapping[str, Any]；`destination`: Path | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `initialize_teacher_run` (L1133) | `run_dir`: str \| Path；`task_ids`: Sequence[str]；`resume`: bool | 标注返回 `Path`；具体值由分支决定。 | 编排训练、采集、评测或 replay 流程。 |
| `teacher_outcome_checkpoint_path` (L1177) | `run_dir`: str \| Path；`task_id`: str | 标注返回 `Path`；具体值由分支决定。 | 校验输入、状态或产物，输出诊断或抛出异常。 |
| `write_teacher_outcome_checkpoint` (L1186) | `outcome`: TeacherTaskOutcome；`run_dir`: str \| Path | 标注返回 `Path`；具体值由分支决定。 | 校验输入、状态或产物，输出诊断或抛出异常。 |
| `load_teacher_outcome_checkpoints` (L1198) | `run_dir`: str \| Path；`task_ids`: Sequence[str] | 标注返回 `dict[str, TeacherTaskOutcome]`；具体值由分支决定。 | 校验输入、状态或产物，输出诊断或抛出异常。 |
| `summarize_teacher_outcomes` (L1217) | `outcomes`: Sequence[TeacherTaskOutcome] | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 计算奖励、指标或聚合统计。 |
| `write_teacher_trajectories` (L1286) | `trajectories`: Sequence[TeacherTrajectory]；`output_path`: str \| Path；`force`: bool | 标注返回 `Path`；具体值由分支决定。 | 序列化并持久化内部结果。 |
| `_write_jsonl_records` (L1326) | `records`: Sequence[Mapping[str, Any]]；`output_path`: str \| Path；`force`: bool | 标注返回 `Path`；具体值由分支决定。 | 序列化并持久化内部结果。 |
| `write_teacher_collection_artifacts` (L1358) | `outcomes`: Sequence[TeacherTaskOutcome]；`accepted_path`: str \| Path；`rejected_path`: str \| Path；`diagnostics_path`: str \| Path；`silver_path`: str \| Path \| None；`force`: bool | 标注返回 `tuple[Path, Path, Path, Path]`；具体值由分支决定。 | 序列化并持久化内部结果。 |

### `src/travel_grpo/training/sft/contracts.py`

职责：Serializable SFT collection contracts and checkpoint schemas.

代码行数：307。

类型：`TeacherTrajectory`、`TeacherAttemptDiagnostic`、`TeacherTaskOutcome`。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `TeacherTrajectory.total_reward` (L51) | 无显式业务参数 | 标注返回 `float`；具体值由分支决定。 | 计算奖励、指标或聚合统计。 |
| `TeacherTrajectory.to_record` (L57) | 无显式业务参数 | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `TeacherTrajectory.from_record` (L119) | `record`: Mapping[str, Any] | 标注返回 `'TeacherTrajectory'`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `TeacherAttemptDiagnostic.to_record` (L173) | 无显式业务参数 | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `TeacherAttemptDiagnostic.from_record` (L198) | `record`: Mapping[str, Any] | 标注返回 `'TeacherAttemptDiagnostic'`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `TeacherTaskOutcome.accepted` (L237) | 无显式业务参数 | 标注返回 `bool`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `TeacherTaskOutcome.gold` (L244) | 无显式业务参数 | 标注返回 `bool`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `TeacherTaskOutcome.rejected_record` (L250) | 无显式业务参数 | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `TeacherTaskOutcome.to_checkpoint_record` (L268) | 无显式业务参数 | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 校验输入、状态或产物，输出诊断或抛出异常。 |
| `TeacherTaskOutcome.from_checkpoint_record` (L284) | `record`: Mapping[str, Any] | 标注返回 `'TeacherTaskOutcome'`；具体值由分支决定。 | 校验输入、状态或产物，输出诊断或抛出异常。 |

### `src/travel_grpo/training/sft/dataset.py`

职责：Strict UserBench trajectory validation and action-only SFT rendering.

代码行数：1096。

类型：`SFTDatasetError`、`SFTTrajectoryTooLongError`、`ChatTemplateTokenizer`、`TrajectoryRejection`、`TrajectoryAudit`、`ActionOnlyExample`、`ActionOnlyDataCollator`。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `ChatTemplateTokenizer.apply_chat_template` (L57) | `conversation`: Sequence[Mapping[str, Any]]；**`kwargs` | 标注返回 `Any`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `TrajectoryAudit.summary` (L78) | 无显式业务参数 | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 计算奖励、指标或聚合统计。 |
| `ActionOnlyExample.sequence_length` (L114) | 无显式业务参数 | 标注返回 `int`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `ActionOnlyExample.label_tokens` (L121) | 无显式业务参数 | 标注返回 `int`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `ActionOnlyExample.to_trainer_dict` (L127) | 无显式业务参数 | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 编排训练、采集、评测或 replay 流程。 |
| `load_tool_schema` (L138) | `path`: str \| Path | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `_message_reasons` (L166) | `messages`: Any | 标注返回 `tuple[str, ...]`；具体值由分支决定。 | 把协议/状态数据渲染成可见文本或消息。 |
| `trajectory_rejection_reasons` (L237) | `record`: Any | 标注返回 `tuple[str, ...]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `sft_admission_reasons` (L364) | `record`: Any；`accepted_quality_tiers`: Sequence[str] | 标注返回 `tuple[str, ...]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `prefix_admission_reasons` (L408) | `record`: Any | 标注返回 `tuple[str, ...]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `recovery_admission_reasons` (L528) | `record`: Any | 标注返回 `tuple[str, ...]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `sft_record_admission_reasons` (L626) | `record`: Any；`record_format`: str；`accepted_quality_tiers`: Sequence[str] | 标注返回 `tuple[str, ...]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `audit_trajectory_file` (L650) | `path`: str \| Path；`limit`: int \| None；`accepted_quality_tiers`: Sequence[str]；`record_format`: str | 标注返回 `TrajectoryAudit`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `load_sft_trajectories` (L711) | `path`: str \| Path；`limit`: int \| None；`accepted_quality_tiers`: Sequence[str]；`record_format`: str | 标注返回 `tuple[dict[str, Any], ...]`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `load_sft_trajectory_files` (L732) | `paths`: Sequence[str \| Path]；`limit`: int \| None；`accepted_quality_tiers`: Sequence[str]；`record_format`: str | 标注返回 `tuple[dict[str, Any], ...]`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `assert_sft_readiness` (L761) | `train`: Sequence[Mapping[str, Any]]；`validation`: Sequence[Mapping[str, Any]]；`minimum_train`: int；`minimum_validation`: int；`required_compositions`: Sequence[str] | 标注返回 `None`；具体值由分支决定。 | 校验输入、状态或产物，输出诊断或抛出异常。 |
| `assert_train_validation_disjoint` (L806) | `train`: Sequence[Mapping[str, Any]]；`validation`: Sequence[Mapping[str, Any]] | 标注返回 `None`；具体值由分支决定。 | 校验输入、状态或产物，输出诊断或抛出异常。 |
| `assert_task_ids_within_split` (L818) | `records`: Sequence[Mapping[str, Any]]；`allowed_task_ids`: Sequence[str]；`split_name`: str | 标注返回 `None`；具体值由分支决定。 | 校验输入、状态或产物，输出诊断或抛出异常。 |
| `_token_ids` (L839) | `value`: Any | 标注返回 `list[int]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_render` (L858) | `tokenizer`: ChatTemplateTokenizer；`messages`: Sequence[Mapping[str, Any]]；`tool_schema`: Mapping[str, Any]；`add_generation_prompt`: bool | 标注返回 `list[int]`；具体值由分支决定。 | 把协议/状态数据渲染成可见文本或消息。 |
| `_messages_for_qwen_template` (L878) | `messages`: Sequence[Mapping[str, Any]] | 标注返回 `list[dict[str, Any]]`；具体值由分支决定。 | 把协议/状态数据渲染成可见文本或消息。 |
| `build_action_only_examples` (L903) | `records`: Sequence[Mapping[str, Any]]；`tokenizer`: ChatTemplateTokenizer；`tool_schema`: Mapping[str, Any]；`max_sequence_length`: int；`accepted_quality_tiers`: Sequence[str]；`record_format`: str | 标注返回 `tuple[ActionOnlyExample, ...]`；具体值由分支决定。 | 根据配置和中间状态构建项目产物。 |
| `build_action_only_dataset` (L978) | `records`: Sequence[Mapping[str, Any]]；`tokenizer`: ChatTemplateTokenizer；`tool_schema`: Mapping[str, Any]；`max_sequence_length`: int；`accepted_quality_tiers`: Sequence[str]；`record_format`: str | 标注返回 `tuple[tuple[ActionOnlyExample, ...], tuple[dict[str, Any], ...]]`；具体值由分支决定。 | 根据配置和中间状态构建项目产物。 |
| `ActionOnlyDataCollator.__call__` (L1032) | `features`: Sequence[Mapping[str, Any] \| ActionOnlyExample] | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `rendered_dataset_summary` (L1070) | `examples`: Sequence[ActionOnlyExample] | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 计算奖励、指标或聚合统计。 |
| `rendered_dataset_summary.percentile` (L1079) | `fraction`: float | 标注返回 `int`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |

### `src/travel_grpo/training/sft/errors.py`

职责：Errors raised by the SFT collection boundary.

代码行数：39。

类型：`TeacherCollectionError`、`TeacherGenerationError`、`TeacherAttemptAbort`。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `TeacherGenerationError.__init__` (L19) | `message`: str；`diagnostics`: Sequence[Mapping[str, Any]]；`reason_code`: str | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `TeacherAttemptAbort.__init__` (L37) | `reason_code`: str；`message`: str | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |

### `src/travel_grpo/training/sft/planning.py`

职责：Task-pool validation and deterministic SFT sampling plans.

代码行数：221。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `load_teacher_task_pool` (L15) | `path`: str \| Path；`expected_source_split`: str | 标注返回 `tuple[dict[str, Any], ...]`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `_largest_remainder_quota` (L68) | `counts`: Mapping[str, int]；`target`: int | 标注返回 `dict[str, int]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `build_stratified_task_plan` (L98) | `tasks`: Sequence[Mapping[str, Any]]；`target`: int；`field`: str；`seed`: str | 标注返回 `tuple[tuple[dict[str, Any], ...], dict[str, int]]`；具体值由分支决定。 | 根据配置和中间状态构建项目产物。 |
| `select_stratified_task_wave` (L146) | `tasks`: Sequence[Mapping[str, Any]]；`quotas`: Mapping[str, int]；`attempted_task_ids`: set[str]；`accepted_task_ids`: set[str]；`field`: str；`wave_size`: int | 标注返回 `tuple[dict[str, Any], ...]`；具体值由分支决定。 | 按固定约束拆分、采样或选择数据。 |
| `assert_disjoint_from_evaluation` (L209) | `tasks`: Sequence[Mapping[str, Any]]；`evaluation_path`: str \| Path | 标注返回 `None`；具体值由分支决定。 | 校验输入、状态或产物，输出诊断或抛出异常。 |

### `src/travel_grpo/training/sft/recovery.py`

职责：Render and audit one-step recovery records for action-only SFT.

代码行数：897。

类型：`RecoverySFTError`、`RecoveryAuditResult`、`CPUChatTemplateTokenizer`。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `CPUChatTemplateTokenizer.__init__` (L91) | `tokenizer`: Any；`template`: Any；`pad_token_id`: int \| None；`eos_token_id`: int \| None | 主要通过副作用更新状态或写出产物，默认返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `CPUChatTemplateTokenizer.apply_chat_template` (L102) | `conversation`: Sequence[Mapping[str, Any]]；`tools`: Sequence[Mapping[str, Any]]；`tokenize`: bool；`add_generation_prompt`: bool；`enable_thinking`: bool；**`_` | 标注返回 `list[int]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `CPUChatTemplateTokenizer.vocab_size` (L127) | 无显式业务参数 | 标注返回 `int`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `load_cpu_chat_template_tokenizer` (L131) | `path`: str \| Path | 标注返回 `CPUChatTemplateTokenizer`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `_jsonl_records` (L171) | `path`: Path | 标注返回 `Iterable[tuple[int, Mapping[str, Any] \| None, str \| None]]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_sha256` (L194) | `path`: Path | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_initial_user_message` (L205) | `messages`: Sequence[Mapping[str, Any]] | 标注返回 `str`；具体值由分支决定。 | 把协议/状态数据渲染成可见文本或消息。 |
| `_coerce_recovery_mode` (L216) | `value`: Any | 标注返回 `RecoveryMode`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `public_state_from_payload` (L235) | `payload`: Mapping[str, Any]；`messages`: Sequence[Mapping[str, Any]]；`phase_hint`: str \| None | 标注返回 `PublicControlState`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `_append_public_control_note` (L334) | `messages`: list[dict[str, Any]]；`note`: str | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_existing_call_ids` (L352) | `messages`: Sequence[Mapping[str, Any]] | 标注返回 `set[str]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_normalise_target` (L369) | `target`: Mapping[str, Any]；`existing_ids`: set[str]；`task_id`: str；`boundary`: str | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `render_recovery_record` (L389) | `target_record`: Mapping[str, Any] | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 把协议/状态数据渲染成可见文本或消息。 |
| `_leakage_hits` (L451) | `value`: Any；`path`: str | 标注返回 `list[str]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_target_action` (L473) | `record`: Mapping[str, Any] | 标注返回 `UserBenchAction`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_answer_visibility` (L492) | `record`: Mapping[str, Any] | 标注返回 `tuple[bool, str \| None]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_sample_hash` (L513) | `record`: Mapping[str, Any] | 标注返回 `str`；具体值由分支决定。 | 按固定约束拆分、采样或选择数据。 |
| `audit_rendered_record` (L526) | `record`: Mapping[str, Any]；`tokenizer`: Any；`tool_schema`: Mapping[str, Any]；`max_sequence_length`: int | 标注返回 `RecoveryAuditResult`；具体值由分支决定。 | 把协议/状态数据渲染成可见文本或消息。 |
| `_percentile` (L623) | `values`: Sequence[int]；`fraction`: float | 标注返回 `int \| None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_length_summary` (L634) | `lengths`: Sequence[int]；`labels`: Sequence[int] | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 计算奖励、指标或聚合统计。 |
| `_rejection_record` (L651) | `value`: Mapping[str, Any] \| None；`line`: int；`reasons`: Sequence[str]；`path`: Path | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_preflight_task_issues` (L669) | `paths`: Mapping[str, Path] | 标注返回 `tuple[dict[str, list[tuple[str, int, str]]], dict[str, int]]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `build_recovery_sft_dataset` (L683) | `target_dir`: str \| Path；`output_dir`: str \| Path；`tokenizer`: Any；`tool_schema`: Mapping[str, Any]；`max_sequence_length`: int | 标注返回 `tuple[dict[str, Path], dict[str, Any]]`；具体值由分支决定。 | 根据配置和中间状态构建项目产物。 |

### `src/travel_grpo/training/sft_collection.py`

职责：Compatibility facade for :mod:`travel_grpo.training.sft.collection`.

代码行数：10。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `__getattr__` (L7) | `name`: str | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |

### `src/travel_grpo/training/sft_dataset.py`

职责：Compatibility facade for :mod:`travel_grpo.training.sft.dataset`.

代码行数：3。

本文件主要提供常量、类型或配置，没有可调用函数。

### `src/travel_grpo/training/teacher_policy.py`

职责：Deterministic phase control for strict UserBench teacher trajectories.

代码行数：833。

类型：`TeacherPhase`、`AttemptStrategy`、`TeacherTurnPlan`、`TeacherPolicyState`。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `AttemptStrategy.for_attempt` (L70) | `attempt`: int | 标注返回 `'AttemptStrategy'`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_search_tokens` (L349) | `text`: str | 标注返回 `tuple[str, ...]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_search_token_matches` (L361) | `actual`: str；`expected`: str | 标注返回 `bool`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_contains_public_alias` (L376) | `text`: str；`alias`: str | 标注返回 `bool`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_search_query_issue` (L395) | `content`: str；`aspect`: str；`public_context`: tuple[str, ...] | 标注返回 `str \| None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `TeacherTurnPlan.choice` (L433) | 无显式业务参数 | 标注返回 `ActionChoice`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `TeacherTurnPlan.canonical_content` (L440) | 无显式业务参数 | 标注返回 `str \| None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `TeacherTurnPlan.elicitation_repair_content` (L457) | 无显式业务参数 | 标注返回 `str \| None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `TeacherTurnPlan.instruction` (L466) | `generation_attempt`: int | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `TeacherTurnPlan.validate` (L524) | `action`: UserBenchAction | 标注返回 `str \| None`；具体值由分支决定。 | 校验输入、状态或产物，输出诊断或抛出异常。 |
| `TeacherPolicyState.__post_init__` (L590) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `TeacherPolicyState._active_complete` (L596) | `aspect`: str；`session`: 'UserBenchSessionState' | 标注返回 `bool`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `TeacherPolicyState._field_order` (L603) | `aspect`: str | 标注返回 `tuple[str, ...]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `TeacherPolicyState._visible_option_ids` (L621) | `aspect`: str；`messages`: list[dict] | 标注返回 `tuple[str, ...]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `TeacherPolicyState._visible_option_details` (L632) | `aspect`: str；`messages`: list[dict]；`option_ids`: tuple[str, ...] | 标注返回 `tuple[str, ...]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `TeacherPolicyState._public_search_requirements` (L671) | `messages`: list[dict] | 标注返回 `tuple[tuple[str, tuple[str, ...]], ...]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `TeacherPolicyState._public_search_context` (L711) | `aspect`: str；`messages`: list[dict] | 标注返回 `tuple[str, ...]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `TeacherPolicyState.next_plan` (L751) | `session`: 'UserBenchSessionState'；`messages`: list[dict] | 标注返回 `TeacherTurnPlan`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `TeacherPolicyState.record_committed` (L791) | `plan`: TeacherTurnPlan | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_public_candidate_satisfies` (L797) | `detail`: str；`aliases`: tuple[str, ...] | 标注返回 `bool`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_public_candidate_satisfies.non_null_key` (L808) | `node`: object；`key`: str | 标注返回 `bool`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `canonical_content_for` (L827) | `plan`: TeacherTurnPlan | 标注返回 `tuple[str, ...]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |

### `src/travel_grpo/trajectory/__init__.py`

职责：Neutral trajectory accounting shared by environments and training.

代码行数：4。

本文件主要提供常量、类型或配置，没有可调用函数。

### `src/travel_grpo/trajectory/turn_credit.py`

职责：Causal turn evidence and bounded GRPO advantage reshaping.

代码行数：655。

类型：`TurnCreditError`、`TurnEvent`、`AspectCausalTrace`、`TurnCreditConfig`、`TurnCreditTrace`。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `TurnEvent.__post_init__` (L67) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `TurnEvent.to_training_record` (L73) | 无显式业务参数 | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 编排训练、采集、评测或 replay 流程。 |
| `TurnCreditConfig.__post_init__` (L118) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `TurnCreditConfig.from_mapping` (L158) | `value`: Mapping[str, Any] \| None | 标注返回 `'TurnCreditConfig'`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `TurnCreditTrace.__post_init__` (L203) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `TurnCreditTrace.to_extra_field` (L215) | `mode`: str | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `TurnCreditTrace.metrics` (L228) | `mode`: str | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 计算奖励、指标或聚合统计。 |
| `validate_turn_credit_mode` (L247) | `value`: Any | 标注返回 `str`；具体值由分支决定。 | 校验输入、状态或产物，输出诊断或抛出异常。 |
| `build_aspect_causal_traces` (L258) | `events`: Sequence[TurnEvent]；`aspects`: Sequence[str]；`blocked_aspects`: Sequence[str] | 标注返回 `tuple[AspectCausalTrace, ...]`；具体值由分支决定。 | 根据配置和中间状态构建项目产物。 |
| `primary_violation_penalty` (L315) | `event`: TurnEvent；`config`: TurnCreditConfig | 标注返回 `float`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `primary_violation_key` (L342) | `event`: TurnEvent | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `allocate_turn_evidence` (L360) | `events`: Sequence[TurnEvent]；`traces`: Sequence[AspectCausalTrace]；`reward_valid`: bool；`config`: TurnCreditConfig \| None | 标注返回 `tuple[float, ...]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `build_turn_credit_trace` (L445) | `events`: Sequence[TurnEvent]；`aspects`: Sequence[str]；`blocked_aspects`: Sequence[str]；`reward_valid`: bool；`config`: TurnCreditConfig \| None | 标注返回 `TurnCreditTrace`；具体值由分支决定。 | 根据配置和中间状态构建项目产物。 |
| `normalized_turn_evidence` (L473) | `evidence`: Sequence[float]；`epsilon`: float | 标注返回 `tuple[float, ...]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `reshape_turn_advantages` (L491) | `sequence_advantage`: float；`evidence`: Sequence[float]；`turn_token_lengths`: Sequence[int] \| None；`config`: TurnCreditConfig \| None | 标注返回 `tuple[float, ...]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `assistant_turn_spans` (L563) | `mask`: Sequence[int \| bool \| float] | 标注返回 `tuple[tuple[int, int], ...]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `reshape_batch_advantages` (L580) | `data`: Any；`algorithm_config`: Any | 标注返回 `Any`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |

### `src/travel_grpo/utils/__init__.py`

职责：Shared infrastructure helpers without domain policy.

代码行数：1。

本文件主要提供常量、类型或配置，没有可调用函数。

### `src/travel_grpo/utils/io.py`

职责：Placeholder for atomic artifact and configuration I/O helpers.

代码行数：1。

本文件主要提供常量、类型或配置，没有可调用函数。

### `src/travel_grpo/utils/logger.py`

职责：Placeholder for run-scoped structured logging helpers.

代码行数：1。

本文件主要提供常量、类型或配置，没有可调用函数。

### `scripts/data/build_dataset_splits.py`

职责：Build or verify the frozen UserBench project-level task splits.

代码行数：100。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `parse_args` (L28) | `argv`: list[str] \| None | 标注返回 `argparse.Namespace`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `main` (L74) | `argv`: list[str] \| None | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |

### `scripts/data/build_recovery_targets.py`

职责：Construct and validate one-step targets for recovery-boundary contexts.

代码行数：62。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `build_parser` (L22) | 无显式业务参数 | 标注返回 `argparse.ArgumentParser`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `main` (L49) | 无显式业务参数 | 标注返回 `int`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |

### `scripts/data/extract_recovery_boundaries.py`

职责：Extract recovery-boundary-v1 contexts from existing local artifacts.

代码行数：63。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `build_parser` (L29) | 无显式业务参数 | 标注返回 `argparse.ArgumentParser`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `main` (L50) | 无显式业务参数 | 标注返回 `int`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |

### `scripts/eval/ab_prompt_test.py`

职责：Small paired A/B test for the production Actor runtime policy.

代码行数：254。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `load_tasks` (L44) | `path`: Path | 标注返回 `list[dict[str, Any]]`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `choose_batch` (L55) | `rows`: Sequence[Mapping[str, Any]]；`size`: int | 标注返回 `list[dict[str, Any]]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `with_condition_prompt` (L88) | `task`: Mapping[str, Any]；`condition`: str | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 把协议/状态数据渲染成可见文本或消息。 |
| `action_choices` (L99) | `result`: Mapping[str, Any] | 标注返回 `list[str]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `condition_summary` (L122) | `records`: Sequence[Mapping[str, Any]] | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 计算奖励、指标或聚合统计。 |
| `condition_summary.rate` (L151) | `name`: str | 标注返回 `float \| None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `condition_summary.mean_metric` (L160) | `name`: str | 标注返回 `float`；具体值由分支决定。 | 计算奖励、指标或聚合统计。 |
| `run` (L197) | `args`: argparse.Namespace | 标注返回 `None`；具体值由分支决定。 | 编排训练、采集、评测或 replay 流程。 |
| `main` (L242) | 无显式业务参数 | 标注返回 `int`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |

### `scripts/eval/compare_stages.py`

职责：Generate the formal paired comparison after all three 471-task runs.

代码行数：59。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `read_jsonl` (L22) | `path`: Path | 标注返回 `list[dict]`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `main` (L30) | 无显式业务参数 | 标注返回 `int`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |

### `scripts/eval/create_task_subset.py`

职责：Create a reproducible composition-stratified UserBench test subset.

代码行数：158。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `_sha256` (L25) | `path`: Path | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_allocate_quotas` (L36) | `counts`: Counter[str]；`target`: int | 标注返回 `dict[str, int]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `create_subset` (L59) | `source`: Path；`output`: Path；`manifest_path`: Path；`count`: int；`seed`: int | 标注返回 `dict`；具体值由分支决定。 | 根据配置和中间状态构建项目产物。 |
| `main` (L134) | 无显式业务参数 | 标注返回 `int`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |

### `scripts/eval/evaluate_userbench.py`

职责：Compatibility CLI for the frozen UserBench evaluation runner.

代码行数：70。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `main` (L27) | 无显式业务参数 | 标注返回 `int`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |

### `scripts/eval/inference_gate.py`

职责：Reproducible SFT Actor inference gate.

代码行数：1172。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `_sha256_bytes` (L88) | `value`: bytes | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_sha256_json` (L95) | `value`: Any | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_sha256_file` (L104) | `path`: Path | 标注返回 `str \| None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_relative` (L117) | `path`: Path | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_source_info` (L127) | `record`: Mapping[str, Any] | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `load_boundary_records` (L145) | `path`: Path | 标注返回 `list[dict[str, Any]]`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `_stable_record_key` (L166) | `record`: Mapping[str, Any] | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_is_grpo_source` (L182) | `record`: Mapping[str, Any] | 标注返回 `bool`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `choose_probe_records` (L190) | `records`: Sequence[Mapping[str, Any]]；`boundary_type`: str；`count`: int；`confused`: bool | 标注返回 `list[dict[str, Any]]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `choose_probe_records._eligible_open_transition` (L255) | `value`: Mapping[str, Any] | 标注返回 `bool`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `load_frozen_tasks` (L299) | `path`: Path | 标注返回 `list[dict[str, Any]]`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `choose_closed_loop_tasks` (L310) | `rows`: Sequence[Mapping[str, Any]]；`count`: int | 标注返回 `list[dict[str, Any]]`；具体值由分支决定。 | 清理资源或恢复状态边界。 |
| `_initial_user_content` (L338) | `messages`: Sequence[Mapping[str, Any]] | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_extract_actions` (L349) | `messages`: Sequence[Mapping[str, Any]] | 标注返回 `list[UserBenchAction]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_append_control_note` (L374) | `messages`: list[dict[str, Any]]；`note`: str | 标注返回 `list[dict[str, Any]]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_public_messages` (L390) | `record`: Mapping[str, Any]；`condition`: str | 标注返回 `tuple[list[dict[str, Any]], dict[str, Any]]`；具体值由分支决定。 | 把协议/状态数据渲染成可见文本或消息。 |
| `_answer_ids` (L430) | `content`: str | 标注返回 `list[str]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_action_record` (L437) | `call`: Any | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_classify_probe` (L453) | `category`: str；`record`: Mapping[str, Any]；`action`: UserBenchAction \| None；`state_payload`: Mapping[str, Any]；`previous_actions`: Sequence[UserBenchAction] | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `run_one_step_probes` (L552) | `samples`: Mapping[str, Sequence[Mapping[str, Any]]]；`actor`: OpenAICompatibleActorClient；`output`: Path；`conditions`: Sequence[str] | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 编排训练、采集、评测或 replay 流程。 |
| `_mean_flags` (L620) | `rows`: Sequence[Mapping[str, Any]] | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 计算奖励、指标或聚合统计。 |
| `_public_reward_summary` (L638) | `report`: Mapping[str, Any] | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 计算奖励、指标或聚合统计。 |
| `_sanitize_transcript` (L661) | `messages`: Sequence[Mapping[str, Any]] | 标注返回 `list[dict[str, Any]]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `guarded_rollout_task` (L665) | `task`: Mapping[str, Any]；`actor`: OpenAICompatibleActorClient；`simulator`: UserSimulatorRuntime；`source_root`: Path | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 校验输入、状态或产物，输出诊断或抛出异常。 |
| `_old_result_summary` (L755) | `result`: Mapping[str, Any] | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 计算奖励、指标或聚合统计。 |
| `run_closed_loop` (L777) | `tasks`: Sequence[Mapping[str, Any]]；`actor`: OpenAICompatibleActorClient；`simulator`: UserSimulatorRuntime；`output`: Path；`conditions`: Sequence[str] | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 编排训练、采集、评测或 replay 流程。 |
| `_transcript_choice_counts` (L821) | `transcript`: Any | 标注返回 `Counter[str]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_guard_reason_count` (L850) | `rows`: Sequence[Mapping[str, Any]]；`predicate`: Any | 标注返回 `tuple[int, int]`；具体值由分支决定。 | 校验输入、状态或产物，输出诊断或抛出异常。 |
| `summarize_closed_loop` (L874) | `rows`: Sequence[Mapping[str, Any]] | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 计算奖励、指标或聚合统计。 |
| `build_comparison_report` (L951) | `output`: Path | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 根据配置和中间状态构建项目产物。 |
| `build_comparison_report.load` (L957) | `relative`: str | 标注返回 `dict[str, Any] \| None`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `build_manifest` (L990) | `args`: argparse.Namespace | 标注返回 `tuple[dict[str, Any], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]`；具体值由分支决定。 | 根据配置和中间状态构建项目产物。 |
| `_requested_conditions` (L1056) | `args`: argparse.Namespace | 标注返回 `tuple[str, ...]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `preserve_baseline_condition` (L1066) | `args`: argparse.Namespace；`conditions`: Sequence[str] | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `run` (L1101) | `args`: argparse.Namespace | 标注返回 `None`；具体值由分支决定。 | 编排训练、采集、评测或 replay 流程。 |
| `main` (L1144) | 无显式业务参数 | 标注返回 `int`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |

### `scripts/eval/recompute_comparable_200_metrics.py`

职责：Replay current Reward-v3 metrics over frozen 200-task evaluation artifacts.

代码行数：423。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `_sha256` (L44) | `path`: Path | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_atomic_json` (L55) | `path`: Path；`value`: Any | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_ratio` (L65) | `numerator`: float；`denominator`: float | 标注返回 `float`；具体值由分支决定。 | 计算奖励、指标或聚合统计。 |
| `_tool_choice` (L72) | `message`: Mapping[str, Any] | 标注返回 `str \| None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `replay_public_metrics` (L83) | `transcript`: Sequence[Mapping[str, Any]] | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 编排训练、采集、评测或 replay 流程。 |
| `_load_preference_counts` (L148) | `compositions`: Sequence[str] | 标注返回 `dict[str, int]`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `_current_penalty` (L167) | `reward`: Mapping[str, Any]；`guard_rejections`: int；`blocked_aspects`: int；`aspect_count`: int；`termination_reason`: str | 标注返回 `tuple[float, dict[str, float]]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `project_current_reward` (L196) | `result`: Mapping[str, Any]；`preference_count`: int | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 计算奖励、指标或聚合统计。 |
| `_extra_summary` (L269) | `records`: Sequence[Mapping[str, Any]]；`denominator`: int | 标注返回 `dict[str, float]`；具体值由分支决定。 | 计算奖励、指标或聚合统计。 |
| `_extra_summary.rate` (L274) | `predicate` | 标注返回 `float`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_task_digest` (L293) | `paths`: Sequence[Path] | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `replay_run` (L304) | `name`: str；`run_dir`: Path；`output_root`: Path；`preference_counts`: Mapping[str, int] | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 编排训练、采集、评测或 replay 流程。 |
| `_report` (L363) | `rows`: Sequence[Mapping[str, Any]] | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `main` (L387) | 无显式业务参数 | 标注返回 `int`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |

### `scripts/eval/recovery_sft_decision.py`

职责：Build the CPU-only Recovery SFT go/no-go decision from frozen gate artifacts.

代码行数：445。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `sha256_file` (L40) | `path`: Path | 标注返回 `str \| None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `read_json` (L53) | `path`: Path | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `rel` (L63) | `path`: Path | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_tool_choices` (L73) | `transcript`: Any | 标注返回 `list[str]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `closed_loop_followup` (L102) | `gate_dir`: Path | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 清理资源或恢复状态边界。 |
| `leakage_scan` (L141) | `gate_dir`: Path | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `gate_rows` (L157) | `comparison`: Mapping[str, Any]；`followup`: Mapping[str, Any]；`leakage`: Mapping[str, Any] | 标注返回 `list[dict[str, Any]]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `build_report` (L177) | `gate_dir`: Path；`boundaries`: Path；`targets`: Path；`sft`: Path | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 根据配置和中间状态构建项目产物。 |
| `render_markdown` (L317) | `report`: Mapping[str, Any]；`machine_path`: str \| None；`gate_path`: str \| None；`output_path`: str \| None；`markdown_path`: str \| None | 标注返回 `str`；具体值由分支决定。 | 把协议/状态数据渲染成可见文本或消息。 |
| `main` (L417) | 无显式业务参数 | 标注返回 `int`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |

### `scripts/eval/replay_search_contract.py`

职责：Replay Actor search calls against the patched UserBench search contract.

代码行数：282。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `_extract_search_calls` (L55) | `path`: Path | 标注返回 `list[str]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_candidate_ids` (L85) | `task`: Mapping[str, Any]；`aspect`: str \| None | 标注返回 `set[str]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_has_candidate_list` (L100) | `feedback`: str；`ids`: set[str] | 标注返回 `bool`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_is_fallback` (L112) | `feedback`: str | 标注返回 `bool`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_initial_state` (L120) | `task`: Mapping[str, Any] | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_model_config` (L132) | `runtime`: UserSimulatorRuntime | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `replay` (L145) | `trace_dir`: Path；`output`: Path；`api_fallback`: bool | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 编排训练、采集、评测或 replay 流程。 |
| `main` (L256) | 无显式业务参数 | 标注返回 `int`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |

### `scripts/eval/run_checkpoint_sequence.py`

职责：Export and evaluate GRPO checkpoints sequentially, then shut down.

代码行数：662。

类型：`Progress`。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `parse_args` (L60) | 无显式业务参数 | 标注返回 `argparse.Namespace`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `utc_now` (L89) | 无显式业务参数 | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `resolve_from_root` (L96) | `path`: Path | 标注返回 `Path`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `relative_model_name` (L100) | `path`: Path | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `read_subset_count` (L108) | `path`: Path | 标注返回 `int`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `task_progress` (L118) | `run_dir`: Path | 标注返回 `Progress`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `evaluation_complete` (L140) | `run_dir`: Path；`expected_tasks`: int；`model_name`: str | 标注返回 `bool`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `merged_model_complete` (L163) | `path`: Path | 标注返回 `bool`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `atomic_status` (L177) | `path`: Path；**`values` | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `stop_process` (L188) | `process`: subprocess.Popen[Any] \| None；`name`: str | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `actor_ready` (L209) | `base_url`: str；`api_key`: str；`model_name`: str | 标注返回 `bool`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `wait_for_actor` (L231) | `process`: subprocess.Popen[Any]；`base_url`: str；`api_key`: str；`model_name`: str；`timeout_seconds`: float；`poll_seconds`: float | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `export_command` (L255) | `actor_dir`: Path；`merged_dir`: Path | 标注返回 `list[str]`；具体值由分支决定。 | 序列化并持久化内部结果。 |
| `actor_command` (L273) | `model_name`: str | 标注返回 `list[str]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `evaluation_command` (L295) | `model_name`: str；`dataset`: Path；`subset_manifest`: Path；`run_dir`: Path；`concurrency`: int | 标注返回 `list[str]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `monitor_evaluation` (L323) | `process`: subprocess.Popen[Any]；`run_dir`: Path；`expected_tasks`: int；`stall_seconds`: float；`poll_seconds`: float；`initial_progress_at`: float；`status_path`: Path；`step`: int | 标注返回 `tuple[bool, float]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `shutdown_after_grace` (L371) | `reason`: str；`delay_seconds`: float；`no_shutdown`: bool；`status_path`: Path | 标注返回 `int`；具体值由分支决定。 | 清理资源或恢复状态边界。 |
| `validate_inputs` (L397) | `args`: argparse.Namespace | 标注返回 `tuple[Path, Path, Path, Path, int]`；具体值由分支决定。 | 校验输入、状态或产物，输出诊断或抛出异常。 |
| `print_dry_run` (L426) | `args`: argparse.Namespace；`run_root`: Path；`dataset`: Path；`subset_manifest`: Path；`output_root`: Path；`expected`: int | 标注返回 `None`；具体值由分支决定。 | 编排训练、采集、评测或 replay 流程。 |
| `main` (L479) | 无显式业务参数 | 标注返回 `int`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |

### `scripts/eval/search_answer_probe.py`

职责：Probe one-step search-to-answer behavior without running live UserBench.

代码行数：318。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `load_task_map` (L55) | 无显式业务参数 | 标注返回 `dict[str, dict[str, Any]]`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `clean_messages` (L69) | `messages`: list[Mapping[str, Any]] | 标注返回 `list[dict[str, Any]]`；具体值由分支决定。 | 把协议/状态数据渲染成可见文本或消息。 |
| `without_teacher_suffix` (L78) | `messages`: list[dict[str, Any]] | 标注返回 `list[dict[str, Any]]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `with_teacher_suffix` (L84) | `messages`: list[dict[str, Any]] | 标注返回 `list[dict[str, Any]]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `collect_contexts` (L94) | `sources`: tuple[Path, ...]；`task_map`: Mapping[str, Mapping[str, Any]]；`limit`: int | 标注返回 `list[dict[str, Any]]`；具体值由分支决定。 | 编排训练、采集、评测或 replay 流程。 |
| `classify` (L165) | `parameters`: Mapping[str, Any] \| None；`context`: Mapping[str, Any] | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `summarize` (L189) | `results`: list[Mapping[str, Any]] | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 计算奖励、指标或聚合统计。 |
| `summarize.count` (L196) | `key`: str | 标注返回 `int`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `summarize.pct` (L202) | `value`: int；`denominator`: int | 标注返回 `float`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `run` (L246) | `args`: argparse.Namespace | 标注返回 `None`；具体值由分支决定。 | 编排训练、采集、评测或 replay 流程。 |
| `main` (L304) | 无显式业务参数 | 标注返回 `int`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |

### `scripts/eval/select_checkpoint.py`

职责：根据 validation summary 选择满足 guard 的 checkpoint。

代码行数：64。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `main` (L19) | 无显式业务参数 | 标注返回 `int`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |

### `scripts/eval/summarize_validation.py`

职责：Convert one veRL validation generation dump into the fixed 132-task summary.

代码行数：37。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `main` (L22) | 无显式业务参数 | 标注返回 `int`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |

### `scripts/simulate/archive_grpo_200task_analysis.py`

职责：Build the six-object 200-task comparison and immutable raw archive.

代码行数：545。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `sha256` (L41) | `path`: Path | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `read_json` (L52) | `path`: Path | 标注返回 `Any`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `read_jsonl` (L59) | `path`: Path | 标注返回 `list[dict[str, Any]]`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `write_json` (L66) | `path`: Path；`value`: Any | 标注返回 `None`；具体值由分支决定。 | 序列化并持久化内部结果。 |
| `copy_tree` (L74) | `source`: Path；`target`: Path | 标注返回 `list[dict[str, Any]]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `copy_file` (L91) | `source`: Path；`target`: Path | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `task_categories` (L102) | `records`: list[Mapping[str, Any]] | 标注返回 `dict[str, int]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `summary_metrics` (L116) | `summary`: Mapping[str, Any]；`records`: list[Mapping[str, Any]] | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 计算奖励、指标或聚合统计。 |
| `load_comparable` (L147) | `name`: str | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `build_comparison` (L190) | 无显式业务参数 | 标注返回 `dict[str, dict[str, Any]]`；具体值由分支决定。 | 根据配置和中间状态构建项目产物。 |
| `write_comparison` (L204) | `comparison`: Mapping[str, Mapping[str, Any]] | 标注返回 `None`；具体值由分支决定。 | 序列化并持久化内部结果。 |
| `write_curve_archive` (L293) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 序列化并持久化内部结果。 |
| `write_readme` (L310) | `comparison`: Mapping[str, Mapping[str, Any]] | 标注返回 `None`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `annotate_copied_files` (L324) | `entries`: list[dict[str, Any]] | 标注返回 `list[dict[str, Any]]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `build_archive` (L374) | `force`: bool | 标注返回 `Path`；具体值由分支决定。 | 根据配置和中间状态构建项目产物。 |
| `refresh_archive` (L460) | 无显式业务参数 | 标注返回 `Path`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `main` (L532) | 无显式业务参数 | 标注返回 `int`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |

### `scripts/simulate/generate_grpo_pipeline_simulation.py`

职责：Generate a deterministic, provenance-labelled GRPO pipeline simulation.

代码行数：1219。

类型：`StaticTask`。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `StaticTask.aspects` (L151) | 无显式业务参数 | 标注返回 `tuple[str, ...]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_sha256` (L158) | `path`: Path | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_hash_float` (L169) | *`parts` | 标注返回 `float`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_normal` (L177) | `rng`: random.Random；`scale`: float | 标注返回 `float`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_clamp` (L184) | `value`: float；`lower`: float；`upper`: float | 标注返回 `float`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_load_subset` (L191) | `path`: Path | 标注返回 `tuple[list[str], list[str]]`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `_load_fixed32` (L203) | `path`: Path | 标注返回 `tuple[list[str], list[str]]`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `_load_static_tasks` (L221) | `ids`: Sequence[str]；`compositions`: Sequence[str] | 标注返回 `list[StaticTask]`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `_weighted_units` (L243) | `count`: int；`aspects`: int | 标注返回 `int`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_choose_category_tasks` (L252) | `tasks`: Sequence[StaticTask]；`full_count`: int；`partial_count`: int；`seed`: int；`aspect_targets`: Mapping[str, int] \| None | 标注返回 `tuple[list[StaticTask], list[StaticTask], list[StaticTask]]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_choose_category_tasks.score` (L267) | `task`: StaticTask；`category`: str | 标注返回 `tuple[float, float, str]`；具体值由分支决定。 | 计算奖励、指标或聚合统计。 |
| `_allocate_partial_counts` (L295) | `partial`: Sequence[StaticTask]；`target_units`: int；`full_count`: int | 标注返回 `dict[str, int]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_select_aspect_subset` (L328) | `task`: StaticTask；`count`: int；`current`: Counter[str]；`targets`: Mapping[str, int] \| None；`seed`: int | 标注返回 `set[str]`；具体值由分支决定。 | 按固定约束拆分、采样或选择数据。 |
| `_allocate_counts` (L360) | `tasks`: Sequence[StaticTask]；`outcomes`: Mapping[str, int]；`seed`: int；`aspect_targets`: Mapping[str, int] \| None | 标注返回 `tuple[dict[str, set[str]], Counter[str]]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_allocate_integer_choices` (L394) | `tasks`: Sequence[StaticTask]；`minimums`: Mapping[str, int]；`target_ratio`: float；`seed`: int；`lower_bound`: int | 标注返回 `dict[str, int]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_constrained_submission_counts` (L423) | `tasks`: Sequence[StaticTask]；`correct`: Mapping[str, set[str]]；`outcomes`: Mapping[str, int]；`target_ratio`: float；`seed`: int | 标注返回 `dict[str, int]`；具体值由分支决定。 | 编排训练、采集、评测或 replay 流程。 |
| `_constrained_submission_counts.total_units` (L453) | 无显式业务参数 | 标注返回 `int`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_preference_state` (L481) | `task`: StaticTask；`target`: float；`seed`: int；`task_index`: int | 标注返回 `tuple[set[str], set[str]]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_transition_state` (L507) | `task`: StaticTask；`target`: float；`seed`: int | 标注返回 `dict[str, int]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_answer_id` (L551) | `task`: StaticTask；`aspect`: str；`correct`: bool；`best`: bool；`seed`: int | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_build_record` (L572) | `task`: StaticTask；`checkpoint`: int；`target`: Mapping[str, float]；`correct_aspects`: set[str]；`submitted_counts`: Mapping[str, int]；`searched_counts`: Mapping[str, int]；`seed`: int；`index`: int | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 根据配置和中间状态构建项目产物。 |
| `_records_for_checkpoint` (L719) | `tasks`: Sequence[StaticTask]；`checkpoint`: int；`target`: Mapping[str, float]；`outcomes`: Mapping[str, int]；`seed`: int；`aspect_targets`: Mapping[str, int] \| None | 标注返回 `tuple[list[dict[str, Any]], dict[str, Any]]`；具体值由分支决定。 | 校验输入、状态或产物，输出诊断或抛出异常。 |
| `_summary_metrics` (L788) | `summary`: Mapping[str, Any] | 标注返回 `dict[str, float]`；具体值由分支决定。 | 计算奖励、指标或聚合统计。 |
| `_add_search_to_summary` (L805) | `summary`: dict[str, Any]；`records`: Sequence[Mapping[str, Any]]；`denominator`: int | 标注返回 `None`；具体值由分支决定。 | 计算奖励、指标或聚合统计。 |
| `_write_summary` (L813) | `path`: Path；`summary`: Mapping[str, Any] | 标注返回 `None`；具体值由分支决定。 | 序列化并持久化内部结果。 |
| `_training_metrics` (L824) | `seed`: int；`validation_by_step`: Mapping[int, Mapping[str, float]] | 标注返回 `list[dict[str, Any]]`；具体值由分支决定。 | 编排训练、采集、评测或 replay 流程。 |
| `_write_training` (L910) | `root`: Path；`validation_by_step`: Mapping[int, Mapping[str, float]] | 标注返回 `None`；具体值由分支决定。 | 序列化并持久化内部结果。 |
| `_load_donor_index` (L929) | `path`: Path | 标注返回 `dict[str, Mapping[str, Any]]`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `_write_readme` (L944) | `root`: Path | 标注返回 `None`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `_publish_swanlab` (L976) | `root`: Path | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `generate` (L1068) | `output`: Path；`force`: bool；`publish_swanlab`: bool | 标注返回 `Path`；具体值由分支决定。 | 根据配置和中间状态构建项目产物。 |
| `main` (L1202) | 无显式业务参数 | 标注返回 `int`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |

### `scripts/simulate/validate_grpo_pipeline_simulation.py`

职责：Independently validate the deterministic GRPO pipeline simulation.

代码行数：444。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `_sha256` (L48) | `path`: Path | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_read_json` (L59) | `path`: Path | 标注返回 `Any`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `_read_jsonl` (L66) | `path`: Path | 标注返回 `list[dict[str, Any]]`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `_close` (L80) | `actual`: float；`expected`: float；`tol`: float | 标注返回 `bool`；具体值由分支决定。 | 清理资源或恢复状态边界。 |
| `_finite` (L87) | `value`: Any | 标注返回 `bool`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_json_close` (L94) | `actual`: Any；`expected`: Any；`path`: str；`tol`: float | 标注返回 `list[str]`；具体值由分支决定。 | 清理资源或恢复状态边界。 |
| `_task_aspects` (L121) | `task_id`: str | 标注返回 `tuple[str, ...]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_penalty_from_public_fields` (L128) | `record`: Mapping[str, Any]；`reward`: Mapping[str, Any]；`aspects`: Sequence[str] | 标注返回 `float`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_validate_record` (L155) | `record`: Mapping[str, Any]；`expected_id`: str | 标注返回 `list[str]`；具体值由分支决定。 | 校验输入、状态或产物，输出诊断或抛出异常。 |
| `_summary_and_records` (L269) | `root`: Path；`split`: str；`step`: int；`task_ids`: Sequence[str]；`compositions`: Sequence[str] | 标注返回 `tuple[dict[str, Any], list[dict[str, Any]], list[str]]`；具体值由分支决定。 | 计算奖励、指标或聚合统计。 |
| `_search_mean` (L289) | `records`: Sequence[Mapping[str, Any]] | 标注返回 `float`；具体值由分支决定。 | 计算奖励、指标或聚合统计。 |
| `_category_counts` (L296) | `records`: Sequence[Mapping[str, Any]] | 标注返回 `dict[str, int]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_check_training` (L309) | `root`: Path | 标注返回 `list[str]`；具体值由分支决定。 | 校验输入、状态或产物，输出诊断或抛出异常。 |
| `_check_training.values` (L327) | `key`: str | 标注返回 `list[float]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_check_training.std` (L331) | `key`: str | 标注返回 `float`；具体值由分支决定。 | 计算奖励、指标或聚合统计。 |
| `validate` (L354) | `root`: Path | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 校验输入、状态或产物，输出诊断或抛出异常。 |
| `validate.check` (L360) | `name`: str；`condition`: bool；`detail`: str | 标注返回 `None`；具体值由分支决定。 | 校验输入、状态或产物，输出诊断或抛出异常。 |
| `main` (L432) | 无显式业务参数 | 标注返回 `int`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |

### `scripts/train/grpo/apply_verl_patch.py`

职责：Apply hash-checked project connections to the pinned veRL 0.8 trainer.

代码行数：121。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `digest` (L63) | `data`: bytes | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `trainer_path` (L70) | 无显式业务参数 | 标注返回 `Path`；具体值由分支决定。 | 编排训练、采集、评测或 replay 流程。 |
| `main` (L80) | 无显式业务参数 | 标注返回 `int`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |

### `scripts/train/grpo/export_actor.py`

职责：Export a veRL FSDP actor checkpoint as a Hugging Face model.

代码行数：98。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `main` (L19) | 无显式业务参数 | 标注返回 `int`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |

### `scripts/train/grpo/prepare_data.py`

职责：Build or verify hidden-label-free veRL 0.8 UserBench datasets.

代码行数：66。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `build_parser` (L25) | 无显式业务参数 | 标注返回 `argparse.ArgumentParser`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `main` (L48) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |

### `scripts/train/grpo/run_grpo_from_sft.py`

职责：Prepare an SFT model and launch the project GRPO training profile.

代码行数：386。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `_project_path` (L41) | `value`: str \| Path；`field`: str | 标注返回 `Path`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_adapter_complete` (L51) | `path`: Path | 标注返回 `bool`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_merged_model_complete` (L65) | `path`: Path | 标注返回 `bool`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_merge_manifest_matches` (L81) | `path`: Path；`adapter`: Path；`base_model`: str | 标注返回 `tuple[bool, str \| None]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_data_artifact_state` (L112) | `output`: Path | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_command_text` (L129) | `command`: Sequence[str] | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_require_grpo_data_dependency` (L136) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_run` (L151) | `label`: str；`command`: Sequence[str] | 标注返回 `None`；具体值由分支决定。 | 编排训练、采集、评测或 replay 流程。 |
| `build_parser` (L163) | 无显式业务参数 | 标注返回 `argparse.ArgumentParser`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `_merge_command` (L206) | `args`: argparse.Namespace；`adapter`: Path；`merged`: Path | 标注返回 `list[str]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_prepare_command` (L225) | `args`: argparse.Namespace | 标注返回 `list[str]`；具体值由分支决定。 | 根据配置和中间状态构建项目产物。 |
| `_train_command` (L246) | `args`: argparse.Namespace；`merged`: Path | 标注返回 `list[str]`；具体值由分支决定。 | 编排训练、采集、评测或 replay 流程。 |
| `run` (L287) | `args`: argparse.Namespace | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 编排训练、采集、评测或 replay 流程。 |
| `main` (L375) | 无显式业务参数 | 标注返回 `int`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |

### `scripts/train/grpo/shutdown_watchdog.py`

职责：Shut down the host after a GRPO run exits or stops making progress.

代码行数：132。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `parse_args` (L20) | 无显式业务参数 | 标注返回 `argparse.Namespace`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `tmux_session_alive` (L38) | `session`: str | 标注返回 `bool`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `latest_completed_step` (L51) | `log_path`: Path | 标注返回 `int \| None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `main` (L62) | 无显式业务参数 | 标注返回 `int`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |

### `scripts/train/grpo/train_grpo.py`

职责：Compatibility CLI for the project GRPO launcher.

代码行数：22。

本文件主要提供常量、类型或配置，没有可调用函数。

### `scripts/train/sft/build_recovery_sft.py`

职责：Build and audit recovery-boundary SFT records without training a model.

代码行数：83。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `build_parser` (L26) | 无显式业务参数 | 标注返回 `argparse.ArgumentParser`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `main` (L60) | 无显式业务参数 | 标注返回 `int`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |

### `scripts/train/sft/check_teacher_trajectories.py`

职责：Audit collected Teacher trajectories for SFT admission.

代码行数：298。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `_read_jsonl` (L48) | `path`: Path | 标注返回 `tuple[list[dict[str, Any]], list[dict[str, Any]]]`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `_load_checkpoint_records` (L76) | `run_dir`: Path | 标注返回 `tuple[list[dict[str, Any]], list[dict[str, Any]]]`；具体值由分支决定。 | 校验输入、状态或产物，输出诊断或抛出异常。 |
| `_load_artifact_records` (L105) | `gold_path`: Path；`silver_path`: Path | 标注返回 `tuple[list[dict[str, Any]], list[dict[str, Any]]]`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `_load_sft_config` (L124) | `path`: Path | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `_render_records` (L139) | `records`: list[dict[str, Any]]；`config`: Mapping[str, Any] | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 把协议/状态数据渲染成可见文本或消息。 |
| `audit` (L197) | `args`: argparse.Namespace | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `build_parser` (L277) | 无显式业务参数 | 标注返回 `argparse.ArgumentParser`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `main` (L292) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |

### `scripts/train/sft/collect_sft_data.py`

职责：Collect DeepSeek-V4-Flash teacher trajectories from UserBench SFT tasks.

代码行数：630。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `build_parser` (L46) | 无显式业务参数 | 标注返回 `argparse.ArgumentParser`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `_sha256_file` (L135) | `path`: Path | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_stratified_manifest_core` (L147) | `input_path`: Path；`tasks`: tuple[dict[str, object], ...]；`quotas`: dict[str, int]；`target`: int；`field`: str；`seed`: str；`wave_size`: int | 标注返回 `dict[str, object]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_write_stratified_state` (L176) | `path`: Path；`core`: dict[str, object]；`status`: str；`outcomes`: dict[str, object]；`waves`: list[dict[str, object]] | 标注返回 `None`；具体值由分支决定。 | 序列化并持久化内部结果。 |
| `_run_stratified_collection` (L213) | `args`: argparse.Namespace；`tasks`: tuple[dict[str, object], ...]；`quotas`: dict[str, int]；`teacher_runtime`: TeacherRuntime；`simulator_runtime`: UserSimulatorRuntime；`output`: Path；`silver_output`: Path；`rejected_output`: Path；`diagnostics_output`: Path；`run_dir`: Path；`summary`: dict[str, object] | 标注返回 `dict[str, object]`；具体值由分支决定。 | 编排训练、采集、评测或 replay 流程。 |
| `_run_stratified_collection.checkpoint` (L333) | `outcome`: object | 标注返回 `None`；具体值由分支决定。 | 校验输入、状态或产物，输出诊断或抛出异常。 |
| `run` (L426) | `args`: argparse.Namespace | 标注返回 `dict[str, object]`；具体值由分支决定。 | 编排训练、采集、评测或 replay 流程。 |
| `run.checkpoint` (L562) | `outcome` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 校验输入、状态或产物，输出诊断或抛出异常。 |
| `main` (L624) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |

### `scripts/train/sft/merge_lora.py`

职责：Merge the SFT LoRA adapter into a standalone Qwen3.5 GRPO starting point.

代码行数：136。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `choose_model_class` (L14) | `config`: Any；`causal_class`: Any；`multimodal_class`: Any | 标注返回 `Any`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `build_merge_manifest` (L27) | `base_model`: str；`adapter`: Path；`output`: Path；`model_type`: str；`dtype`: str | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 根据配置和中间状态构建项目产物。 |
| `build_parser` (L44) | 无显式业务参数 | 标注返回 `argparse.ArgumentParser`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `main` (L65) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |

### `scripts/train/sft/prepare_stage1_prefix_sft.py`

职责：Extract safe decision prefixes from failed Teacher attempts for Stage-1 SFT.

代码行数：380。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `_read_jsonl` (L61) | `path`: Path | 标注返回 `tuple[list[dict[str, Any]], list[dict[str, Any]]]`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `_valid_tool_pairs` (L85) | `messages`: Any | 标注返回 `tuple[bool, str \| None]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_action_parameters` (L132) | `message`: Mapping[str, Any] | 标注返回 `Mapping[str, Any] \| None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_composition` (L151) | `task_id`: str | 标注返回 `str \| None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_has_final_answer_failure` (L167) | `reasons`: Sequence[Any] | 标注返回 `bool`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_candidate` (L179) | `envelope`: Mapping[str, Any]；`line_number`: int | 标注返回 `tuple[dict[str, Any] \| None, str \| None]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `prepare` (L287) | `diagnostics`: Path；`output`: Path；`manifest`: Path；`keep_all`: bool | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 根据配置和中间状态构建项目产物。 |
| `build_parser` (L358) | 无显式业务参数 | 标注返回 `argparse.ArgumentParser`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `main` (L374) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |

### `scripts/train/sft/sft_train.py`

职责：Validate, render, or LoRA-finetune Qwen3.5 on strict UserBench trajectories.

代码行数：637。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `build_parser` (L33) | 无显式业务参数 | 标注返回 `argparse.ArgumentParser`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `_mapping` (L59) | `value`: Any；`name`: str | 标注返回 `Mapping[str, Any]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_project_path` (L69) | `value`: Any；`name`: str | 标注返回 `Path`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_project_paths` (L80) | `value`: Any；`name`: str | 标注返回 `tuple[Path, ...]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `load_config` (L90) | `path`: Path | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `_audit` (L223) | `config`: Mapping[str, Any]；`limit`: int \| None；`allow_small_smoke`: bool | 标注返回 `tuple[dict[str, Any], tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], dict[str, Any]]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_load_tokenizer` (L325) | `model`: Mapping[str, Any] | 返回运行时计算结果；无值分支可能返回 None。 | 读取/解析外部数据并转为内部结构。 |
| `_render` (L358) | `config`；`train`；`validation`；`schema`；`limit`；`allow_small_smoke` | 返回运行时计算结果；无值分支可能返回 None。 | 把协议/状态数据渲染成可见文本或消息。 |
| `_train` (L406) | `config`；`tokenizer`；`train_examples`；`validation_examples`；`resume` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 编排训练、采集、评测或 replay 流程。 |
| `run` (L563) | `args`: argparse.Namespace | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 编排训练、采集、评测或 replay 流程。 |
| `main` (L629) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |

### `scripts/train/sft/split_train_holdout.py`

职责：Create a deterministic internal SFT validation holdout from train trajectories.

代码行数：442。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `build_parser` (L28) | 无显式业务参数 | 标注返回 `argparse.ArgumentParser`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `_load_jsonl` (L83) | `path`: Path；`tier`: str | 标注返回 `list[dict[str, Any]]`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `_largest_remainder` (L109) | `weights`: dict[str, int]；`total`: int；`capacities`: dict[str, int] | 标注返回 `dict[str, int]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_composition_quotas` (L144) | `records`: list[dict[str, Any]]；`count`: int | 标注返回 `dict[str, int]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_gold_quotas` (L158) | `records`: list[dict[str, Any]]；`composition_quotas`: dict[str, int]；`gold_target`: int | 标注返回 `dict[str, int]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_rank` (L197) | `seed`: str；`record`: dict[str, Any] | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_select` (L206) | `records`: list[dict[str, Any]]；`count`: int；`seed`: str | 标注返回 `list[dict[str, Any]]`；具体值由分支决定。 | 按固定约束拆分、采样或选择数据。 |
| `_sha256` (L235) | `path`: Path | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_atomic_write_lines` (L246) | `path`: Path；`lines`: Iterable[str] | 标注返回 `None`；具体值由分支决定。 | 序列化并持久化内部结果。 |
| `_atomic_write_json` (L256) | `path`: Path；`value`: dict[str, Any] | 标注返回 `None`；具体值由分支决定。 | 序列化并持久化内部结果。 |
| `_write_task_splits` (L268) | `source`: Path；`train_output`: Path；`validation_output`: Path；`selected_ids`: set[str] | 标注返回 `None`；具体值由分支决定。 | 序列化并持久化内部结果。 |
| `run` (L296) | `args`: argparse.Namespace | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 编排训练、采集、评测或 replay 流程。 |
| `main` (L434) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |

### `scripts/train/sft/two_stage_sft.py`

职责：Audit, render, or train the two-stage Qwen3.5-2B SFT curriculum.

代码行数：177。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `build_parser` (L21) | 无显式业务参数 | 标注返回 `argparse.ArgumentParser`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `_load_config` (L48) | `path`: Path | 标注返回 `dict[str, Any]`；具体值由分支决定。 | 读取/解析外部数据并转为内部结构。 |
| `_project_path` (L65) | `value`: Any；`field`: str | 标注返回 `Path`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_stage_paths` (L76) | `config_path`: Path | 标注返回 `tuple[Path, Path \| None]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_adapter_complete` (L93) | `path`: Path | 标注返回 `bool`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_command` (L103) | `config`: Path；`args`: argparse.Namespace；`resume`: Path \| None | 标注返回 `list[str]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_run` (L125) | `label`: str；`command`: list[str] | 标注返回 `None`；具体值由分支决定。 | 编排训练、采集、评测或 replay 流程。 |
| `run` (L135) | `args`: argparse.Namespace | 标注返回 `None`；具体值由分支决定。 | 编排训练、采集、评测或 replay 流程。 |
| `main` (L169) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |

### `tests/smoke/test_userbench_real.py`

职责：Opt-in smoke test for the editable pinned UserBench installation.

代码行数：61。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `test_real_userbench_without_network` (L29) | `monkeypatch` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_real_userbench_without_network.fake_async_evaluator` (L38) | *`args`；**`kwargs` | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |

### `tests/smoke/test_verl_080.py`

职责：Opt-in import contract for the pinned veRL 0.8 runtime.

代码行数：26。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `test_external_verl_version_and_adapter_subclasses` (L18) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |

### `tests/test_actor_policy.py`

职责：CPU contracts for the shared production Actor policy.

代码行数：139。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `_prompt` (L24) | 无显式业务参数 | 标注返回 `list[dict[str, str]]`；具体值由分支决定。 | 把协议/状态数据渲染成可见文本或消息。 |
| `test_runtime_policy_contains_the_production_behavior_contract` (L34) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_runtime_policy_injection_is_idempotent_and_deduplicated` (L63) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_teacher_instruction_is_request_only_and_removed_for_actor_messages` (L77) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_sft_trajectory_records_actor_policy_version` (L99) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_grpo_rows_use_the_same_runtime_policy_without_hidden_labels` (L119) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |

### `tests/test_agent_loop_policy.py`

职责：CPU tests for Agent Loop prompt preparation and policy provenance.

代码行数：253。

类型：`test_agent_loop_run_passes_private_policy_prompt_to_parent.FakeRuntime`。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `prompt` (L29) | 无显式业务参数 | 标注返回 `list[dict[str, object]]`；具体值由分支决定。 | 把协议/状态数据渲染成可见文本或消息。 |
| `test_agent_loop_injects_policy_into_the_system_message_only` (L40) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_agent_loop_policy_injection_is_idempotent` (L51) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_agent_loop_requires_a_system_message_when_enabled` (L62) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_policy_can_be_disabled_without_changing_prompt_content` (L71) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_train_and_validation_use_the_same_default_policy` (L87) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_raw_prompt_is_never_modified_in_place` (L101) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_policy_version_is_recorded_without_hidden_state` (L113) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_agent_loop_config_pins_one_default_for_train_and_validation` (L131) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_agent_loop_run_passes_private_policy_prompt_to_parent` (L152) | `monkeypatch`: pytest.MonkeyPatch | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_agent_loop_run_passes_private_policy_prompt_to_parent.fake_parent_run` (L164) | `sampling_params`；**`kwargs` | 返回运行时计算结果；无值分支可能返回 None。 | 编排训练、采集、评测或 replay 流程。 |
| `test_agent_loop_run_passes_private_policy_prompt_to_parent.FakeRuntime.astart_session` (L175) | `task_id` | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |

### `tests/test_dataset_split.py`

职责：Contracts for the pinned UserBench task split builder.

代码行数：324。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `split_spec` (L54) | 无显式业务参数 | 返回运行时计算结果；无值分支可能返回 None。 | 按固定约束拆分、采样或选择数据。 |
| `split_bundle` (L62) | `split_spec` | 返回运行时计算结果；无值分支可能返回 None。 | 按固定约束拆分、采样或选择数据。 |
| `test_exact_counts_and_disjointness` (L69) | `split_bundle` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_exact_composition_quotas` (L83) | `split_bundle` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_records_follow_example_contract` (L95) | `split_bundle` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_reference_example_records_are_reproduced` (L112) | `split_bundle` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_hash_split_is_deterministic` (L133) | `split_spec`；`split_bundle` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_write_verify_and_refuse_overwrite` (L148) | `tmp_path`；`split_spec`；`split_bundle` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_source_count_drift_fails` (L180) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `_write_small_source` (L189) | `tmp_path`: Path；`rows`: list[dict]；`task`: dict；`task_id`: str | 标注返回 `Path`；具体值由分支决定。 | 序列化并持久化内部结果。 |
| `_first_source_row_and_task` (L214) | 无显式业务参数 | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_corrupt_source_rows_fail` (L241) | `tmp_path`；`mutation`；`message` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_duplicate_source_task_id_fails` (L261) | `tmp_path` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_missing_source_column_fails` (L274) | `tmp_path` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_cli_dry_run_writes_nothing` (L288) | `tmp_path` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |

### `tests/test_evaluation_pipeline.py`

职责：项目自有实现文件，负责 `tests/test_evaluation_pipeline.py` 对应阶段的逻辑。

代码行数：253。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `result` (L27) | `task_id`: str；`reward`: float；`valid`: bool；`composition`: str | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_summary_uses_fixed_denominator_for_invalid_and_missing` (L57) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_summary_keeps_metric_schema_when_every_task_is_invalid` (L72) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_actor_runtime_rejects_wrong_frozen_stage_model` (L85) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_explicit_infrastructure_retry_preserves_attempt_diagnostics` (L100) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `summary` (L117) | `correct`；`aligned`；`reward`；`valid`；`efficiency` | 返回运行时计算结果；无值分支可能返回 None。 | 计算奖励、指标或聚合统计。 |
| `test_checkpoint_selection_applies_gates_and_tiebreaks` (L134) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_validation_directory_is_summarized_and_selected_atomically` (L152) | `tmp_path` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_validation_directory_is_summarized_and_selected_atomically.rows` (L167) | `reward` | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_subset_contract_and_comparison_use_subset_denominator` (L213) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_formal_comparison_requires_complete_matching_contract` (L242) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |

### `tests/test_evaluation_rollout.py`

职责：项目自有实现文件，负责 `tests/test_evaluation_rollout.py` 对应阶段的逻辑。

代码行数：181。

类型：`_FakeWrapper`、`_FakeActor`。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `_FakeWrapper.__init__` (L27) | `task_id`；`simulator`；`config`；`source_root` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_FakeWrapper.areset` (L37) | 无显式业务参数 | 返回运行时计算结果；无值分支可能返回 None。 | 清理资源或恢复状态边界。 |
| `_FakeWrapper.reward_task` (L45) | 无显式业务参数 | 返回运行时计算结果；无值分支可能返回 None。 | 计算奖励、指标或聚合统计。 |
| `_FakeWrapper.reward_snapshot` (L51) | 无显式业务参数 | 返回运行时计算结果；无值分支可能返回 None。 | 计算奖励、指标或聚合统计。 |
| `_FakeWrapper.astep` (L57) | `action` | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_FakeWrapper.close` (L76) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 清理资源或恢复状态边界。 |
| `_FakeActor.__init__` (L85) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_FakeActor.generate_action` (L99) | `messages` | 返回运行时计算结果；无值分支可能返回 None。 | 根据配置和中间状态构建项目产物。 |
| `test_guard_rejects_before_simulator_and_renders_public_feedback` (L129) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_raw_rollout_remains_explicitly_available_for_ablation` (L168) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |

### `tests/test_grpo_adapter.py`

职责：Offline tests for provider-neutral portions of the veRL adapter.

代码行数：405。

类型：`FakeWrapper`。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `FakeWrapper.__init__` (L39) | `task_id`；`result`；`snapshot` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `FakeWrapper.astep` (L49) | `action` | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `FakeWrapper.reward_snapshot` (L56) | 无显式业务参数 | 返回运行时计算结果；无值分支可能返回 None。 | 计算奖励、指标或聚合统计。 |
| `FakeWrapper.close` (L62) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 清理资源或恢复状态边界。 |
| `_recovery_session` (L70) | `answered_aspect` | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_rollout_extra_info_duplicates_and_validates_task_id` (L111) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_tool_returns_zero_tool_reward_and_terminal_reward_v2` (L123) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_tool_returns_zero_tool_reward_and_terminal_reward_v2.scenario` (L128) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_malformed_tool_call_returns_stable_error_without_stepping_env` (L166) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_malformed_tool_call_returns_stable_error_without_stepping_env.scenario` (L171) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_parallel_tool_calls_terminate_before_environment_step` (L194) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_no_tool_output_is_a_penalized_protocol_error` (L219) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_answer_only_recovery_rejects_without_environment_step` (L247) | `parameters` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_answer_only_recovery_rejects_without_environment_step.scenario` (L252) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_visible_but_wrong_answer_is_executed_and_recovery_succeeds` (L271) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_visible_but_wrong_answer_is_executed_and_recovery_succeeds.scenario` (L276) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_answer_only_rejects_an_already_answered_aspect` (L307) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_answer_only_rejects_an_already_answered_aspect.scenario` (L312) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_recovery_generation_cannot_be_started_twice` (L331) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_verl_yaml_paths_and_simulator_roles_are_consistent` (L345) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |

### `tests/test_grpo_pipeline.py`

职责：项目自有实现文件，负责 `tests/test_grpo_pipeline.py` 对应阶段的逻辑。

代码行数：1038。

类型：`test_groups_are_restored_to_original_prompt_order.Scalar`、`test_bounded_sampler_restores_cross_batch_group_order.Scores`、`test_bounded_sampler_restores_cross_batch_group_order.Output`、`test_bounded_sampler_restores_cross_batch_group_order.DataProto`、`test_bounded_sampler_keeps_valid_rows_across_batches_and_preserves_fields.Scores`、`test_bounded_sampler_keeps_valid_rows_across_batches_and_preserves_fields.Output`、`test_bounded_sampler_keeps_valid_rows_across_batches_and_preserves_fields.DataProto`、`test_bounded_sampler_uses_degraded_rows_only_when_clean_candidates_are_insufficient.Scores`、`test_bounded_sampler_uses_degraded_rows_only_when_clean_candidates_are_insufficient.Output`、`test_bounded_sampler_uses_degraded_rows_only_when_clean_candidates_are_insufficient.DataProto`。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `_fake_verl_padding_modules` (L35) | 无显式业务参数 | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_grpo_worker_padding_hook_uses_pure_torch_only_when_flash_attn_is_missing` (L61) | `monkeypatch` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_grpo_worker_padding_hook_preserves_flash_attn_baseline` (L86) | `monkeypatch` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_grpo_numpy_override_tracks_vllm_and_verl_metadata_conflict` (L116) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_dynamic_sampling_discards_invalid_and_equal_groups` (L132) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_bounded_sampling_three_batches_and_skip_limit` (L149) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_reward_valid_false_is_sampling_invalid` (L167) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_stalled_valid_trajectory_remains_a_dynamic_sampling_candidate` (L180) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_training_and_validation_sampling_profiles_are_disjoint` (L201) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_groups_are_restored_to_original_prompt_order` (L212) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_groups_are_restored_to_original_prompt_order.Scalar.__init__` (L218) | `value` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_groups_are_restored_to_original_prompt_order.Scalar.item` (L224) | 无显式业务参数 | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_bounded_sampler_restores_cross_batch_group_order` (L234) | `monkeypatch` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_bounded_sampler_restores_cross_batch_group_order.Scores.__init__` (L240) | `values` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_bounded_sampler_restores_cross_batch_group_order.Scores.sum` (L246) | `dim` | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_bounded_sampler_restores_cross_batch_group_order.Scores.detach` (L252) | 无显式业务参数 | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_bounded_sampler_restores_cross_batch_group_order.Scores.cpu` (L258) | 无显式业务参数 | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_bounded_sampler_restores_cross_batch_group_order.Scores.tolist` (L264) | 无显式业务参数 | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_bounded_sampler_restores_cross_batch_group_order.Output.__init__` (L272) | `uids`；`rewards`；`timing` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_bounded_sampler_restores_cross_batch_group_order.Output.slice` (L287) | `start`；`stop` | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_bounded_sampler_restores_cross_batch_group_order.DataProto.concat` (L300) | `outputs` | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_bounded_sampler_keeps_valid_rows_across_batches_and_preserves_fields` (L346) | `monkeypatch` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_bounded_sampler_keeps_valid_rows_across_batches_and_preserves_fields.Scores.__init__` (L352) | `values` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_bounded_sampler_keeps_valid_rows_across_batches_and_preserves_fields.Scores.sum` (L358) | `dim` | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_bounded_sampler_keeps_valid_rows_across_batches_and_preserves_fields.Scores.detach` (L364) | 无显式业务参数 | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_bounded_sampler_keeps_valid_rows_across_batches_and_preserves_fields.Scores.cpu` (L370) | 无显式业务参数 | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_bounded_sampler_keeps_valid_rows_across_batches_and_preserves_fields.Scores.tolist` (L376) | 无显式业务参数 | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_bounded_sampler_keeps_valid_rows_across_batches_and_preserves_fields.Output.__init__` (L384) | `row_ids`；`uids`；`rewards`；`valid`；`degraded` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_bounded_sampler_keeps_valid_rows_across_batches_and_preserves_fields.Output.slice` (L416) | `start`；`stop` | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_bounded_sampler_keeps_valid_rows_across_batches_and_preserves_fields.DataProto.concat` (L439) | `outputs` | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_bounded_sampler_keeps_valid_rows_across_batches_and_preserves_fields.generate` (L484) | `batch` | 返回运行时计算结果；无值分支可能返回 None。 | 根据配置和中间状态构建项目产物。 |
| `test_bounded_sampler_uses_degraded_rows_only_when_clean_candidates_are_insufficient` (L526) | `monkeypatch` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_bounded_sampler_uses_degraded_rows_only_when_clean_candidates_are_insufficient.Scores.__init__` (L532) | `values` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_bounded_sampler_uses_degraded_rows_only_when_clean_candidates_are_insufficient.Scores.sum` (L538) | `dim` | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_bounded_sampler_uses_degraded_rows_only_when_clean_candidates_are_insufficient.Scores.detach` (L544) | 无显式业务参数 | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_bounded_sampler_uses_degraded_rows_only_when_clean_candidates_are_insufficient.Scores.cpu` (L550) | 无显式业务参数 | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_bounded_sampler_uses_degraded_rows_only_when_clean_candidates_are_insufficient.Scores.tolist` (L556) | 无显式业务参数 | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_bounded_sampler_uses_degraded_rows_only_when_clean_candidates_are_insufficient.Output.__init__` (L564) | `uids`；`rewards`；`degraded` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_bounded_sampler_uses_degraded_rows_only_when_clean_candidates_are_insufficient.Output.slice` (L585) | `start`；`stop` | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_bounded_sampler_uses_degraded_rows_only_when_clean_candidates_are_insufficient.DataProto.concat` (L600) | `outputs` | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_grpo_profile_wires_tool_budget_and_rollout_reuse` (L648) | `tmp_path` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_grpo_profile_dry_preflight_is_static_only` (L665) | `tmp_path` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_preflight_rejects_incoherent_rollout_reuse_settings` (L687) | `tmp_path` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_preflight_rejects_sampling_profile_drift` (L705) | `tmp_path` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_grpo_dry_run_exposes_stall_configuration` (L731) | `tmp_path`；`flags`；`enabled` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `_write_fake_sft_adapter` (L765) | `path`: Path | 标注返回 `None`；具体值由分支决定。 | 序列化并持久化内部结果。 |
| `test_grpo_from_sft_dry_run_chains_merge_data_and_training` (L775) | `tmp_path` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_grpo_from_sft_rejects_missing_adapter_before_any_write` (L830) | `tmp_path` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_grpo_dry_run_accepts_custom_model_and_data_paths` (L858) | `tmp_path` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_verl_data_has_no_hidden_labels` (L895) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_verl_data_manifest_pins_runtime_and_generator_versions` (L918) | `tmp_path` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_actor_export_requires_passed_selected_checkpoint` (L945) | `tmp_path` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_preflight_rejects_turn_credit_mode_enabled_drift` (L976) | `tmp_path` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_grpo_dry_run_connects_turn_credit_mode_to_loop_and_trainer` (L998) | `tmp_path`；`mode` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_hash_checked_verl_connection_contains_turn_credit_hook` (L1030) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |

### `tests/test_imports.py`

职责：Project import checks.

代码行数：67。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `test_stage_packages_import` (L7) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_dataset_split_public_api_imports` (L24) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_refactored_stage_facades_preserve_public_symbols` (L43) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |

### `tests/test_inference_gate.py`

职责：项目自有实现文件，负责 `tests/test_inference_gate.py` 对应阶段的逻辑。

代码行数：138。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `_args` (L25) | `tmp_path`: Path | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_fixed_manifest_counts_and_task_ids` (L47) | `tmp_path`: Path | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_prompt_conditions_are_public_and_nonduplicating` (L64) | `tmp_path`: Path | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_normal_result_probes_are_answer_required_with_visible_ids` (L92) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_manifest_supports_32_task_closed_loop_validation` (L114) | `tmp_path`: Path | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_probe_metric_definitions` (L127) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |

### `tests/test_public_control.py`

职责：CPU tests for the public-only control state boundary.

代码行数：449。

类型：`_FeedbackWrapper`。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `_action` (L31) | `choice`: str；`content`: str；`thought`: str | 标注返回 `UserBenchAction`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_public_aspects_only_use_explicit_initial_message_mentions` (L41) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_observation_classifier_uses_text_not_hidden_diagnostics` (L57) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_public_signatures_do_not_use_actor_thought_or_hidden_aspects` (L81) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_normal_search_records_only_visible_ids_for_public_aspects_and_requires_answer` (L96) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_visible_wrong_answer_is_publicly_answered_without_correctness_lookup` (L115) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_unseen_answer_is_recorded_as_actor_input_but_not_marked_answered` (L134) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_first_and_second_public_search_fallbacks_are_separate_recovery_signals` (L146) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_no_public_aspect_does_not_invent_a_search_recovery_target` (L175) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_no_progress_threshold_is_public_and_does_not_need_reward_snapshot` (L189) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_public_aspect_advance_follows_initial_message_order_and_terminates` (L204) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_renderer_contains_public_evidence_only` (L223) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `_FeedbackWrapper.close` (L243) | 无显式业务参数 | 返回运行时计算结果；无值分支可能返回 None。 | 清理资源或恢复状态边界。 |
| `test_session_feedback_entry_is_unified_and_idempotent` (L251) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_session_public_phase_ledger_counts_opportunities_and_guard_rejections` (L270) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_actor_control_renderer_has_stable_normal_snapshot` (L294) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_actor_control_renderer_names_each_recovery_phase_and_allowlist` (L311) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_actor_control_renderer_names_each_recovery_phase_and_allowlist.action` (L315) | `choice`: str；`content`: str | 标注返回 `UserBenchAction`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_actor_control_renderer_never_serializes_hidden_reward_fields` (L385) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_session_feedback_leakage_guard_excludes_hidden_values_and_reward_fields` (L403) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_public_reducer_api_has_no_hidden_reward_inputs` (L445) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |

### `tests/test_public_entrypoints.py`

职责：Placeholder for future public script and CLI contract tests.

代码行数：1。

本文件主要提供常量、类型或配置，没有可调用函数。

### `tests/test_public_phase_guard.py`

职责：CPU tests for the finite public recovery phase guard.

代码行数：342。

类型：`_PublicGuardWrapper`。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `_action` (L29) | `choice`: str；`content`: str | 标注返回 `UserBenchAction`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_normal_eliciting_turn_preserves_progress_and_then_requires_answer` (L39) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_synonymous_no_preference_repeats_force_search_required` (L63) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_threshold_forces_search_and_rejects_action` (L83) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_normal_candidates_force_one_visible_answer_only` (L100) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_fallback_retry_is_one_substantive_rewrite_then_blocks` (L122) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_first_fallback_rewritten_query_can_recover_to_answer_required` (L157) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_blocked_aspect_rejects_old_search_but_accepts_next_aspect_after_advance` (L178) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_query_normalization_ignores_order_but_accepts_new_public_token` (L216) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_answered_and_blocked_aspects_advance_and_terminate_separately` (L226) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `_PublicGuardWrapper.__init__` (L262) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_PublicGuardWrapper.astep` (L269) | `action` | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_PublicGuardWrapper.reward_snapshot` (L277) | 无显式业务参数 | 返回运行时计算结果；无值分支可能返回 None。 | 计算奖励、指标或聚合统计。 |
| `_PublicGuardWrapper.close` (L283) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 清理资源或恢复状态边界。 |
| `test_session_render_actor_feedback_is_unified_and_idempotent` (L291) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_tool_guard_rejects_answer_required_search_before_environment` (L311) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_tool_guard_rejects_answer_required_search_before_environment.scenario` (L316) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |

### `tests/test_public_rendering_fix.py`

职责：项目自有实现文件，负责 `tests/test_public_rendering_fix.py` 对应阶段的逻辑。

代码行数：157。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `_action` (L29) | `choice`: str；`content`: str | 标注返回 `UserBenchAction`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_public_completion_hint_renders_search_required_and_rejects_action` (L38) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_advance_preserves_explicit_answered_switch_note` (L57) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_advance_preserves_explicit_blocked_switch_note` (L76) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_boundary_phase_hint_round_trips_without_hidden_fields` (L93) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_inference_gate_renders_corrected_public_phases` (L121) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_switch_note_clears_on_next_public_event` (L148) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |

### `tests/test_recovery_boundaries.py`

职责：项目自有实现文件，负责 `tests/test_recovery_boundaries.py` 对应阶段的逻辑。

代码行数：224。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `_call` (L25) | `thought`: str；`choice`: str；`content`: str | 标注返回 `dict`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_tool` (L47) | `content`: str | 标注返回 `dict`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_messages` (L54) | 无显式业务参数 | 标注返回 `list[dict]`；具体值由分支决定。 | 把协议/状态数据渲染成可见文本或消息。 |
| `test_schema_boundaries_and_public_replay` (L80) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_grpo_transcript_parser_keeps_only_public_call_fields` (L115) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_split_assignment_precedes_extraction_and_eval_is_not_training` (L133) | `tmp_path`: Path | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_normalizer_drops_record_level_hidden_fields` (L178) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_write_extraction_manifest_and_targets_deferred` (L200) | `tmp_path`: Path | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |

### `tests/test_recovery_sft.py`

职责：项目自有实现文件，负责 `tests/test_recovery_sft.py` 对应阶段的逻辑。

代码行数：274。

类型：`FakeQwenTokenizer`。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `FakeQwenTokenizer.apply_chat_template` (L43) | `conversation`；`tools`；`tokenize`；`add_generation_prompt`；`enable_thinking` | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_call` (L64) | `call_id`: str；`choice`: str；`content`: str | 标注返回 `dict`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_tool` (L86) | `call_id`: str；`content`: str | 标注返回 `dict`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_target_record` (L98) | `choice`: str；`content`: str；`split`: str | 标注返回 `dict`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_recovery_prompt_injects_policy_note_and_does_not_mutate_source` (L137) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_recovery_reuses_action_only_loss_mask_for_final_target` (L161) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_recovery_audit_checks_visible_answer_and_teacher_leakage` (L182) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_recovery_audit_rejects_overlong_without_truncation` (L210) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_recovery_requires_public_system_and_final_target` (L227) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_public_warning_history_is_kept_as_recovery_context` (L238) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_build_quarantines_exact_duplicates_but_allows_same_task_boundaries` (L255) | `tmp_path`: Path | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |

### `tests/test_recovery_sft_decision.py`

职责：项目自有实现文件，负责 `tests/test_recovery_sft_decision.py` 对应阶段的逻辑。

代码行数：79。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `test_followup_gate_is_not_vacuously_passed` (L23) | `tmp_path`: Path | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_leakage_scan_detects_forbidden_public_text` (L42) | `tmp_path`: Path | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_gate_rows_fail_known_recovery_boundaries` (L57) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |

### `tests/test_recovery_targets.py`

职责：项目自有实现文件，负责 `tests/test_recovery_targets.py` 对应阶段的逻辑。

代码行数：238。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `_call` (L21) | `choice`: str；`content`: str；`thought`: str | 标注返回 `dict`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_tool` (L43) | `content`: str | 标注返回 `dict`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_context` (L51) | `boundary_type`: str；`messages`: list[dict]；`state`: dict；`split`: str；`provenance`: list[dict] \| None | 标注返回 `dict`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_accepted_teacher_search_and_answer_are_reused` (L78) | `tmp_path`: Path | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_first_fallback_gets_one_substantive_same_aspect_retry` (L121) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_second_fallback_switches_to_next_public_aspect` (L155) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_pending_visible_options_is_quarantined_instead_of_guessing` (L191) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_evaluation_targets_are_excluded_from_train_and_hidden_keys_absent` (L216) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |

### `tests/test_repository_layout.py`

职责：Repository-layout contracts adapted from the stage-oriented reference.

代码行数：62。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `test_stage_oriented_directories_exist` (L12) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_runtime_boundaries_have_separate_configs` (L38) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_legacy_flat_entrypoints_are_absent` (L50) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |

### `tests/test_reward.py`

职责：Deterministic completion-priority Travel Reward v3 contracts.

代码行数：239。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `_task` (L20) | 无显式业务参数 | 标注返回 `TravelRewardTask`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_score` (L39) | **`overrides` | 返回运行时计算结果；无值分支可能返回 None。 | 计算奖励、指标或聚合统计。 |
| `test_completion_dominates_and_gold_remains_one` (L56) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_completion_is_correct_answer_rate_not_submission_rate` (L71) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_policy_penalties_are_decomposed_and_bounded` (L100) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_error_counts_decrease_reward_until_each_cap` (L121) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_passive_preference_coverage_is_positive_and_not_a_penalty` (L133) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_blocked_aspect_is_not_completion` (L143) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_public_phase_transition_score_is_vacuously_one_without_opportunities` (L153) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_public_phase_failures_are_small_relative_to_completion` (L163) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_negative_terminal_rewards_are_smooth_and_distinct` (L180) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_search_coverage_distinguishes_progress_and_actor_attempts_affect_efficiency` (L191) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_infrastructure_invalid_is_zero_not_a_negative_training_example` (L214) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_reward_trace_remains_raw_diagnostic_only` (L224) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_invalid_raw_rewards_fail` (L237) | `value` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |

### `tests/test_sft_collection.py`

职责：Offline teacher-trajectory collection tests.

代码行数：1799。

类型：`FakeTeacher`、`FakeWrapper`、`SequenceTeacher`、`RetryWrapper`、`SearchRepairWrapper`、`ConstraintEchoTeacher`、`TwoAspectWrapper`、`ElicitationRepairWrapper`、`UnrecordedJudgmentThenRecoveryWrapper`、`UnrecordedVagueThenRecoveryWrapper`、`NeverRecordsElicitationWrapper`、`JudgmentFallbackWrapper`、`OneJudgmentFallbackWrapper`。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `test_collection_cli_rejects_existing_output_before_runtime_loading` (L43) | `tmp_path`: Path；`monkeypatch`: pytest.MonkeyPatch | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_collection_cli_rejects_existing_output_before_runtime_loading.unexpected_runtime` (L61) | *`args`；**`kwargs` | 标注返回 `None`；具体值由分支决定。 | 编排训练、采集、评测或 replay 流程。 |
| `test_collection_cli_checkpoints_and_resumes_without_recollecting` (L73) | `tmp_path`: Path；`monkeypatch`: pytest.MonkeyPatch | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_collection_cli_checkpoints_and_resumes_without_recollecting.collect` (L100) | `tasks`；`teacher`；`simulator`；`on_outcome`；**`kwargs` | 返回运行时计算结果；无值分支可能返回 None。 | 编排训练、采集、评测或 replay 流程。 |
| `test_collection_cli_checkpoints_and_resumes_without_recollecting.unexpected_collect` (L137) | *`args`；**`kwargs` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 编排训练、采集、评测或 replay 流程。 |
| `test_adaptive_collection_reaches_per_composition_quota_in_waves` (L152) | `tmp_path`: Path；`monkeypatch`: pytest.MonkeyPatch | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_adaptive_collection_reaches_per_composition_quota_in_waves.collect` (L185) | `tasks`；`teacher`；`simulator`；`on_outcome`；**`kwargs` | 返回运行时计算结果；无值分支可能返回 None。 | 编排训练、采集、评测或 replay 流程。 |
| `test_stratified_plan_uses_largest_remainder_and_stable_order` (L243) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_stratified_wave_refills_rejected_stratum_without_repeating_tasks` (L278) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_collection_script_runs_from_source_checkout` (L312) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_collection_cli_uses_fixed_development_batch` (L349) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_teacher_smoke_batches_are_unique_train_tasks_and_pairwise_disjoint` (L392) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_upstream_preference_fields_narrow_local_phase_without_exposing_values` (L422) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_generic_amenity_question_is_rejected_before_environment_consumes_a_step` (L456) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_answer_instruction_can_include_only_public_visible_candidate_facts` (L481) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_answer_validation_rejects_visible_option_missing_public_search_requirement` (L500) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_primary_and_repair_questions_cover_global_preference_taxonomy_regressions` (L531) | `aspect`；`field`；`required_phrase` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_search_validation_rejects_preferences_and_years_absent_from_public_context` (L543) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_search_validation_accepts_inflected_and_paraphrased_public_evidence` (L608) | `aspect`；`context`；`query` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_search_validation_does_not_equate_generic_shared_bed_with_king_bed` (L626) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `FakeTeacher.__init__` (L649) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `FakeTeacher.generate_action` (L658) | `messages`；`force_answer`；`constraint` | 返回运行时计算结果；无值分支可能返回 None。 | 根据配置和中间状态构建项目产物。 |
| `FakeTeacher.close` (L692) | 无显式业务参数 | 返回运行时计算结果；无值分支可能返回 None。 | 清理资源或恢复状态边界。 |
| `FakeWrapper.__init__` (L703) | `task_id`；`runtime`；`config`；**`kwargs` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `FakeWrapper.reset` (L715) | 无显式业务参数 | 返回运行时计算结果；无值分支可能返回 None。 | 清理资源或恢复状态边界。 |
| `FakeWrapper.reward_task` (L722) | 无显式业务参数 | 返回运行时计算结果；无值分支可能返回 None。 | 计算奖励、指标或聚合统计。 |
| `FakeWrapper.reward_snapshot` (L734) | 无显式业务参数 | 返回运行时计算结果；无值分支可能返回 None。 | 计算奖励、指标或聚合统计。 |
| `FakeWrapper.astep` (L747) | `action` | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `FakeWrapper.close` (L767) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 清理资源或恢复状态边界。 |
| `task` (L774) | `task_id` | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `simulator` (L790) | 无显式业务参数 | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_collects_tool_only_messages_and_raw_rewards` (L803) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `write_pool` (L842) | `path`；`records` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 序列化并持久化内部结果。 |
| `test_task_pool_disjointness_and_atomic_output` (L852) | `tmp_path` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_collection_config_pins_both_deepseek_roles` (L879) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_teacher_pool_rejects_frozen_test_rows` (L900) | `tmp_path` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `SequenceTeacher.__init__` (L912) | `actions` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `SequenceTeacher.generate_action` (L922) | `messages`；`force_answer`；`constraint` | 返回运行时计算结果；无值分支可能返回 None。 | 根据配置和中间状态构建项目产物。 |
| `test_wrong_phase_choice_is_retried_without_stepping_environment` (L940) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_search_with_invented_preference_is_retried_before_environment_step` (L963) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_state_machine_stops_eliciting_after_active_coverage` (L989) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_bundled_preference_question_is_retried_before_environment_step` (L1008) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_strict_gate_rejects_truncation_fallback_and_missing_answer` (L1038) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_teacher_strict_gate_rejects_reward_policy_failures` (L1083) | `field`；`value`；`reason` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `RetryWrapper.__init__` (L1107) | *`args`；**`kwargs` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `RetryWrapper.astep` (L1115) | `action` | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `SearchRepairWrapper.reward_snapshot` (L1137) | 无显式业务参数 | 返回运行时计算结果；无值分支可能返回 None。 | 计算奖励、指标或聚合统计。 |
| `SearchRepairWrapper.astep` (L1154) | `action` | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_search_not_recorded_is_retried_once_and_failed_turn_is_loss_masked` (L1174) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_whole_trajectory_retry_and_artifact_routing` (L1211) | `tmp_path` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_failed_attempt_diagnostic_contains_safe_partial_trajectory` (L1240) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `ConstraintEchoTeacher.generate_action` (L1279) | `messages`；`force_answer`；`constraint` | 返回运行时计算结果；无值分支可能返回 None。 | 根据配置和中间状态构建项目产物。 |
| `TwoAspectWrapper.__init__` (L1308) | *`args`；**`kwargs` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `TwoAspectWrapper.reward_task` (L1317) | 无显式业务参数 | 返回运行时计算结果；无值分支可能返回 None。 | 计算奖励、指标或聚合统计。 |
| `TwoAspectWrapper.reward_snapshot` (L1335) | 无显式业务参数 | 返回运行时计算结果；无值分支可能返回 None。 | 计算奖励、指标或聚合统计。 |
| `TwoAspectWrapper.astep` (L1351) | `action` | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_composition_22_state_machine_completes_in_eight_steps` (L1387) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_every_canonical_preference_template_satisfies_its_own_contract` (L1421) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_every_elicitation_repair_template_is_distinct_and_satisfies_contract` (L1439) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `ElicitationRepairWrapper.reward_snapshot` (L1465) | 无显式业务参数 | 返回运行时计算结果；无值分支可能返回 None。 | 计算奖励、指标或聚合统计。 |
| `UnrecordedJudgmentThenRecoveryWrapper.astep` (L1487) | `action` | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `UnrecordedVagueThenRecoveryWrapper.astep` (L1509) | `action` | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_uncommitted_fallback_rephrases_without_duplicate_exhaustion` (L1541) | `wrapper_factory`；`repair_marker` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `NeverRecordsElicitationWrapper.reward_snapshot` (L1580) | 无显式业务参数 | 返回运行时计算结果；无值分支可能返回 None。 | 计算奖励、指标或聚合统计。 |
| `test_elicitation_not_recorded_retries_same_field_once_and_masks_failed_turn` (L1595) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_second_unrecorded_elicitation_aborts_without_consuming_another_field` (L1640) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `JudgmentFallbackWrapper.astep` (L1671) | `action` | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `OneJudgmentFallbackWrapper.__init__` (L1689) | *`args`；**`kwargs` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `OneJudgmentFallbackWrapper.astep` (L1696) | `action` | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_one_judgment_fallback_is_admitted_as_silver_and_loss_masked` (L1715) | `tmp_path` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_second_simulator_fallback_aborts_after_the_consumed_step` (L1756) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_per_task_checkpoint_round_trip_and_manifest_resume` (L1779) | `tmp_path` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |

### `tests/test_sft_dataset.py`

职责：Offline action-only SFT dataset and dry-run contracts.

代码行数：767。

类型：`FakeQwenTokenizer`、`test_chat_template_prefix_mismatch_fails.BrokenTokenizer`、`test_no_assistant_completion_tokens_fails.EmptyCompletionTokenizer`。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `FakeQwenTokenizer.encode` (L48) | `text` | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `FakeQwenTokenizer.decode` (L55) | `tokens` | 返回运行时计算结果；无值分支可能返回 None。 | 读取/解析外部数据并转为内部结构。 |
| `FakeQwenTokenizer.apply_chat_template` (L62) | `conversation`；`tools`；`tokenize`；`add_generation_prompt`；`enable_thinking` | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_call` (L90) | `call_id`；`choice`；`content` | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `valid_record` (L111) | `task_id` | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_legacy_reward_v2_fixture_without_new_diagnostics_remains_sft_compatible` (L174) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `valid_prefix_record` (L211) | `task_id` | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `write_jsonl` (L256) | `path`；`records` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 序列化并持久化内部结果。 |
| `test_valid_multiturn_empty_content_renders_only_tool_calls_as_labels` (L267) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_safe_prefix_has_a_separate_gate_and_uses_action_only_rendering` (L291) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_prefix_gate_rejects_a_retained_failed_answer` (L310) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_loss_masked_assistant_turn_is_kept_in_context_but_not_supervised` (L341) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_message_contract_failures` (L403) | `mutate`；`reason` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_reward_admission_failures` (L427) | `field`；`value`；`reason` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_missing_reward_and_vague_feedback_are_both_reported` (L437) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_loader_rejects_duplicate_task_ids` (L452) | `tmp_path` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_sft_gate_ignores_serialized_quality_tier_for_gold_evidence` (L466) | `declared` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_legacy_silver_judgment_fallback_without_marker_is_admitted` (L481) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_overlength_fails_without_truncation` (L503) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_dataset_builder_rejects_whole_overlong_trajectory_and_reports_it` (L515) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_collator_uses_minus_100_for_label_padding` (L541) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_merge_lora_model_class_manifest_and_output_guard` (L559) | `tmp_path` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_train_validation_overlap_fails` (L601) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_sft_task_must_remain_in_its_frozen_split` (L610) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_chat_template_prefix_mismatch_fails` (L621) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_chat_template_prefix_mismatch_fails.BrokenTokenizer.apply_chat_template` (L627) | `conversation`；**`kwargs` | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_no_assistant_completion_tokens_fails` (L646) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_no_assistant_completion_tokens_fails.EmptyCompletionTokenizer.apply_chat_template` (L652) | `conversation`；**`kwargs` | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_rendering_does_not_rewrite_archived_json_arguments` (L676) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_dry_run_is_offline_and_writes_no_checkpoint` (L695) | `tmp_path` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |

### `tests/test_teacher_api.py`

职责：Offline tests for the DeepSeek teacher API boundary.

代码行数：331。

类型：`FakeCompletions`、`FakeClient`。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `runtime` (L22) | 无显式业务参数 | 返回运行时计算结果；无值分支可能返回 None。 | 编排训练、采集、评测或 replay 流程。 |
| `FakeCompletions.__init__` (L35) | `response`；`error` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `FakeCompletions.create` (L43) | **`kwargs` | 返回运行时计算结果；无值分支可能返回 None。 | 根据配置和中间状态构建项目产物。 |
| `FakeClient.__init__` (L57) | `completions` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `response` (L64) | `arguments`；`name`；`count` | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_teacher_runtime_is_secret_safe_and_role_is_pinned` (L80) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_teacher_runtime_environment_defaults_to_three_action_retries` (L92) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_teacher_client_sends_required_official_tool_and_parses_action` (L107) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_teacher_client_sends_required_official_tool_and_parses_action.scenario` (L112) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_teacher_protocol_requires_exactly_one_tool_call` (L142) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_teacher_protocol_requires_exactly_one_tool_call.scenario` (L147) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_teacher_protocol_retries_before_accepting_one_call` (L165) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_teacher_protocol_retries_before_accepting_one_call.scenario` (L170) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_thought_retry_locks_action_and_gives_specific_length_correction` (L195) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_thought_retry_locks_action_and_gives_specific_length_correction.scenario` (L200) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_teacher_force_answer_narrows_tool_schema` (L236) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_teacher_force_answer_narrows_tool_schema.scenario` (L241) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_teacher_request_constraint_narrows_choice_and_content` (L263) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_teacher_request_constraint_narrows_choice_and_content.scenario` (L268) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_teacher_rejects_nonempty_assistant_prose_with_tool_call` (L293) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_teacher_rejects_nonempty_assistant_prose_with_tool_call.scenario` (L298) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_teacher_api_error_does_not_echo_secret_or_endpoint` (L316) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_teacher_api_error_does_not_echo_secret_or_endpoint.scenario` (L321) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |

### `tests/test_turn_credit.py`

职责：CPU contract tests for conservative-turn-credit-v2.

代码行数：462。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `event` (L29) | `index`: int；`aspect`: str；`choice`: str \| None；**`kwargs` | 标注返回 `TurnEvent`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `completed_hotel_events` (L49) | 无显式业务参数 | 标注返回 `list[TurnEvent]`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_completed_aspect_splits_preference_search_answer_budget` (L68) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_one_preference_turn_receives_the_whole_preference_budget` (L89) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_successful_aspect_does_not_add_partial_progress_twice` (L103) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_incomplete_aspect_uses_partial_progress_and_wrong_answer_penalty` (L113) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_blocked_aspect_never_gets_completion_evidence` (L134) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_specific_violation_does_not_stack_semantic_and_guard_penalties` (L151) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_error_root_cause_weights` (L179) | `kwargs`；`expected` | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_infrastructure_failure_and_invalid_reward_zero_credit` (L189) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_multi_aspect_chains_never_cross` (L202) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_assistant_turn_spans_are_separated_by_tool_tokens` (L220) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_bounded_reshaping_preserves_sign_and_lambda_zero_parity` (L232) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_token_weighted_reshaping_exactly_conserves_sequence_advantage` (L251) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_token_weighted_reshaping_exactly_conserves_sequence_advantage.weighted_mean` (L260) | `values` | 返回运行时计算结果；无值分支可能返回 None。 | 计算奖励、指标或聚合统计。 |
| `test_repeated_same_root_cause_is_blamed_once_per_aspect` (L275) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_zero_sequence_advantage_stays_zero` (L289) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_batch_reshaping_uses_turn_records_and_keeps_tool_tokens_zero` (L297) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_batch_turn_span_mismatch_fails_closed` (L340) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_extra_field_contains_no_hidden_ids_or_values` (L364) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_session_ledger_records_public_guard_root_cause_without_environment` (L387) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_session_ledger_off_mode_has_no_runtime_recording` (L420) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_rejected_action_uses_guard_penalty_not_accepted_no_progress_penalty` (L438) | 无显式业务参数 | 标注返回 `None`；具体值由分支决定。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |

### `tests/test_userbench_context.py`

职责：Pinned provenance, simulator isolation, and ContextVar trajectory tests.

代码行数：470。

类型：`CloseOnlyWrapper`。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `CloseOnlyWrapper.__init__` (L32) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `CloseOnlyWrapper.close` (L38) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 清理资源或恢复状态边界。 |
| `_stall_state` (L46) | `threshold`；`aspects` | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_step` (L84) | `state`；`snapshot`；`action`；`feedback`；`step` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_embedded_source_commit_and_license_are_pinned` (L103) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_simulator_repr_hides_api_key_and_process_rejects_mixing` (L115) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_deepseek_simulator_roles_load_from_separate_variables` (L144) | `role`；`prefix` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_grpo_simulator_rejects_a_different_model` (L163) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_concurrent_contexts_keep_task_and_raw_rewards_isolated` (L179) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_concurrent_contexts_keep_task_and_raw_rewards_isolated.trajectory` (L184) | `task_id`；`values` | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_concurrent_contexts_keep_task_and_raw_rewards_isolated.scenario` (L210) | 无显式业务参数 | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_complete_fallback_is_degraded_but_still_reward_valid` (L228) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_fallback_with_missing_snapshot_is_hard_invalid` (L273) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_progress_from_preference_search_and_answer_resets_streak` (L313) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_no_progress_and_invalid_protocol_events_increment_streak` (L369) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_stall_without_visible_answer_evidence_hard_cuts_valid_reward` (L392) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_stall_with_visible_options_enters_one_recovery_pending_state` (L409) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_infrastructure_invalid_does_not_become_stall` (L438) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_feature_off_keeps_no_progress_control_flow_unchanged` (L450) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_second_stall_after_recovery_use_hard_cuts_without_retry` (L464) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |

### `tests/test_userbench_search_contract.py`

职责：CPU tests for the project-authorized UserBench search compatibility patch.

代码行数：110。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `_task` (L13) | 无显式业务参数 | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `_query` (L35) | 无显式业务参数 | 标注返回 `str`；具体值由分支决定。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_base_arguments_plus_preferences_match_one_aspect` (L46) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_wrong_base_argument_does_not_get_positive_deterministic_judgement` (L60) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_sync_search_and_preference_refinement_reuse_visible_candidates` (L70) | `monkeypatch` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_async_judge_uses_same_base_plus_preferences_contract` (L105) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |

### `tests/test_userbench_tools.py`

职责：UserBench action and official tool-schema contracts.

代码行数：162。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `test_action_renders_the_upstream_protocol` (L25) | `choice` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_invalid_actions_fail_stably` (L58) | `parameters`；`message` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_python_and_yaml_schemas_match_the_pinned_official_schema` (L67) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_action_query_issue_normalizes_plural_field_words` (L87) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_longer_service_hint_shadows_embedded_seat_hint_only_at_same_span` (L105) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_layover_duration_is_a_flight_time_question_not_a_path_bundle` (L128) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_carry_on_allowance_is_flight_amenities_not_service` (L138) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_delivery_is_a_restaurant_tags_question` (L149) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_visible_option_extraction_uses_official_boundaries_only` (L159) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |

### `tests/test_userbench_wrapper.py`

职责：Offline wrapper tests using an injected TravelEnv-compatible fake.

代码行数：291。

类型：`FakeTravelEnv`、`test_async_step_reconciles_pinned_one_choice_termination.AsyncOneChoiceEnv`、`test_collection_mode_captures_upstream_fallback_without_raw_stdout.PrintingTravelEnv`。

| 函数/方法 | 输入 | 输出 | 功能 |
|---|---|---|---|
| `FakeTravelEnv.__init__` (L24) | `task_id`；`rewards` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `FakeTravelEnv._observation` (L33) | `index`；`reward`；`complete` | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `FakeTravelEnv.reset` (L47) | `seed`；`options` | 返回运行时计算结果；无值分支可能返回 None。 | 清理资源或恢复状态边界。 |
| `FakeTravelEnv.step` (L54) | `action_input` | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `FakeTravelEnv.step_async` (L73) | `action_input` | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `FakeTravelEnv.close` (L80) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 清理资源或恢复状态边界。 |
| `reset_process_binding` (L89) | 无显式业务参数 | 生成器/迭代器，逐步产出值。 | 清理资源或恢复状态边界。 |
| `runtime` (L98) | 无显式业务参数 | 返回运行时计算结果；无值分支可能返回 None。 | 编排训练、采集、评测或 replay 流程。 |
| `build_wrapper` (L110) | `fake`；`config` | 返回运行时计算结果；无值分支可能返回 None。 | 根据配置和中间状态构建项目产物。 |
| `test_reset_and_sync_steps_project_actor_safe_feedback_only` (L123) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_async_step_uses_the_async_environment_path` (L149) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_async_step_uses_the_async_environment_path.scenario` (L154) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_async_reset_uses_non_blocking_wrapper_entrypoint` (L172) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_async_reset_uses_non_blocking_wrapper_entrypoint.scenario` (L177) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_async_step_reconciles_pinned_one_choice_termination` (L191) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_async_step_reconciles_pinned_one_choice_termination.AsyncOneChoiceEnv.step_async` (L200) | `action_input` | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_async_step_reconciles_pinned_one_choice_termination.scenario` (L209) | 无显式业务参数 | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_collection_mode_captures_upstream_fallback_without_raw_stdout` (L228) | `capsys` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_collection_mode_captures_upstream_fallback_without_raw_stdout.PrintingTravelEnv.step_async` (L234) | `action_input` | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_collection_mode_captures_upstream_fallback_without_raw_stdout.scenario` (L245) | 无显式业务参数 | 返回运行时计算结果；无值分支可能返回 None。 | 实现所在模块的局部业务逻辑并维护相关不变量。 |
| `test_step_before_reset_and_step_after_close_fail_loudly` (L266) | 无显式业务参数 | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |
| `test_environment_contract_rejects_unpinned_modes` (L289) | `override` | 主要通过副作用更新状态或写出产物，默认返回 None。 | 构造 fixture/输入并断言目标行为，承担回归测试职责。 |

## 8. 运行和验证建议

```bash
# 先验证固定数据和 Python 契约
python scripts/data/build_dataset_splits.py --verify-only
python -m compileall -q src scripts tests
pytest -q

# dry-run 只检查配置和边界，不启动正式训练
python scripts/train/sft/sft_train.py --dry-run
bash scripts/train/grpo/run_grpo.sh --dry-run

# 正式阶段（需 Linux/GPU/外部 simulator 凭据）
bash scripts/train/sft/run_sft.sh
bash scripts/train/grpo/run_grpo.sh
python scripts/eval/select_checkpoint.py --validation-dir outputs/models/grpo/validation_rollouts
```

正式训练、模型服务、UserBench simulator 和评测必须保持独立进程及配置命名空间；任何输出应写入 `outputs/`，不要把 checkpoint、rollout、cache 写入源码目录。

## 9. 注释改造说明

- 已为项目自有 `src/`、`scripts/`、`tests/` 中缺少说明的模块、类和函数补充 `[项目注释]`，每个函数至少说明功能、输入和输出；原有高质量 docstring 保留不覆盖。
- 注释是导航层，不替代 Reward、turn-credit、UserBench public-control 等规范文档；发生冲突时以实现契约和对应 `docs/` 规范为准。
- 没有修改 `environments/UserBench/`、已有评测原始目录或生成模型产物。

