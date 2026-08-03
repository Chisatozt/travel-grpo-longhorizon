# GRPO environment adapter

The adapter targets veRL 0.6.1 and keeps veRL outside the core package dependency graph. Install UserBench from the pinned snapshot and veRL from a separate checkout:

```bash
pip install -e environments/UserBench
pip install -e /path/to/verl
```

`configs/tool_config/userbench_tools.yaml` exposes exactly one native tool. The schema is copied verbatim from the pinned `environments/UserBench/schema/interact_tool.yaml`; each model turn may execute at most one call. Dataset rows must provide matching task IDs through both `extra_info.interaction_kwargs.task_id` and `tools_kwargs.interact_with_env.create_kwargs.id`. Use `build_rollout_extra_info()` to construct this payload.

Each asyncio trajectory owns a separate `UserBenchWrapper` and `TravelEnv`. A `ContextVar` carries a mutable session reference into veRL's child tool task while keeping concurrent trajectories isolated. The custom AgentLoop detects upstream `terminated` or `truncated` immediately after tool processing and does not request another actor turn.

TravelGym step rewards are preserved unchanged. The tool reward is always `0.0`; raw step rewards are retained in tool metadata and the interaction score is their sum. This prevents veRL from counting the same environment reward as both a tool reward and a final interaction score.

The pinned UserBench OpenAI clients read `OPENAI_BASE_URL` from process state. The wrapper therefore binds one simulator runtime once per process. Rebinding the same role/model/endpoint is idempotent; switching any of them fails. Start training and formal evaluation in different processes, using `simulator_train.yaml` and `simulator_eval.yaml` respectively.

Training execution and optimizer configuration are not implemented yet. `configs/train/grpo/vanilla_grpo.yaml` only supplies the UserBench-specific multi-turn overlay.

## Optional smoke tests

Default tests are offline. After installing the optional editable dependencies, the dependency-specific checks can be enabled explicitly:

```bash
TRAVEL_GRPO_REAL_USERBENCH_SMOKE=1 pytest tests/smoke/test_userbench_real.py
TRAVEL_GRPO_VERL_SMOKE=1 pytest tests/smoke/test_verl_061.py
```

The real-UserBench smoke test replaces the evaluator before stepping the environment, so it does not make a network call.
