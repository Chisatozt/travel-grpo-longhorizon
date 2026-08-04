# Repository layout

The layout follows the stage-oriented organization of
`qiqihezh/agentic-grpo-longhorizon` at reference commit
`2004fcbc747b9b282bc0a8ce0c006683c7c42751`, adapted to UserBench and the
`travel_grpo` Python package.

## Ownership boundaries

- `configs/interaction_config/` owns UserBench interaction and the two user
  simulator configurations. Training and evaluation simulator endpoints remain
  independent.
- `configs/tool_config/` owns the actor-visible tool schema.
- `configs/train/` and `configs/eval/` own stage-specific reproducibility
  settings.
- `src/travel_grpo/envs/` owns project integration code around the pinned
  UserBench snapshot; it must not modify `environments/UserBench/`.
- `src/travel_grpo/models/` owns actor inference and external teacher clients.
  Simulator clients stay in the environment interaction boundary.
- `scripts/` contains thin, stage-grouped launchers. Reusable logic belongs in
  `src/travel_grpo/`.
- `experiments/` is reserved for small, auditable summaries and configurations.
  Large checkpoints, rollouts, and logs belong under ignored output paths.

## Implementation status

The deterministic task-splitting implementation lives in
`src/travel_grpo/data/`, with its entry point under `scripts/data/`. The
project-owned UserBench lifecycle wrapper lives in `src/travel_grpo/envs/`, and
the optional veRL 0.8 adapter lives in
`src/travel_grpo/training/grpo/adapter/`. The DeepSeek teacher collector lives
in `src/travel_grpo/training/sft_collection.py`. SFT merge, GRPO launch/export,
and resumable frozen-evaluation entry points are implemented; formal GPU/API
runs remain external operations and no benchmark result is claimed.
