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
`src/travel_grpo/data/userbench.py`, with its entry point under `scripts/data/`.
Recovery-boundary extraction and target construction form the adjacent
`src/travel_grpo/data/recovery/` stage package. The actor-visible message
normalizer is shared from `src/travel_grpo/protocols/` so evaluation and SFT do
not depend on the recovery extractor to parse protocol messages.

The project-owned UserBench lifecycle wrapper lives in `src/travel_grpo/envs/`.
Trajectory accounting shared by the environment ledger and GRPO lives in
`src/travel_grpo/trajectory/`, while the optional veRL 0.8 adapter and its
profile launcher live in `src/travel_grpo/training/grpo/`. Teacher collection,
SFT rendering, and recovery SFT are grouped under
`src/travel_grpo/training/sft/`. Evaluation runtime orchestration lives in
`src/travel_grpo/evaluation/`; `scripts/` retains the historical CLI paths as
thin compatibility shims.

Within the SFT package, `contracts.py` owns trajectory, diagnostic, and
checkpoint schemas; `planning.py` owns task-pool validation, evaluation
disjointness, deterministic quotas, and adaptive waves; `collection.py` owns
runtime collection, strict admission, retries, and artifact persistence.
`dataset.py` owns action-only rendering and `recovery.py` owns recovery-SFT
target construction. The historical flat SFT paths remain import-compatible.

SFT merge, GRPO launch/export, and resumable frozen-evaluation entry points
remain available under `scripts/`; formal GPU/API runs remain external
operations and no benchmark result is claimed without reproducible artifacts.
