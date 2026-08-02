# Repository guidance

This repository is an early-stage UserBench-based travel-agent post-training project. The reproducible task-splitting pipeline is implemented; model training and rollout pipelines remain scaffolds.

- Keep the actor model, training user simulator, and evaluation user simulator as separate runtime boundaries.
- Treat `environments/UserBench/` as a pinned third-party snapshot. Do not edit it during normal project work.
- When upgrading UserBench, replace the complete snapshot and update `environments/UserBench/EMBEDDED_SOURCE.json` in the same change.
- Do not claim training or benchmark results without committed artifacts and reproducible configuration.
- Keep generated models, rollouts, logs, and caches under ignored output directories.
