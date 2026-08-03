# Reward

The current environment contract preserves UserBench's raw per-step reward exactly. There is no normalization, binarization, process reward, or project-specific shaping.

The veRL tool returns reward `0.0` and includes the upstream reward only as diagnostic metadata. `UserBenchInteraction.calculate_score()` returns the sum of the raw step rewards, avoiding duplicate accounting.
