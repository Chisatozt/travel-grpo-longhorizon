# Checkpoint scenario pipeline provenance

This directory contains the checkpoint scenario artifacts used by the standalone 200-task analysis. It preserves deterministic task-result generation, Reward-v3 recomputation, checkpoint summaries, non-monotonic step 1--200 metrics, and validator outputs.

No actual model training or evaluation was executed. The authoritative machine-readable record is `PROVENANCE.json`, where `scenario_config.json` records the generation settings and `consistency_report.json` records the independent validation result.
