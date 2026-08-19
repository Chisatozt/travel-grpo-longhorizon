# Recovery SFT decision (inference-gate-v1)

> **归档 / Historical evaluation record.** 本报告对应已结束且当前输出目录已清理的 recovery/inference gate rerun，不是当前 Baseline/SFT/GRPO 200-Task 结果。

**Decision: NO-GO for the current controller/prompt contract.**

No model parameters were changed and no training was started. The public-state rendering fix was applied and rechecked in this rerun; the remaining failures are policy/grounding gates.

## Gate results (condition B)

| Gate | Observed | Threshold | Result |
|---|---:|---:|---|
| normal answer@1 | 1.0000 | >= 0.95 | PASS |
| answer is exactly one visible option ID | 0.8750 | == 1.00 | FAIL/UNPROVEN |
| preference-complete search@1 | 1.0000 | >= 0.85 | PASS |
| first fallback exact query repeat | 0.3333 | <= 0.05 | FAIL/UNPROVEN |
| second fallback same-aspect search | 0.0417 | == 0 | FAIL/UNPROVEN |
| answer then action/search | 0.0000 | == 0 (must be observable) | FAIL/UNPROVEN |
| hidden-state leakage | 0 | == 0 | PASS |
| 8-task max_steps | 0.1250 | <= 0.25 | PASS |

The B closed loop reached max_steps on 12% of tasks and completion on 0%; the answer-follow-up gate is unproven because no closed-loop answer was emitted.

## Root-cause classification

- **Deterministic controller: PARTIAL** — B public guard rejected 32 calls before simulator; repeated-action/search occurred in 87.5% of closed-loop tasks, while completion remained 0.0%.
- **Model policy execution: BLOCKER** — B first-fallback exact repeats=33.3%, confused-history repeated actions=28.1%, and answer calls in the 8-task loop=0.
- **Fallback infrastructure: NOT_PRIMARY_BLOCKER** — B reward_degraded=0.0%; no simulator infrastructure error was recorded in the public summaries.
- **Semantic option selection: BLOCKER** — B normal answer@1=100.0%, but exactly-one-visible-ID=87.5%; the remaining failures are option-grounding errors.
- **Prompt/state rendering: FIXED_RECHECKED** — The public rendering fix was rechecked in this B rerun: preference-complete search@1=100.0% and second-fallback same-aspect search=4.2%. Remaining failures are policy behavior, not the previously stale phase snapshot.
- **Recovery SFT necessity: TARGETED_YES_AFTER_FIX** — Fallback rewrite/switch, visible-ID grounding, and closed-loop completion still fail hard gates; keep recovery SFT targeted and do not start broad training from this gate.

## Required ordering

1. Keep the public reducer/rendering fix and its CPU/public-snapshot assertions; this rerun no longer shows the stale ELICITING phase defect.
2. Repair fallback query rewriting, visible-option-ID grounding, and answer emission. Do not promote rejected/quarantine records by guessing.
3. Run only targeted Recovery SFT after those policy fixes, then rerun this deterministic gate and a 32-task confirmation set.

## Proposed targeted Recovery SFT quota

| Boundary | Train | Validation | Current accepted pool |
|---|---:|---:|---:|
| preference_complete_to_search | 400 | 100 | 131 |
| first_fallback | 200 | 50 | 50 |
| second_fallback | 200 | 50 | 31 |
| valid_search_to_answer | 200 | 50 | 101 |
| repeated_no_progress_action | 200 | 50 | 4729 |

The first pass is 1,200 train / 300 validation records, task-disjoint. The current pools can only support part of this quota; target generation must be repaired or the quota reduced without fabricating targets.

## Reproduction

```bash
PYTHONPATH=src .venv/bin/python scripts/eval/recovery_sft_decision.py \
  --gate outputs/evaluation/inference_gate_sft_merged_b_rerun \
  --output outputs/evaluation/recovery_sft_decision_v2 \
  --markdown docs/evaluation/recovery-sft-decision-v2.md
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src .venv/bin/pytest -q tests/test_recovery_sft_decision.py
```

The machine-readable report is `outputs/evaluation/recovery_sft_decision_v2/report.json`. Its inputs include hashes for the rerun gate and recovery manifests.
