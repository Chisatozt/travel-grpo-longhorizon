# Recovery SFT decision (inference-gate-v1)

**Decision: NO-GO for the current controller/prompt contract.**

No model parameters were changed and no training was started. Targeted Recovery SFT is recommended only after the public-state rendering defects are corrected and the gate is rerun.

## Gate results (condition B)

| Gate | Observed | Threshold | Result |
|---|---:|---:|---|
| normal answer@1 | 1.0000 | >= 0.95 | PASS |
| answer is exactly one visible option ID | 0.8750 | == 1.00 | FAIL/UNPROVEN |
| preference-complete search@1 | 0.0833 | >= 0.85 | FAIL/UNPROVEN |
| first fallback exact query repeat | 0.3333 | <= 0.05 | FAIL/UNPROVEN |
| second fallback same-aspect search | 0.3333 | == 0 | FAIL/UNPROVEN |
| answer then action/search | 0.0000 | == 0 (must be observable) | FAIL/UNPROVEN |
| hidden-state leakage | 0 | == 0 | PASS |
| 8-task max_steps | 0.2500 | <= 0.25 | PASS |

The max-steps gate is exactly 2/8 (the allowed 25%), but completion is 0/8. The answer-follow-up gate is unproven because no closed-loop answer was emitted.

## Root-cause classification

- **Deterministic controller: PARTIAL** — B guard rejected 23 calls before simulator; hard invariants reduce repeats, but closed-loop completion remained 0.0 and the state/render contract still loses phase information.
- **Model policy execution: BLOCKER** — B first-fallback exact repeats 8/24; confused repeated actions 10/32; no answer call in any 8-task B closed loop.
- **Fallback infrastructure: NOT_PRIMARY_BLOCKER** — B reward_degraded=0/8 and infrastructure_errors=0; only A task 06 showed three simulator search fallbacks.
- **Semantic option selection: BLOCKER** — B normal answer@1=24/24 but visible-ID-only=21/24; C5, C16, and C4 were not in the current visible option-ID sets.
- **Prompt/state rendering: BLOCKER** — Preference-complete boundary snapshots render ELICITING with action allowed (20/24 B calls were action); after second fallback the next aspect is rendered as ELICITING rather than an explicit switch contract.
- **Recovery SFT necessity: TARGETED_YES_AFTER_FIX** — Fallback rewrite/switch and visible-ID grounding still fail hard gates, but low target acceptance (preference 131/668; first fallback 50/80; second fallback 31/103) makes immediate broad training unsafe.

## Required ordering

1. Fix the public reducer/rendering so a preference-complete snapshot is `SEARCH_REQUIRED`, and preserve an explicit `SWITCH_ASPECT_REQUIRED` note after the second fallback (including the next public aspect).
2. Regenerate and validate public-only targets. Do not promote rejected/quarantine records by guessing; current acceptance is too low for the affected phases.
3. Run the targeted Recovery SFT described below, then rerun the same deterministic gate and a 32-task confirmation set.

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
PYTHONPATH=src .venv/bin/python scripts/eval/recovery_sft_decision.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src .venv/bin/pytest -q tests/test_recovery_sft_decision.py
```

The machine-readable report is `outputs/evaluation/recovery_sft_decision_v1/report.json`. Its inputs include hashes for the frozen gate and recovery manifests.
