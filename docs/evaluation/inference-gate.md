# SFT-merged Actor inference gate

`inference_gate.py` is an inference-only A/B gate for the frozen
`outputs/models/sft-merged` checkpoint. It does not update parameters and does
not start GRPO. A sends the historical prompt (runtime policy removed); B
sends the versioned `ACTOR_RUNTIME_POLICY`, the public control note, and the
public phase guard.

The fixed manifest contains 24 normal-search-result contexts, 24 first
fallback contexts, 24 second-fallback contexts, 24 preference-complete
contexts, 32 Actor-confused histories, and the same eight frozen evaluation
tasks for both closed-loop conditions. Formal evaluation contexts are not
used for the boundary probes; the eight closed-loop rows are selected by the
existing balanced rule and exclude compositions `333`, `334`, `444`, and
`2222`.

Run (with the local Actor endpoint and the configured `EVAL_USER_SIM_*`
variables available):

```bash
set -a; source .env; set +a
export ACTOR_MODEL=outputs/models/sft-merged
export ACTOR_BASE_URL=http://127.0.0.1:8000/v1
export ACTOR_API_KEY=local
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
PYTHONPATH=src .venv/bin/python scripts/eval/inference_gate.py \
  --output outputs/evaluation/inference_gate_sft_merged
```

`--dry-run` writes only the fixed manifest and sample index. `--probes-only`
and `--closed-loop-only` run the corresponding portions. The manifest records
the policy/phase versions, policy hash, model merge-manifest hash, decoding
parameters, task IDs, and output paths; credentials and endpoint values are
never written.

The output directory contains per-sample JSON, `A/probes-summary.json`,
`B/probes-summary.json`, closed-loop summaries, `comparison.json`, and the
fixed `manifest.json`. Transcripts are normalized to actor-visible messages;
hidden preference IDs, correctness labels, reward snapshots, and terminal
reward values are not copied into them or into the closed-loop summaries.
## Public phase rendering audit

The B prompt reconstruction carries the public boundary phase explicitly.
`preference_complete_to_search` records render `SEARCH_REQUIRED`; terminal
contexts are excluded from the search@1 probe. Second-fallback records advance
only after retaining a one-turn `SWITCH_ASPECT_REQUIRED` transition note, which
identifies the blocked/answered public aspect and the next public aspect without
using reward state.

