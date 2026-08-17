# Conservative turn credit v2

`conservative-turn-credit-v2` is an optional trainer-only GRPO advantage routing layer. Travel Reward v3 remains the sole trajectory objective. The implementation never changes terminal reward, `rm_scores`, dynamic-sampling scores, public control state, or Actor-visible feedback.

## Routing contract

Standard GRPO first computes one sequence advantage `A`. Turn evidence then produces non-negative relative weights. Positive trajectories route through positive causal evidence; negative trajectories route through sparse root-cause blame. No turn may reverse the sign of `A`.

For a correctly completed public aspect, success evidence is allocated as follows:

| Cause | Budget |
| --- | ---: |
| preference-resolving causal chain | 0.20 |
| successful search producing visible candidates | 0.45 |
| correct visible answer | 0.35 |

Failure evidence remains numeric and trainer-only. The same violation class is blamed at most once per public aspect, generic guard rejection does not stack with a more specific violation on the same turn, and infrastructure-invalid turns receive zero evidence. Evidence is routing metadata, not additive reward.

## Exact conservation

For assistant turn `t`, let `L_t` be the number of response-mask tokens and `w_t` the raw routing weight. The implementation normalizes weights by their token-weighted mean:

```text
route_t = max(evidence_t, 0)              if A > 0
route_t = max(-evidence_t, 0)             if A < 0
raw_t   = 1 + lambda * band * route_t / max(route)
mean_w  = sum(L_t * raw_t) / sum(L_t)
w_t     = raw_t / mean_w
A_t     = A * w_t
```

If no applicable evidence exists, all weights are 1. Therefore:

```text
sum(L_t * A_t) / sum(L_t) == A
sign(A_t) == sign(A)
A == 0  =>  every A_t == 0
```

Tool-observation and padding tokens stay zero. Turn/span or evidence/length mismatch fails closed. `advantages` and GRPO `returns` are reshaped; rewards are untouched.

## Modes and compatibility

- `off`: no ledger and no transform.
- `shadow`: records v2 evidence and metrics without changing optimization.
- `train`: applies conservative routing after standard GRPO advantage computation.

The command-line switches remain unchanged. V1 rollout records deliberately fail the v2 version check, so do not resume a v1 turn-credit optimizer run as v2. Start a new output directory or explicitly keep the old code/config for an exact v1 resume.

```bash
bash scripts/train/grpo/run_grpo_from_sft.sh \
  --turn-credit-mode shadow \
  --output outputs/models/grpo-turn-credit-v2-shadow

bash scripts/train/grpo/run_grpo_from_sft.sh \
  --turn-credit-mode train \
  --turn-credit-lambda 0.5 \
  --turn-credit-band 0.2 \
  --output outputs/models/grpo-turn-credit-v2
```

Before training, audit reward validity, turn/span alignment, evidence sparsity by outcome, token-weighted conservation error, completion, answer submission, guard rejection, KL, and entropy.
