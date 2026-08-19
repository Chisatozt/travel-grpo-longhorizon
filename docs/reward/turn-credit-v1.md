# Causal turn credit v1

> **归档 / Historical reference.** 本页描述 v1 turn-credit 设计；当前 turn-credit 参考以 `turn-credit-v2.md` 和当前 GRPO 配置为准，不应将本页作为现行训练契约。

`causal-turn-credit-v1` is an optional, trainer-only credit assignment layer for multi-turn GRPO. Travel Reward v3 remains the sole trajectory objective and the sole value used by dynamic sampling. Turn credit does not add step rewards and does not alter `rm_scores`.

## Modes

- `off`: default. The AgentLoop records no turn ledger and standard GRPO advantages are unchanged.
- `shadow`: the AgentLoop records hidden-ID-free turn evidence and metrics, but standard advantages are unchanged.
- `train`: records the same evidence, then reshapes standard GRPO advantages after `compute_advantage`.

Training and validation use the same Actor policy and public guard. The turn-credit mode is also the same at rollout time so validation can audit traces; validation does not execute the training advantage transform.

## Causal allocation

For each publicly named aspect that reaches a correct answer after a normal candidate list:

| Cause | Total evidence |
| --- | ---: |
| preference-resolving action turns before search | `+0.20`, split across contributors |
| last normal search before the accepted answer | `+0.35` |
| correct visible answer | `+0.45` |

An incomplete aspect can receive provisional evidence: `+0.05` per resolved preference field capped at `+0.10`, `+0.20` for a normal candidate search, and `+0.05` for submitting a visible answer ID. A wrong visible answer also receives `-0.15`, so its default net evidence is `-0.10`.

Only one root-cause error penalty is selected per turn. Exact query repetition, wrong aspect, and invisible answer are `-0.20`; generic guard rejection, wrong answer, malformed/no-tool output are `-0.15`; no-progress and semantic repetition are `-0.10`. Infrastructure-invalid turns receive zero evidence. All evidence is clipped to `[-0.50, +0.50]`.

Actor-visible public control continues to depend only on the public whitelist. Hidden preference counts and correct-answer membership can inform trainer-only evidence, but cannot alter public state, guard decisions, prompts, simulator calls, or Actor feedback. The serialized `turn_credit` record contains only version, mode, validity, turn count, and numeric evidence. It contains no preference IDs, correct IDs, best IDs, reward snapshots, or hidden values.

## Advantage transform

For a trajectory sequence advantage `A` and turn evidence `c_t`:

```text
z_t = (c_t - mean(c)) / (std(c) + epsilon)
aligned_t = sign(A) * z_t
m_t = clip(1 + band * aligned_t, 1 - band, 1 + band)
A_t = A * ((1 - lambda) + lambda * m_t)
```

Defaults are `lambda=0.50` and `band=0.20`, so each assistant turn stays within `0.90x` to `1.10x` of the standard sequence advantage. The transform cannot reverse its sign. Tool-observation and padding tokens remain masked to zero. A turn-count/token-span mismatch fails closed.

This sign alignment means that useful turns in a positive trajectory receive more positive weight, while useful turns in a negative trajectory are punished less. Error turns in a negative trajectory are punished more. A zero or constant sequence advantage remains zero.

## Runtime flow

```text
AgentLoop opens TurnEvent before generation
  -> public guard or simulator result closes the event
  -> terminal Reward v3 is computed once
  -> causal traces allocate numeric evidence
  -> veRL computes standard GRPO advantage
  -> train mode only: bounded reshape over assistant-token spans
```

The hash-checked project patch in `scripts/train/grpo/apply_verl_patch.py` installs the final hook immediately after veRL standard advantage computation. Do not edit `.venv` manually. Run project setup to install or verify the pinned connection patch before formal training.

## Commands

```bash
# Verify logging and allocations without changing optimization
bash scripts/train/grpo/run_grpo.sh \
  --turn-credit-mode shadow \
  --output outputs/models/grpo-turn-shadow

# Enable bounded turn-level optimization
bash scripts/train/grpo/run_grpo.sh \
  --turn-credit-mode train \
  --turn-credit-lambda 0.5 \
  --turn-credit-band 0.2 \
  --output outputs/models/grpo-turn-v1

# The from-SFT entry point forwards the same options
bash scripts/train/grpo/run_grpo_from_sft.sh \
  --turn-credit-mode train \
  --turn-credit-lambda 0.5 \
  --turn-credit-band 0.2 \
  --output outputs/models/grpo-turn-v1
```

Recommended rollout is `off -> shadow -> train`. Before enabling `train`, audit turn/span mismatch count, reward-valid rate, evidence distribution by outcome, completion, preference coverage, guard rejection, and KL/entropy stability.
