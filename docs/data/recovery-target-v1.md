# recovery-target-v1

> **归档 / Historical artifact contract.** 这是 recovery target 的历史派生格式；对应输出目录已不再保留，不属于当前 200-Task 最终测试主链路。

`recovery-target-v1` is the derived, one-step target format built from
`recovery-boundary-v1` contexts. It is an offline training/evaluation aid; it
does not alter the source trajectories and it never consults hidden reward
state when constructing a target.

## Files and record shape

The target builder writes three JSONL files and a manifest under an ignored
output directory:

* `train.jsonl` — accepted targets from `grpo_train` and `sft_train`.
* `validation.jsonl` — accepted targets from `grpo_validation`.
* `rejected.jsonl` — invalid/quarantined records and valid records belonging
  to the formal `evaluation` split. Evaluation records are marked
  `excluded_evaluation` and are never copied to either training file.
* `manifest.json` — generator version, split counts, boundary counts,
  rejection reasons, and quality checks.

Each record retains the boundary fields and adds:

```json
{
  "schema_version": "recovery-target-v1",
  "target_status": "accepted",
  "task_id": "...",
  "boundary_type": "first_fallback",
  "policy_version": "...",
  "messages": [],
  "public_state_before": {},
  "target_assistant": {
    "role": "assistant",
    "content": "",
    "tool_calls": [{
      "type": "function",
      "function": {"name": "interact_with_env", "arguments": "{...}"}
    }]
  },
  "source": {},
  "quality_checks": {}
}
```

The assistant has no explanatory text and exactly one `interact_with_env`
call. Its JSON arguments contain only the public action shape (`thought`,
`choice`, and `content`); `choice` is a valid action/search/answer choice and
`content` is the corresponding public query or visible option ID.

## Deterministic target rules

Targets are generated in this order:

* `preference_complete_to_search`: reuse the accepted source search when its
  actor-visible prefix matches exactly and the phase/aspect checks pass.
* `valid_search_to_answer`: reuse the accepted source answer only when it is
  exactly one option ID visible in the preceding candidate list.
* `first_fallback`: issue one same-aspect search with a deterministic rewritten
  query; the normalized query must differ substantively from the failed query.
* `second_fallback`: issue an action/search for the next open *public* aspect;
  the failed aspect is not mentioned in the target.
* `repeated_no_progress_action`: search the current public aspect.
* `explicit_no_preference`: ask the next unasked public field, or search the
  current aspect when no field can be selected safely.
* `visible_options_pending_answer`: quarantine. Correctness cannot be inferred
  from public text alone, so no answer ID is guessed.

If a source action cannot be matched to the public context, a phase/aspect
guard fails, or a target would require hidden correctness, the record is
written to `rejected.jsonl` with `target_status: rejected` and a stable
`rejection_reason`.

## Split and leakage contract

Task split assignment is resolved before target construction. The builder
does not perform sample-level random splitting. Formal evaluation tasks are
always excluded from `train.jsonl` and `validation.jsonl`. A quality-check
scan rejects any target containing hidden fields such as
`remaining_preference_ids`, `correct_ids`, `best_ids`, `reward_snapshot`,
reward deltas, or hidden preference values. The final manifest records the
number of hidden-key hits and whether every emitted target has one tool call.

The command-line entry point is:

```bash
PYTHONPATH=src python scripts/data/build_recovery_targets.py \
  --project-root . \
  --contexts outputs/recovery_boundaries/recovery-boundary-v1/contexts.jsonl \
  --output outputs/recovery_targets/recovery-target-v1
```

The implementation may use a Teacher service only as an explicitly added,
public-context fallback in a future extension. The current generator uses no
Teacher API and no GPU; ambiguous records are quarantined.
