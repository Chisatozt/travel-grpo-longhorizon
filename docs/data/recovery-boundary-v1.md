# Recovery boundary contexts (`recovery-boundary-v1`)

`scripts/data/extract_recovery_boundaries.py` creates a derived, CPU-only
context set for recovery and phase-boundary work. It reads existing accepted
Teacher trajectories, failed 20-step GRPO rollouts, and A/B probe artifacts;
source files are never rewritten. The default output is the ignored directory
`outputs/recovery_boundaries/recovery-boundary-v1/`.

## Record contract

Each JSONL record contains at least:

| Field | Meaning |
| --- | --- |
| `schema_version` | Exactly `recovery-boundary-v1`. |
| `task_id` | Upstream task identifier. |
| `boundary_type` | One of the seven public boundary labels listed below. |
| `policy_version` | Policy metadata from the source; `unknown` or `mixed` is explicit when unavailable or merged. |
| `messages` | Actor-visible system/user/assistant/tool messages up to the boundary. Tool-call arguments retain only `thought`, `choice`, and `content`. |
| `public_state_before` | Public phase, current public aspect, visible option IDs, fallback count, search counters, and public terminal lists. |
| `target_assistant` | Always `null` in this step. Target generation is intentionally deferred. |
| `source_provenance` | Source kind, relative artifact path, line/record/context references, and evaluation marker. |
| `quality_checks` | Public-only, split, replay, target-deferred, and deduplication checks. |

`composition` and `project_split` are included as auditable convenience fields;
they do not replace the required split information in provenance/checks.

## Boundary labels

- `preference_complete_to_search`: an actor-visible elicitation sequence is
  followed by a search for the current public aspect. This is a public
  transcript heuristic; it does not consult hidden completion fields.
- `explicit_no_preference`: a simulator response explicitly says that the
  user has no specific preference (or is open to any choice).
- `valid_search_to_answer`: a normal candidate list is followed by an answer
  tool call.
- `first_fallback` / `second_fallback`: the first or second public search
  fallback for an aspect. A/B fallback probes synthesize only the public
  fallback sentence recorded in their manifest.
- `repeated_no_progress_action`: an exact public action or identical public
  feedback repeats.
- `visible_options_pending_answer`: a normal visible candidate list is
  followed by a non-answer action.

The SFT train and holdout task inventories are also recorded as manifest-only
sources for split resolution; they contain no assistant/tool turns and thus do
not create synthetic boundaries. The extractor replays each context through the
public control reducer. A malformed or historically invalid sequence is retained with
`quality_checks.public_replay_ok=false`; the source is not repaired silently.

## Split and leakage rules

The task map is loaded before any trajectory is parsed. `task_id` determines
one of `sft_train`, `sft_validation`, `grpo_train`, `grpo_validation`, or
`evaluation`; there is no sample-level random split. Any task not resolved by
the frozen map is marked `unresolved` and is not training-eligible. Artifacts
under `outputs/evaluation/` are marked formal evaluation even if a task ID is
also present in a training inventory. The manifest fails the audit if an
evaluation context is marked training-eligible.

Record-level reward/correctness fields (for example `correct_ids`,
`best_ids`, `reward_snapshot`, `reward delta`, or hidden preference values)
are not copied. Only visible messages and option IDs literally present in
those messages may enter a context.

## Manifest

`manifest.json` records source SHA-256 values, raw and unique candidate counts,
boundary/composition/source distributions, an inventory summary for the SFT
task files, the canonical deduplication key, split checks, policy versions, and
the fact that all targets remain deferred.
The generated `contexts.jsonl` and manifest are disposable derived artifacts;
the source trajectories remain unchanged.

## Public phase rendering

`public_state_before` may include the public-only `preference_complete_aspects`
list. For a `preference_complete_to_search` boundary, the renderer reconstructs
that aspect as `SEARCH_REQUIRED`; it does not use reward snapshots, preference IDs,
or correctness labels. Older records without this field are rendered with the
explicit boundary type as a compatibility phase hint.

When `ANSWERED` or `BLOCKED` is advanced to the next open aspect, the public
ledger retains one transition window. Actor feedback includes a stable
`SWITCH_ASPECT_REQUIRED` transition line naming only the previous public aspect,
its public terminal status, and the next public aspect. The note is cleared on
the next public event, so it cannot accumulate or leak hidden state.
