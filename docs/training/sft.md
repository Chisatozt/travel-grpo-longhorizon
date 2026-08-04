# Supervised fine-tuning

## Data stages

The SFT pipeline deliberately keeps two different artifacts:

1. A `userbench-teacher-trajectory-v4` JSONL record is an auditable raw trajectory. It contains the original `system → user → assistant(tool_calls) → tool` messages, raw UserBench step rewards, and a top-level Travel Reward v2 admission report.
2. The in-memory training dataset contains tokenized per-assistant-turn examples: `input_ids`, `attention_mask`, and action-only `labels`, plus task/trajectory/turn identifiers. It is generated from the raw trajectories and is not committed.

Version 3 trajectories are archival inputs. The default loader reports `legacy_or_unknown_schema` and `missing_reward_evidence`; it never silently trains on them.

## Teacher trajectory collection

Collection uses two independent OpenAI-compatible clients:

- `TEACHER_*`: `deepseek-v4-flash` emits exactly one `interact_with_env` call per turn.
- `COLLECTION_USER_SIM_*`: a separate `deepseek-v4-flash` runtime supplies UserBench feedback.

Install collection dependencies and the pinned environment:

```bash
pip install -e ".[api,data]"
pip install -e environments/UserBench
```

Validate contracts without an API request:

```bash
python scripts/train/sft/collect_sft_data.py --dry-run --limit 1
```

Collect train and validation in separate commands:

```bash
python scripts/train/sft/collect_sft_data.py \
  --input data/sft/tasks_train.jsonl \
  --output outputs/teacher_trajectories/sft_train.accepted.jsonl

python scripts/train/sft/collect_sft_data.py \
  --input data/sft/tasks_validation.jsonl \
  --output outputs/teacher_trajectories/sft_validation.accepted.jsonl
```

Collection policy `teacher-state-machine-v2` controls each aspect through `ELICIT → SEARCH → ANSWER`. The hidden Reward-v2 ledger only tells the controller when active preference coverage for an aspect is complete; preference IDs, values, best IDs, coverage, and reward evidence are never inserted into Teacher messages. DeepSeek generates the official three-field tool call under a request-only phase constraint.

Collection has two accepted quality tiers. `gold` satisfies the strict gate and is written to `*.accepted.jsonl`. `silver` is accepted separately when the trajectory still terminates with a correct, fully answered itinerary and zero policy/answer errors, but contains one explicitly bounded recovery: one search repair, one vague-action repair, or one simulator judgment fallback. Repaired turns are retained for context and marked `loss_mask=true`; silver with an infrastructure fallback keeps `reward_valid=false` and is never counted as gold. Silver records are written to the sibling `*.silver.jsonl` artifact for explicit downstream inspection and policy-controlled filtering; the default strict SFT loader continues to reject infrastructure-invalid records.

Request-local retries become progressively stricter: natural phase guidance, a field-specific instruction with forbidden alternatives, and finally a canonical content allowlist in the temporary request schema. Search is keyed by aspect rather than query wording, and answer IDs must come from visible search output. Response fallback, invalid feedback, wrong answers, and unrecorded answer transitions abort the attempt immediately. A single judgment fallback or recoverable vague/search transition is retained as silver only when the final hard correctness checks pass; a second fallback/repair remains fail-loud.

Each completed task is atomically checkpointed under a run directory before the batch finishes. Resume a stopped run without recollecting completed tasks:

```bash
python scripts/train/sft/collect_sft_data.py \
  --limit 10 \
  --run-dir outputs/teacher_trajectories/runs/smoke-10 \
  --output outputs/teacher_trajectories/smoke-10.accepted.jsonl

python scripts/train/sft/collect_sft_data.py \
  --limit 10 \
  --run-dir outputs/teacher_trajectories/runs/smoke-10 \
  --output outputs/teacher_trajectories/smoke-10.accepted.jsonl \
  --resume
```

The ordered task IDs and policy version must match the run manifest. Gold (`accepted`), silver, rejected, and diagnostic JSONL files are materialized in input order from the checkpoints. Progress events go to stderr and contain no prompts, credentials, hidden labels, or reward evidence.

Use staged real-API validation before a large collection: first one task must reach a terminal Reward-v2 report, then three tasks should confirm retry behavior, and only then run ten tasks. Do not scale if completion is below 80%, strict acceptance below 60%, action-exhaustion above 10%, or composition-22 step p95 exceeds 14. These are collection-readiness gates, not benchmark claims.

## Travel Reward v2 admission

Every environment transition updates the same hidden evidence ledger used by GRPO. Reward labels and hidden preferences stay outside `messages`; the Actor never sees them. A trajectory is admitted only when all of the following hold:

- `reward_valid=true`, `completion_rate=1`, and `correct_itinerary=true`;
- `terminal_reward >= 0.7` and `policy_penalty=0`;
- invalid, exact-repeat, semantic-repeat, ambiguous, unsearched-answer, and wrong-answer counters are zero;
- the environment terminated without truncation or simulator/search/judgment/response fallback;
- every requested aspect was answered and every assistant/tool pair is valid.

Known deterministic feedback such as “too vague”, duplicate recommendations, and invalid option IDs produces a stable rejection reason. Rejected and diagnostic artifacts must never be passed to SFT.

Audit an arbitrary trajectory file without loading a model or using the network:

```bash
python scripts/train/sft/sft_train.py \
  --audit-only outputs/teacher_trajectories/smoke_strict_v2_deepseek_v4_flash.accepted.jsonl
```

## Action-only rendering

`src/travel_grpo/training/sft_dataset.py` loads the sole tool schema from `configs/tool_config/userbench_tools.yaml` and proves that it equals the Python `interact_with_env` contract. The same schema is passed to the Qwen3.5 chat template.

The raw archive keeps OpenAI-compatible JSON-string function arguments. Qwen3.5's official template iterates function arguments as a mapping, so the renderer converts a defensive copy to a mapping immediately before templating; it never rewrites the archive. SFT and GRPO both pin `enable_thinking: false`, ensuring the same generation prefix and XML tool-call format.

One raw trajectory produces one example per assistant decision. For each decision, the renderer tokenizes the complete prior context with an assistant generation prefix, then tokenizes that context plus the assistant tool call. The first token sequence must be an exact prefix of the second. Only the verified completion suffix receives labels; system, user, tool observation, prior assistant turns, and padding use `-100`. This avoids string matching and correctly supervises an empty `assistant.content` whose actual target is in `assistant.tool_calls`.

The renderer measures the true token length before training. A sample longer than `max_sequence_length` fails with its task and turn number. There is no silent left or right truncation, so search evidence, final answers, and tool-call pairing cannot be lost.

## LoRA and QLoRA

Install the training stack separately from the lightweight core package:

```bash
pip install -e ".[sft]"
```

For Linux QLoRA, also install the optional bitsandbytes extra:

```bash
pip install -e ".[sft,qlora]"
```

Configuration lives in `configs/train/sft/sft_lora.yaml`. It pins `Qwen/Qwen3.5-2B`, action-only assistant-turn examples, LoRA targets, optimizer parameters, evaluation/save cadence, and an ignored `outputs/sft/` destination.

Audit both configured splits without loading a tokenizer or model:

```bash
python scripts/train/sft/sft_train.py --dry-run
```

After explicitly caching the Qwen3.5 tokenizer, test real chat-template rendering without loading the model:

```bash
export HF_HOME=outputs/cache/huggingface
python scripts/train/sft/sft_train.py --dry-run --render-smoke --limit 1
```

The configured Transformers cache is `outputs/cache/huggingface`. The render smoke uses `local_files_only=True`; it never downloads implicitly. Formal LoRA training is an explicit command:

```bash
bash scripts/train/sft/run_sft.sh

# Resume only from a checkpoint under outputs/
bash scripts/train/sft/run_sft.sh \
  --resume-from-checkpoint outputs/sft/qwen3.5-2b-lora/checkpoint-100
```

Set `model.qlora: true` for 4-bit NF4 QLoRA. Training, checkpoints, logs, and caches belong under ignored output directories. No model-quality or benchmark result is claimed until reproducible artifacts exist.
