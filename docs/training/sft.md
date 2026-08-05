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

Collection policy `teacher-state-machine-v5` controls each aspect through `ELICIT → SEARCH → ANSWER`. The hidden Reward-v2 ledger only tells the controller when active preference coverage for an aspect is complete; preference IDs, values, best IDs, coverage, and reward evidence are never inserted into Teacher messages. An elicitation field is committed only after the ledger reports a positive active-preference delta. If UserBench does not record an otherwise valid question, the collector retains and loss-masks that turn, retries the same field once with a distinct deterministic template, and aborts if the repair is also unrecorded. The v5 preference templates cover the complete global UserBench field taxonomy, vague/judgment recovery uses a distinct retry without colliding with duplicate detection, and search generation is constrained to public trip facts and simulator disclosures with inflection, rating/star, and common paraphrase normalization. The action guard allows three generation retries per turn. DeepSeek generates the official three-field tool call under a request-only phase constraint.

Collection has two accepted quality tiers. `gold` satisfies the strict gate and is written to `*.accepted.jsonl`. `silver` is accepted separately when the trajectory still terminates with a correct, fully answered itinerary and zero policy/answer errors, but contains an explicitly bounded recovery: at most one search repair per aspect, one vague-action repair per field/phase, one elicitation-not-recorded repair per field, and at most one simulator judgment fallback in the trajectory. Repaired turns are retained for context and marked `loss_mask=true`; silver with an infrastructure fallback keeps `reward_valid=false` and is never counted as gold. Formal SFT reloads both tier files, ignores the serialized tier as authority, and reruns the Gold/Silver gates. A Silver judgment fallback without a masked assistant repair is rejected.

For a fixed-size, composition-proportional train collection, use adaptive stratification instead of `--limit`. `--target-accepted` counts unique Gold plus Silver trajectories; rejected tasks are checkpointed and replaced by the next task from the same composition. The candidate order, largest-remainder quotas, wave history, and source SHA-256 are recorded in `selection_manifest.json`.

```bash
python scripts/train/sft/collect_sft_data.py \
  --input data/sft/tasks_train.jsonl \
  --target-accepted 50 \
  --stratify-by composition \
  --stratified-wave-size 32 \
  --sampling-seed sft-train-composition-v4 \
  --run-dir outputs/teacher_trajectories/runs/sft-train-composition-v4 \
  --output outputs/teacher_trajectories/sft_train.accepted.jsonl
```

The command stops only when every composition reaches its quota. For the current 716-task train pool, the quotas are `22=86`, `2222=15`, `233=51`, `33=74`, `333=45`, `334=40`, `44=57`, and `444=32`. Resume the same adaptive run with the identical arguments plus `--resume`; do not add `--limit` or change the seed.

Request-local retries become progressively stricter: natural phase guidance, a field-specific instruction with forbidden alternatives, and finally a canonical content allowlist in the temporary request schema. Search is keyed by aspect rather than query wording, and answer IDs must come from visible search output. Response fallback, invalid feedback, wrong answers, and unrecorded answer transitions abort the attempt immediately. A single judgment fallback or recoverable vague/search/elicitation transition is retained as silver only when the final hard correctness checks pass; a second fallback/repair remains fail-loud.

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
  --audit-only outputs/teacher_trajectories/your-run.accepted.jsonl
```

## Action-only rendering

`src/travel_grpo/training/sft_dataset.py` loads the sole tool schema from `configs/tool_config/userbench_tools.yaml` and proves that it equals the Python `interact_with_env` contract. The same schema is passed to the Qwen3.5 chat template.

The raw archive keeps OpenAI-compatible JSON-string function arguments. Qwen3.5's official template iterates function arguments as a mapping, so the renderer converts a defensive copy to a mapping immediately before templating; it never rewrites the archive. SFT and GRPO both pin `enable_thinking: false`, ensuring the same generation prefix and XML tool-call format.

One raw trajectory produces one example per assistant decision. For each decision, the renderer tokenizes the complete prior context with an assistant generation prefix, then tokenizes that context plus the assistant tool call. The first token sequence must be an exact prefix of the second. Only the verified completion suffix receives labels; system, user, tool observation, prior assistant turns, and padding use `-100`. This avoids string matching and correctly supervises an empty `assistant.content` whose actual target is in `assistant.tool_calls`.

The renderer measures the true token length before training. If any assistant decision exceeds `max_sequence_length`, the entire trajectory is excluded and recorded in `overlong_rejections` with its task and turn number. Readiness is checked again after this exclusion. There is no silent left or right truncation, so search evidence, final answers, and tool-call pairing cannot be lost.

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

Formal readiness requires at least 400 accepted train trajectories, 40 validation trajectories, and all eight compositions in both splits. Trajectories are deduplicated across tier files and train/validation must be disjoint. No GRPO or final-evaluation task may be used as backfill. The maximum rendered length is 32768; an overlength trajectory is rejected whole and never truncated.

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

Merge the trained adapter before GRPO with `python scripts/train/sft/merge_lora.py`. The fixed target is `outputs/models/sft-merged`; the merger supports Qwen3.5's multimodal Auto model class, saves model/tokenizer-or-processor plus `merge_manifest.json`, and refuses a non-empty destination.
