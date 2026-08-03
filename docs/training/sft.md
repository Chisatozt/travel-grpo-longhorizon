# Supervised fine-tuning

## Teacher trajectory collection

Teacher collection has two independent OpenAI-compatible API clients:

- `TEACHER_*`: `deepseek-v4-flash` generates exactly one `interact_with_env` call per turn.
- `COLLECTION_USER_SIM_*`: a separate `deepseek-v4-flash` runtime supplies UserBench feedback and rewards.

The variables may point to the same provider account, but they are deliberately not aliased in code. This prevents a future teacher or simulator change from silently changing the other role. Copy `.env.example`, supply real keys and the exact endpoint/model identifier exposed by the selected provider, then install the optional API dependencies:

```bash
pip install -e ".[api,data]"
pip install -e environments/UserBench
```

Validate the task, model, and environment contracts without issuing requests:

```bash
python scripts/train/sft/collect_sft_data.py --dry-run --limit 1
```

Collect the SFT train and validation pools in separate processes:

```bash
python scripts/train/sft/collect_sft_data.py \
  --input data/sft/tasks_train.jsonl \
  --output outputs/teacher_trajectories/sft_train.accepted.jsonl

python scripts/train/sft/collect_sft_data.py \
  --input data/sft/tasks_validation.jsonl \
  --output outputs/teacher_trajectories/sft_validation.accepted.jsonl
```

The collector requires one valid tool call per teacher turn and executes it asynchronously. Request-local protocol retries preserve the one-tool contract. If only `thought` exceeds its limit, the retry locks the already-valid `choice` and `content`, reports the measured length, and asks the Teacher to replace only `thought`. Exact and semantic action repeats are corrected without entering the accepted message history; semantic corrections name the repeated aspect/field and list completed and still-available fields. The Teacher policy reserves enough turns to answer every remaining travel aspect and forces `answer` when only that reserve remains.

Every task receives up to three complete trajectory attempts by default. Admission is strict: the environment must terminate without truncation, every task aspect must be answered, simulator fallback text and captured UserBench judge/search fallbacks must be absent, and assistant/tool messages must have a valid one-call pairing. The primary output contains only accepted trajectories. Sibling `*.rejected.jsonl` and `*.diagnostics.jsonl` files contain exhausted tasks and per-attempt reasons respectively; they must never be used as SFT examples. Failed-attempt diagnostics include the failure turn, committed actions, partial messages, raw step rewards, answered aspects, and fallback counters, but never credentials or endpoints. Evaluation task IDs are checked for overlap before any API call.

`configs/train/sft/sft_lora.yaml` pins the eventual SFT base model to `Qwen/Qwen3.5-2B`. Action-only label rendering, LoRA optimization, filtering, and model export remain unimplemented.
