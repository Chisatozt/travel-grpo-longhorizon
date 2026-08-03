#!/usr/bin/env bash
set -euo pipefail

MODEL="${ACTOR_MODEL:-${GRPO_ACTOR_MODEL:-Qwen/Qwen3.5-2B}}"
SERVED_NAME="${SERVED_MODEL_NAME:-travel-agent}"
PORT="${ACTOR_PORT:-8000}"

exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --port "$PORT" \
  --trust-remote-code
