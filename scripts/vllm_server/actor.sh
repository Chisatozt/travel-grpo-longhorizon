#!/usr/bin/env bash
# [项目注释] 文件职责：启动 actor vLLM 服务。
set -euo pipefail
if [[ $# -ne 1 ]]; then echo "usage: $0 MODEL_OR_PATH" >&2; exit 2; fi
exec python -m vllm.entrypoints.openai.api_server \
  --model "$1" \
  --served-model-name "$1" \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
