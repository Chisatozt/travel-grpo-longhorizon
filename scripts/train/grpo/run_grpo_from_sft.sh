#!/usr/bin/env bash
# [项目注释] 文件职责：从 merged SFT 模型串联 GRPO 数据准备和训练。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT/.venv/bin/python"
  else
    PYTHON_BIN="$(command -v python || true)"
  fi
fi
[[ -n "$PYTHON_BIN" ]] || {
  echo "[grpo-from-sft] ERROR: Python not found; run scripts/setup.sh or set PYTHON_BIN" >&2
  exit 1
}
exec "$PYTHON_BIN" scripts/train/grpo/run_grpo_from_sft.py "$@"
