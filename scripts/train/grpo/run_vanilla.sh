#!/usr/bin/env bash
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
  echo "[grpo] ERROR: Python not found; run scripts/setup.sh or set PYTHON_BIN" >&2
  exit 1
}
exec "$PYTHON_BIN" scripts/train/grpo/train_grpo.py --config configs/train/grpo/vanilla_grpo.yaml "$@"
