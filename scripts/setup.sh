#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Formal SFT/GRPO setup is supported only on Linux. Use the existing dev environment for offline checks." >&2
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required; install it from https://docs.astral.sh/uv/" >&2
  exit 1
fi

uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[api,data,sft,qlora,eval,grpo,dev]"
uv pip install --python .venv/bin/python -e environments/UserBench
.venv/bin/python scripts/train/grpo/apply_verl_patch.py

.venv/bin/python - <<'PY'
import sys
if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"expected Python 3.12, found {sys.version.split()[0]}")
import travel_grpo
import travelgym
print("Travel GRPO Linux environment installed successfully")
PY
