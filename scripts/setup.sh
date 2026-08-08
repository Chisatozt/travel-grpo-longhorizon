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
GRPO_OVERRIDES="$ROOT/configs/train/grpo/uv-overrides.txt"
if [[ ! -f "$GRPO_OVERRIDES" ]]; then
  echo "missing GRPO dependency override file: $GRPO_OVERRIDES" >&2
  exit 1
fi

uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  --overrides "$GRPO_OVERRIDES" \
  -e ".[api,data,sft,qlora,eval,grpo,dev]"
uv pip install --python .venv/bin/python -e environments/UserBench
.venv/bin/python scripts/train/grpo/apply_verl_patch.py

.venv/bin/python - <<'PY'
import importlib.metadata
import sys
if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"expected Python 3.12, found {sys.version.split()[0]}")
expected = {
    "verl": "0.8.0",
    "vllm": "0.25.1",
    "opencv-python-headless": "4.13.0.90",
    "numpy": "2.2.6",
}
for distribution, required in expected.items():
    found = importlib.metadata.version(distribution)
    if found != required:
        raise SystemExit(f"{distribution}=={required} required, found {found}")
import cv2
import numpy
import verl
import vllm
import travel_grpo
import travelgym
print(f"Travel GRPO Linux environment installed successfully (numpy={numpy.__version__}, cv2={cv2.__version__})")
PY
