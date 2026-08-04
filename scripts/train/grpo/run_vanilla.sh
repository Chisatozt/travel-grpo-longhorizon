#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
exec python scripts/train/grpo/train_grpo.py --config configs/train/grpo/vanilla_grpo.yaml "$@"
