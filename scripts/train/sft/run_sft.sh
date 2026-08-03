#!/usr/bin/env bash
set -euo pipefail

python scripts/train/sft/sft_train.py --config configs/train/sft/sft_lora.yaml "$@"
