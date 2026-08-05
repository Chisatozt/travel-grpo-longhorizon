#!/usr/bin/env bash
set -euo pipefail

python scripts/train/sft/two_stage_sft.py "$@"
