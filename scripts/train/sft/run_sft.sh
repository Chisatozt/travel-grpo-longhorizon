#!/usr/bin/env bash
# [项目注释] 文件职责：启动 SFT 训练入口。
set -euo pipefail

python scripts/train/sft/two_stage_sft.py "$@"
