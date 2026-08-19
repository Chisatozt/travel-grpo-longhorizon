#!/usr/bin/env bash
# [项目注释] 文件职责：调用 actor 导出脚本生成推理模型。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
exec python scripts/train/grpo/export_actor.py "$@"
