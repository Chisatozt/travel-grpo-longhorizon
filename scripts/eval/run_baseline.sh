#!/usr/bin/env bash
# [项目注释] 文件职责：启动 baseline frozen evaluation。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "$ROOT/scripts/eval/run_evaluation.sh" baseline "$@"
