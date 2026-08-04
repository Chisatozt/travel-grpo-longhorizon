#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
if [[ $# -lt 1 ]]; then echo "usage: $0 baseline|sft|grpo [options]" >&2; exit 2; fi
STAGE="$1"; shift
case "$STAGE" in baseline|sft|grpo) ;; *) echo "unknown stage: $STAGE" >&2; exit 2;; esac
exec python scripts/eval/evaluate_userbench.py --stage "$STAGE" "$@"
