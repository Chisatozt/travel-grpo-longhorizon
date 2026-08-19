#!/usr/bin/env bash
# [项目注释] 文件职责：串联两个 SFT stage 的 shell 入口。
set -Eeuo pipefail

# One-command Linux GPU launcher for the local Transformers two-stage SFT.
# It never collects data and never changes the pinned UserBench snapshot.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${SFT_VENV_DIR:-$ROOT_DIR/.venv-sft}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$ROOT_DIR/outputs/cache/huggingface}"
MODEL_ID="Qwen/Qwen3.5-2B"
MIN_VRAM_GIB="${SFT_MIN_VRAM_GIB:-24}"
LOG_DIR="$ROOT_DIR/outputs/sft/logs"

MODE="train"
SKIP_INSTALL=0
SKIP_DOWNLOAD=0
ALLOW_CPU_PREFLIGHT="${ALLOW_CPU_PREFLIGHT:-0}"
FORCE_MODEL_DOWNLOAD=0
TWO_STAGE_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts/train/sft/launch_two_stage_sft.sh [options]

Default behavior:
  create .venv-sft, install the [sft] extra, verify the GPU, cache Qwen3.5-2B,
  audit both stages, render both stages, then run Stage 1 -> Stage 2.

Options:
  --preflight                 stop after audit and full tokenizer rendering
  --dry-run                   run record-level audit only, do not train
  --render-smoke              run audit plus full tokenizer rendering, no train
  --stage 1|2|all             run only one stage or both (default: all)
  --stage1-resume PATH        resume Stage 1 from a checkpoint under outputs/
  --stage2-resume PATH        resume Stage 2 from a checkpoint under outputs/
  --limit N                   limit records for an explicit smoke run
  --allow-small-smoke         bypass configured count gates for smoke only
  --skip-install              reuse the active/created virtual environment
  --skip-download             do not download the model; require a local cache
  --force-download             refresh the model cache with hf download
  -h, --help                  show this help

Environment overrides:
  PYTHON_BIN=python3
  SFT_VENV_DIR=.venv-sft
  HF_CACHE_DIR=outputs/cache/huggingface
  SFT_MIN_VRAM_GIB=24
  ALLOW_CPU_PREFLIGHT=1       allow --dry-run without CUDA (not real training)
EOF
}

die() {
  echo "[two-stage-sft] ERROR: $*" >&2
  exit 1
}

log() {
  echo "[two-stage-sft] $*" >&2
}

while (($#)); do
  case "$1" in
    --preflight)
      MODE="preflight"
      ;;
    --dry-run)
      MODE="dry-run"
      TWO_STAGE_ARGS+=("--dry-run")
      ;;
    --render-smoke)
      MODE="render-smoke"
      TWO_STAGE_ARGS+=("--render-smoke")
      ;;
    --stage)
      (($# >= 2)) || die "--stage requires 1, 2, or all"
      TWO_STAGE_ARGS+=("--stage" "$2")
      shift
      ;;
    --stage1-resume)
      (($# >= 2)) || die "--stage1-resume requires a checkpoint path"
      TWO_STAGE_ARGS+=("--stage1-resume-from-checkpoint" "$2")
      shift
      ;;
    --stage2-resume)
      (($# >= 2)) || die "--stage2-resume requires a checkpoint path"
      TWO_STAGE_ARGS+=("--stage2-resume-from-checkpoint" "$2")
      shift
      ;;
    --limit)
      (($# >= 2)) || die "--limit requires a positive integer"
      TWO_STAGE_ARGS+=("--limit" "$2")
      shift
      ;;
    --allow-small-smoke)
      TWO_STAGE_ARGS+=("--allow-small-smoke")
      ;;
    --skip-install)
      SKIP_INSTALL=1
      ;;
    --skip-download)
      SKIP_DOWNLOAD=1
      ;;
    --force-download)
      FORCE_MODEL_DOWNLOAD=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1 (use --help)"
      ;;
  esac
  shift
done

command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python executable not found: $PYTHON_BIN"
command -v git >/dev/null 2>&1 || die "git is required by the pinned Transformers dependency"

if [[ ! -d "$VENV_DIR" ]]; then
  log "Creating virtual environment: $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
PYTHON_BIN="$(command -v python)"

if ((SKIP_INSTALL == 0)); then
  log "Installing the pinned local SFT stack"
  "$PYTHON_BIN" -m pip install --upgrade pip
  "$PYTHON_BIN" -m pip install -e ".[sft]"
else
  log "Skipping package installation"
fi

log "Checking NVIDIA and CUDA"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found; use an NVIDIA GPU server"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

"$PYTHON_BIN" - "$MIN_VRAM_GIB" "$ALLOW_CPU_PREFLIGHT" <<'PY'
import sys

minimum_gib = float(sys.argv[1])
allow_cpu = sys.argv[2] == "1"
recommended_gib = 48.0
try:
    import torch
except ImportError as exc:
    raise SystemExit("PyTorch is not installed; rerun without --skip-install") from exc

if not torch.cuda.is_available():
    if allow_cpu:
        print("[two-stage-sft] CUDA unavailable; allowed for offline preflight only")
        raise SystemExit(0)
    raise SystemExit("CUDA is unavailable; real SFT requires an NVIDIA CUDA GPU")

for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    memory_gib = props.total_memory / (1024**3)
    print(
        f"[two-stage-sft] GPU {index}: {props.name}; "
        f"VRAM={memory_gib:.1f} GiB; bf16={torch.cuda.is_bf16_supported()}"
    )
    if memory_gib < minimum_gib:
        raise SystemExit(
            f"GPU {index} has {memory_gib:.1f} GiB, below SFT_MIN_VRAM_GIB={minimum_gib:.1f}"
        )
    if memory_gib < recommended_gib:
        print(
            f"[two-stage-sft] WARNING: {memory_gib:.1f} GiB is below the "
            f"recommended {recommended_gib:.1f} GiB; reduce batch/length if OOM occurs"
        )
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("Current SFT config requires bf16, but this GPU does not support it")
PY

mkdir -p "$HF_CACHE_DIR" "$LOG_DIR"

required_files=(
  "outputs/teacher_trajectories/sft_train.accepted.jsonl"
  "outputs/teacher_trajectories/sft_train.silver.jsonl"
  "outputs/teacher_trajectories/sft_validation.from_train.accepted.jsonl"
  "outputs/teacher_trajectories/sft_validation.from_train.silver.jsonl"
  "outputs/teacher_trajectories/sft_train.from_train_holdout.tasks.parquet"
  "outputs/teacher_trajectories/sft_validation.from_train.tasks.parquet"
)

prefix_file="outputs/teacher_trajectories/sft_stage1_prefix.jsonl"
if [[ ! -s "$prefix_file" ]]; then
  diagnostics="outputs/teacher_trajectories/sft_train.diagnostics.jsonl"
  [[ -s "$diagnostics" ]] || die "missing Stage-1 prefix and diagnostics: $prefix_file"
  log "Generating Stage-1 prefix data from diagnostics"
  "$PYTHON_BIN" scripts/train/sft/prepare_stage1_prefix_sft.py
fi
required_files+=("$prefix_file")

missing=()
for path in "${required_files[@]}"; do
  [[ -s "$path" ]] || missing+=("$path")
done
if ((${#missing[@]})); then
  printf '[two-stage-sft] Missing required artifact: %s\n' "${missing[@]}" >&2
  die "copy the generated teacher artifacts and internal task split parquet files to the server"
fi

tokenizer_probe=(
  "from transformers import AutoTokenizer; "
  "AutoTokenizer.from_pretrained("
  "'Qwen/Qwen3.5-2B', local_files_only=True, "
  "cache_dir=r'$HF_CACHE_DIR'); print('cached')"
)
if ((FORCE_MODEL_DOWNLOAD == 1)); then
  cached=0
else
  if "$PYTHON_BIN" -c "${tokenizer_probe[*]}" >/dev/null 2>&1; then
    cached=1
    log "Qwen3.5-2B tokenizer is already cached"
  else
    cached=0
  fi
fi

if [[ "$MODE" == "dry-run" ]]; then
  log "Skipping model download for record-only dry-run"
elif ((cached == 0)); then
  ((SKIP_DOWNLOAD == 0)) || die "Qwen3.5-2B is not cached and --skip-download was supplied"
  command -v hf >/dev/null 2>&1 || die "hf CLI not found; install the SFT extra or remove --skip-install"
  log "Downloading Qwen3.5-2B into $HF_CACHE_DIR"
  hf download "$MODEL_ID" --cache-dir "$HF_CACHE_DIR"
else
  log "Using the existing Qwen3.5-2B cache"
fi

if [[ "$MODE" == "dry-run" ]]; then
  log "Running record-level two-stage audit"
  "$PYTHON_BIN" scripts/train/sft/two_stage_sft.py "${TWO_STAGE_ARGS[@]}"
  exit 0
fi

if [[ "$MODE" == "preflight" || "$MODE" == "render-smoke" ]]; then
  log "Running record-level two-stage audit"
  "$PYTHON_BIN" scripts/train/sft/two_stage_sft.py --dry-run "${TWO_STAGE_ARGS[@]}"
  log "Rendering every Stage-1 and Stage-2 split with the local tokenizer"
  "$PYTHON_BIN" scripts/train/sft/two_stage_sft.py --render-smoke "${TWO_STAGE_ARGS[@]}"
  log "Preflight completed; no model weights were updated"
  exit 0
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="$LOG_DIR/two_stage_${timestamp}.log"
log "Running full two-stage SFT; log: $log_file"
set -o pipefail
"$PYTHON_BIN" scripts/train/sft/two_stage_sft.py "${TWO_STAGE_ARGS[@]}" 2>&1 | tee "$log_file"
log "Two-stage SFT completed; final adapter: outputs/sft/qwen3.5-2b-lora-stage2"
