#!/usr/bin/env bash
# Train a single registry model for 1 epoch with mid-epoch evaluation.
# Run from the repository root (AURA_Bench).
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <model-name> [extra-args]" >&2
  exit 1
fi

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

MODEL="$1"
shift || true

DATASET_ROOT="${DATASET_ROOT:-processing/outputs/official_ttl300k_cap10M_sf10k_postprocessed_balanced}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EVAL_FRACTION="${EVAL_FRACTION:-0.5}"

echo ">>> Training ${MODEL} ..."
python eval/train.py \
  --model "$MODEL" \
  --epochs 1 \
  --batch-size "$BATCH_SIZE" \
  --eval-fraction-epoch "$EVAL_FRACTION" \
  --eval-every-epoch \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
