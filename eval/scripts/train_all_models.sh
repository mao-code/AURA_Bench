#!/usr/bin/env bash
# Train every registered model for 1 epoch with mid-epoch evaluation.
# Run from the repository root (AURA_Bench).
set -euo pipefail

# sbatch -p nlplarge-sasha-highpri --nodelist=nlplarge-compute-01 --gres=gpu:1 --ntasks=1 --cpus-per-task=4 --mem=128G -t 720:00:00 eval/scripts/train_all_models.sh

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# Weights & Biases defaults (override via env).
WANDB_PROJECT="${WANDB_PROJECT:-AURA_Bench}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_PREFIX="${WANDB_RUN_PREFIX:-train-all}"
WANDB_TAGS="${WANDB_TAGS:-AURA_Bench train-all}"

DATASET_ROOT="${DATASET_ROOT:-processing/outputs/official_ttl300k_cap10M_sf10k_postprocessed_balanced}"
OUTPUT_DIR="${OUTPUT_DIR:-eval/results/training_summary}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EVAL_FRACTION="${EVAL_FRACTION:-0.5}"

MODELS=(
  "e5-large-v2"
  "multilingual-e5-large"
  "bge-large-en-v1.5"
  # "bge-base-en-v1.5"
  # "bge-m3"
  # "snowflake-arctic/-embed-l-v2"
  # "jina-embeddings-v2-base-en"
  # "mxbai-embed-large-v1"
  # "gte-large-en-v1.5"
  # "nv-embed-v1"
  "qwen3-embedding-0.6b"
  # "nomic-embed-text-v1"
  # "sfr-embedding-mistral" # 7B parameters
  # "all-minilm-l12-v2"
)

for MODEL in "${MODELS[@]}"; do
  echo ">>> Training ${MODEL} ..."
  WANDB_ARGS=(
    --wandb-project "$WANDB_PROJECT"
    --wandb-run-name "${WANDB_RUN_PREFIX}-${MODEL}"
  )
  # Only include entity flag if set.
  if [[ -n "$WANDB_ENTITY" ]]; then
    WANDB_ARGS+=(--wandb-entity "$WANDB_ENTITY")
  fi
  # Split space-delimited tags into separate args.
  for TAG in $WANDB_TAGS; do
    WANDB_ARGS+=(--wandb-tags "$TAG")
  done

  python eval/train.py \
    --model "$MODEL" \
    --epochs 1 \
    --batch-size "$BATCH_SIZE" \
    --eval-fraction-epoch "$EVAL_FRACTION" \
    --eval-every-epoch \
    --dataset-root "$DATASET_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --skip-checkpoint \
    "${WANDB_ARGS[@]}"
done
