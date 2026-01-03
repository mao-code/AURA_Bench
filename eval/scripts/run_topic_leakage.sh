#!/usr/bin/env bash
# Run topic-matched evaluations and export CSV diagnostics.
# Usage: eval/scripts/run_topic_leakage.sh
set -euo pipefail

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

DATASET_ROOT="${DATASET_ROOT:-processing/outputs/official_ttl300k_cap10M_sf10k_postprocessed_balanced}"
SPLIT="${SPLIT:-test}"
DEFAULT_RESULTS_DIR="${DEFAULT_RESULTS_DIR:-eval/results}"
TOPIC_RESULTS_DIR="${TOPIC_RESULTS_DIR:-eval/results/topic_matched}"
OUTPUT_DIR="${OUTPUT_DIR:-eval/results/topic_leakage}"
NEG_PER_QUERY="${NEG_PER_QUERY:-10}"
MAX_TOPIC_CANDIDATES="${MAX_TOPIC_CANDIDATES:-}"

DEFAULT_METRICS="${DEFAULT_METRICS:-success@10 recall@10 ndcg@10 eer@10}"
TOPIC_METRICS="${TOPIC_METRICS:-success@1 success@5 success@10 ndcg@10 eer@10}"

mkdir -p "${TOPIC_RESULTS_DIR}"

if [[ -n "${MODELS:-}" ]]; then
  read -r -a MODEL_LIST <<<"${MODELS}"
else
  MODEL_LIST=()
fi

if [[ "${#MODEL_LIST[@]}" -gt 0 ]]; then
  for MODEL in "${MODEL_LIST[@]}"; do
    OUTPUT_PATH="${TOPIC_RESULTS_DIR}/${MODEL//\//_}.json"
    RUN_ARGS=(
      --task both
      --split "${SPLIT}"
      --dataset-root "${DATASET_ROOT}"
      --candidate-pool topic
      --negatives-per-query "${NEG_PER_QUERY}"
    )
    if [[ -n "${MAX_TOPIC_CANDIDATES}" ]]; then
      RUN_ARGS+=(--max-topic-candidates "${MAX_TOPIC_CANDIDATES}")
    fi
    python -m eval.runner \
      "${RUN_ARGS[@]}" \
      --models "${MODEL}" \
      --output-json "${OUTPUT_PATH}"
  done
fi

python -m eval.tfidf_runner \
  --dataset-root "${DATASET_ROOT}" \
  --split "${SPLIT}" \
  --candidate-pool all \
  --negatives-per-query "${NEG_PER_QUERY}" \
  --output-json "${DEFAULT_RESULTS_DIR}/tfidf.json"

python -m eval.tfidf_runner \
  --dataset-root "${DATASET_ROOT}" \
  --split "${SPLIT}" \
  --candidate-pool topic \
  --negatives-per-query "${NEG_PER_QUERY}" \
  --output-json "${TOPIC_RESULTS_DIR}/tfidf.json"

python -m eval.export_results \
  --results-dir "${DEFAULT_RESULTS_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --metrics ${DEFAULT_METRICS} \
  --prefix "default_"

python -m eval.export_results \
  --results-dir "${TOPIC_RESULTS_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --metrics ${TOPIC_METRICS} \
  --prefix "topic_"

echo "Wrote CSVs to ${OUTPUT_DIR}"
