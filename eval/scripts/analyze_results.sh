#!/usr/bin/env bash
# Run evaluation analysis: export CSVs and plots from eval/results JSON files.
# Usage (from repo root): eval/scripts/analyze_results.sh
set -euo pipefail

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

RESULTS_DIR="${RESULTS_DIR:-eval/results}"
OUTPUT_DIR="${OUTPUT_DIR:-eval/results/analysis}"
TASK="${TASK:-representation}" # representation | attribution (used for non-EER metrics)
METRICS_DEFAULT="success@10 recall@10 ndcg@10 eer@10"

# Allow overriding metrics via METRICS="m1 m2 ..."
read -r -a METRICS_ARR <<<"${METRICS:-$METRICS_DEFAULT}"

# If the requested directory has no JSON files but contains a nested "results" dir, use that.
if [[ ! -e "${RESULTS_DIR}" || "$(ls -1 "${RESULTS_DIR}"/*.json 2>/dev/null | wc -l)" -eq 0 ]]; then
  if [[ -d "${RESULTS_DIR}/results" ]]; then
    RESULTS_DIR="${RESULTS_DIR}/results"
  fi
fi

echo "Analyzing results from ${RESULTS_DIR}"
echo "Writing plots + CSVs to ${OUTPUT_DIR}"
echo "Task=${TASK} Metrics=${METRICS_ARR[*]}"

python -m eval.export_results \
  --results-dir "${RESULTS_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --metrics "${METRICS_ARR[@]}"

python -m post_analysis.plot_results \
  --results-dir "${RESULTS_DIR}" \
  --performance-metrics "${METRICS_ARR[@]}" \
  --performance-out "${OUTPUT_DIR}"

echo "Done. Check ${OUTPUT_DIR} for CSVs and PNGs."
