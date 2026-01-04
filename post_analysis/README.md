## Post-analysis utilities

This folder contains a reusable script for summarizing the final AuthBench release.

### Usage

```bash
python -m AURA_Bench.post_analysis.analyze_dataset \
    --dataset-dir AURA_Bench/processing/outputs/official_ttl300k_cap10M_sf10k_postprocessed \
    --output-dir AURA_Bench/post_analysis/outputs
```

Arguments:
- `--dataset-dir`: Path to the split directories (`train/dev/test`) containing `queries.jsonl`, `candidates.jsonl`, and `ground_truth.jsonl`.
- `--output-dir`: Destination for CSV tables and figures (defaults to `AURA_Bench/post_analysis/outputs`).
- `--splits`: Optional list of splits to include.

### Outputs
- CSV tables under `outputs/csv/` covering language, genre, primary-genre, token length, source, author, and positive-pair alignment statistics (counts and percentages).
- Figures under `outputs/figures/` showing language mix, primary-genre heatmap, token-length boxplots, and top primary genres.

Dependencies for this script are listed in `post_analysis/requirements.txt`.
