from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export eval results to CSV tables.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("eval/results"),
        help="Directory containing per-model evaluation JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval/results/analysis"),
        help="Directory to write metric CSVs.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["success@10", "recall@10", "ndcg@10", "eer@10"],
        help="Metrics to export (eer@* maps to attribution.eer).",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Optional filename prefix (e.g., default_ or topic_).",
    )
    return parser.parse_args()


def language_order_from_results(results_dir: Path) -> List[str]:
    langs: set[str] = set()
    for path in sorted(results_dir.glob("*.json")):
        with path.open() as f:
            data = json.load(f)
        if not data:
            continue
        _, payload = next(iter(data.items()))
        block = payload.get("representation") or payload.get("attribution")
        if not block:
            continue
        langs.update(block.get("by_language", {}).keys())
    return sorted(langs)


def resolve_metric(metric: str) -> Tuple[str, str]:
    metric_lower = metric.lower()
    if "eer" in metric_lower:
        return "attribution", "eer"
    return "representation", metric


def load_results(results_dir: Path) -> Dict[str, Dict[str, object]]:
    results: Dict[str, Dict[str, object]] = {}
    for path in sorted(results_dir.glob("*.json")):
        with path.open() as f:
            data = json.load(f)
        if not data:
            continue
        model, payload = next(iter(data.items()))
        results[model] = payload
    return results


def write_metric_csv(
    output_path: Path,
    rows: List[Dict[str, object]],
    columns: List[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def build_rows(
    results: Dict[str, Dict[str, object]],
    metric: str,
    langs: List[str],
) -> List[Dict[str, object]]:
    task, metric_key = resolve_metric(metric)
    rows: List[Dict[str, object]] = []
    for model, payload in sorted(results.items()):
        block = payload.get(task)
        if not block:
            continue
        by_lang = block.get("by_language", {})
        row: Dict[str, object] = {"model": model, "overall": block.get(metric_key)}
        values = []
        for lang in langs:
            value = by_lang.get(lang, {}).get(metric_key)
            row[lang] = value
            if value is not None:
                values.append(float(value))
        row["macro_avg"] = sum(values) / len(values) if values else None
        rows.append(row)
    return rows


def _get_block(payload: Dict[str, object], task: str) -> Dict[str, object] | None:
    block = payload.get(task)
    if not isinstance(block, dict):
        return None
    return block


def _collect_keys(block: Dict[str, object], field: str) -> List[str]:
    bucket = block.get(field, {})
    if not isinstance(bucket, dict):
        return []
    return sorted(bucket.keys())


def _primary_genre(genre: str) -> str:
    return genre.split("/", 1)[0]


def _aggregate_primary_genres(genre_block: Dict[str, Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    aggregated: Dict[str, Dict[str, List[float]]] = {}
    for genre, metrics in genre_block.items():
        if not isinstance(metrics, dict):
            continue
        primary = _primary_genre(genre)
        for key, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue
            aggregated.setdefault(primary, {}).setdefault(key, []).append(float(value))
    collapsed: Dict[str, Dict[str, float]] = {}
    for primary, metrics in aggregated.items():
        collapsed[primary] = {key: sum(values) / len(values) for key, values in metrics.items()}
    return collapsed


def build_bucket_rows(
    results: Dict[str, Dict[str, object]],
    metric: str,
    field: str,
    buckets: List[str],
    aggregate_primary_genres: bool = False,
) -> List[Dict[str, object]]:
    task, metric_key = resolve_metric(metric)
    rows: List[Dict[str, object]] = []
    for model, payload in sorted(results.items()):
        block = _get_block(payload, task)
        if not block:
            continue
        bucket_data = block.get(field, {})
        if not isinstance(bucket_data, dict):
            continue
        if aggregate_primary_genres:
            bucket_data = _aggregate_primary_genres(bucket_data)
        row: Dict[str, object] = {"model": model, "overall": block.get(metric_key)}
        values = []
        for bucket in buckets:
            value = bucket_data.get(bucket, {}).get(metric_key)
            row[bucket] = value
            if value is not None:
                values.append(float(value))
        row["macro_avg"] = sum(values) / len(values) if values else None
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    results = load_results(args.results_dir)
    if not results:
        raise ValueError(f"No results found in {args.results_dir}")

    for metric in args.metrics:
        task, _ = resolve_metric(metric)
        langs = language_order_from_results(args.results_dir)
        lang_columns = ["model", "overall", "macro_avg"] + langs
        lang_rows = build_rows(results, metric, langs)
        lang_path = args.output_dir / f"{args.prefix}{metric}.csv"
        write_metric_csv(lang_path, lang_rows, lang_columns)

        genre_keys: List[str] = []
        length_keys: List[str] = []
        for payload in results.values():
            block = _get_block(payload, task)
            if not block:
                continue
            if not genre_keys:
                genre_keys = _collect_keys(block, "by_genre")
            if not length_keys:
                length_keys = _collect_keys(block, "by_length_bucket")
            if genre_keys and length_keys:
                break

        if genre_keys:
            primary_genres = sorted({_primary_genre(g) for g in genre_keys})
            genre_columns = ["model", "overall", "macro_avg"] + primary_genres
            genre_rows = build_bucket_rows(
                results,
                metric,
                field="by_genre",
                buckets=primary_genres,
                aggregate_primary_genres=True,
            )
            genre_path = args.output_dir / f"{args.prefix}genre_{metric}.csv"
            write_metric_csv(genre_path, genre_rows, genre_columns)

        if length_keys:
            length_columns = ["model", "overall", "macro_avg"] + length_keys
            length_rows = build_bucket_rows(
                results,
                metric,
                field="by_length_bucket",
                buckets=length_keys,
            )
            length_path = args.output_dir / f"{args.prefix}length_{metric}.csv"
            write_metric_csv(length_path, length_rows, length_columns)


if __name__ == "__main__":
    main()

# Example (export CSVs + plots):
# python -m eval.export_results --results-dir eval/results --output-dir eval/results/analysis --metrics success@10 recall@10 ndcg@10 eer@10
# python -m post_analysis.plot_results --results-dir eval/results --performance-metrics success@10 recall@10 ndcg@10 eer@10 --performance-out eval/results/analysis
