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


def main() -> None:
    args = parse_args()
    results = load_results(args.results_dir)
    if not results:
        raise ValueError(f"No results found in {args.results_dir}")

    langs = language_order_from_results(args.results_dir)
    columns = ["model", "overall", "macro_avg"] + langs

    for metric in args.metrics:
        rows = build_rows(results, metric, langs)
        output_path = args.output_dir / f"{args.prefix}{metric}.csv"
        write_metric_csv(output_path, rows, columns)


if __name__ == "__main__":
    main()
