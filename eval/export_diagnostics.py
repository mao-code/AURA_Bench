from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Set, Tuple


METRIC_COLUMNS = ("success@10", "recall@10", "ndcg@10", "eer@10")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export overall and grouped diagnostics metrics to CSV."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("eval/results"),
        help="Directory containing per-model evaluation JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval/results"),
        help="Directory to write diagnostic CSVs.",
    )
    return parser.parse_args()


def load_results(results_dir: Path) -> Dict[str, Dict[str, object]]:
    results: Dict[str, Dict[str, object]] = {}
    for path in sorted(results_dir.glob("*.json")):
        with path.open() as handle:
            data = json.load(handle)
        if not data:
            continue
        model, payload = next(iter(data.items()))
        results[model] = payload
    return results


def _get_metric(block: Mapping[str, object], key: str) -> Optional[float]:
    value = block.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _iter_group_keys(results: Dict[str, Dict[str, object]], group_key: str) -> List[str]:
    keys: Set[str] = set()
    for payload in results.values():
        for section in ("representation", "attribution"):
            block = payload.get(section) or {}
            grouped = block.get(group_key) or {}
            keys.update(grouped.keys())
    return sorted(keys)


def _write_csv(path: Path, rows: Iterable[Dict[str, object]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def export_overall(results: Dict[str, Dict[str, object]], output_dir: Path) -> None:
    rows: List[Dict[str, object]] = []
    for model, payload in sorted(results.items()):
        rep = payload.get("representation") or {}
        attr = payload.get("attribution") or {}
        rows.append(
            {
                "model": model,
                "success@10": _get_metric(rep, "success@10"),
                "recall@10": _get_metric(rep, "recall@10"),
                "ndcg@10": _get_metric(rep, "ndcg@10"),
                "eer@10": _get_metric(attr, "eer"),
            }
        )
    _write_csv(output_dir / "overall_metrics.csv", rows, ["model", *METRIC_COLUMNS])


def export_grouped(
    results: Dict[str, Dict[str, object]],
    output_dir: Path,
    group_key: str,
    output_name: str,
    group_label: str,
) -> None:
    group_values = _iter_group_keys(results, group_key)
    rows: List[Dict[str, object]] = []
    for model, payload in sorted(results.items()):
        rep = payload.get("representation") or {}
        attr = payload.get("attribution") or {}
        rep_group = rep.get(group_key) or {}
        attr_group = attr.get(group_key) or {}
        for group in group_values:
            rep_metrics = rep_group.get(group) or {}
            attr_metrics = attr_group.get(group) or {}
            rows.append(
                {
                    "model": model,
                    group_label: group,
                    "success@10": _get_metric(rep_metrics, "success@10"),
                    "recall@10": _get_metric(rep_metrics, "recall@10"),
                    "ndcg@10": _get_metric(rep_metrics, "ndcg@10"),
                    "eer@10": _get_metric(attr_metrics, "eer"),
                }
            )
    columns = ["model", group_label, *METRIC_COLUMNS]
    _write_csv(output_dir / output_name, rows, columns)


def main() -> None:
    args = parse_args()
    results = load_results(args.results_dir)
    if not results:
        raise ValueError(f"No results found in {args.results_dir}")

    export_overall(results, args.output_dir)
    export_grouped(results, args.output_dir, "by_language", "by_language_metrics.csv", "lang")
    export_grouped(results, args.output_dir, "by_genre", "by_genre_metrics.csv", "genre")
    export_grouped(
        results,
        args.output_dir,
        "by_length_bucket",
        "by_length_bucket_metrics.csv",
        "length_bucket",
    )


if __name__ == "__main__":
    main()
