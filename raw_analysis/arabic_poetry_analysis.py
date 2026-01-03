"""
Authorship-style stats for the Arabic Classical Poetry dataset (Kaggle: mdanok/arabic-poetry-dataset).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from utils import count_tokens, length_bucket, truncate_text


def download_dataset(target_dir: Path):
    target_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            "mdanok/arabic-poetry-dataset",
            "-p",
            str(target_dir),
            "--unzip",
        ],
        check=True,
    )


def find_csv(data_dir: Path) -> Path:
    for path in data_dir.rglob("*.csv"):
        return path
    raise FileNotFoundError(f"No CSV file found under {data_dir}")


def analyze(csv_path: Path, max_rows: int | None, output_path: Path):
    df_iter = pd.read_csv(csv_path, chunksize=5000, encoding_errors="ignore")
    lang = "ar"
    aggregates = defaultdict(
        lambda: {
            "documents": 0,
            "authors": set(),
            "genres": Counter(),
            "length_buckets": Counter(),
        }
    )
    sample = None
    scanned = 0

    for chunk in df_iter:
        if sample is None and not chunk.empty:
            sample = chunk.iloc[0].to_dict()
        for _, row in chunk.iterrows():
            text = row.get("poem_text")
            author = row.get("poet_name")
            tags = row.get("poem_tags")
            if not isinstance(text, str) or not text.strip():
                continue
            text = truncate_text(text)
            bucket = length_bucket(count_tokens(text))
            aggregates[lang]["documents"] += 1
            aggregates[lang]["length_buckets"][bucket] += 1
            if isinstance(author, str) and author.strip():
                aggregates[lang]["authors"].add(author.strip())
            if isinstance(tags, str) and tags.strip():
                aggregates[lang]["genres"][tags] += 1
            scanned += 1
            if max_rows and scanned >= max_rows:
                break
        if max_rows and scanned >= max_rows:
            break

    finalized = {
        lang: {
            "documents": data["documents"],
            "unique_authors": len(data["authors"]),
            "genres": dict(data["genres"]),
            "length_buckets": dict(data["length_buckets"]),
        }
        for lang, data in aggregates.items()
    }
    summary = {"languages": finalized, "sample": sample}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Download and aggregate stats for the Arabic poetry dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / "outputs" / "arabic_poetry_raw",
        help="Where to download/unzip the dataset.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=200000,
        help="Row cap for quick scans; None scans full file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "outputs" / "arabic_poetry_stats.json",
        help="Where to write the summary JSON.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Set if the dataset is already present.",
    )
    args = parser.parse_args()

    if not args.skip_download:
        download_dataset(args.data_dir)
    csv_path = find_csv(args.data_dir)
    summary = analyze(csv_path, args.max_rows, args.output)
    print(f"Wrote Arabic poetry stats to {args.output.resolve()}")


if __name__ == "__main__":
    main()

    """
    python raw_analysis/arabic_poetry_analysis.py --max-rows 10000 --skip-download
    """
