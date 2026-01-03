"""
Authorship-style stats for the Douban review dataset
(Kaggle: fengzhujoey/douban-datasetratingreviewside-information).
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from utils import (
    TEXT_EXTS,
    count_tokens,
    iter_tabular_rows,
    length_bucket,
    pick_column,
    truncate_text,
)

TEXT_CANDIDATES = ["comment", "self_statement", "review", "review_text", "content", "Sentence", "text"]
AUTHOR_CANDIDATES = ["user_id", "uid", "userid", "userID", "reviewer"]
SPECIAL_TEXT_EXTS = {".txt"}


def download_dataset(target_dir: Path):
    target_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            "fengzhujoey/douban-datasetratingreviewside-information",
            "-p",
            str(target_dir),
            "--unzip",
        ],
        check=True,
    )


def analyze(data_dir: Path, max_rows: int | None, output_path: Path):
    lang = "zh"
    aggregates = defaultdict(
        lambda: {
            "documents": 0,
            "authors": set(),
            "genres": Counter(),
            "length_buckets": Counter(),
        }
    )
    sample = None

    files_scanned = 0
    allowed_exts = TEXT_EXTS | SPECIAL_TEXT_EXTS
    scanned = 0

    for path in sorted(data_dir.rglob("*")):
        suffix = path.suffix.lower()
        if not path.is_file() or suffix not in allowed_exts:
            continue
        files_scanned += 1
        row_iter = (
            iter_douban_txt_rows(path) if suffix in SPECIAL_TEXT_EXTS else iter_tabular_rows(path)
        )
        for chunk in row_iter:
            if chunk.empty:
                continue
            text_col = pick_column(chunk.columns.tolist(), TEXT_CANDIDATES)
            if not text_col:
                raise ValueError(
                    f"No known text column found in {path}; columns={chunk.columns.tolist()}"
                )
            author_col = pick_column(chunk.columns.tolist(), AUTHOR_CANDIDATES)
            if sample is None:
                sample = chunk.iloc[0].to_dict()
            for _, row in chunk.iterrows():
                text = row.get(text_col)
                if not isinstance(text, str) or not text.strip():
                    continue
                text = truncate_text(text)
                author = row.get(author_col) if author_col else None
                bucket = length_bucket(count_tokens(text))
                aggregates[lang]["documents"] += 1
                aggregates[lang]["length_buckets"][bucket] += 1
                if author and str(author).strip():
                    aggregates[lang]["authors"].add(str(author).strip())
                scanned += 1
                if max_rows and scanned >= max_rows:
                    break
            if max_rows and scanned >= max_rows:
                break
        if max_rows and scanned >= max_rows:
            break

    if files_scanned == 0:
        raise RuntimeError(f"No tabular files found under {data_dir}")

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


def iter_douban_txt_rows(path: Path, chunksize: int = 5000):
    """Yield DataFrame chunks from the quoted, tab-delimited Douban .txt dump."""

    rows = []
    with path.open(encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t", quotechar='"')
        for row in reader:
            rows.append(row)
            if len(rows) >= chunksize:
                yield pd.DataFrame(rows)
                rows = []
    if rows:
        yield pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Download and aggregate stats for the Douban review dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / "outputs" / "douban_raw",
        help="Where to download/unzip the dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "outputs" / "douban_stats.json",
        help="Where to write the summary JSON.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on rows scanned across files.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Set if the dataset is already present in data-dir.",
    )
    args = parser.parse_args()

    if not args.skip_download:
        download_dataset(args.data_dir)
    summary = analyze(args.data_dir, args.max_rows, args.output)
    print(f"Wrote Douban stats to {args.output.resolve()}")


if __name__ == "__main__":
    main()

    """
    Deleted (Dropped Dataset): books_cleaned.txt movies_cleaned.txt music_cleaned.txt

    python raw_analysis/douban_reviews_analysis.py --data-dir "raw_analysis/outputs/douban_raw/douban_dataset/douban_dataset(text information)" --skip-download 
    """
