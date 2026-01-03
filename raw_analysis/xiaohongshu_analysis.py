"""
Authorship-style stats for the Xiaohongshu/Weibo dataset
(Kaggle: yuanchunhong/xiaohongshu-aigc-comments-including-postsdataset).
"""

from __future__ import annotations

import argparse
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from utils import (
    TEXT_EXTS,
    count_tokens,
    iter_tabular_rows,
    length_bucket,
    pick_column,
    truncate_text,
)
import json

TEXT_CANDIDATES = ["content", "desc"]
AUTHOR_CANDIDATES = ["user_id"]

def download_dataset(target_dir: Path):
    target_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            "yuanchunhong/xiaohongshu-aigc-comments-including-postsdataset",
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
    scanned = 0

    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_EXTS:
            continue
        files_scanned += 1
        for chunk in iter_tabular_rows(path):
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


def main():
    parser = argparse.ArgumentParser(
        description="Download and aggregate stats for the Xiaohongshu AIGC dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / "outputs" / "xiaohongshu_raw",
        help="Where to download/unzip the dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "outputs" / "xiaohongshu_stats.json",
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
    print(f"Wrote Xiaohongshu stats to {args.output.resolve()}")


if __name__ == "__main__":
    main()

    """
    python raw_analysis/xiaohongshu_analysis.py --data-dir raw_analysis/outputs/xiaohongshu_raw/data --skip-download
    """
