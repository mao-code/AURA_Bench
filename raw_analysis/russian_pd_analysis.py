"""
Authorship-style stats for the Russian Public Domain corpus (HF: PleIAs/Russian-PD).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset

from utils import count_tokens, length_bucket, truncate_text


def analyze(max_rows: int | None, output_path: Path):
    ds = load_dataset("PleIAs/Russian-PD", split="train", streaming=True)
    lang = "ru"
    aggregates = defaultdict(
        lambda: {
            "documents": 0,
            "authors": set(),
            "genres": Counter(),
            "length_buckets": Counter(),
        }
    )
    sample = None
    valid = 0

    for record in ds:
        sample = sample or record
        text = record.get("text")
        author = record.get("creator")
        if not text:
            continue
        text = truncate_text(text)
        bucket = length_bucket(count_tokens(text))
        aggregates[lang]["documents"] += 1
        aggregates[lang]["length_buckets"][bucket] += 1
        if author:
            aggregates[lang]["authors"].add(author)
        valid += 1
        if max_rows and valid >= max_rows:
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
        description="Stream and aggregate stats for PleIAs/Russian-PD."
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=500,
        help="How many rows to stream for the summary.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "outputs" / "russian_pd_stats.json",
        help="Where to write the summary JSON.",
    )
    args = parser.parse_args()
    summary = analyze(args.max_rows, args.output)
    print(f"Wrote Russian PD stats to {args.output.resolve()}")


if __name__ == "__main__":
    main()

    """
    python raw_analysis/russian_pd_analysis.py --max-rows 1000
    """
