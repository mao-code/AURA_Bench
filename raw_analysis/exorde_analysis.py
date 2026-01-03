"""
Streaming analysis for the Exorde HF dataset.

Outputs per-language aggregates:
- number of documents and unique authors
- genre/source counts when available
- token length buckets (short<=10, medium 11-100, long 101-500, extra-long>500)

The script uses HF streaming so it does not download the full dataset.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from utils import count_tokens, length_bucket, truncate_text


def extract_language(sample: dict) -> str:
    key = "language"
    val = sample.get(key)
    if val:
        return str(val).lower()
    return "unknown"


def extract_author(sample: dict):
    key = "author_hash"
    val = sample.get(key)
    if val:
        return str(val)
    return None


def extract_genre(sample: dict):
    key = "primary_theme"
    val = sample.get(key)
    if val:
        if isinstance(val, list):
            return ",".join(map(str, val))
        return str(val)
    return None


def analyze(split: str, max_records: int | None, output_path: Path) -> dict:
    dataset = load_dataset(
        "Exorde/exorde-social-media-december-2024-week1", split=split, streaming=True
    )
    valid = 0

    aggregates = defaultdict(
        lambda: {
            "documents": 0,
            "authors": set(),
            "genres": Counter(),
            "length_buckets": Counter(),
        }
    )

    progress = tqdm(
        dataset,
        desc=f"Scanning Exorde split={split}",
        unit="docs",
        total=max_records if max_records else None,
    )
    for record in progress:
        if max_records and valid >= max_records:
            break

        text = record.get("original_text")
        lang = extract_language(record)
        author = extract_author(record)
        genre = extract_genre(record)

        if not isinstance(text, str) or not text.strip() or not lang or not author:
            continue

        text = truncate_text(text)
        bucket = length_bucket(count_tokens(text))
        aggregates[lang]["documents"] += 1
        aggregates[lang]["length_buckets"][bucket] += 1
        if author:
            aggregates[lang]["authors"].add(author)
        if genre:
            aggregates[lang]["genres"][genre] += 1
        valid += 1

    finalized = {
        lang: {
            "documents": data["documents"],
            "unique_authors": len(data["authors"]),
            "genres": dict(data["genres"]),
            "length_buckets": dict(data["length_buckets"]),
        }
        for lang, data in aggregates.items()
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(finalized, f, indent=2, ensure_ascii=False)
    return finalized


def main():
    parser = argparse.ArgumentParser(
        description="Stream and aggregate language/genre/length stats for Exorde."
    )
    parser.add_argument("--split", default="train", help="HF split to stream.")
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional cap for quick dry runs; omit to scan full stream.",
    )
    parser.add_argument(
        "--output",
        default=Path(__file__).parent / "outputs" / "exorde_stats.json",
        type=Path,
        help="Where to write the JSON summary.",
    )
    args = parser.parse_args()

    results = analyze(args.split, args.max_records, args.output)
    print(f"Wrote summary to {args.output.resolve()} with {len(results)} languages.")


if __name__ == "__main__":
    main()

    """
    python raw_analysis/exorde_analysis.py --max-records 100000
    """
