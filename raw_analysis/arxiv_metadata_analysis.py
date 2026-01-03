"""
Authorship-style stats for the arXiv metadata snapshot (Kaggle: Cornell-University/arxiv).

We read `arxiv-metadata-oai-snapshot.json` (JSONL), extract the first author
as the author id, take the primary category as genre, and compute token
length buckets on the abstract using tiktoken when available.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from utils import count_tokens, length_bucket, truncate_text


def download_arxiv(target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            "Cornell-University/arxiv",
            "-p",
            str(target_dir),
            "-f",
            "arxiv-metadata-oai-snapshot.json",
            "--unzip",
        ],
        check=True,
    )
    json_path = target_dir / "arxiv-metadata-oai-snapshot.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"Expected arxiv-metadata-oai-snapshot.json in {target_dir}"
        )
    return json_path


def analyze(json_path: Path, max_rows: int | None, output_path: Path) -> dict:
    aggregates = defaultdict(
        lambda: {
            "documents": 0,
            "authors": set(),
            "genres": Counter(),
            "length_buckets": Counter(),
        }
    )
    lang = "en"
    sample = None
    valid = 0

    with json_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            sample = sample or record

            abstract = record.get("abstract")
            authors = record.get("authors") or ""
            author = authors.split(",")[0].strip() if authors else None # Choose the first author of this paper
            categories = record.get("categories") or ""
            primary_cat = categories.split()[0] if categories else None

            if abstract:
                abstract = truncate_text(abstract)
                bucket = length_bucket(count_tokens(abstract))
                aggregates[lang]["documents"] += 1
                aggregates[lang]["length_buckets"][bucket] += 1
                if author:
                    aggregates[lang]["authors"].add(author)
                if primary_cat:
                    aggregates[lang]["genres"][primary_cat] += 1
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
        description="Download and inspect arXiv metadata JSON."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / "outputs" / "arxiv_raw",
        help="Where to download/unzip the JSONL file.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=20000,
        help="How many lines to scan to confirm keys.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "outputs" / "arxiv_stats.json",
        help="Where to write the summary JSON.",
    )
    args = parser.parse_args()

    json_path = download_arxiv(args.data_dir)
    summary = analyze(json_path, args.max_rows, args.output)
    print(f"Wrote arXiv stats to {args.output.resolve()}")


if __name__ == "__main__":
    main()

    """
    python raw_analysis/arxiv_metadata_analysis.py --max-rows 10000
    """