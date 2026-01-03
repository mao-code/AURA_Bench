"""
Authorship-style stats for the Enron Email Dataset (Kaggle: wcukierski/enron-email-dataset).

We read `emails.csv`, extract the sender from the message headers, and bucket
lengths using tiktoken when available. Aggregates are keyed under language
`en`.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from utils import count_tokens, length_bucket, truncate_text


def download_enron(target_dir: Path):
    target_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            "wcukierski/enron-email-dataset",
            "-p",
            str(target_dir),
            "-f",
            "emails.csv",
        ],
        check=True,
    )
    csv_path = target_dir / "emails.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Expected emails.csv after download in {target_dir} but did not find it."
        )
    return csv_path


def parse_sender(raw_message: str) -> str | None:
    # minimal header parse to avoid pulling in the full email module cost for every row
    for line in raw_message.splitlines():
        if line.lower().startswith("from:"):
            return line.split(":", 1)[1].strip()
        if not line.strip():
            break
    return None


def analyze(csv_path: Path, max_rows: int | None, output_path: Path) -> dict:
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

    with csv_path.open(newline="", encoding="utf-8", errors="ignore") as f:
        # Skip rows containing NUL bytes that csv cannot parse
        filtered_lines = (line for line in f if "\0" not in line)
        reader = csv.DictReader(filtered_lines)
        if "file" not in reader.fieldnames or "message" not in reader.fieldnames:
            raise ValueError(
                f"emails.csv columns are {reader.fieldnames}; expected 'file' and 'message'."
            )
        for row in reader:
            sample = sample or row
            msg = row["message"]
            if not msg:
                continue

            sender = parse_sender(msg)
            msg = truncate_text(msg)
            bucket = length_bucket(count_tokens(msg))
            aggregates[lang]["documents"] += 1
            aggregates[lang]["length_buckets"][bucket] += 1
            if sender:
                aggregates[lang]["authors"].add(sender)
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
        description="Download and inspect the Enron Email dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / "outputs" / "enron_raw",
        help="Where to download/unzip emails.csv.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=10000,
        help="How many rows to scan for the summary.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "outputs" / "enron_summary.json",
        help="Where to write the summary JSON.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use existing emails.csv in --data-dir instead of downloading.",
    )
    args = parser.parse_args()

    csv_path = args.data_dir / "emails.csv" if args.skip_download else download_enron(args.data_dir)
    if args.skip_download and not csv_path.exists():
        raise FileNotFoundError(
            f"--skip-download set but {csv_path} does not exist. Run without the flag to fetch it."
        )
    summary = analyze(csv_path, args.max_rows, args.output)
    print(f"Wrote Enron stats to {args.output.resolve()}")


if __name__ == "__main__":
    main()

    """
    python raw_analysis/enron_email_analysis.py --max-rows 10000 --skip-download
    """
