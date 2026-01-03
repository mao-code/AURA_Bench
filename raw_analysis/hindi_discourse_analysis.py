"""
Authorship-style stats for the Hindi Discourse Dataset from
https://github.com/midas-research/hindi-discourse.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import requests

from utils import count_tokens, length_bucket, truncate_text

SOURCE_URL = "https://raw.githubusercontent.com/midas-research/hindi-discourse/master/discourse_dataset.json"
_UNICODE_ESCAPE_RE = re.compile(r"\\u[0-9a-fA-F]{4}|\\x[0-9a-fA-F]{2}")


def download_json(target_path: Path):
    target_path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(SOURCE_URL, timeout=30)
    resp.raise_for_status()
    target_path.write_bytes(resp.content)
    return target_path


def _maybe_unescape_text(text: str) -> str:
    if not isinstance(text, str) or not _UNICODE_ESCAPE_RE.search(text):
        return text
    try:
        return text.encode("utf-8").decode("unicode_escape")
    except Exception:
        return text


def analyze(json_path: Path, output_path: Path, max_rows: int | None):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    records = list(data.values())
    if not records:
        raise ValueError("No records found in discourse_dataset.json")

    lang = "hi"
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

    for rec in records:
        sample = sample or rec
        text = rec.get("Sentence")
        # story_no is the only stable grouping; use as proxy author id
        author = f"story_{rec.get('Story_no')}"
        mode = rec.get("Discourse Mode")
        if not text:
            continue
        text = _maybe_unescape_text(text)
        text = truncate_text(text)
        bucket = length_bucket(count_tokens(text))
        aggregates[lang]["documents"] += 1
        aggregates[lang]["length_buckets"][bucket] += 1
        if author:
            aggregates[lang]["authors"].add(author)
        if mode:
            aggregates[lang]["genres"][mode] += 1
        scanned += 1
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
        description="Download and aggregate stats for the Hindi Discourse dataset."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path(__file__).parent
        / "outputs"
        / "hindi_discourse"
        / "discourse_dataset.json",
        help="Where to save the downloaded JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent
        / "outputs"
        / "hindi_discourse_stats.json",
        help="Where to write the summary JSON.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on number of documents processed.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Set if the JSON has already been fetched.",
    )
    args = parser.parse_args()

    json_path = args.data_path
    if not args.skip_download:
        json_path = download_json(args.data_path)

    summary = analyze(json_path, args.output, args.max_rows)
    print(f"Wrote Hindi discourse stats to {args.output.resolve()}")


if __name__ == "__main__":
    main()

    """
    python raw_analysis/hindi_discourse_analysis.py --max-rows 10000 --skip-download
    """
