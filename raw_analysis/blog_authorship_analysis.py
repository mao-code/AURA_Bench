"""
Authorship-style stats for the Blog Authorship Corpus (HF: barilan/blog_authorship_corpus).

We stream posts directly from the source zip to retain the blogger id that
is present in the file name but omitted in the HF examples. Stats are
aggregated per language (English only), with counts of documents, authors,
optional job buckets, and length buckets computed with tiktoken when
available.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download

from utils import count_tokens, length_bucket, truncate_text


def stream_posts(limit: int | None):
    """Yield posts with recovered author ids from the zip archive."""
    archive_path = hf_hub_download(
        repo_id="barilan/blog_authorship_corpus",
        repo_type="dataset",
        filename="data/blogs.zip",
    )
    with zipfile.ZipFile(archive_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".xml"):
                continue
            file_id, gender, age, job, horoscope = name.split(".")[:-1]
            with zf.open(name) as fh:
                date = ""
                for raw in fh:
                    line = raw.decode("latin-1").strip()
                    if line.startswith("<date>"):
                        date = line.replace("<date>", "").replace("</date>", "")
                        continue
                    if not line or line.startswith("<"):
                        continue
                    yield {
                        "author_id": file_id,
                        "gender": gender,
                        "age": int(age),
                        "job": job,
                        "horoscope": horoscope,
                        "date": date,
                        "text": line,
                    }
                    if limit:
                        limit -= 1
                        if limit <= 0:
                            return


def analyze(max_records: int | None, output_path: Path) -> dict:
    lang = "en"
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

    for post in stream_posts(max_records):
        if not post["text"]:
            continue
        post["text"] = truncate_text(post["text"])
        sample = sample or post
        bucket = length_bucket(count_tokens(post["text"]))
        aggregates[lang]["documents"] += 1
        aggregates[lang]["length_buckets"][bucket] += 1
        aggregates[lang]["authors"].add(post["author_id"])
        if post["job"]:
            aggregates[lang]["genres"][post["job"]] += 1
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

    summary = {"languages": finalized, "sample": sample}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate author/length stats for the Blog Authorship Corpus."
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=200_000,
        help="Optional cap for quick scans; set None to stream all posts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "outputs" / "blog_authorship_stats.json",
        help="Where to write the JSON summary.",
    )
    args = parser.parse_args()
    summary = analyze(args.max_records, args.output)
    print(
        f"Wrote summary to {args.output.resolve()} with {len(summary['languages'])} language entry."
    )


if __name__ == "__main__":
    main()

    """
    python raw_analysis/blog_authorship_analysis.py --max-records 10000
    """