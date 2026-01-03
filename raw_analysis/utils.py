"""
Shared helpers for raw analysis scripts.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

# Allow very large text fields in CSV/TSV files.
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)

try:
    import tiktoken

    _ENCODER = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_ENCODER.encode(text, disallowed_special=()))

except Exception:  # pragma: no cover - tiktoken not installed

    def count_tokens(text: str) -> int:
        return len(text.split())


def length_bucket(token_count: int) -> str:
    if token_count <= 10:
        return "short"
    if token_count <= 100:
        return "medium"
    if token_count <= 500:
        return "long"
    return "extra-long"


TEXT_EXTS = {".csv", ".tsv", ".json", ".jsonl", ".xlsx", ".xls"}
MAX_TEXT_LEN = int(os.getenv("MAX_TEXT_LEN", "2000"))


def pick_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    return None


def truncate_text(text: str) -> str:
    """Trim text to the global MAX_TEXT_LEN when set to a positive value."""
    if not isinstance(text, str):
        return text
    if MAX_TEXT_LEN and MAX_TEXT_LEN > 0 and len(text) > MAX_TEXT_LEN:
        return text[:MAX_TEXT_LEN]
    return text


def iter_tabular_rows(path: Path, chunksize: int = 5000) -> Iterator[pd.DataFrame]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        # Pure-Python CSV reader to avoid segfaults from pandas' C engine on noisy files.
        rows: list[dict] = []
        with path.open(encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
                if len(rows) >= chunksize:
                    yield pd.DataFrame(rows)
                    rows = []
        if rows:
            yield pd.DataFrame(rows)
    elif suffix == ".tsv":
        rows = []
        with path.open(encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                rows.append(row)
                if len(rows) >= chunksize:
                    yield pd.DataFrame(rows)
                    rows = []
        if rows:
            yield pd.DataFrame(rows)
    elif suffix in {".xlsx", ".xls"}:
        yield pd.read_excel(path)
    elif suffix in {".json", ".jsonl"}:
        rows = []
        with path.open(encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                rows.append(rec)
                if len(rows) >= chunksize:
                    yield pd.DataFrame(rows)
                    rows = []
        if rows:
            yield pd.DataFrame(rows)
