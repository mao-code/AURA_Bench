from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Iterable, Iterator, Sequence

logger = logging.getLogger(__name__)

import tiktoken

_ENCODER = tiktoken.get_encoding("cl100k_base")
def count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text, disallowed_special=()))

def length_bucket(token_count: int) -> str:
    if token_count <= 10:
        return "short"
    if token_count <= 100:
        return "medium"
    if token_count <= 500:
        return "long"
    return "extra_long"


def hash_author(source: str, raw_author: str) -> str:
    canonical = f"{source}:{raw_author}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def deterministic_shuffle(items: Sequence, rng) -> list:
    copied = list(items)
    rng.shuffle(copied)
    return copied


def slugify(value: str) -> str:
    cleaned = value.strip().lower().replace(" ", "_")
    return cleaned.replace("\\", "/")
