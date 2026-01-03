from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable


@dataclass
class RawDocument:
    """
    Minimal representation of a raw record before cleaning/chunking.
    """

    raw_id: str
    author: str
    text: str
    lang: str
    source: str
    genre: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessedDocument:
    """
    Document that satisfies the unified schema (minus the final integer id).
    """

    raw_id: str
    author_id: str
    text: str
    lang: str
    source: str
    genre: str
    token_length: int
    length_bucket: str
    metadata: dict[str, Any] = field(default_factory=dict)
    doc_id: str | None = None


@dataclass
class DatasetConfig:
    """
    Configuration describing how to read one dataset.
    """

    name: str
    source: str
    loader: str  # jsonl, csv, tsv, hf_streaming
    path: Path | None = None  # For local files
    split: str | None = None  # For HF splits
    text_field: str = "text"
    author_field: str = "author"
    lang_field: str | None = "lang"
    genre_field: str | None = None
    static_lang: str | None = None
    raw_id_field: str | None = None
    preprocess_row: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    max_documents: int | None = None  # Used for sanity checks


@dataclass
class SamplingTargets:
    total_docs: int
    language_targets: dict[str, int]
    genre_percents: dict[str, dict[str, float]]
    length_bucket_percents: dict[str, float]


@dataclass
class SplitRatios:
    train: float = 0.8
    dev: float = 0.1
    test: float = 0.1

    def as_list(self) -> list[tuple[str, float]]:
        return [("train", self.train), ("dev", self.dev), ("test", self.test)]


class DatasetLoader:
    """
    Protocol used by dataset loader functions.
    """

    def __call__(
        self, config: DatasetConfig, sanity_limit: int | None = None
    ) -> Iterable[RawDocument]:
        raise NotImplementedError
