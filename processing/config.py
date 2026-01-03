from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .types import SamplingTargets, SplitRatios

TARGET_TOTAL_DOCS = 100_000

LANGUAGE_PERCENTS = {
    "en": 0.35,
    "zh": 0.10,
    "es": 0.10,
    "ar": 0.08,
    "fr": 0.08,
    "ru": 0.08,
    "de": 0.08,
    "ja": 0.05,
    "ko": 0.05,
    "hi": 0.03,
}

LENGTH_BUCKET_PERCENTS = {
    "short": 0.15,
    "medium": 0.50,
    "long": 0.20,
    "extra_long": 0.15,
}

GENRE_PERCENTS = {
    "en": {
        "social_media": 0.45,
        "blog": 0.15,
        "ecommerce_reviews": 0.10,
        "research_paper": 0.10,
        "news": 0.20,
    },
    "zh": {
        "social_media/xiaohongshu": 0.35,
        "social_media": 0.20,
        "media_reviews/douban": 0.25,
        "news": 0.20,
    },
    "es": {
        "social_media": 0.45,
        "literature": 0.35,
        "news": 0.20,
    },
    "ar": {
        "social_media": 0.45,
        "literature": 0.25,
        "poetry": 0.10,
        "news": 0.20,
    },
    "fr": {
        "social_media": 0.55,
        "literature": 0.25,
        "news": 0.20,
    },
    "ru": {
        "social_media": 0.55,
        "literature": 0.25,
        "news": 0.20,
    },
    "de": {
        "social_media": 0.55,
        "literature": 0.25,
        "news": 0.20,
    },
    "ja": {
        "social_media": 0.55,
        "literature": 0.25,
        "news": 0.20,
    },
    "ko": {
        "social_media": 0.55,
        "literature": 0.25,
        "news": 0.20,
    },
    "hi": {
        "social_media": 0.55,
        "literature": 0.25,
        "news": 0.20,
    },
}

DEFAULT_SPLIT_RATIOS = SplitRatios()

CHUNKING_DEFAULTS = {
    "max_tokens": 500,
    "target_chunk_tokens": 100,
    "min_chunk_tokens": 10,
    "chunk_probability": 1.0,
}

DIRTY_DEFAULTS = {
    "unique_token_ratio": 0.2,
    "symbol_ratio": 0.4,
    "max_consecutive_symbols": 5,
    "max_repeated_char_run": 20,
}

AUTHOR_DOC_LIMITS = {
    "min_docs": 3,
    "fallback_min_docs": 2,
    "max_docs": 5,
}

SUPPORTED_LOADERS = {"jsonl", "csv", "tsv", "hf_streaming", "blog_authorship"}


def build_sampling_targets(total_docs: int = TARGET_TOTAL_DOCS) -> SamplingTargets:
    language_targets = {k: round(total_docs * v) for k, v in LANGUAGE_PERCENTS.items()}
    return SamplingTargets(
        total_docs=total_docs,
        language_targets=language_targets,
        genre_percents=GENRE_PERCENTS,
        length_bucket_percents=LENGTH_BUCKET_PERCENTS,
    )


def default_manifest_path() -> Path:
    return Path(__file__).parent / "datasets_manifest.json"


def make_split_ratios(train: float, dev: float, test: float) -> SplitRatios:
    total = train + dev + test
    if not total:
        raise ValueError("Split ratios must sum to >0.")
    return SplitRatios(train=train / total, dev=dev / total, test=test / total)


def genre_mapper_for_source(source: str) -> Callable[[str | None], str]:
    """
    Standardize genre values by dataset.
    """

    def _normalize(val: str | None) -> str:
        if not val:
            return source_default.get(source, "unknown")
        cleaned = str(val).strip().lower().replace("\\", "/")
        cleaned = cleaned.replace(" ", "_")
        if source == "exorde":
            return f"social_media/{cleaned}" if not cleaned.startswith("social_media") else cleaned
        if source == "babel_briefings":
            return "news"
        if source == "amazon_multi":
            return "ecommerce_reviews"
        if source == "blog_authorship":
            return f"blog/{cleaned}" if not cleaned.startswith("blog") else cleaned
        if source == "arxiv":
            return "research_paper"
        if source == "xiaohongshu":
            return "social_media/xiaohongshu"
        if source == "douban":
            return "media_reviews/douban"
        if source in {"spanish_pd_books", "french_pd_books", "russian_pd", "german_pd"}:
            return "literature"
        if source == "arabic_poetry":
            return "poetry"
        if source == "hindi_discourse":
            return "literature"
        return cleaned or source_default.get(source, "unknown")

    source_default = {
        "exorde": "social_media",
        "babel_briefings": "news",
        "amazon_multi": "ecommerce_reviews",
        "blog_authorship": "blog/general",
        "arxiv": "research_paper",
        "xiaohongshu": "social_media/xiaohongshu",
        "douban": "media_reviews/douban",
        "spanish_pd_books": "literature",
        "french_pd_books": "literature",
        "arabic_poetry": "poetry",
        "russian_pd": "literature",
        "german_pd": "literature",
        "hindi_discourse": "literature",
    }
    return _normalize
