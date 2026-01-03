from __future__ import annotations

import random
import re
from typing import Iterable, List, Optional

from .types import RawDocument
from .utils import count_tokens


def _split_paragraphs(text: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if paragraphs:
        return paragraphs
    sentences = [s.strip() for s in re.split(r"(?<=[。！？!?\.])\s+", text) if s.strip()]
    if sentences:
        return sentences
    return [text.strip()]


def _split_on_punctuation(text: str) -> list[str]:
    # Split on common punctuation boundaries while keeping delimiters attached.
    # This is a light-weight semantic cut to avoid chopping mid-sentence.
    parts = re.split(r"(?<=[。！？!?\.])\s+|\n+", text)
    parts = [p.strip() for p in parts if p and p.strip()]
    if parts:
        return parts
    return [text.strip()]


def _bounded_segments(text: str, max_tokens: int) -> list[str]:
    """Split text into segments each no longer than max_tokens using punctuation fallbacks."""
    segments: list[str] = []
    for chunk in _split_paragraphs(text):
        if count_tokens(chunk) <= max_tokens:
            segments.append(chunk)
            continue
        for sentence in _split_on_punctuation(chunk):
            if count_tokens(sentence) <= max_tokens:
                segments.append(sentence)
                continue
            # Fallback: break long sentences into token-sized pieces.
            words = sentence.split()
            current: list[str] = []
            current_len = 0
            for w in words:
                w_len = count_tokens(w)
                if current_len + w_len > max_tokens and current:
                    segments.append(" ".join(current))
                    current = []
                    current_len = 0
                current.append(w)
                current_len += w_len
            if current:
                segments.append(" ".join(current))
    return segments


def chunk_document(
    doc: RawDocument,
    max_tokens: int,
    target_chunk_tokens: int,
    min_chunk_tokens: int,
    *,
    chunk_probability: float = 1.0,
    rng: Optional[random.Random] = None,
) -> list[RawDocument]:
    token_len = count_tokens(doc.text)
    if token_len <= max_tokens:
        return [doc]

    rng = rng or random.Random()
    if chunk_probability < 1.0 and rng.random() > chunk_probability:
        # keep as a single long doc
        return [doc]

    segments = _bounded_segments(doc.text, max_tokens)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for seg in segments:
        seg_tokens = count_tokens(seg)
        if seg_tokens == 0:
            continue
        if current_tokens + seg_tokens > max_tokens and current_tokens >= min_chunk_tokens:
            chunks.append(" ".join(current))
            current = []
            current_tokens = 0

        current.append(seg)
        current_tokens += seg_tokens

        if current_tokens >= target_chunk_tokens and current_tokens >= min_chunk_tokens:
            chunks.append(" ".join(current))
            current = []
            current_tokens = 0

    if current:
        if chunks and current_tokens < min_chunk_tokens:
            chunks[-1] = f"{chunks[-1]} {' '.join(current)}"
        else:
            chunks.append(" ".join(current))

    split_docs: list[RawDocument] = []
    for idx, chunk_text in enumerate(chunks):
        chunk_tokens = count_tokens(chunk_text)
        if chunk_tokens < min_chunk_tokens and idx != len(chunks) - 1:
            continue
        split_docs.append(
            RawDocument(
                raw_id=f"{doc.raw_id}#chunk_{idx}",
                author=doc.author,
                text=chunk_text,
                lang=doc.lang,
                source=doc.source,
                genre=doc.genre,
                metadata={**doc.metadata, "chunk_tokens": chunk_tokens},
            )
        )
    return split_docs


def truncate_raw_document(doc: RawDocument, max_tokens: int) -> RawDocument:
    """
    Truncate a document to <= max_tokens using punctuation/paragraph boundaries.
    """

    if count_tokens(doc.text) <= max_tokens:
        return doc

    segments = _bounded_segments(doc.text, max_tokens)
    kept: list[str] = []
    total = 0
    for seg in segments:
        seg_tokens = count_tokens(seg)
        if total + seg_tokens > max_tokens:
            break
        kept.append(seg)
        total += seg_tokens
        if total >= max_tokens:
            break

    truncated_text = " ".join(kept).strip()
    if not truncated_text:
        truncated_text = doc.text[: max_tokens * 4]  # crude fallback if segmentation failed

    return RawDocument(
        raw_id=doc.raw_id,
        author=doc.author,
        text=truncated_text,
        lang=doc.lang,
        source=doc.source,
        genre=doc.genre,
        metadata={**doc.metadata, "truncated_to": max_tokens},
    )
