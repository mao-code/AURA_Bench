from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from typing import Dict, List, Mapping, Optional, Sequence


def extract_primary_genre(genre: object) -> Optional[str]:
    if isinstance(genre, str):
        value = genre.strip()
        if not value:
            return None
        return value.split("/")[0]
    if isinstance(genre, list):
        for item in genre:
            if isinstance(item, str) and item.strip():
                return item.strip().split("/")[0]
    return None


def topic_label_for_record(record: Mapping[str, object]) -> Optional[str]:
    return extract_primary_genre(record.get("genre"))


def build_topic_candidate_index(
    candidates: Sequence[Mapping[str, object]],
) -> Dict[str, List[int]]:
    grouped: Dict[str, List[int]] = defaultdict(list)
    for idx, record in enumerate(candidates):
        label = topic_label_for_record(record)
        if label:
            grouped[label].append(idx)
    return dict(grouped)


def _stable_int_seed(seed: int, query_id: str) -> int:
    digest = hashlib.sha256(query_id.encode("utf-8")).digest()
    extra = int.from_bytes(digest[:8], "big")
    return (seed + extra) % (2**32)


def build_topic_pool(
    query_record: Mapping[str, object],
    query_id: str,
    candidate_ids: Sequence[str],
    candidate_indices_by_topic: Mapping[str, Sequence[int]],
    positive_indices: Sequence[int],
    max_candidates: Optional[int],
    seed: int = 13,
) -> Optional[List[int]]:
    label = topic_label_for_record(query_record)
    if not label:
        return None
    pool = candidate_indices_by_topic.get(label)
    if not pool:
        return None

    filtered = [idx for idx in pool if candidate_ids[idx] != query_id]
    if not filtered:
        return None

    if max_candidates is None or max_candidates >= len(filtered):
        return list(filtered)
    if max_candidates <= 0:
        return []

    pos_set = set(int(idx) for idx in positive_indices)
    pool_pos = [idx for idx in filtered if idx in pos_set]
    rng = random.Random(_stable_int_seed(seed, query_id))

    if max_candidates <= len(pool_pos):
        return sorted(rng.sample(pool_pos, max_candidates))

    remaining = [idx for idx in filtered if idx not in pos_set]
    sample_count = min(max_candidates - len(pool_pos), len(remaining))
    sampled = pool_pos + (rng.sample(remaining, sample_count) if sample_count > 0 else [])
    return sorted(sampled)
