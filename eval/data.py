from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset


def load_jsonl(path: Path) -> List[dict]:
    """Load a JSONL file into memory."""

    records: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


@dataclass
class AuthBenchSplit:
    """Container for one split of the processed AuthBench benchmark."""

    name: str
    root: Path
    queries: List[dict]
    candidates: List[dict]
    ground_truth: List[dict]
    candidate_by_id: Dict[str, dict] = field(init=False)
    query_by_id: Dict[str, dict] = field(init=False)
    positives_by_query: Dict[str, List[str]] = field(init=False)
    author_by_query: Dict[str, str] = field(init=False)

    def __post_init__(self) -> None:
        self.root = self.root.expanduser()
        self.candidate_by_id = {c["candidate_id"]: c for c in self.candidates}
        self.query_by_id = {q["query_id"]: q for q in self.queries}
        self.positives_by_query = {}
        self.author_by_query = {}
        for entry in self.ground_truth:
            query_id = entry["query_id"]
            positives = [cid for cid in entry["positive_ids"] if cid in self.candidate_by_id]
            if positives:
                self.positives_by_query[query_id] = positives
                if "author_id" in entry:
                    self.author_by_query[query_id] = entry["author_id"]

    def limited(
        self,
        max_queries: Optional[int] = None,
        max_candidates: Optional[int] = None,
        seed: int = 13,
    ) -> "AuthBenchSplit":
        """Return a copy of the split with optional subsampling."""

        rng = random.Random(seed)

        queries = self.queries
        candidates = self.candidates
        if max_queries is not None and max_queries < len(queries):
            queries = rng.sample(queries, max_queries)
        if max_candidates is not None and max_candidates < len(candidates):
            candidates = rng.sample(candidates, max_candidates)

        allowed_query_ids = {q["query_id"] for q in queries}
        allowed_candidate_ids = {c["candidate_id"] for c in candidates}

        filtered_gt: List[dict] = []
        for entry in self.ground_truth:
            if entry["query_id"] not in allowed_query_ids:
                continue
            positives = [cid for cid in entry["positive_ids"] if cid in allowed_candidate_ids]
            if not positives:
                continue
            filtered_entry = {
                "query_id": entry["query_id"],
                "positive_ids": positives,
            }
            if "author_id" in entry:
                filtered_entry["author_id"] = entry["author_id"]
            filtered_gt.append(filtered_entry)

        return AuthBenchSplit(
            name=self.name,
            root=self.root,
            queries=queries,
            candidates=candidates,
            ground_truth=filtered_gt,
        )

    def to_summary(self) -> Mapping[str, object]:
        """Lightweight summary for logs."""

        return {
            "split": self.name,
            "root": str(self.root),
            "num_queries": len(self.queries),
            "num_candidates": len(self.candidates),
            "num_ground_truth": len(self.ground_truth),
        }


def load_split(dataset_root: Path, split: str) -> AuthBenchSplit:
    """Load queries, candidates, and ground-truth for one split."""

    split_path = Path(dataset_root).expanduser() / split
    if not split_path.exists():
        raise FileNotFoundError(f"Split directory does not exist: {split_path}")

    queries_path = split_path / "queries.jsonl"
    candidates_path = split_path / "candidates.jsonl"
    ground_truth_path = split_path / "ground_truth.jsonl"
    if not queries_path.exists() or not candidates_path.exists() or not ground_truth_path.exists():
        raise FileNotFoundError(
            f"Expected queries/candidates/ground_truth JSONL files under {split_path}"
        )

    queries = load_jsonl(queries_path)
    candidates = load_jsonl(candidates_path)
    ground_truth = load_jsonl(ground_truth_path)

    return AuthBenchSplit(
        name=split,
        root=split_path,
        queries=queries,
        candidates=candidates,
        ground_truth=ground_truth,
    )


def build_positive_pairs(
    split: AuthBenchSplit, max_pairs: Optional[int] = None, seed: int = 13
) -> List[Tuple[str, str, str, str]]:
    """
    Convert ground truth into (query_text, positive_text, query_id, candidate_id) tuples.
    The dataset already caps positives to 1–2 per query; we optionally subsample pairs.
    """

    rng = random.Random(seed)
    pairs: List[Tuple[str, str, str, str]] = []

    for entry in split.ground_truth:
        query_id = entry["query_id"]
        positive_ids = split.positives_by_query.get(query_id, [])
        if not positive_ids:
            continue
        candidate_id = rng.choice(positive_ids)
        query = split.query_by_id.get(query_id)
        candidate = split.candidate_by_id.get(candidate_id)
        if not query or not candidate:
            continue
        pairs.append((query["content"], candidate["content"], query_id, candidate_id))
        if max_pairs is not None and len(pairs) >= max_pairs:
            break

    return pairs


def iter_positive_pairs(split: AuthBenchSplit) -> Iterable[Tuple[str, str]]:
    """Yield (query_text, candidate_text) for every positive link."""

    for entry in split.ground_truth:
        query = split.query_by_id.get(entry["query_id"])
        if not query:
            continue
        for cid in entry["positive_ids"]:
            candidate = split.candidate_by_id.get(cid)
            if candidate:
                yield (query["content"], candidate["content"])


class PairDataset(Dataset):
    """PyTorch dataset wrapping (query, positive) text pairs."""

    def __init__(self, pairs: Sequence[Tuple[str, str, str, str]]):
        self.pairs = list(pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        query_text, candidate_text, query_id, candidate_id = self.pairs[idx]
        return {
            "query_text": query_text,
            "candidate_text": candidate_text,
            "query_id": query_id,
            "candidate_id": candidate_id,
        }
