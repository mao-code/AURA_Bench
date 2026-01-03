# AuthBench Processing Pipeline

This package turns the raw datasets listed in `DATASET.md` into the unified benchmark described in `PROCESSING.md`.

## CLI

```
python -m authbench.processing.build_benchmark \
  --manifest authbench/processing/datasets_manifest.json \
  --output-dir /path/to/output \
  --total-docs 100000 \
  --allow-other-languages \
  --max-chunk-tokens 500 \
  --target-chunk-tokens 350 \
  --min-chunk-tokens 50 \
  --chunk-probability 0.8 \
  --truncate-to-tokens 2000 \
  --sanity-check --sanity-limit 500  # optional quick run
```

Key flags:
- `--manifest`: JSON manifest describing where each raw dataset lives and which fields contain text/author/lang/genre.
- `--sanity-check` + `--sanity-limit`: cap records per dataset to validate the end-to-end flow on a small sample.
- `--total-docs`: global target size (defaults to 100k); language/genre/length buckets follow `PROCESSING.md`.
- `--train-ratio/--dev-ratio/--test-ratio`: split ratios (default 0.8/0.1/0.1).
- `--allow-other-languages`: optionally fill unused budget with languages outside the top-10 table.
- `--max-chunk-tokens` / `--target-chunk-tokens` / `--min-chunk-tokens`: control chunking for long docs; chunks are built on paragraph/sentence boundaries and capped at `max-chunk-tokens`.
- `--chunk-probability`: for docs exceeding `max-chunk-tokens`, probability of chunking; set <1 to retain some very long docs intact while still chunking most.
- `--truncate-to-tokens`: after chunking, optionally cap each document to this size using punctuation-aware truncation (avoids mid-sentence cuts).

Outputs per split (`train|dev|test`):
- `candidates.jsonl`: full documents with `candidate_id`, `author_id`, `lang`, `genre`, `content`, `source`, `token_length`.
- `queries.jsonl`: one query per eligible author in the split (no `author_id` field).
- `ground_truth.jsonl`: `query_id` → positive candidate ids + `author_id`.
- Logs: `dirty_docs.log`, `sampling_log.json`, `processing_summary.json`.

## Dataset manifest

See `datasets_manifest.example.json` for a template. Each entry needs:
- `loader`: one of `jsonl`, `csv`, `tsv`, `hf_streaming`.
- `path`: local file path (for tabular/jsonl) or HF dataset name via `extra.hf_dataset`.
- `split`: HF split when using `hf_streaming`.
- `text_field`, `author_field`, `lang_field` (or `static_lang`), optional `genre_field`, `raw_id_field`.
- `preprocess_row`: optional built-in helpers (e.g., `arxiv_first_author`).

## Processing steps implemented

- Standardizes genres using the mapping in `PROCESSING.md` (Section 6).
- Tokenizes with `tiktoken` (`cl100k_base`) when available; falls back to whitespace.
- Splits long docs (>500 tokens) into 100–500 token chunks, preserving author/source/genre.
- Dirty data filters (unique token ratio, symbol ratio, dominant token ratio, zero-length) with logging to `dirty_docs.log`.
- Enforces 3–5 docs per author (Section 9) with a fallback down to 2 when data are scarce.
- Samples to language, genre, and length-bucket targets (Sections 5, 8, 12) up to the global doc budget.
- Deterministic train/dev/test split per language (Section 4) and IR files for queries/candidates/ground truth (Section 15).
