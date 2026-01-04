# AuthBench Dataset Processing Plan

This document specifies how to process the raw source datasets into a unified benchmark for authorship representation and attribution. It is written so that an AI agent can implement the pipeline deterministically.

---

## 1. Benchmark Overview

The goal is to build a general benchmark for authorship representation and attribution that:

- Covers multiple languages (en, zh, hi, es, fr, ar, ru, de, ja, ko, plus others when available)
- Spans multiple genres (social media, blogs, reviews, literature, research papers, etc.)
- Covers a wide token length range (short to extra long)
- Uses explicit author identifiers

The benchmark size is fixed to approximately **100,000 documents**, sampled from the configured datasets.

We create **three splits**:

- `train`
- `dev`
- `test`

Default split ratios:

- `train`: 80%
- `dev`: 10%
- `test`: 10%

These ratios should be configurable via arguments, but 80/10/10 is the default.

All datasets used are described in `DATASET.md`. This plan assumes that the datasets listed there match the ones described in the context above.

---

## 2. Unified Document Schema

All processed documents must follow the same JSON schema:

```json
{
  "id": "<global_unique_doc_id>",
  "author_id": "<hashed_author_id>",
  "content": "<document text>",
  "genre": "<genre_tag>",
  "lang": "<language_code>",
  "source": "<source_dataset_tag>",
  "token_length": <integer_token_length>
}
```

Implementation notes:

- `id`: a globally unique document identifier within the benchmark.
  - Recommended: an integer index (0-based) or a string like `"doc_000123"`.
- `lang`: use ISO-like language codes (e.g. `en`, `zh`, `hi`, `es`, `fr`, `ar`, `ru`, `de`, `ja`, `ko`).
- `source`: short dataset tag (e.g. `exorde`, `babel_briefings`, `amazon_multi`, `blog_authorship`, `arxiv`, `xiaohongshu`, `douban`, `spanish_pd_books`, `french_pd_books`, `arabic_poetry`, `russian_pd`, `german_pd`, etc.).
- `token_length`:
  - Use a single tokenizer (for example, `tiktoken` with `cl100k_base`) for all languages.
  - `token_length = len(tokenizer.encode(content))`.

---

## 3. Prerequisite
All scripts under AURA_Bench/raw_analysis must be able to:
1. Load each dataset listed in `AURA_Bench/DATASET.md`.
2. Extract author identifier, raw text, language, and any available genre metadata.
3. Compute basic statistics per dataset:
   1. number of documents
   2. number of unique authors
   3. token length distribution
   4. per author document count

You must take a deep look at these scripts of how we read the file, extract the fields and aggregate information.

There are some output for a subset of these datasets in `AURA_Bench/raw_analysis/output`.
You can check the fields name, values, etc. for each dataset

## 4. Target Benchmark Size and Splits

### 4.1 Total Number of Documents

- Total target documents: **100,000** (after cleaning and splitting).
- If the available clean documents exceed this number, sample according to the language and genre distributions described below.
- If the available clean documents are fewer than 100,000, use as many as possible while preserving the proportions approximately.

### 4.2 Split Logic

- Use a **document-level split**:
  - A single `author_id` can appear across splits, but the same `id` must not.
  - For reproducibility, sort authors and documents deterministically (for example by `(lang, source, author_id, raw_id)`), then assign them to splits in a fixed order.
- Default ratios:
  - `train`: 80% of documents
  - `dev`: 10% of documents
  - `test`: 10% of documents

Keep the ratio approximately consistent **within each language** so that every language has representation in all splits.

---

## 5. Language Distribution

We focus on 10 high-resource languages:

`en, zh, hi, es, fr, ar, ru, de, ja, ko`

We also allow additional languages if:

- The dataset provides explicit author identifiers.
- The dataset has a reasonable scale (enough authors and documents).

Target distribution for the 10 main languages (around the 300k documents):

| Language      | Percent |
| ------------- | ------- |
| English (en)  | 35%     |
| Chinese (zh)  | 10%     |
| Spanish (es)  | 10%     |
| Arabic (ar)   | 8%      |
| French (fr)   | 8%      |
| Russian (ru)  | 8%      |
| German (de)   | 8%      |
| Japanese (ja) | 5%      |
| Korean (ko)   | 5%      |
| Hindi (hi)    | 3%      |

Implementation steps:

1. Compute `target_docs_per_language[L] = round(100000 * percent[L])`.
2. For any additional languages not on the list, either:
   - Group them into an `"other"` bucket, or
   - Ignore them unless explicitly configured.
3. During sampling, enforce these targets per language as closely as possible.

Because English has more genres and high quality data, it is intentionally assigned the largest share.

---

## 6. Genre Encoding and Dataset Mapping

We treat a coarse-grained `domain` as the main notion of genre, and optionally attach sub-genre information if available.

### 6.1 `genre` Field Encoding

Standardized genre values:

1. Exorde (multilingual):  
   - `social_media/{subgenre}`  
   - Where `{subgenre}` is derived from Exorde metadata (for example `social_media/news`, `social_media/comments`, `social_media/forums`).

2. Babel Briefings (multilingual):  
   - `news`

3. Amazon Reviews Multi (multilingual):  
   - `ecommerce_reviews`

4. Blog Authorship (English):  
   - `blog/{subgenre}`  
   - If explicit sub-genres are not available, use `blog/general`.

5. ArXiv (English):  
   - `research_paper`

6. Xiaohongshu / Weibo (Chinese):  
   - `social_media/xiaohongshu` or `social_media/weibo` depending on source.

7. Douban Reviews (Chinese):  
   - `media_reviews/douban`

8. Hindi Discourse Dataset (Hindi):  
   - `literature`

9. Spanish Public Domain Books (Spanish):  
   - `literature`

10. French Public Domain Books (French):  
    - `literature`

11. Arabic Poetry Dataset (Arabic):  
    - `poetry`

12. Russian Public Domain Corpus (Russian):  
    - `literature`

13. German Public Domain Corpus (German):  
    - `literature`

Implementation notes:

- If a dataset provides multiple categories or tags, map them to a small set of sub-genres (for example, up to 10 sub-genres per `social_media` or `blog`).
- Keep `genre` values consistent and machine-readable, all lowercase with `/` separators.

---

## 7. High-level Sampling Strategy

Sampling proceeds in this order:

1. Filter out dirty or unusable documents (see Section 11).
2. Split long documents into meaningful chunks (see Section 10).
3. Compute per-language pools of candidate documents.
4. For each language, subdivide by `genre` using the mappings above.
5. Within each `(language, genre)` bucket, enforce:
   - Token length distribution (Section 8)
   - Documents per author constraints (Section 9)
6. Sample documents to meet the language and genre distribution targets.

---

## 8. Token Length Buckets and Distribution

Token length buckets (by `token_length`):

- `short`: 1 to 10 tokens
- `medium`: 11 to 100 tokens
- `long`: 101 to 500 tokens
- `extra_long`: more than 500 tokens

Target distribution **within each language**:

| Length      | Percent |
| ----------- | ------- |
| short       | 15%     |
| medium      | 50%     |
| long        | 20%     |
| extra_long  | 15%     |

Implementation steps:

1. For each candidate document, compute `token_length`.
2. Assign each document to a bucket based on the thresholds above.
3. For each language `L`, and optionally each `(L, genre)`:
   - Compute `target_docs[L, bucket] = round(target_docs_per_language[L] * length_percent[bucket])`.
4. When sampling, prefer documents in underrepresented buckets.
5. If a bucket is underfull due to data scarcity:
   - Borrow from adjacent buckets (for example, use long when extra_long is scarce), but keep track of the deviation.

---

## 9. Documents per Author and Author Sampling

We want approximately **3 to 5 documents per author** in the final benchmark.

Implementation steps:

1. After splitting and cleaning:
   - For each `(lang, source)` combination, group documents by `author_id`.
2. Filter authors:
   - Keep only authors who have at least 3 candidate documents after splitting.
   - Drop authors with fewer than 3 documents for the main benchmark (they can be logged for analysis only).
3. For each remaining author:
   - Let `n` be the number of documents for that author.
   - If `n` is between 3 and 5, keep all documents.
   - If `n` is greater than 5, randomly sample 5 documents for that author (stratified by token length if possible).
4. Ensure that the total number of documents still meets the language and length targets approximately. If not, relax the 3-to-5 rule slightly (for example allow 2 documents for rare languages) as a fallback, but log this explicitly.

This strategy keeps the per-author contribution balanced and makes the benchmark suitable for authorship tasks.

---

## 10. Long Context Splitting

Long documents (lengthy papers, books, etc.) must be split into smaller chunks that behave as independent documents from the same author.

Implementation steps:

1. For each raw document with `token_length > 500`:
   - Parse the text into sentences or paragraphs if possible.
   - Greedily group sentences into chunks with a target size between 100 and 500 tokens.
   - Do not create chunks shorter than 50 tokens unless it is the last remainder.
2. Each chunk becomes a new document with:
   - the same `author_id`
   - the same `lang`, `source`, and `genre`
   - a new `raw_chunk_id` such as `"{raw_id}#chunk_{k}"`.
3. Recompute `token_length` for each chunk.
4. Use these chunks in the sampling pipeline just like native documents.

This splitting is especially important for:

- ArXiv papers
- Public domain books (Spanish, French, Russian, German, etc.)

---

## 11. Dirty Data Detection and Exclusion

Some datasets, especially large public domain corpora, contain noisy or corrupted text. We need automatic heuristics to exclude such entries.

### 11.1 Heuristics

For each candidate document (after chunking):

1. **Duplicate token ratio**:
   - Compute `unique_token_ratio = (#unique_tokens) / (#tokens)`.
   - If `unique_token_ratio < 0.2`, mark the document as dirty.
2. **Symbol ratio**:
   - Compute the proportion of characters that are non-letter and non-digit (`symbols / total_chars`).
   - If the symbol ratio exceeds a threshold (for example `> 0.5`), mark as dirty.
3. **Repetition**:
   - If any single token accounts for more than 50% of all tokens in the document, mark as dirty.
4. **Length sanity**:
   - Drop documents with `token_length == 0`.

Thresholds should be configurable, but use the values above as defaults.

### 11.2 Dataset-specific Notes

- For Spanish and French public domain books (`spanish_pd_books`, `french_pd_books`), apply these heuristics aggressively and log the fraction of entries removed.
- Maintain a log file `dirty_docs.log` listing:
  - `source`, `raw_id` (or `raw_chunk_id`), and the reason for exclusion.

---

## 12. Language and Genre Target Distributions

After cleaning and splitting, sample documents to approximate the following ideal distributions.

### 12.1 Language Distribution

Same as in Section 5:

| Language      | Percent |
| ------------- | ------- |
| English (en)  | 35%     |
| Chinese (zh)  | 10%     |
| Spanish (es)  | 10%     |
| Arabic (ar)   | 8%      |
| French (fr)   | 8%      |
| Russian (ru)  | 8%      |
| German (de)   | 8%      |
| Japanese (ja) | 5%      |
| Korean (ko)   | 5%      |
| Hindi (hi)    | 3%      |

### 12.2 Genre Distribution Per Language

Because available genres differ by language, we define **per-language genre targets**. Within each language, the percentages below should sum to 100%.

**English**

| Genre                 | Percent |
| --------------------- | ------- |
| social_media/{genre}  | 45%     |
| blog/{genre}          | 15%     |
| ecommerce_reviews     | 10%     |
| research_paper        | 10%     |
| news                  | 20%     |

**Chinese**

| Genre                        | Percent |
| ---------------------------- | ------- |
| social_media/xiaohongshu     | 35%     |
| social_media/{genre}         | 20%     |
| media_reviews/douban         | 25%     |
| news                         | 20%     |

Use `Xiaohongshu` (or similar) preferentially to fill the `social_media/xiaohongshu` portion.

**Spanish**

| Genre                 | Percent |
| --------------------- | ------- |
| social_media/{genre}  | 45%     |
| literature            | 35%     |
| news                  | 20%     |

**Arabic**

| Genre                 | Percent |
| --------------------- | ------- |
| social_media/{genre}  | 45%     |
| literature            | 25%     |
| poetry                | 10%     |
| news                  | 20%     |

**French**

| Genre                 | Percent |
| --------------------- | ------- |
| social_media/{genre}  | 55%     |
| literature            | 25%     |
| news                  | 20%     |

**Russian**

| Genre                 | Percent |
| --------------------- | ------- |
| social_media/{genre}  | 55%     |
| literature            | 25%     |
| news                  | 20%     |

**German**

| Genre                 | Percent |
| --------------------- | ------- |
| social_media/{genre}  | 55%     |
| literature            | 25%     |
| news                  | 20%     |

**Japanese**

| Genre                 | Percent |
| --------------------- | ------- |
| social_media/{genre}  | 55%     |
| literature            | 25%     |
| news                  | 20%     |

**Korean**

| Genre                 | Percent |
| --------------------- | ------- |
| social_media/{genre}  | 55%     |
| literature            | 25%     |
| news                  | 20%     |

**Hindi**

| Genre                 | Percent |
| --------------------- | ------- |
| social_media/{genre}  | 55%     |
| literature            | 25%     |
| news                  | 20%     |

Implementation steps:

1. For each language `L`, compute `target_docs_per_language[L]`.
2. For each genre `G` in that language, compute `target_docs[L, G] = round(target_docs_per_language[L] * genre_percent[L, G])`.
3. Sample within each `(L, G)` bucket, respecting token length and author constraints.
4. If a given `(L, G)` lacks sufficient clean documents, log the deficit and redistribute the surplus to other genres within the same language.

---

## 13. ID and Author Hashing

### 13.1 Document IDs

- Define a global ordering of documents after sampling.
- For example, sort by `(lang, source, author_id, raw_id or raw_chunk_id)`.
- Assign `id` as `0, 1, 2, ...` in this order, or as `"doc_{:06d}".format(index)`.

### 13.2 Author IDs

To anonymize authors while preserving consistent identity across datasets:

1. For each raw author identifier:
   - Build a canonical string:
     ```text
     canonical_author_key = f"{source_dataset_tag}:{raw_author_identifier}"
     ```
2. Compute:
   ```text
   author_id = sha256(canonical_author_key.encode("utf-8")).hexdigest()
   ```
3. Use `author_id` in all downstream files.

This ensures:

- Same author in the same dataset always maps to the same `author_id`.
- Different datasets with similar author names still have distinct `author_id`s.

---

## 14. ArXiv Author Handling

For the `arxiv` dataset:

- Use the **first author** in the author list as the `author_identifier` for the purpose of authorship attribution.
- If the author list is empty or malformed, drop the document.

This simplifies authorship attribution for multi-author scientific papers.

---

## 15. Queries, Candidates, and Ground Truth Files

For each split (`train`, `dev`, `test`), we create three JSONL files:

1. `queries.jsonl`
2. `candidates.jsonl`
3. `ground_truth.jsonl`

### 15.1 `candidates.jsonl`

Contains all documents in the split, one per line:

```json
{
  "candidate_id": "<same_as_doc_id>",
  "author_id": "<hashed_author_id>",
  "lang": "<language_code>",
  "genre": "<genre_tag>",
  "content": "<document text>",
  "source": "<dataset_tag>",
  "token_length": <integer_token_length>
}
```

By default, `candidate_id` equals `id` from the unified schema.

### 15.2 `queries.jsonl`

Contains a subset of documents selected as queries:

```json
{
  "query_id": "<query_document_id>",
  //   "author_id": "<hashed_author_id>", // no author_id in query file
  "lang": "<language_code>",
  "genre": "<genre_tag>",
  "content": "<query text>",
  "source": "<dataset_tag>",
  "token_length": <integer_token_length>
}
```

Selection strategy:

- For each author with at least 3 documents in a split:
  - Choose 1 document as a query (for example at random or based on a fixed rule).
  - The remaining 2 to 4 documents for that author remain as candidate documents.
- Ensure that:
  - Each `query_id` exists in `candidates.jsonl`.
  - Each author contributes at most one query per split by default (this can be parameterized).

### 15.3 `ground_truth.jsonl`

Maps each query to its positive candidates:

```json
{
  "query_id": "<query_document_id>",
  "positive_ids": ["<candidate_id_1>", "<candidate_id_2>", "..."],
  "author_id": "<hashed_author_id>"
}
```

Construction rule:

- For a given `query_id` with author `A`:
  - `positive_ids` is the list of candidate documents in the same split whose `author_id` equals `author_id(query_id)` and whose `candidate_id` is not equal to `query_id`.
- There is no need to explicitly list negative candidates. All other candidates in the same split are considered negatives for retrieval tasks.

This makes the benchmark suitable for retrieval and ranking methods for authorship attribution.

---

## 16. Special Note for Chinese Social Media

For Chinese (`zh`):

- To fill `social_media/xiaohongshu`, prioritize documents from the Xiaohongshu / Weibo dataset with valid author IDs and clean content.
- If additional Chinese social media data is available via Exorde or Babel, map them to `social_media/{genre}` and use them to fill the remaining `social_media` quota in the Chinese genre table.