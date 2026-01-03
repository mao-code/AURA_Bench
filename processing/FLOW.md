Source ingestion & normalization: Collect candidate items from the raw corpora (multiple languages/domains). Normalize text (Unicode NFC), strip control characters, collapse whitespace, and retain minimal metadata (language, source domain, length in tokens, genre/task type). Tokenize with the target tokenizer; store token length.

Deduplication & near-duplicate removal: Hash exact texts for exact dedup; use locality-sensitive hashing (e.g., MinHash) over character 5-grams to drop near-duplicates above a Jaccard threshold. Remove boilerplate via regex rules (copyright headers, navigation lists).

Safety & quality filtering:
Toxicity/policy: Apply classifier(s) to flag unsafe content; drop anything above a conservative score.
Language ID & script checks: Keep only items whose predicted language/script matches the target language; drop mixed-script anomalies.
Length bounds: Enforce per-task token limits (min/max) to avoid trivially short or overly long items.
Content heuristics: Remove low-information items (e.g., URL lists, code dumps if task is natural language, high emoji/markup ratio).

Safety & quality metrics (heuristics are thresholded; values below/above the thresholds are filtered):
- **Unique token ratio**: $r\_{uniq} = \\tfrac{\\lvert\\text{unique tokens}\\rvert}{\\lvert\\text{tokens}\\rvert}$. Filter if $r\_{uniq} < \\tau\_{uniq}$ (default $\\tau\_{uniq}=0.2$).
- **Symbol ratio**: $r\_{sym} = \\tfrac{\\#\\text{symbols}}{\\lvert\\text{text chars}\\rvert}$ where symbols are non-letters/non-digits (whitespace excluded for most sources, included for public-domain sources). Filter if $r\_{sym} > \\tau\_{sym}$ (default $\\tau\_{sym}=0.4$).
- **Max consecutive symbols** (public-domain sources only): $m\_{sym} = \\max$ run length of consecutive symbols. Filter if $m\_{sym} > K\_{sym}$ (default $K\_{sym}=5$).
- **Max repeated character run**: $m\_{rep} = \\max$ run length of identical non-space characters. Filter if $m\_{rep} > K\_{rep}$ (default $K\_{rep}=20$).
- **Zero/empty content**: drop if $\\text{token\\_length}=0$ or no non-empty tokens.
- **Alphabetic character ratio**: $r\_{\\alpha} = \\tfrac{\\#\\text{alphabetic chars}}{\\#\\text{non-space chars}}$. Filter if $r\_{\\alpha} < \\tau\_{\\alpha}$.
- **Alphabetic token ratio**: $r\_{tok} = \\tfrac{\\#\\text{tokens containing an alphabetic char}}{\\lvert\\text{tokens}\\rvert}$. Filter if $r\_{tok} < \\tau\_{tok}$.
- **Single-letter density**: $r\_{1} = \\tfrac{\\#\\text{single-letter tokens}}{\\lvert\\text{tokens}\\rvert}$ and $m\_{1}$ the max consecutive run of single-letter tokens. Filter if $r\_{1} > \\tau\_{1}$ or $m\_{1} \\ge K\_{1}$.
- **Script match ratio**: $r\_{script} = \\tfrac{\\#\\text{letters whose Unicode script}\\in S\_{lang}}{\\#\\text{letters}}$, where $S\_{lang}$ is the expected script set for the language (e.g., Latin for en/es/fr/de, Cyrillic for ru, Arabic for ar, CJK for zh, etc.). Filter if $r\_{script} < \\tau\_{script}(lang)$ when there are at least 8 letters.
- **Language ID agreement (fallback)**: If language detection is enabled and $r\_{script}$ is low/unknown, drop if the detected language $\\hat{\\ell}$ is not compatible with the target language $\\ell$.

Task-specific structuring: For each task type (e.g., QA, summarization, classification), parse fields (prompt, answer/label). Validate schema completeness; discard malformed entries.

Bucket definition for balanced sampling:
Define buckets along key axes: language, token length bins (e.g., quantiles or fixed ranges), domain/genre, and task type.
Compute per-bucket counts to understand the available distribution.

Bucketed random sampling:
Set target proportions per axis (e.g., equal mass across languages; length buckets balanced within each language; domain/task proportional or capped).
Within each (language × length bucket × task) cell, sample uniformly at random without replacement up to the target quota.
If a cell is underfull, redistribute the deficit proportionally to sibling cells in the same language and task while keeping global balance constraints.
Use this bucket acting like a sliding window to slide over the whole dataset. Sample uniformly at random without replacement and add next instance in to the bucket. This sampling technique can mimic global random sampling without exploding the memory.

Post-processing:
Canonicalize formatting (trim, single trailing newline, consistent quotes).
Re-tokenize and re-check length after trimming; drop items that fall outside bounds.
Final de-dup pass on prompt fields to prevent leakage across splits.

Split construction (train/dev/test or dev/test):
Stratify splits across the same buckets to preserve balance; use disjoint hashing of IDs to avoid cross-split collisions.
Enforce source disjointness when required (e.g., no same-document leakage across splits).

This algorithm yields a balanced, diverse benchmark with controlled length and language coverage, minimized duplicates, and safety/quality guarantees.
