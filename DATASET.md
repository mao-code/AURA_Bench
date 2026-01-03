# Multilingual Datasets

## Exorde
- **Languages**: 122+ (global)
- **Genres**: Social media, comments, news, forums
- **Scale**: 65M+ items per week
- **Length**: Short-long
- **Author Labels**: Yes (explicit author hash)
- **Link**: https://huggingface.co/datasets/Exorde/exorde-social-media-december-2024-week1
- **Notes**: Multi-genre; ideal for multilingual authorship benchmarking

## Babel Briefings
- **Languages**: 30+
- **Genres**: News headlines, social media
- **Scale**: 4.7M
- **Length**: Short
- **Author Labels**: Partial (source/org-level grouping only)
- **Link**: https://huggingface.co/datasets/felixludos/babel-briefings
- **Notes**: No per-post user ID

## Amazon Reviews Multi
- **Languages**: 6 (English, Spanish, French, German, Japanese, Chinese)
- **Genres**: Product reviews (e-commerce)
- **Scale**: ~200k reviews (≈33k per language)
- **Length**: Short-medium (1-5 sentences per review)
- **Author Labels**: Yes (reviewer IDs provided)
- **Link**: https://www.kaggle.com/datasets/mexwell/amazon-reviews-multi
- **Notes**: Balanced multilingual dataset with consistent structure across languages; suitable for cross-lingual authorship benchmarking

---

# Monolingual Datasets

## English - Blog Authorship Corpus
- **Languages**: English
- **Genres**: Blogs (personal posts)
- **Scale**: ~681k posts by 19,320 bloggers (≈140M words)
- **Length**: Medium (avg ~35 posts; ~7,250 words per blogger)
- **Author Labels**: Yes (one blog per author; demographics available)
- **Link**: https://huggingface.co/datasets/barilan/blog_authorship_corpus
- **Notes**: Classic author-profiling dataset from Blogger.com

## English - arXiv Papers
- **Languages**: English
- **Genres**: Scientific research papers
- **Scale**: ~1.7M papers
- **Length**: Long (5-15 pages, full papers)
- **Author Labels**: Yes (full author list per paper)
- **Link**: https://www.kaggle.com/datasets/Cornell-University/arxiv
- **Notes**: Scholarly preprints across scientific domains

---

## Chinese - Xiaohongshu / Weibo
- **Languages**: Chinese
- **Genres**: Social media (Weibo/XHS)
- **Scale**: ~11,329 posts + comments
- **Length**: Short (1-3 sentence posts)
- **Author Labels**: Yes (user IDs included)
- **Link**: https://www.kaggle.com/datasets/yuanchunhong/xiaohongshu-aigc-comments-including-postsdataset
- **Notes**: Chinese microblog content with metadata

## Chinese - Douban Reviews
- **Languages**: Chinese
- **Genres**: Reviews (books, movies, music)
- **Scale**: ~13.5M reviews
- **Length**: Short-medium
- **Author Labels**: Yes (~383k unique reviewers)
- **Link**: https://www.kaggle.com/datasets/fengzhujoey/douban-datasetratingreviewside-information
- **Notes**: Multi-domain review dataset with ratings
- **Deleted (Dropped Dataset)**: books_cleaned.txt movies_cleaned.txt music_cleaned.txt because of data quality issues

---

## Hindi - Hindi Discourse Dataset
- **Languages**: Hindi
- **Genres**: Literature (short stories)
- **Scale**: 53 stories by 11 authors
- **Length**: Medium (multi-page stories)
- **Author Labels**: Yes (famous 20th-century authors)
- **Link**: https://github.com/midas-research/hindi-discourse?tab=readme-ov-file
- **Notes**: Public-domain Hindi literature labeled for discourse modes

---

## Spanish - Public Domain Books
- **Languages**: Spanish
- **Genres**: Literature (books)
- **Scale**: ~302,640 texts; 13.9B words
- **Length**: Long (essays to novels)
- **Author Labels**: Yes (public domain metadata)
- **Link**: https://huggingface.co/datasets/PleIAs/Spanish-PD-Books
- **Notes**: Large corpus from national libraries

---

## French - Public Domain Books
- **Languages**: French
- **Genres**: Literature (books)
- **Scale**: ~289,000 books (~16.4B words)
- **Length**: Long (novels, nonfiction)
- **Author Labels**: Yes
- **Link**: https://huggingface.co/datasets/PleIAs/French-PD-Books
- **Notes**: From BnF Gallica archives

---

## Arabic - Classical Poetry
- **Languages**: Arabic
- **Genres**: Poetry (classical)
- **Scale**: ~70,000 poems by 750+ poets
- **Length**: Short to long (lines to stanzas)
- **Author Labels**: Yes (poet name included)
- **Link**: https://www.kaggle.com/datasets/mdanok/arabic-poetry-dataset
- **Notes**: Comprehensive poetry collection from 6th-20th century

---

## Russian - Public Domain Corpus
- **Languages**: Russian
- **Genres**: Literature (books, periodicals)
- **Scale**: 8,525 titles; ~995M words
- **Length**: Long
- **Author Labels**: Yes
- **Link**: https://huggingface.co/datasets/PleIAs/Russian-PD
- **Notes**: Mostly 19th-century works from Internet Archive

---

## German - Public Domain Corpus
- **Languages**: German
- **Genres**: Literature (books, newspapers)
- **Scale**: ~260,638 texts; ~37.65B words
- **Length**: Long
- **Author Labels**: Yes
- **Link**: https://huggingface.co/datasets/PleIAs/German-PD
- **Notes**: Largest open German corpus; 17th-19th century works