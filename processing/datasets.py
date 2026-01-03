from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path
from typing import Callable, Iterable, Iterator

from huggingface_hub import hf_hub_download
from datasets import load_dataset

from .config import SUPPORTED_LOADERS, genre_mapper_for_source
from .types import DatasetConfig, RawDocument
from .utils import read_jsonl

logger = logging.getLogger(__name__)


def dataset_from_dict(cfg: dict) -> DatasetConfig:
    loader = cfg.get("loader")
    if loader not in SUPPORTED_LOADERS:
        raise ValueError(f"Unsupported loader '{loader}'. Supported: {sorted(SUPPORTED_LOADERS)}")

    preprocess_name = cfg.get("preprocess_row")
    preprocess_fn = PREPROCESSORS.get(preprocess_name) if preprocess_name else None

    return DatasetConfig(
        name=cfg.get("name") or cfg["source"],
        source=cfg["source"],
        loader=loader,
        path=Path(cfg["path"]) if cfg.get("path") else None,
        split=cfg.get("split"),
        text_field=cfg.get("text_field", "text"),
        author_field=cfg.get("author_field", "author"),
        lang_field=cfg.get("lang_field", "lang"),
        genre_field=cfg.get("genre_field"),
        static_lang=cfg.get("static_lang"),
        raw_id_field=cfg.get("raw_id_field"),
        preprocess_row=preprocess_fn,
        extra=cfg.get("extra", {}),
        max_documents=cfg.get("max_documents"),
    )


def load_manifest(path: Path) -> list[DatasetConfig]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {path}")
    with path.open() as f:
        data = json.load(f)
    datasets = data.get("datasets") if isinstance(data, dict) else data
    if not isinstance(datasets, list):
        raise ValueError("Manifest must be a list or a dict with key 'datasets'.")
    return [dataset_from_dict(cfg) for cfg in datasets]


def _expand_paths(path: Path, allowed_exts: set[str]) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        paths = [
            p for p in sorted(path.rglob("*")) if p.is_file() and p.suffix.lower() in allowed_exts
        ]
        if not paths:
            raise FileNotFoundError(f"No files with extensions {allowed_exts} found under {path}")
        return paths
    raise FileNotFoundError(f"Path {path} not found")


def _normalize_row(row: dict, config: DatasetConfig, idx: int) -> RawDocument | None:
    if config.preprocess_row:
        row = config.preprocess_row(dict(row))

    text = row.get(config.text_field)
    author = row.get(config.author_field)
    if text is None or author is None:
        return None

    lang_val = config.static_lang if config.static_lang else (row.get(config.lang_field) if config.lang_field else None)
    lang = str(lang_val).strip().lower() if lang_val else "unknown"
    genre_val = row.get(config.genre_field) if config.genre_field else None
    raw_id_val = row.get(config.raw_id_field) if config.raw_id_field else None
    raw_id = str(raw_id_val) if raw_id_val is not None else f"{config.source}:{idx}"
    genre_mapper = genre_mapper_for_source(config.source)
    genre = genre_mapper(genre_val)

    return RawDocument(
        raw_id=raw_id,
        author=str(author),
        text=str(text),
        lang=lang,
        source=config.source,
        genre=genre,
        metadata={"dataset": config.name},
    )


def iter_jsonl_dataset(config: DatasetConfig, sanity_limit: int | None) -> Iterator[RawDocument]:
    limit = min(filter(None, [config.max_documents, sanity_limit])) if sanity_limit or config.max_documents else None
    count = 0
    paths = _expand_paths(config.path, {".jsonl", ".json"})  # type: ignore[arg-type]
    for path in paths:
        if path.suffix.lower() == ".jsonl":
            rows_iter = read_jsonl(path)
        else:
            try:
                obj = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(obj, list):
                    rows_iter = obj
                elif isinstance(obj, dict):
                    rows_iter = obj.values()
                else:
                    rows_iter = []
            except Exception:
                rows_iter = read_jsonl(path)
        for idx, row in enumerate(rows_iter):
            doc = _normalize_row(row, config, idx)
            if doc:
                yield doc
                count += 1
            if limit and count >= limit:
                return


def iter_tabular_dataset(
    config: DatasetConfig,
    sanity_limit: int | None,
    delimiter: str,
    allowed_exts: set[str],
) -> Iterator[RawDocument]:
    import csv

    limit = min(filter(None, [config.max_documents, sanity_limit])) if sanity_limit or config.max_documents else None
    count = 0
    paths = _expand_paths(config.path, allowed_exts)  # type: ignore[arg-type]
    for path in paths:
        suffix = path.suffix.lower()
        file_delim = "," if suffix == ".csv" else delimiter
        with path.open(encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f, delimiter=file_delim)
            for idx, row in enumerate(reader):
                doc = _normalize_row(row, config, idx)
                if doc:
                    yield doc
                    count += 1
                if limit and count >= limit:
                    return


def iter_hf_streaming_dataset(config: DatasetConfig, sanity_limit: int | None) -> Iterator[RawDocument]:
    if not config.split:
        raise ValueError(f"HF loader for {config.name} requires 'split'.")
    ds_name = config.extra.get("hf_dataset") or (
        config.path.as_posix() if isinstance(config.path, Path) else None
    ) or config.name
    config_name = config.extra.get("hf_config")
    ds_kwargs = {k: v for k, v in config.extra.items() if k not in {"hf_dataset", "hf_config"}}
    if config_name:
        dataset = load_dataset(ds_name, config_name, split=config.split, streaming=True, **ds_kwargs)
    else:
        dataset = load_dataset(ds_name, split=config.split, streaming=True, **ds_kwargs)
    limit = min(filter(None, [config.max_documents, sanity_limit])) if sanity_limit or config.max_documents else None
    count = 0
    for idx, row in enumerate(dataset):
        doc = _normalize_row(row, config, idx)
        if doc:
            yield doc
            count += 1
        if limit and count >= limit:
            break


def iter_dataset(config: DatasetConfig, sanity_limit: int | None = None) -> Iterable[RawDocument]:
    if config.loader == "jsonl":
        if not config.path:
            raise ValueError(f"jsonl loader for {config.name} requires 'path'.")
        return iter_jsonl_dataset(config, sanity_limit)
    if config.loader == "csv":
        if not config.path:
            raise ValueError(f"csv loader for {config.name} requires 'path'.")
        return iter_tabular_dataset(config, sanity_limit, delimiter=",", allowed_exts={".csv"})
    if config.loader == "tsv":
        if not config.path:
            raise ValueError(f"tsv loader for {config.name} requires 'path'.")
        return iter_tabular_dataset(config, sanity_limit, delimiter="\t", allowed_exts={".tsv", ".txt", ".csv"})
    if config.loader == "hf_streaming":
        return iter_hf_streaming_dataset(config, sanity_limit)
    if config.loader == "blog_authorship":
        return iter_blog_authorship(config, sanity_limit)
    raise ValueError(f"Unsupported loader: {config.loader}")


def preprocess_arxiv_first_author(row: dict) -> dict:
    authors = row.get("authors") or row.get("author") or ""
    if isinstance(authors, list):
        author_val = authors[0] if authors else None
    else:
        author_val = str(authors).split(",")[0].strip() if authors else None
    row["author"] = author_val
    return row


def preprocess_hindi_story(row: dict) -> dict:
    story_no = row.get("Story_no")
    row["author"] = f"story_{story_no}" if story_no is not None else None
    return row


PREPROCESSORS: dict[str, Callable[[dict], dict]] = {
    "arxiv_first_author": preprocess_arxiv_first_author,
    "hindi_story_author": preprocess_hindi_story,
}


def iter_blog_authorship(config: DatasetConfig, sanity_limit: int | None) -> Iterator[RawDocument]:
    repo_id = config.extra.get("hf_repo", "barilan/blog_authorship_corpus")
    filename = config.extra.get("filename", "data/blogs.zip")
    archive_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=filename,
    )

    limit = min(filter(None, [config.max_documents, sanity_limit])) if sanity_limit or config.max_documents else None
    count = 0
    lang = config.static_lang or "en"
    genre_mapper = genre_mapper_for_source(config.source)

    with zipfile.ZipFile(archive_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".xml"):
                continue
            try:
                file_id, gender, age, job, horoscope = name.split(".")[:-1]
            except ValueError:
                file_id, gender, age, job, horoscope = name, "", "", "", ""
            with zf.open(name) as fh:
                date = ""
                line_idx = 0
                for raw in fh:
                    line = raw.decode("latin-1").strip()
                    if line.startswith("<date>"):
                        date = line.replace("<date>", "").replace("</date>", "")
                        continue
                    if not line or line.startswith("<"):
                        continue
                    doc = RawDocument(
                        raw_id=f"{name}:{date}:{line_idx}",
                        author=file_id,
                        text=line,
                        lang=lang,
                        source=config.source,
                        genre=genre_mapper(job or "general"),
                        metadata={
                            "gender": gender,
                            "age": age,
                            "job": job,
                            "horoscope": horoscope,
                            "date": date,
                        },
                    )
                    yield doc
                    count += 1
                    line_idx += 1
                    if limit and count >= limit:
                        return
