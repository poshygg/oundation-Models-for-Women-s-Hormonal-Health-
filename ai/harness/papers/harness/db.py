"""Chroma 클라이언트 + 임베딩 모델 싱글톤."""
from __future__ import annotations

import functools
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]


@functools.lru_cache(maxsize=1)
def _load_cfg() -> dict:
    with (_ROOT / "configs" / "harness.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@functools.lru_cache(maxsize=1)
def get_embedder():
    from sentence_transformers import SentenceTransformer

    cfg = _load_cfg()
    return SentenceTransformer(cfg["embedding"]["model"], device=cfg["embedding"].get("device", "cpu"))


@functools.lru_cache(maxsize=1)
def get_collection():
    import chromadb

    cfg = _load_cfg()
    persist_dir = _ROOT / cfg["paths"]["chroma_db"]
    client = chromadb.PersistentClient(path=str(persist_dir))
    return client.get_collection(cfg["vector_db"]["collection"])


def root() -> Path:
    return _ROOT
