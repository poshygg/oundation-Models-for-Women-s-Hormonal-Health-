"""papers.json / capabilities.json / categories.json 로더."""
from __future__ import annotations

import functools
import json
from pathlib import Path

from .db import root
from .models import Paper

_INDEX = root() / "index"


@functools.lru_cache(maxsize=1)
def load_papers() -> list[Paper]:
    path = _INDEX / "papers.json"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return [Paper.from_dict(d) for d in raw]


@functools.lru_cache(maxsize=1)
def papers_by_id() -> dict[str, Paper]:
    out: dict[str, Paper] = {}
    for p in load_papers():
        key = p.arxiv_id or f"legacy::{p.pdf_path}"
        out[key] = p
    return out


@functools.lru_cache(maxsize=1)
def load_capabilities_index() -> dict[str, list[str]]:
    path = _INDEX / "capabilities.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@functools.lru_cache(maxsize=1)
def load_categories() -> dict[str, dict]:
    path = _INDEX / "categories.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def papers_for_capability(capability: str) -> list[Paper]:
    idx = load_capabilities_index()
    ids = idx.get(capability, [])
    by_id = papers_by_id()
    out: list[Paper] = []
    for i in ids:
        p = by_id.get(i)
        if p:
            out.append(p)
    return out
