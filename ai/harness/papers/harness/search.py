"""자연어 질의 → Chroma semantic search + 메타 필터."""
from __future__ import annotations

from .db import get_collection, get_embedder
from .models import SearchResult


def _passes(meta: dict, category: str | None, capability: str | None,
            min_citations: int | None, min_year: int | None) -> bool:
    if category:
        cats = (meta.get("categories") or "").split(",")
        if category not in cats:
            return False
    if capability:
        caps = (meta.get("capabilities") or "").split(",")
        if capability not in caps:
            return False
    if min_citations is not None and (meta.get("citations", 0) or 0) < min_citations:
        return False
    if min_year is not None and (meta.get("year", 0) or 0) < min_year:
        return False
    return True


def search(
    query: str,
    top_k: int = 5,
    category: str | None = None,
    capability: str | None = None,
    min_citations: int | None = None,
    min_year: int | None = None,
) -> list[SearchResult]:
    """자연어 질의로 논문 검색.

    Chroma 는 pre-filter 메타 조건을 잘 지원하지 않는 경우가 있어서
    over-fetch 후 파이썬 side 에서 필터링한다.
    """
    coll = get_collection()
    embedder = get_embedder()
    qvec = embedder.encode([query], convert_to_numpy=True).tolist()

    fetch = max(top_k * 5, 25)
    res = coll.query(query_embeddings=qvec, n_results=fetch)

    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    out: list[SearchResult] = []
    for _id, doc, meta, dist in zip(ids, docs, metas, dists):
        if not _passes(meta, category, capability, min_citations, min_year):
            continue
        out.append(SearchResult(
            arxiv_id=meta.get("arxiv_id", "") or "",
            title=meta.get("title", "") or "",
            abstract=doc,
            categories=[c for c in (meta.get("categories") or "").split(",") if c],
            capabilities=[c for c in (meta.get("capabilities") or "").split(",") if c],
            citations=int(meta.get("citations", 0) or 0),
            year=int(meta.get("year")) if meta.get("year") else None,
            pdf_path=meta.get("pdf_path", "") or "",
            arxiv_url=meta.get("arxiv_url", "") or "",
            distance=float(dist),
        ))
        if len(out) >= top_k:
            break
    return out
