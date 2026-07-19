"""02_download_papers.py — arxiv + Semantic Scholar 를 이용해 카테고리별 상위 인용 논문 다운로드.

각 카테고리(4개)마다:
  1) 여러 검색 쿼리로 arxiv 후보 수집 (2024-01-01 이후)
  2) 중복 제거
  3) Semantic Scholar API 로 인용수 조회
  4) 인용수 상위 N편 선택 (기본 20)
  5) PDF 다운로드 → focused/<category>/{arxiv_id}_{slug}.pdf
  6) focused/<category>/manifest.json 기록

주의: arxiv/Semantic Scholar 는 rate limit 있음. 재시도 3회.
샌드박스 네트워크가 차단된 경우 로컬에서 실행하세요.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, get_logger, load_config, slugify, write_json  # noqa: E402

log = get_logger("download_papers", "02_download_papers.log")


@dataclass
class PaperMeta:
    arxiv_id: str
    title: str
    authors: list[str]
    summary: str
    published: str
    updated: str
    primary_category: str
    categories: list[str]
    pdf_url: str
    citations: int = -1
    influential_citations: int = -1
    matched_queries: list[str] = field(default_factory=list)


def normalize_arxiv_id(raw: str) -> str:
    # arxiv Result.entry_id 는 "http://arxiv.org/abs/2401.12345v2" 형태
    tail = raw.rsplit("/", 1)[-1]
    return tail.split("v")[0]


def search_arxiv(query: str, since_date: str, max_results: int, rate_sec: float):
    import arxiv

    client = arxiv.Client(page_size=50, delay_seconds=rate_sec, num_retries=3)
    q = f"({query}) AND submittedDate:[{since_date.replace('-', '')} TO 99991231]"
    search = arxiv.Search(
        query=q,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending,
    )
    return list(client.results(search))


def fetch_citations_s2(arxiv_ids: list[str], rate_sec: float, timeout: int) -> dict[str, dict[str, int]]:
    """Semantic Scholar batch endpoint 로 인용수 조회."""
    import requests

    out: dict[str, dict[str, int]] = {}
    if not arxiv_ids:
        return out
    url = "https://api.semanticscholar.org/graph/v1/paper/batch"
    fields = "citationCount,influentialCitationCount,externalIds"
    for i in range(0, len(arxiv_ids), 100):
        batch = arxiv_ids[i : i + 100]
        payload = {"ids": [f"ARXIV:{aid}" for aid in batch]}
        for attempt in range(3):
            try:
                r = requests.post(url, params={"fields": fields}, json=payload, timeout=timeout)
                if r.status_code == 429:
                    time.sleep(rate_sec * 4)
                    continue
                r.raise_for_status()
                data = r.json()
                for aid, item in zip(batch, data):
                    if item is None:
                        out[aid] = {"citations": 0, "influential": 0}
                    else:
                        out[aid] = {
                            "citations": item.get("citationCount", 0) or 0,
                            "influential": item.get("influentialCitationCount", 0) or 0,
                        }
                break
            except Exception as e:
                log.warning(f"S2 batch 실패 (재시도 {attempt+1}/3): {e}")
                time.sleep(rate_sec * 2)
        else:
            for aid in batch:
                out.setdefault(aid, {"citations": 0, "influential": 0})
        time.sleep(rate_sec)
    return out


def download_pdf(url: str, dest: Path, retries: int, timeout: int) -> bool:
    import requests

    if dest.exists() and dest.stat().st_size > 1024:
        return True
    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".tmp")
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 15):
                        if chunk:
                            f.write(chunk)
                tmp.rename(dest)
            return True
        except Exception as e:
            log.warning(f"PDF 다운로드 실패 {url} (재시도 {attempt+1}/{retries}): {e}")
            time.sleep(2 * (attempt + 1))
    return False


def process_category(cat_key: str, cat_cfg: dict, dl_cfg: dict, out_dir: Path, per_category: int) -> dict[str, Any]:
    log.info(f"[{cat_key}] 검색 시작 ({cat_cfg.get('name', cat_key)})")
    candidates: dict[str, PaperMeta] = {}

    for q in cat_cfg["queries"]:
        log.info(f"  arxiv 쿼리: {q}")
        try:
            results = search_arxiv(q, dl_cfg["since_date"], max_results=50, rate_sec=dl_cfg["arxiv_rate_sec"])
        except Exception as e:
            log.error(f"  arxiv 검색 실패 '{q}': {e}")
            continue
        for r in results:
            aid = normalize_arxiv_id(r.entry_id)
            if aid in candidates:
                candidates[aid].matched_queries.append(q)
                continue
            candidates[aid] = PaperMeta(
                arxiv_id=aid,
                title=r.title.strip().replace("\n", " "),
                authors=[a.name for a in r.authors],
                summary=(r.summary or "").strip().replace("\n", " "),
                published=r.published.isoformat() if r.published else "",
                updated=r.updated.isoformat() if r.updated else "",
                primary_category=r.primary_category or "",
                categories=list(r.categories or []),
                pdf_url=r.pdf_url,
                matched_queries=[q],
            )

    log.info(f"[{cat_key}] 후보 {len(candidates)}편, Semantic Scholar 인용수 조회")
    cite_map = fetch_citations_s2(list(candidates.keys()), dl_cfg["s2_rate_sec"], dl_cfg["request_timeout_sec"])
    for aid, meta in candidates.items():
        c = cite_map.get(aid, {"citations": 0, "influential": 0})
        meta.citations = c["citations"]
        meta.influential_citations = c["influential"]

    ranked = sorted(candidates.values(), key=lambda m: (m.citations, m.influential_citations), reverse=True)
    top = ranked[:per_category]
    log.info(f"[{cat_key}] 상위 {len(top)}편 선정, PDF 다운로드")

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    succ = fail = 0
    for meta in top:
        fname = f"{meta.arxiv_id}_{slugify(meta.title)}.pdf"
        dest = out_dir / fname
        ok = download_pdf(meta.pdf_url, dest, retries=dl_cfg["retries"], timeout=dl_cfg["request_timeout_sec"])
        rec = asdict(meta)
        rec["pdf_path"] = str(dest.relative_to(ROOT)).replace("\\", "/") if ok else None
        rec["downloaded"] = ok
        manifest.append(rec)
        if ok:
            succ += 1
        else:
            fail += 1
        time.sleep(dl_cfg["arxiv_rate_sec"])

    write_json(out_dir / "manifest.json", manifest)
    log.info(f"[{cat_key}] 완료 — 성공 {succ}, 실패 {fail}")
    return {"category": cat_key, "success": succ, "fail": fail, "candidates": len(candidates)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-category", type=int, default=None, help="카테고리당 논문 수 (기본 설정값)")
    parser.add_argument("--only", nargs="*", default=None, help="특정 카테고리만 처리 (예: llm_finetune_rag)")
    args = parser.parse_args()

    cfg = load_config()
    dl_cfg = cfg["download"]
    per_category = args.per_category or dl_cfg["per_category"]
    focused_root = ROOT / cfg["paths"]["focused"]

    stats = []
    for cat_key, cat_cfg in cfg["categories"].items():
        if args.only and cat_key not in args.only:
            continue
        out_dir = focused_root / cat_key
        stats.append(process_category(cat_key, cat_cfg, dl_cfg, out_dir, per_category))

    log.info("=" * 60)
    log.info("전체 다운로드 요약:")
    for s in stats:
        log.info(f"  {s['category']:24s}  후보 {s['candidates']:4d}  성공 {s['success']:3d}  실패 {s['fail']:3d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
