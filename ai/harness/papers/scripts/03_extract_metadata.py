"""03_extract_metadata.py — focused/ 와 legacy/ 의 PDF 를 스캔해서 index/papers.json 생성.

- focused/<cat>/manifest.json 있으면 우선 사용 (arxiv 메타 정확)
- legacy/ 는 PyMuPDF 로 첫 2페이지에서 title/authors/abstract 를 추정
- 최종 결과를 index/papers.json 에 통합, index/categories.json 도 생성
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, get_logger, load_config, read_json, write_json  # noqa: E402

log = get_logger("extract_metadata", "03_extract_metadata.log")

ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})")


def parse_pdf_head(pdf_path: Path) -> dict[str, Any]:
    """PyMuPDF 로 첫 2페이지에서 title/authors/abstract 대충 추출."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return {}
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        log.warning(f"PDF 열기 실패 {pdf_path.name}: {e}")
        return {}
    try:
        pages = min(2, doc.page_count)
        text = "\n".join(doc[i].get_text() for i in range(pages))
    finally:
        doc.close()

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    title = lines[0] if lines else pdf_path.stem
    if len(title) < 8 and len(lines) >= 2:
        title = f"{lines[0]} {lines[1]}"

    abstract = ""
    lower = text.lower()
    idx = lower.find("abstract")
    if idx >= 0:
        chunk = text[idx : idx + 3000]
        chunk = re.sub(r"(?i)^abstract[:.]?", "", chunk).strip()
        stop = re.search(r"(?i)\n(1\s+introduction|introduction\b)", chunk)
        if stop:
            chunk = chunk[: stop.start()]
        abstract = " ".join(chunk.split())[:2500]

    return {"title": title[:300], "abstract": abstract, "raw_text_head": text[:500]}


def guess_arxiv_id(pdf_path: Path) -> str | None:
    m = ARXIV_ID_RE.search(pdf_path.name)
    return m.group(1) if m else None


def collect_focused(focused_root: Path) -> list[dict[str, Any]]:
    papers: list[dict[str, Any]] = []
    for cat_dir in sorted(p for p in focused_root.iterdir() if p.is_dir()):
        manifest = read_json(cat_dir / "manifest.json", default=[])
        for entry in manifest:
            pdf_rel = entry.get("pdf_path")
            if not pdf_rel:
                continue
            pdf_abs = ROOT / pdf_rel
            if not pdf_abs.exists():
                continue
            pub = entry.get("published") or ""
            year = int(pub[:4]) if pub[:4].isdigit() else None
            month = int(pub[5:7]) if len(pub) >= 7 and pub[5:7].isdigit() else None
            papers.append({
                "arxiv_id": entry["arxiv_id"],
                "title": entry["title"],
                "authors": entry.get("authors", []),
                "abstract": entry.get("summary", ""),
                "year": year,
                "month": month,
                "citations": entry.get("citations", 0),
                "influential_citations": entry.get("influential_citations", 0),
                "primary_category": entry.get("primary_category", ""),
                "arxiv_categories": entry.get("categories", []),
                "categories": [cat_dir.name],
                "capabilities": [],
                "tags": [],
                "pdf_path": pdf_rel,
                "arxiv_url": f"https://arxiv.org/abs/{entry['arxiv_id']}",
                "source": "arxiv_focused",
            })
    return papers


def collect_legacy(legacy_root: Path) -> list[dict[str, Any]]:
    papers: list[dict[str, Any]] = []
    if not legacy_root.exists():
        return papers
    pdfs = list(legacy_root.rglob("*.pdf"))
    for pdf in tqdm(pdfs, desc="legacy PDF 파싱", unit="pdf"):
        head = parse_pdf_head(pdf)
        aid = guess_arxiv_id(pdf) or ""
        papers.append({
            "arxiv_id": aid,
            "title": head.get("title") or pdf.stem,
            "authors": [],
            "abstract": head.get("abstract", ""),
            "year": None,
            "month": None,
            "citations": 0,
            "influential_citations": 0,
            "primary_category": "",
            "arxiv_categories": [],
            "categories": ["legacy"],
            "capabilities": [],
            "tags": ["legacy"],
            "pdf_path": str(pdf.relative_to(ROOT)).replace("\\", "/"),
            "arxiv_url": f"https://arxiv.org/abs/{aid}" if aid else "",
            "source": "legacy_trading",
        })
    return papers


def main() -> int:
    cfg = load_config()
    focused_root = ROOT / cfg["paths"]["focused"]
    legacy_root = ROOT / cfg["paths"]["legacy"]
    index_dir = ROOT / cfg["paths"]["index"]

    log.info("focused/ 메타 수집")
    focused_papers = collect_focused(focused_root)
    log.info(f"  focused 논문 {len(focused_papers)}편")

    log.info("legacy/ PDF 파싱")
    legacy_papers = collect_legacy(legacy_root)
    log.info(f"  legacy 논문 {len(legacy_papers)}편")

    # dedup by arxiv_id (focused 우선)
    by_key: dict[str, dict[str, Any]] = {}
    for p in focused_papers + legacy_papers:
        key = p["arxiv_id"] or f"legacy::{p['pdf_path']}"
        if key not in by_key:
            by_key[key] = p
        else:
            # merge categories
            existing = by_key[key]
            for cat in p["categories"]:
                if cat not in existing["categories"]:
                    existing["categories"].append(cat)

    papers = list(by_key.values())
    write_json(index_dir / "papers.json", papers)
    log.info(f"papers.json 저장 — {len(papers)}편")

    cat_index = {
        k: {"name": v.get("name", k), "description": v.get("description", ""), "count": 0}
        for k, v in cfg["categories"].items()
    }
    cat_index["legacy"] = {"name": "Legacy (trading/papers)", "description": "기존 논문 폴더에서 복사", "count": 0}
    for p in papers:
        for c in p["categories"]:
            if c in cat_index:
                cat_index[c]["count"] += 1
    write_json(index_dir / "categories.json", cat_index)
    log.info("categories.json 저장")
    return 0


if __name__ == "__main__":
    sys.exit(main())
