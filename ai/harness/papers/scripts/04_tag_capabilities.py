"""04_tag_capabilities.py — 초록+제목에서 capability 키워드를 매칭해 태그 부여.

- configs/harness.yaml 의 capabilities 사전 사용
- 매칭된 capability 를 papers.json 의 각 논문 capabilities 에 저장
- index/capabilities.json 역인덱스 (capability -> [arxiv_id or pdf_path])
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, get_logger, load_config, read_json, write_json  # noqa: E402

log = get_logger("tag_capabilities", "04_tag_capabilities.log")


def build_matchers(cfg: dict) -> list[tuple[str, str, list[str]]]:
    """(category, tag, patterns) 튜플 리스트."""
    out = []
    for cat, tags in cfg["capabilities"].items():
        for entry in tags:
            out.append((cat, entry["tag"], [p.lower() for p in entry["patterns"]]))
    return out


def match_capabilities(text: str, matchers) -> list[str]:
    lowered = " " + text.lower() + " "
    hits: list[str] = []
    for _cat, tag, patterns in matchers:
        if any(p in lowered for p in patterns):
            hits.append(tag)
    return sorted(set(hits))


def main() -> int:
    cfg = load_config()
    index_dir = ROOT / cfg["paths"]["index"]
    papers_path = index_dir / "papers.json"
    papers = read_json(papers_path, default=[])
    if not papers:
        log.error("papers.json 이 비어있습니다. 먼저 03_extract_metadata.py 를 실행하세요.")
        return 1

    matchers = build_matchers(cfg)

    reverse: dict[str, list[str]] = defaultdict(list)
    for p in papers:
        text = f"{p.get('title','')} {p.get('abstract','')}"
        caps = match_capabilities(text, matchers)
        p["capabilities"] = caps
        key = p.get("arxiv_id") or p.get("pdf_path", "")
        for c in caps:
            reverse[c].append(key)

    write_json(papers_path, papers)
    write_json(index_dir / "capabilities.json", dict(sorted(reverse.items())))
    log.info(f"완료 — 논문 {len(papers)}편, capability {len(reverse)}종")
    for cap, ids in sorted(reverse.items(), key=lambda kv: -len(kv[1]))[:10]:
        log.info(f"  {cap}: {len(ids)}편")
    return 0


if __name__ == "__main__":
    sys.exit(main())
