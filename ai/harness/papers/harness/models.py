"""데이터 클래스."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Paper:
    arxiv_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    year: int | None = None
    month: int | None = None
    citations: int = 0
    influential_citations: int = 0
    categories: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    pdf_path: str = ""
    arxiv_url: str = ""
    source: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "Paper":
        return cls(
            arxiv_id=d.get("arxiv_id", ""),
            title=d.get("title", ""),
            authors=d.get("authors", []) or [],
            abstract=d.get("abstract", "") or "",
            year=d.get("year"),
            month=d.get("month"),
            citations=d.get("citations", 0) or 0,
            influential_citations=d.get("influential_citations", 0) or 0,
            categories=d.get("categories", []) or [],
            capabilities=d.get("capabilities", []) or [],
            tags=d.get("tags", []) or [],
            pdf_path=d.get("pdf_path", "") or "",
            arxiv_url=d.get("arxiv_url", "") or "",
            source=d.get("source", "") or "",
        )


@dataclass
class SearchResult:
    arxiv_id: str
    title: str
    abstract: str
    categories: list[str]
    capabilities: list[str]
    citations: int
    year: int | None
    pdf_path: str
    arxiv_url: str
    distance: float

    def __repr__(self) -> str:
        return f"<SearchResult {self.arxiv_id or '(no-id)'} d={self.distance:.3f} cite={self.citations} '{self.title[:60]}'>"
