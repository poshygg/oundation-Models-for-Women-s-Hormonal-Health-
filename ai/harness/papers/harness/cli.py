"""CLI 진입점.

사용:
  python -m harness "how to quantize a 70B model?"
  python -m harness "LoRA best practices" --top 10 --category llm_finetune_rag
  python -m harness "vllm serving" --min-citations 50 --min-year 2024
"""
from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table

from .loader import load_categories, load_capabilities_index, papers_for_capability
from .search import search


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.argument("query")
@click.option("--top", "-k", "top_k", default=5, type=int, help="반환 개수")
@click.option("--category", "-c", default=None, help="카테고리 필터 (예: llm_finetune_rag)")
@click.option("--capability", "-p", default=None, help="capability 필터 (예: lora)")
@click.option("--min-citations", default=None, type=int, help="최소 인용수")
@click.option("--min-year", default=None, type=int, help="최소 연도")
@click.option("--list-categories", is_flag=True, help="카테고리 목록 출력 후 종료")
@click.option("--list-capabilities", is_flag=True, help="capability 목록 출력 후 종료")
@click.option("--by-capability", default=None, help="지정 capability 로 필터, 벡터 검색 없이 인용순 반환")
def main(query, top_k, category, capability, min_citations, min_year,
         list_categories, list_capabilities, by_capability):
    console = Console()

    if list_categories:
        cats = load_categories()
        t = Table(title="Categories")
        t.add_column("key"); t.add_column("name"); t.add_column("count", justify="right")
        for k, v in cats.items():
            t.add_row(k, v.get("name", ""), str(v.get("count", 0)))
        console.print(t)
        return

    if list_capabilities:
        idx = load_capabilities_index()
        t = Table(title="Capabilities")
        t.add_column("tag"); t.add_column("count", justify="right")
        for k, v in sorted(idx.items(), key=lambda kv: -len(kv[1])):
            t.add_row(k, str(len(v)))
        console.print(t)
        return

    if by_capability:
        papers = sorted(papers_for_capability(by_capability), key=lambda p: -p.citations)[:top_k]
        t = Table(title=f"papers for capability={by_capability}")
        t.add_column("rank", width=4); t.add_column("title", overflow="fold")
        t.add_column("cite", justify="right"); t.add_column("pdf")
        for i, p in enumerate(papers, 1):
            t.add_row(str(i), p.title[:100], str(p.citations), p.pdf_path)
        console.print(t)
        return

    results = search(query, top_k=top_k, category=category, capability=capability,
                     min_citations=min_citations, min_year=min_year)
    if not results:
        console.print("[yellow]결과 없음. 벡터 DB 가 비었거나 필터가 너무 엄격할 수 있습니다.[/]")
        return

    t = Table(title=f"top {len(results)} for: {query}")
    t.add_column("rank", width=4)
    t.add_column("title", overflow="fold")
    t.add_column("cat"); t.add_column("caps", overflow="fold")
    t.add_column("cite", justify="right"); t.add_column("dist", justify="right")
    for i, r in enumerate(results, 1):
        t.add_row(
            str(i),
            r.title[:120],
            ",".join(r.categories),
            ",".join(r.capabilities[:5]),
            str(r.citations),
            f"{r.distance:.3f}",
        )
    console.print(t)

    console.print("\n[dim]PDF paths:[/]")
    for i, r in enumerate(results, 1):
        console.print(f"  {i}. {r.pdf_path}   {r.arxiv_url}")


if __name__ == "__main__":
    main()
