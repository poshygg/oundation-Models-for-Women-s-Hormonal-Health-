"""06_smoke_test.py — 하네스 검색 파이프라인 sanity check.

세 개 샘플 질의에 대해 top 5 결과를 예쁘게 출력한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # papers/ 를 import 경로에

QUERIES = [
    "How to efficiently fine-tune LLaMA-3 on a single 8GB GPU?",
    "Best practices for serving quantized LLMs with vLLM",
    "Multi-agent coordination for complex reasoning tasks",
]


def main() -> int:
    try:
        from harness.search import search
    except Exception as e:
        print(f"[!] harness import 실패: {e}")
        return 1
    from rich.console import Console
    from rich.table import Table

    console = Console()
    for q in QUERIES:
        console.rule(f"[bold cyan]{q}")
        results = search(q, top_k=5)
        if not results:
            console.print("[yellow]결과 없음 — 벡터 DB 가 비어있을 수 있습니다.[/]")
            continue
        table = Table(show_lines=False)
        table.add_column("rank", width=4)
        table.add_column("title", overflow="fold")
        table.add_column("cat")
        table.add_column("cite", justify="right")
        table.add_column("dist", justify="right")
        for i, r in enumerate(results, 1):
            table.add_row(
                str(i),
                r.title[:100],
                ",".join(r.categories),
                str(r.citations),
                f"{r.distance:.3f}",
            )
        console.print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
