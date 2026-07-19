"""papers harness — RAG 검색 API.

Quick start
-----------
>>> from harness.search import search
>>> for r in search("how to quantize a 70B model?", top_k=5):
...     print(r.title, r.citations)
"""
from .models import Paper, SearchResult
from .search import search

__all__ = ["Paper", "SearchResult", "search"]
