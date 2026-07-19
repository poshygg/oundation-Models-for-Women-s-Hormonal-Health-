"""05_build_vector_db.py — 각 논문의 title+abstract 를 임베딩해 Chroma 에 저장.

- 임베딩 모델은 configs/harness.yaml 의 embedding.model
- Chroma persist 디렉토리는 configs/harness.yaml 의 paths.chroma_db
- 메타데이터: arxiv_id, categories(문자열 join), capabilities(문자열 join), year, citations, pdf_path
"""
from __future__ import annotations

import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, get_logger, load_config, read_json  # noqa: E402

log = get_logger("build_vector_db", "05_build_vector_db.log")


def main() -> int:
    cfg = load_config()
    papers = read_json(ROOT / cfg["paths"]["index"] / "papers.json", default=[])
    if not papers:
        log.error("papers.json 이 비어있습니다. 03/04 스크립트 먼저 실행.")
        return 1

    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        log.error(f"의존성 부족: {e}. pip install -r requirements.txt 실행하세요.")
        return 2

    model_name = cfg["embedding"]["model"]
    device = cfg["embedding"].get("device", "cpu")
    log.info(f"임베딩 모델 로드: {model_name} ({device})")
    model = SentenceTransformer(model_name, device=device)

    persist_dir = ROOT / cfg["paths"]["chroma_db"]
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))

    coll_name = cfg["vector_db"]["collection"]
    # 재빌드
    try:
        client.delete_collection(coll_name)
    except Exception:
        pass
    collection = client.create_collection(
        name=coll_name,
        metadata={"hnsw:space": cfg["vector_db"]["distance"]},
    )

    ids, docs, metas = [], [], []
    for i, p in enumerate(papers):
        doc = (p.get("title", "") + ". " + p.get("abstract", "")).strip()
        if not doc:
            continue
        pid = p.get("arxiv_id") or f"legacy-{i}"
        ids.append(pid)
        docs.append(doc)
        metas.append({
            "arxiv_id": p.get("arxiv_id", "") or "",
            "title": p.get("title", "")[:400],
            "categories": ",".join(p.get("categories", [])),
            "capabilities": ",".join(p.get("capabilities", [])),
            "year": p.get("year") or 0,
            "citations": p.get("citations") or 0,
            "pdf_path": p.get("pdf_path", "") or "",
            "source": p.get("source", ""),
            "arxiv_url": p.get("arxiv_url", "") or "",
        })

    log.info(f"임베딩 계산 — {len(docs)}개")
    batch = cfg["embedding"].get("batch_size", 32)
    embs = []
    for i in tqdm(range(0, len(docs), batch), desc="embed", unit="batch"):
        chunk = docs[i : i + batch]
        embs.extend(model.encode(chunk, show_progress_bar=False, convert_to_numpy=True).tolist())

    log.info("Chroma add")
    # Chroma add 는 큰 배치도 OK 지만 안전하게 나눔
    for i in range(0, len(ids), 500):
        collection.add(
            ids=ids[i : i + 500],
            documents=docs[i : i + 500],
            embeddings=embs[i : i + 500],
            metadatas=metas[i : i + 500],
        )
    log.info(f"완료 — collection={coll_name}, count={collection.count()}, persist={persist_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
