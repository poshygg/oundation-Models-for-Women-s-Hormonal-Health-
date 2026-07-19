# Papers — 논문 라이브러리 + RAG 검색 하네스

해커톤용 AI 논문 도서관. 기존 `trading/papers/` 를 legacy 로 흡수하고, 4개 서브카테고리에서 2024년 이후 인용수 상위 논문을 자동 수집한 뒤, 자연어 질의로 관련 논문을 찾을 수 있는 검색 하네스를 제공합니다.

## 폴더 구조

```
papers/
├── README.md
├── requirements.txt
├── legacy/                     # trading/papers/ 에서 복사된 기존 논문 (구조 유지)
├── focused/
│   ├── llm_finetune_rag/       # A. LLM 파인튜닝·RAG (~20편)
│   ├── vlm_multimodal/         # B. 비전-언어 멀티모달 (~20편)
│   ├── mlops_serving/          # C. MLOps·서빙·양자화 (~20편)
│   └── agents_tooluse/         # D. 에이전트·툴유즈 (~20편)
├── index/
│   ├── papers.json             # 전체 논문 메타데이터
│   ├── categories.json         # 카테고리 트리 + 카운트
│   └── capabilities.json       # capability → arxiv_id 역인덱스
├── chroma_db/                  # 벡터 DB (자동 생성, .gitignore)
├── scripts/                    # 01~06 파이프라인 스크립트
├── harness/                    # 파이썬 검색 API + CLI
└── configs/harness.yaml        # 임베딩 모델, 카테고리, capability 사전
```

## 최초 셋업

```bash
cd Hackathon_2026Summer/ai/harness/papers
python -m pip install -r requirements.txt

# 1) 기존 논문 복사 (trading/papers → legacy/)
python scripts/01_copy_legacy.py

# 2) 카테고리별 상위 인용 논문 다운로드 (arxiv + Semantic Scholar)
#    - 총 ~80편, rate limit 때문에 15~25분 소요
python scripts/02_download_papers.py

# 3) PDF 메타데이터 추출 → index/papers.json
python scripts/03_extract_metadata.py

# 4) capability 태그 자동 부여 → index/capabilities.json
python scripts/04_tag_capabilities.py

# 5) 벡터 DB 빌드 (임베딩 + Chroma)
python scripts/05_build_vector_db.py

# 6) 동작 확인
python scripts/06_smoke_test.py
```

옵션 커맨드:

```bash
# 소스 폴더 오버라이드
python scripts/01_copy_legacy.py --source "C:/other/papers" --overwrite

# 특정 카테고리만, 카테고리당 5편만
python scripts/02_download_papers.py --only mlops_serving --per-category 5
```

## 카테고리

| key | 이름 | 설명 |
|-----|-----|-----|
| `llm_finetune_rag` | LLM 파인튜닝·RAG | SFT/LoRA/QLoRA/DPO/RLHF, RAG(dense/hybrid), long context, tool use in LLMs |
| `vlm_multimodal` | 비전-언어 멀티모달 (VLM) | LLaVA류, SigLIP/CLIP 후속, VLM 파인튜닝, 멀티모달 RAG, 시각 추론 |
| `mlops_serving` | MLOps·서빙·양자화 | vLLM/TGI/TorchServe, GPTQ/AWQ/GGUF/FP8, pruning, distillation, LLM inference optimization |
| `agents_tooluse` | 에이전트·툴유즈 | ReAct 후속, function calling, multi-agent, computer use, tool learning, agent evals |
| `legacy` | Legacy | `trading/papers/` 에서 복사된 기존 논문 |

## Capability 태그

`configs/harness.yaml` 의 `capabilities` 섹션에 카테고리별 태그와 매칭 패턴이 정의되어 있습니다. 예:

- **LLM 파인튜닝·RAG**: `lora`, `qlora`, `dpo`, `ppo`, `rlhf`, `sft`, `ift`, `rag`, `dense_retrieval`, `hybrid_retrieval`, `reranker`, `long_context`, `kv_cache`, `flash_attention`
- **VLM**: `clip`, `siglip`, `llava`, `vlm`, `multimodal`, `vision_language`, `visual_reasoning`, `ocr_vlm`, `video_llm`
- **MLOps·서빙**: `vllm`, `tgi`, `torchserve`, `triton`, `quantization_int8`, `quantization_int4`, `gptq`, `awq`, `gguf`, `fp8`, `pruning`, `distillation`, `kv_cache_optimization`, `speculative_decoding`, `batching`, `serving`
- **에이전트·툴유즈**: `react`, `tool_calling`, `function_calling`, `agent`, `multi_agent`, `computer_use`, `code_agent`, `planning`, `swe_bench`, `gaia`, `tool_learning`

전체 목록은 `python -m harness --list-capabilities` 로 확인.

## 파이썬 API 사용 예시

```python
from harness import search

# 예시 1: 자연어 검색
for r in search("How to fine-tune LLaMA-3 on limited VRAM?", top_k=5):
    print(r.arxiv_id, r.title, r.citations, r.pdf_path)

# 예시 2: 카테고리 필터
results = search("distillation for small language models",
                 top_k=10, category="mlops_serving")

# 예시 3: capability + 최소 인용수 필터
results = search("preference optimization",
                 top_k=5, capability="dpo", min_citations=50)
```

capability 로 pre-filter 만 하고 벡터 검색 없이 인용순 상위를 원할 때는 loader 사용:

```python
from harness.loader import papers_for_capability
for p in sorted(papers_for_capability("quantization_int4"),
                key=lambda p: -p.citations)[:10]:
    print(p.title, p.citations)
```

## CLI 사용 예시

```bash
# 기본 검색
python -m harness "how to quantize a 70B model?"

# 카테고리 필터 + top 10
python -m harness "LoRA best practices" --top 10 --category llm_finetune_rag

# capability + 최소 인용수
python -m harness "long context extension" --capability long_context --min-citations 30

# 목록 조회
python -m harness --list-categories
python -m harness --list-capabilities

# capability 로 인용순 정렬 (벡터 검색 우회)
python -m harness "" --by-capability vllm --top 10
```

## "특정 기능을 짤 때 관련 논문 찾기" 워크플로우

1. **정확한 검색 — capability 필터**
   짜려는 기능이 특정 태그로 정의될 때 (`quantization_int4`, `tool_calling`, `speculative_decoding` 등):
   ```bash
   python -m harness "" --by-capability quantization_int4 --top 10
   ```

2. **발견적 검색 — 자연어 질의**
   태그로 딱 잡히지 않을 때:
   ```bash
   python -m harness "reduce KV cache memory during long-context inference" --top 8
   ```

3. **파생 검색 — 한 논문 → 관련 논문**
   좋은 논문 한 편을 찾은 뒤, 그 논문의 `capabilities` 로 역인덱스 조회:
   ```python
   from harness.loader import papers_by_id, papers_for_capability
   seed = papers_by_id()["2401.12345"]
   for cap in seed.capabilities:
       print(cap, "→", [p.arxiv_id for p in papers_for_capability(cap)[:3]])
   ```

## 재실행 팁

- 논문 목록 갱신: `02` 를 다시 돌리면 `manifest.json` 이 새로 쓰이고 이미 받은 PDF 는 skip.
- 태그 규칙 변경 후: `configs/harness.yaml` 수정 → `04` 만 다시 실행.
- 벡터 DB 만 리빌드: `05` 만 다시 실행 (기존 collection 을 지우고 새로 만듦).

## 구성 요소 요약

- 검색 백엔드: **Chroma (persistent)**
- 임베딩: **sentence-transformers/all-MiniLM-L6-v2** (기본), YAML 에서 교체 가능
- 메타 소스: **arxiv API + Semantic Scholar Graph API** (인용수)
- PDF 파싱: **PyMuPDF (fitz)** — legacy 폴더의 비-arxiv 논문용
- CLI: **click + rich**
