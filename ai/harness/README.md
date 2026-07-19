# ai/harness/ — 자체 eval harness

24시간 안에 여러 모델을 공정하게 비교하려면 harness가 있어야 함. 무엇을 쓸지 상황별 결정.

## 언제 뭘 쓰나
| 상황                                            | 쓸 것                          |
| ----------------------------------------------- | ------------------------------ |
| 표준 LLM 벤치마크 (MMLU, HellaSwag, ARC 등)     | `lm-eval` (EleutherAI harness) |
| Agent / tool-use / 안전성 평가                  | `inspect-ai` (UK AISI)         |
| 우리 프로덕트만의 커스텀 태스크                 | 자체 harness (아래 구조)       |
| 순수 텍스트 지표 (ROUGE, BLEU, BERTScore)       | `evaluate` 라이브러리          |
| LLM-as-judge                                    | OpenAI/Anthropic API + rubric  |

## 자체 harness 구조
```
harness/
├── tasks/       # 태스크 정의 (input/output 스키마 + 평가 함수)
├── runners/     # 모델별 어댑터 (HF, OpenAI, Anthropic, local)
└── configs/     # 벤치 스위트 조합
```

## 태스크 등록 절차 (권장 스켈레톤)
`tasks/<task_name>.py` 에 아래 인터페이스:
```python
from dataclasses import dataclass

@dataclass
class TaskExample:
    input: str
    expected: str
    metadata: dict

def load_examples() -> list[TaskExample]: ...
def score(prediction: str, example: TaskExample) -> dict: ...
```
runner가 `load_examples()` → 예측 생성 → `score()` 를 순회. 결과를 wandb.Table 로 push.

## 평가 지표
- **LLM**: accuracy, exact-match, ROUGE-L, BERTScore, LLM-as-judge (rubric 5점)
- **CNN**: top-1, top-5, macro-F1, confusion matrix (wandb.plot.confusion_matrix)
- **VLM**: image-text retrieval R@k, VQA accuracy
- **Audio (STT)**: WER, CER

## 실행 예시 (자체 harness가 준비되면)
```bash
python -m ai.harness.runners.run \
    --task ai/harness/tasks/qa_baseline.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --config ai/harness/configs/quick.yaml
```
