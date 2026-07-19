# 실험 워크플로우

## 핵심 원칙
**1개 실험 = 1개 git branch + 1개 wandb run + 1개 config 파일.**
이 3개가 항상 1:1:1 로 맞아야 실험 재현이 됨.

## 브랜치 네이밍
```
exp/<YYMMDD>-<짧은설명>
예) exp/260718-cnn-baseline
    exp/260718-llm-lora-mistral7b
    exp/260719-vlm-siglip-eval
```
데모/피치 브랜치는 `demo/<이름>`.

## Config 규약
- 공용 default: `configs/base.yaml`
- 실험별 오버라이드: `ai/cnn/configs/<exp_name>.yaml` 또는 `ai/llm/configs/<exp_name>.yaml`
- Hydra 로드: `python train.py --config-path ../configs --config-name <exp_name>`

## WandB
- Project: `hacknation-2026`
- Run name = 브랜치 이름 (자동 매핑)
- 태그: `[cnn|llm|vlm|audio]`, `[baseline|tuned|final]`
- 팀 공유용 대시보드 — 심사 때 실험 트랙 기록 보여줄 수 있음

## MLflow
- 로컬 대시보드 전용 (`scripts/launch_mlflow.ps1` 로 실행)
- 오프라인 상황에서도 항상 뜨는 백업 트래킹
- `experiments/mlruns/` 에 저장 (gitignore)

## Git 커밋 규칙
- 실험 시작: `exp: start <name> — <가설 1줄>`
- 결과 커밋: `exp: <name> — <핵심 수치>` (예: `exp: cnn-baseline — top1 0.72`)
- 병합 전 `memory/experiment_log.md` 에 결과 요약 1문단 추가

## 실험 로그
`memory/experiment_log.md` 에 아래 포맷으로:
```
### exp/260718-cnn-baseline
- 가설: ResNet50 fine-tune 이 EfficientNet-B0 스크래치보다 top-1 5%p 이상 높다
- 결과: top-1 0.71 vs 0.66. 가설 성립.
- 다음: EfficientNet-B3 로 스케일업, 데이터 aug 추가
- 링크: [wandb run](...) / commit hash
```

## 병렬 실험 관리 (16h 지점 이후 필수)
- worktree 활용: `git worktree add ../exp-A exp/260718-A` 로 여러 실험 동시 실행
- WandB sweep 도 옵션 (그리드 서치용)
