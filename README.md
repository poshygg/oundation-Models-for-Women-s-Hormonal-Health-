# Hack Nation MIT — 2026 Summer

24시간 안에 실제 AI 프로덕트를 처음부터 빌드. 개인 작업환경 (팀 공유 X).

## 지금 바로
1. `powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1`
2. `wandb login` → `huggingface-cli login` → `copy .env.example .env`
3. `pytest -q tests\test_smoke.py`

## 폴더
- **docs/** — 대회 정보, 셋업 가이드, 24h 플랜, 모델 레지스트리, 워크플로우, 포터빌리티 가이드, 시간 예산 게이트
- **ai/** — harness(논문 검색 + eval) / CNN / LLM (실험 코드)
- **ml/** — PyTorch / TensorFlow 트레이닝 스켈레톤 (둘 다 준비)
- **scripts/** — 환경 셋업, MLflow/WandB 실행, quick_start, time_probe
- **requirements/** — 목적별 분리 (base / pytorch / tensorflow / ai_harness / ml_extra / dev)
- **data/** · **experiments/** — gitignore
- **memory/experiment_log.md** — 실험별 결과 노트

## Training code: search papers -> time budget gate -> write code
1. `python -m harness "<query>"` to find a reference paper (`ai/harness/papers/README.md`)
2. `python scripts\time_probe.py --steps-per-epoch N --epochs N --step-seconds N --phase N` for a go/no-go check (`docs/time_budgeting.md`)
3. If it passes, extend the `ml/`/`ai/` skeletons — full rules in `CLAUDE.md`

## 실험 = 브랜치 + wandb run + config 1:1:1
새 실험: `.\scripts\quick_start.ps1 -Name "cnn-baseline" -Kind "cnn"`

## 하드웨어 바뀔 때
`.env` 와 `requirements/pytorch.txt` 딱 2개만 손봄. `docs/portability_guide.md` 참조.

## 팀 저장소
아직 미정. 팀장이 만들면 `git remote add origin <url>` 후 push.
