# 포터빌리티 가이드 — 로컬(4060 8GB) → 클라우드/타 GPU 이동

24시간 안에 하드웨어를 갈아탈 수 있어야 함. 아래 원칙과 절차를 따르면 **환경 재현 시간이 15분 이내**.

## 원칙
1. 코드는 어디서든 동일하게 실행. **하드웨어 의존값은 `.env` 로만 조절.**
2. requirements는 목적별로 쪼개져 있음. GPU 아키텍처 바뀌어도 `pytorch.txt` 만 재설치.
3. Docker 이미지 하나로 재현 가능하도록 `Dockerfile` 유지.
4. 데이터/체크포인트는 gitignore. 이동 시 별도 sync (S3, HF Hub, gdrive rclone).

## 시나리오 1 — Lambda Labs / RunPod (Ubuntu + 임의 GPU)
```bash
git clone <team-repo>   # 또는 rsync/scp 로 로컬 → 원격 복사
cd Hackathon_2026Summer
bash scripts/setup_env.sh
source venv/bin/activate
cp .env.example .env    # 원격 GPU에 맞게 BATCH_SIZE 등 조정
```
- H100/A100 이면 `.env` 에서 `BATCH_SIZE`↑, `MIXED_PRECISION=bf16` 로 변경.
- xformers가 안 붙는 GPU면 `requirements/pytorch.txt` 에서 xformers 라인 주석 처리.

## 시나리오 2 — Google Colab (임시 실험용)
1. Colab에 리포 클론
2. `!pip install -r requirements/base.txt` + Colab pre-installed torch 그대로 사용 (torch 재설치 X)
3. 노트북에서 `notebooks/` 폴더 활용, 결과 checkpoint는 gdrive 마운트로 저장

## 시나리오 3 — Docker 컨테이너 (재현성 최고)
```bash
# 빌드 (초 1회)
docker build -t hacknation-2026 .

# 로컬 GPU로 실행
docker compose up -d
docker compose exec app bash

# 클라우드에서: 같은 이미지 pull 후 동일 명령
```

## `.env` 로 분리하는 하드웨어 의존값
| 키                    | 4060 8GB | H100 80GB | 설명                           |
| --------------------- | -------- | --------- | ------------------------------ |
| `BATCH_SIZE`          | 4        | 64        | 모델 크기 따라 조정            |
| `GRAD_ACCUM_STEPS`    | 8        | 1         | effective batch 유지           |
| `MIXED_PRECISION`     | fp16     | bf16      | Ampere 이상은 bf16 권장        |
| `LOAD_IN_4BIT`        | true     | false     | LLM 로컬 로드 시 4-bit 양자화  |
| `MAX_SEQ_LEN`         | 1024     | 4096      | 컨텍스트 길이                  |
| `NUM_WORKERS`         | 2        | 8         | DataLoader worker              |

## 환경 이동 시 바꿔야 하는 파일 — 딱 2개
1. `.env` — 위 표대로 수정
2. `requirements/pytorch.txt` — GPU 아키텍처가 CUDA 11.x인 경우 index-url을 `cu118` 로 교체

그 외 코드/문서는 그대로 유지.

## 데이터/체크포인트 이동
- **작은 데이터셋 (<1GB)**: HF Datasets 로 push → 어디서든 pull
- **큰 데이터셋**: S3 / rclone gdrive → 원격에서 pull
- **모델 체크포인트**: HF Hub private repo 로 push
