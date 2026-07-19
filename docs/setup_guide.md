# 환경 셋업 가이드 (Windows + RTX 4060 8GB)

## 0. 사전 요구사항
- Windows 11 (10도 가능)
- **Python 3.10** (3.11 도 OK, 3.12 는 TF 호환 이슈로 비추천)
- **CUDA 12.1** 드라이버 설치 (`nvidia-smi` 로 확인)
- Git for Windows
- (선택) WSL2 Ubuntu 22.04 — 리눅스 도구가 필요할 때

## 1. 저장소 클론 / 이 폴더로 이동
```powershell
cd C:\Users\HyeonYongLEE\Desktop\Hackathon_2026Summer
```

## 2. 가상환경 생성
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip wheel setuptools
```

## 3. 의존성 설치 (분리해서 순서대로)
CUDA 인덱스가 필요한 PyTorch만 별도 설치:
```powershell
pip install -r requirements\base.txt
pip install -r requirements\pytorch.txt --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple
pip install -r requirements\tensorflow.txt
pip install -r requirements\ai_harness.txt
pip install -r requirements\ml_extra.txt
pip install -r requirements\dev.txt
```

## 4. pre-commit 훅 설치
```powershell
pre-commit install
```

## 5. 계정 로그인
```powershell
wandb login          # WandB 대시보드 붙기
huggingface-cli login  # HF 게이티드 모델 접근용
```
OpenAI 등 API 키는 `.env` 에 넣기:
```powershell
copy .env.example .env
# 편집기로 열어서 채우기
```

## 6. GPU 인식 확인
PyTorch:
```powershell
python -c "import torch; print('cuda?', torch.cuda.is_available(), 'device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```
TensorFlow:
```powershell
python -c "import tensorflow as tf; print('gpus:', tf.config.list_physical_devices('GPU'))"
```

## 7. 스모크 테스트
```powershell
pytest -q tests\test_smoke.py
```

## 8. 문제 해결 (자주 발생)
- **PyTorch가 CPU만 잡힘** → `pytorch.txt` 를 `--index-url` 없이 설치했을 확률. venv 지우고 다시.
- **TF가 GPU를 못 봄** → `tensorflow[and-cuda]` 대신 `tensorflow` 만 깔았거나, CUDA/cuDNN 버전 불일치. `nvidia-smi` 로 드라이버 버전 확인.
- **bitsandbytes 에러** → Windows에서는 최신 wheel이 필요. `pip install -U bitsandbytes` 재시도.
- **4060 8GB OOM** → `.env` 의 `BATCH_SIZE` 낮추고 `MIXED_PRECISION=fp16` 설정.
