# ml/pytorch/ — PyTorch 트레이닝 스켈레톤

`train.py` 는 accelerate 호환 최소 루프. 실제 모델·데이터를 채워넣어 사용.

## 실행
```powershell
python ml\pytorch\train.py
# 또는 multi-GPU
accelerate launch ml\pytorch\train.py
```

## 채워야 할 곳
- `build_model()` — 실제 아키텍처
- `build_dataloaders()` — HF Datasets 또는 자체 Dataset

## 하이퍼파라
`.env` 의 `BATCH_SIZE`, `EPOCHS`, `LR`, `MIXED_PRECISION` 로 오버라이드.
