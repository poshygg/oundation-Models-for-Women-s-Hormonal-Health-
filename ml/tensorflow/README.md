# ml/tensorflow/ — TensorFlow 트레이닝 스켈레톤

`train.py` 는 tf.keras `Model.fit` 최소 루프. TensorBoard + WandB callback.

## 실행
```powershell
python ml\tensorflow\train.py
tensorboard --logdir experiments\tf-logs
```

## 채워야 할 곳
- `build_model()` — Sequential/Functional/Subclassed 모델
- `build_dataset()` — `tf.data.Dataset` 파이프라인 (tfds 활용 권장)

## Mixed precision
`.env` 의 `MIXED_PRECISION` = `fp16` (일반) 또는 `bf16` (Ampere+).
