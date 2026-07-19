# ai/cnn/ — CNN 실험 (timm, torchvision, keras_cv)

## 폴더
- `models/` — 커스텀 아키텍처 (backbone 조합, custom head)
- `training/` — 학습 루프 (accelerate 기반 권장)
- `configs/` — 실험별 yaml (quick_start.ps1 이 복사)

## 시작점
- 베이스라인: `torchvision.models.resnet50(weights="DEFAULT")` fine-tune
- 성능/파라 균형: `timm.create_model("tf_efficientnet_b3", pretrained=True)`
- SOTA 근처: `timm.create_model("convnextv2_tiny.fcmae_ft_in22k_in1k", pretrained=True)`

## 4060 8GB 팁
- input resolution 224 유지
- `bs=32` for B0, `bs=8` for B3, grad-accum 으로 effective bs 조절
- torch.compile 켜면 15~30% 가속
