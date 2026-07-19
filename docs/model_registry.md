# 모델 레지스트리 — 후보 리스트

라이선스 필독. 상업적 사용 조건이 트랙별 상금 조건과 충돌할 수 있음.

## CNN (비전 인식·분류·검출)
| 모델          | HF repo / 소스                                    | 라이선스   | 용도                          |
| ------------- | ------------------------------------------------- | ---------- | ----------------------------- |
| ResNet-50     | `torchvision.models.resnet50`                     | BSD-3      | 이미지 분류 베이스라인        |
| EfficientNet  | `timm/tf_efficientnet_b0` ~ `b7`                  | Apache-2.0 | 성능/파라미터 균형            |
| ConvNeXt-V2   | `timm/convnextv2_tiny.fcmae_ft_in22k_in1k`        | MIT        | 최신 CNN, 트랜스포머급 성능   |
| ViT           | `google/vit-base-patch16-224`                     | Apache-2.0 | 트랜스포머 비전               |
| Swin-V2       | `microsoft/swinv2-tiny-patch4-window8-256`        | MIT        | 계층적 attention              |
| DINOv2        | `facebook/dinov2-base`                            | Apache-2.0 | Self-supervised feature       |

## LLM (텍스트 생성·이해)
| 모델              | HF repo                                    | 라이선스           | 용도                        |
| ----------------- | ------------------------------------------ | ------------------ | --------------------------- |
| Llama 3.1 8B      | `meta-llama/Llama-3.1-8B-Instruct`         | Llama 3.1 (gated)  | 로컬 4060 4-bit 가능        |
| Llama 3.1 70B     | `meta-llama/Llama-3.1-70B-Instruct`        | Llama 3.1 (gated)  | 클라우드 전용               |
| Mistral 7B        | `mistralai/Mistral-7B-Instruct-v0.3`       | Apache-2.0         | 상업적 사용 자유            |
| Mixtral 8x7B      | `mistralai/Mixtral-8x7B-Instruct-v0.1`     | Apache-2.0         | MoE, 클라우드 전용          |
| Qwen 2.5 7B       | `Qwen/Qwen2.5-7B-Instruct`                 | Apache-2.0         | 다국어 강함, 코드 강함      |
| Phi-3.5-mini      | `microsoft/Phi-3.5-mini-instruct`          | MIT                | 3.8B, 4060 로컬 학습 가능   |
| Gemma 2 9B        | `google/gemma-2-9b-it`                     | Gemma (gated)      | 성능 좋으나 라이선스 유의   |

## VLM (비전-언어)
| 모델         | HF repo                               | 라이선스   | 용도                         |
| ------------ | ------------------------------------- | ---------- | ---------------------------- |
| SigLIP       | `google/siglip-base-patch16-224`      | Apache-2.0 | CLIP 대체, retrieval 강함    |
| CLIP         | `openai/clip-vit-large-patch14`       | MIT        | 표준 baseline                |
| LLaVA-1.6    | `llava-hf/llava-v1.6-mistral-7b-hf`   | Apache-2.0 | 이미지 QA / caption          |
| Qwen2-VL 7B  | `Qwen/Qwen2-VL-7B-Instruct`           | Apache-2.0 | 최신 VLM, 비디오도 처리      |

## 오디오
| 모델            | HF repo                            | 라이선스   | 용도                         |
| --------------- | ---------------------------------- | ---------- | ---------------------------- |
| Whisper large-v3| `openai/whisper-large-v3`          | MIT        | 다국어 STT                   |
| Whisper small   | `openai/whisper-small`             | MIT        | 로컬 실시간 STT              |
| MMS             | `facebook/mms-1b-all`              | CC-BY-NC   | 1000+ 언어 (비상업)          |
| Bark            | `suno/bark`                        | MIT        | TTS 생성                     |

## 사용 가이드
- **4060 8GB 로컬**: Phi-3.5, Mistral 7B (4-bit), Whisper small, SigLIP, EfficientNet-B0~B3
- **클라우드 A100/H100**: Llama 3.1 70B, Mixtral, Qwen2-VL 7B (fp16), Whisper large-v3
- 게이티드 모델은 HF 토큰 필수 → `.env` 의 `HF_TOKEN` 세팅
