# ai/llm/ — LLM 파인튜닝 & 추론

## 폴더
- `models/` — 어댑터 저장소, 프롬프트 템플릿
- `training/` — SFT / LoRA / RLHF 스켈레톤
- `inference/` — 서빙 (vLLM / TGI / HF pipeline / OpenAI-호환 API)
- `configs/` — 실험별 yaml

## 4060 8GB 로컬 파인튜닝 조합
- **Phi-3.5-mini (3.8B)** — full precision LoRA 가능
- **Mistral 7B / Llama 3.1 8B** — 4-bit QLoRA (`bnb_4bit_quant_type="nf4"`)
- 시퀀스 길이 1024~2048, grad-accum 8~16

## 라이브러리 조합
- SFT: `transformers` + `trl.SFTTrainer`
- LoRA: `peft.LoraConfig`
- 데이터: `datasets` + `apply_chat_template`
- 추론(로컬): `transformers.pipeline` 또는 `vllm` (클라우드에서만)

## 클라우드로 옮길 때
`bnb_config` 제거 → fp16/bf16 full precision, batch size 대폭 상향. `.env` 만 바꾸면 코드 동일.
