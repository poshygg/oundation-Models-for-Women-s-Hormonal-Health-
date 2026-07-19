#!/usr/bin/env bash
# scripts/setup_env.sh
# Linux / macOS / WSL / 클라우드 GPU 셋업
# 사용: 프로젝트 루트에서 `bash scripts/setup_env.sh`

set -euo pipefail

echo "=== 1/6 Python 버전 ==="
python3 --version

echo "=== 2/6 venv ==="
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

echo "=== 3/6 base + tools ==="
pip install -r requirements/base.txt
pip install -r requirements/dev.txt

echo "=== 4/6 PyTorch (CUDA 12.1) ==="
pip install -r requirements/pytorch.txt \
    --index-url https://download.pytorch.org/whl/cu121 \
    --extra-index-url https://pypi.org/simple

echo "=== 5/6 TensorFlow + AI harness + ML extras ==="
pip install -r requirements/tensorflow.txt
pip install -r requirements/ai_harness.txt
pip install -r requirements/ml_extra.txt

echo "=== 6/6 pre-commit ==="
pre-commit install

echo ""
echo "완료. 다음:"
echo "  1) wandb login"
echo "  2) huggingface-cli login"
echo "  3) cp .env.example .env  (편집)"
echo "  4) pytest -q tests/test_smoke.py"
