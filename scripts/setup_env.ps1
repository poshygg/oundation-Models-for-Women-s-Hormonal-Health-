# scripts/setup_env.ps1
# Windows PowerShell: Hack Nation MIT 로컬 환경 셋업
# 사용: 프로젝트 루트에서 `powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1`

$ErrorActionPreference = "Stop"

Write-Host "=== 1/6 Python 버전 확인 ===" -ForegroundColor Cyan
python --version

Write-Host "=== 2/6 venv 생성 ===" -ForegroundColor Cyan
if (-not (Test-Path ".\venv")) {
    python -m venv venv
}
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip wheel setuptools

Write-Host "=== 3/6 base + tools ===" -ForegroundColor Cyan
pip install -r requirements\base.txt
pip install -r requirements\dev.txt

Write-Host "=== 4/6 PyTorch (CUDA 12.1) ===" -ForegroundColor Cyan
pip install -r requirements\pytorch.txt `
    --index-url https://download.pytorch.org/whl/cu121 `
    --extra-index-url https://pypi.org/simple

Write-Host "=== 5/6 TensorFlow + AI harness + ML extras ===" -ForegroundColor Cyan
pip install -r requirements\tensorflow.txt
pip install -r requirements\ai_harness.txt
pip install -r requirements\ml_extra.txt

Write-Host "=== 6/6 pre-commit ===" -ForegroundColor Cyan
pre-commit install

Write-Host ""
Write-Host "완료. 다음 단계:" -ForegroundColor Green
Write-Host "  1) wandb login"
Write-Host "  2) huggingface-cli login"
Write-Host "  3) copy .env.example .env  (그리고 편집)"
Write-Host "  4) pytest -q tests\test_smoke.py"
