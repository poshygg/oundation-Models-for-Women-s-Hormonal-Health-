# scripts/launch_wandb.ps1
# WandB 로그인 안내. API 키는 https://wandb.ai/authorize 에서 발급.

.\venv\Scripts\Activate.ps1

Write-Host "WandB API 키를 https://wandb.ai/authorize 에서 복사한 뒤 입력하세요." -ForegroundColor Cyan
wandb login

Write-Host ""
Write-Host "Project: hacknation-2026" -ForegroundColor Green
Write-Host "Runs: https://wandb.ai/<your-entity>/hacknation-2026"
