# scripts/quick_start.ps1
# 새 실험 시작 헬퍼: 브랜치 생성 + config 복사 + wandb run init 스니펫 안내
# 사용: .\scripts\quick_start.ps1 -Name "cnn-baseline" -Kind "cnn"

param(
    [Parameter(Mandatory=$true)][string]$Name,
    [Parameter(Mandatory=$true)][ValidateSet("cnn","llm","vlm","audio")][string]$Kind
)

$date = Get-Date -Format "yyMMdd"
$branch = "exp/$date-$Name"

Write-Host "=== 브랜치 생성: $branch ===" -ForegroundColor Cyan
git checkout -b $branch

$srcConfig = "configs\base.yaml"
$dstDir = "ai\$Kind\configs"
$dstConfig = "$dstDir\$Name.yaml"

if (-not (Test-Path $dstDir)) { New-Item -ItemType Directory -Path $dstDir | Out-Null }
if (-not (Test-Path $dstConfig)) {
    Copy-Item $srcConfig $dstConfig
    Write-Host "config 복사: $dstConfig" -ForegroundColor Green
}

Write-Host ""
Write-Host "WandB 초기화 스니펫 (Python):" -ForegroundColor Yellow
Write-Host @"
import wandb
wandb.init(
    project="hacknation-2026",
    name="$branch",
    tags=["$Kind"],
    config={"config_file": "$dstConfig"},
)
"@

Write-Host ""
Write-Host "Before writing training code, check the time budget gate with time_probe.py (docs\time_budgeting.md)." -ForegroundColor Yellow
Write-Host "실험 종료 후 memory\experiment_log.md 에 결과 기록 잊지 말기." -ForegroundColor Yellow
