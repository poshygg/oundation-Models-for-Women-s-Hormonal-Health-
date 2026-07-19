# scripts/launch_mlflow.ps1
# 로컬 MLflow UI 실행 (http://localhost:5000)

.\venv\Scripts\Activate.ps1

$storeUri = "file:./experiments/mlruns"
Write-Host "MLflow UI on http://localhost:5000  (store: $storeUri)" -ForegroundColor Cyan
mlflow ui --backend-store-uri $storeUri --port 5000
