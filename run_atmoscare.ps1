# Start backend + frontend (portable — works from any PC with internet)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

if (-not (Test-Path ".env")) {
    Copy-Item "portable.env" ".env"
}
Copy-Item "portable.env" "Frontend\.env" -Force

$py = if (Test-Path ".\.venv\Scripts\python.exe") {
    ".\.venv\Scripts\python.exe"
} else {
    "python"
}

Write-Host "Using Python: $py" -ForegroundColor Cyan
Write-Host "Backend listens on ALL interfaces: 0.0.0.0:8000" -ForegroundColor Cyan
Write-Host "Database: MongoDB Atlas" -ForegroundColor Cyan

$env:PYTHONPATH = $Root
$backend = Start-Process -FilePath $py -ArgumentList "-m", "uvicorn", "Backend.main:app", "--host", "0.0.0.0", "--port", "8000" -WorkingDirectory $Root -PassThru -WindowStyle Minimized
Write-Host "Backend PID $($backend.Id) starting (LAN discovery UDP 3847)..."

Start-Sleep -Seconds 6

try {
    $health = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -TimeoutSec 15
    Write-Host "Backend health: $($health.status) / db=$($health.database)" -ForegroundColor Green
    if ($health.urls) {
        Write-Host "Phone/APK can auto-detect these URLs:" -ForegroundColor Green
        $health.urls | ForEach-Object { Write-Host "  $_" }
    }
} catch {
    Write-Host "Backend still starting (continuing)..." -ForegroundColor Yellow
}

Set-Location "$Root\Frontend"
& $py main.py

Write-Host "Closing backend..."
Stop-Process -Id $backend.Id -Force -ErrorAction SilentlyContinue
