# AtmosCare portable setup for any Windows PC
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

Write-Host "=== AtmosCare portable setup ===" -ForegroundColor Cyan

if (-not (Test-Path ".env")) {
    Copy-Item "portable.env" ".env"
    Write-Host "Created .env from portable.env (MongoDB Atlas)" -ForegroundColor Green
} else {
    Write-Host ".env already exists — keeping it" -ForegroundColor Yellow
}

Copy-Item "portable.env" "Frontend\.env" -Force
Write-Host "Synced Frontend/.env" -ForegroundColor Green

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv .venv
}

& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements-backend.txt
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host ""
Write-Host "Setup complete. Run: .\run_atmoscare.ps1" -ForegroundColor Green
Write-Host "MongoDB Atlas is already configured — no local MongoDB needed." -ForegroundColor Green
