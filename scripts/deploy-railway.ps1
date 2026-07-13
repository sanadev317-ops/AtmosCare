# Deploy AtmosCare backend to Railway with MongoDB Atlas
# Prerequisites:
#   1. railway login   (or complete browser activation)
#   2. MongoDB Atlas cluster + connection string
# Usage:
#   .\scripts\deploy-railway.ps1 -DatabaseUri "mongodb+srv://..." -WaqiApiKey "your_key"

param(
    [Parameter(Mandatory = $true)]
    [string]$DatabaseUri,
    [string]$DatabaseName = "AtmosCareDB",
    [string]$WaqiApiKey = "",
    [string]$AccuweatherApiKey = "",
    [string]$GoogleApiKey = "",
    [string]$ProjectName = "atmoscare"
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

Write-Host "Checking Railway CLI..."
railway whoami | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Run: railway login"
    exit 1
}

Write-Host "Linking Railway project (create if new)..."
if (-not (Test-Path ".railway")) {
    railway init --name $ProjectName
}

Write-Host "Setting environment variables..."
railway variables set "DATABASE_URI=$DatabaseUri"
railway variables set "DATABASE_NAME=$DatabaseName"
if ($WaqiApiKey) { railway variables set "WAQI_API_KEY=$WaqiApiKey" }
if ($AccuweatherApiKey) { railway variables set "ACCUWEATHER_API_KEY=$AccuweatherApiKey" }
if ($GoogleApiKey) { railway variables set "GOOGLE_API_KEY=$GoogleApiKey" }

Write-Host "Deploying from GitHub-connected service or local upload..."
railway up --detach

Write-Host "Done. Open dashboard for public URL:"
railway open
