param(
    [switch]$SkipVision,
    [switch]$SkipHermes
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "=== AegisFlow setup ===" -ForegroundColor Cyan

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Creating Python 3.11 virtual environment..."
    py -3.11 -m venv .venv
}

$Python = Join-Path $Root ".venv\Scripts\python.exe"

& $Python -m pip install --upgrade pip
& $Python -m pip install -r requirements.txt

if (-not $SkipVision) {
    Write-Host "Installing local OpenCLIP vision dependencies..."
    & $Python -m pip install -r requirements-vision.txt
}

if (-not $SkipHermes) {
    Write-Host "Installing Hermes MCP integration dependencies..."
    & $Python -m pip install -r requirements-hermes.txt
}

Write-Host "Installing frontend dependencies..."
Push-Location frontend
npm install
Pop-Location

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Run: .\scripts\start_windows.ps1"
Write-Host "Hermes guide: docs\HERMES_INTEGRATION.md"
