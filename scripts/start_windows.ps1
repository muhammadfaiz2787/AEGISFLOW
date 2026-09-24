param(
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Virtual environment not found. Run .\scripts\setup_windows.ps1 first."
}

$BackendCommand = "Set-Location '$Root'; & '$Python' -m uvicorn backend.main:app --host 127.0.0.1 --port 8000"
$FrontendRoot = Join-Path $Root "frontend"
$FrontendCommand = "Set-Location '$FrontendRoot'; npm run dev"

Write-Host "Starting AegisFlow backend..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", $BackendCommand

Start-Sleep -Seconds 2

Write-Host "Starting AegisFlow frontend..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", $FrontendCommand

if (-not $NoBrowser) {
    Start-Sleep -Seconds 3
    Start-Process "http://localhost:5173/"
}

Write-Host ""
Write-Host "AegisFlow launched." -ForegroundColor Green
Write-Host "Dashboard       : http://localhost:5173/"
Write-Host "Secure Transfer : http://localhost:5173/secure.html"
Write-Host "API docs        : http://127.0.0.1:8000/docs"
