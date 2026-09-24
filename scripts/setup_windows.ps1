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
$Extras = @()
if (-not $SkipVision) { $Extras += "vision" }
if (-not $SkipHermes) { $Extras += "hermes" }

if ($Extras.Count -gt 0) {
    $ExtraSpec = ".[{0}]" -f ($Extras -join ",")
    Write-Host "Installing AegisFlow editable package with extras: $($Extras -join ', ')..."
    & $Python -m pip install -e $ExtraSpec
}
else {
    Write-Host "Installing AegisFlow editable package..."
    & $Python -m pip install -e .
}

Write-Host "Installing frontend dependencies..."
Push-Location frontend
npm install
Pop-Location

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Run: .\scripts\start_windows.ps1"
Write-Host "Hermes guide: docs\HERMES_INTEGRATION.md"
