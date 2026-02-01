# DomainBench Installation Script for Windows PowerShell
# This script installs domainbench globally using pipx (recommended)
# or falls back to pip --user installation
#
# Usage: .\install.ps1
# Or: powershell -ExecutionPolicy Bypass -File install.ps1

$ErrorActionPreference = "Stop"

# Colors
function Write-Cyan { param($Text) Write-Host $Text -ForegroundColor Cyan }
function Write-Green { param($Text) Write-Host $Text -ForegroundColor Green }
function Write-Yellow { param($Text) Write-Host $Text -ForegroundColor Yellow }
function Write-Red { param($Text) Write-Host $Text -ForegroundColor Red }

# Banner
Write-Cyan @"

    ,---,                        ____                                      ,---,.                                  ,---,
  .'  .' ``\                    ,'  , ``.             ,--,                 ,'  .'  \                               ,--.' |
,---.'     \    ,---.       ,-+-,.' _ |           ,--.'|         ,---, ,---.' .' |               ,---,           |  |  :
|   |  .``\  |  '   ,'\   ,-+-. ;   , ||           |  |,      ,-+-. /  ||   |  |: |           ,-+-. /  |          :  :  :
:   : |  '  | /   /   | ,--.'|'   |  || ,--.--.   ``--'_     ,--.'|'   |:   :  :  /   ,---.  ,--.'|'   |   ,---.  :  |  |,--.
|   ' '  ;  :.   ; ,. :|   |  ,', |  |,/       \  ,' ,'|   |   |  ,"' |:   |    ;   /     \|   |  ,"' |  /     \ |  :  '   |
'   | ;  .  |'   | |: :|   | /  | |--'.--.  .-. | '  | |   |   | /  | ||   :     \ /    /  |   | /  | | /    / ' |  |   /' :
|   | :  |  ''   | .; :|   : |  | ,    \__\/: . . |  | :   |   | |  | ||   |   . |.    ' / |   | |  | |.    ' /  '  :  | | |
'   : | /  ; |   :    ||   : |  |/     ," .--.; | '  : |__ |   | |  |/ '   :  '; |'   ;   /|   | |  |/ '   ; :__ |  |  ' | :
|   | '`` ,/   \   \  / |   | |``-'     /  /  ,.  | |  | '.'||   | |--'  |   |  | ; '   |  / |   | |--'  '   | '.'||  :  :_:,'
;   :  .'      ``----'  |   ;/        ;  :   .'   \;  :    ;|   |/      |   :   /  |   :    |   |/      |   :    :|  | ,'
|   ,.'                '---'         |  ,     .-./|  ,   / '---'       |   | ,'    \   \  /'---'        \   \  / ``--''
'---'                                 ``--``---'     ---``-'              ``----'       ``----'               ``----'

"@

Write-Green "DomainBench Installation Script"
Write-Host "=================================="
Write-Host ""

# Get script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Check for Python
$PythonCmd = $null
try {
    $null = Get-Command python -ErrorAction Stop
    $PythonCmd = "python"
} catch {
    try {
        $null = Get-Command python3 -ErrorAction Stop
        $PythonCmd = "python3"
    } catch {
        Write-Red "Error: Python is not installed. Please install Python 3.8+ first."
        exit 1
    }
}

$PythonVersion = & $PythonCmd --version 2>&1
Write-Host "Python version: " -NoNewline
Write-Cyan $PythonVersion
Write-Host ""

function Install-WithPipx {
    Write-Host ""
    Write-Green "Installing with pipx (recommended)..."

    # Check if pipx is installed
    $pipxInstalled = $false
    try {
        $null = Get-Command pipx -ErrorAction Stop
        $pipxInstalled = $true
    } catch {}

    if (-not $pipxInstalled) {
        Write-Yellow "pipx not found. Installing pipx first..."
        & $PythonCmd -m pip install --user pipx
        & $PythonCmd -m pipx ensurepath

        # Refresh PATH
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
    }

    # Install domainbench with pipx
    Write-Host "Installing domainbench from " -NoNewline
    Write-Cyan $ScriptDir
    Write-Host "..."

    & pipx install $ScriptDir --force

    Write-Host ""
    Write-Green "Installation complete!"
    Write-Host "Run " -NoNewline
    Write-Cyan "domainbench --help" -NoNewline
    Write-Host " to get started."
}

function Install-WithPipUser {
    Write-Host ""
    Write-Green "Installing with pip --user..."

    & $PythonCmd -m pip install --user $ScriptDir

    # Get user scripts path
    $UserScripts = & $PythonCmd -c "import site; print(site.getusersitepackages().replace('site-packages', 'Scripts'))"

    # Check if user scripts is in PATH
    $CurrentPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if ($CurrentPath -notlike "*$UserScripts*") {
        Write-Host ""
        Write-Yellow "Adding $UserScripts to PATH..."

        $NewPath = $CurrentPath + ";" + $UserScripts
        [Environment]::SetEnvironmentVariable("Path", $NewPath, "User")

        Write-Yellow "PATH updated. Please restart your terminal for changes to take effect."
    }

    Write-Host ""
    Write-Green "Installation complete!"
    Write-Host "Run " -NoNewline
    Write-Cyan "domainbench --help" -NoNewline
    Write-Host " to get started."
}

# Main installation logic
Write-Host ""
Write-Host "Select installation method:"
Write-Host "  1) pipx (recommended - isolated environment, auto PATH)"
Write-Host "  2) pip --user (installs to user site-packages)"
Write-Host ""
$choice = Read-Host "Enter choice [1]"
if ([string]::IsNullOrWhiteSpace($choice)) { $choice = "1" }

switch ($choice) {
    "1" { Install-WithPipx }
    "2" { Install-WithPipUser }
    default {
        Write-Red "Invalid choice. Defaulting to pipx."
        Install-WithPipx
    }
}

Write-Host ""
Write-Green "Don't forget to set your API keys:"
Write-Host '  $env:OPENAI_API_KEY = "your_key"'
Write-Host '  $env:GEMINI_API_KEY = "your_key"'
Write-Host '  $env:ANTHROPIC_API_KEY = "your_key"'
Write-Host ""
Write-Host "Or add them permanently to your environment variables."
Write-Host ""
