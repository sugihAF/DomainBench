@echo off
REM DomainBench Installation Script for Windows Command Prompt
REM This script runs the PowerShell installer

echo.
echo DomainBench Installation
echo ========================
echo.

REM Run PowerShell script with bypass execution policy
powershell -ExecutionPolicy Bypass -File "%~dp0install.ps1"

pause
