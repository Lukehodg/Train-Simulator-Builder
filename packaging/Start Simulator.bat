@echo off
rem Train Link Simulator - double-click launcher (Windows). No installation required.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start-Simulator.ps1"
if errorlevel 1 pause
