@echo off
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0claude-v17-coder-island.ps1" %*
exit /b %ERRORLEVEL%
