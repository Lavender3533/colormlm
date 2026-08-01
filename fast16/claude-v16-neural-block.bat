@echo off
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0claude-v16-neural-block.ps1" %*
exit /b %ERRORLEVEL%
