@echo off
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run-colormlm-v33-qwen36-global-moe.ps1" %*
exit /b %ERRORLEVEL%
