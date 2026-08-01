@echo off
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run-polaris-v0.1-preview.ps1" %*
exit /b %ERRORLEVEL%
