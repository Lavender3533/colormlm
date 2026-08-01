@echo off
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run-colormlm-v36-qwen36-shared-backbone.ps1" %*
exit /b %ERRORLEVEL%
