@echo off
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run-colormlm-v38-qwen36-sequence-policy.ps1" %*
exit /b %ERRORLEVEL%
