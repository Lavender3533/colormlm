@echo off
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0claude-v29-sequence-policy.ps1" %*
exit /b %ERRORLEVEL%
