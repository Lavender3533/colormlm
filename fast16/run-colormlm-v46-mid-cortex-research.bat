@echo off
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run-colormlm-v46-mid-cortex-research.ps1" %*
exit /b %ERRORLEVEL%
