@echo off
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run-colormlm-v20-k3-coder-alloy.ps1" %*
set "RESULT=%ERRORLEVEL%"
if "%RESULT%"=="0" echo ColorLM v20 K3+Coder Alloy: http://127.0.0.1:8107/v1
exit /b %RESULT%
