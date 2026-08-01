@echo off
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run-colormlm-v18-deep-activation-island.ps1" %*
set "RESULT=%ERRORLEVEL%"
if "%RESULT%"=="0" echo ColorLM v18.1 Nullspace Anchored Island: http://127.0.0.1:8106/v1
exit /b %RESULT%
