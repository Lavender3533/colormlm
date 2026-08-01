@echo off
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run-colormlm-v17-coder-island.ps1" %*
set "RESULT=%ERRORLEVEL%"
if "%RESULT%"=="0" echo ColorLM v17 Coder Neural Island: http://127.0.0.1:8105/v1
exit /b %RESULT%
