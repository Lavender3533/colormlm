@echo off
chcp 65001 >nul
setlocal
set "ROOT=%~dp0.."
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run-colormlm-v16-neural-block.ps1" %*
set "RESULT=%ERRORLEVEL%"
if "%RESULT%"=="0" echo ColorLM v16 Neural Block: http://127.0.0.1:8104/v1
endlocal & exit /b %RESULT%
