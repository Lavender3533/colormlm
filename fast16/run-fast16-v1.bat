@echo off
chcp 65001 >nul
setlocal

set "ROOT=%~dp0.."
pushd "%ROOT%"
python fast16\start_fast16_runtime.py %*
set "RESULT=%ERRORLEVEL%"
if "%RESULT%"=="0" echo Fast16 Runtime v1: http://127.0.0.1:8096

popd
endlocal & exit /b %RESULT%
