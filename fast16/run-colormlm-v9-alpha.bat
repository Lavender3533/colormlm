@echo off
chcp 65001 >nul
setlocal

set "ROOT=%~dp0.."
pushd "%ROOT%"
python fast16\start_fast16_runtime.py --port 8097 --ctx-size 32768 --neural-bus --runtime-alias ColorLM-v9-alpha %*
set "RESULT=%ERRORLEVEL%"
if "%RESULT%"=="0" echo ColorLM v9-alpha: http://127.0.0.1:8097/v1

popd
endlocal & exit /b %RESULT%
