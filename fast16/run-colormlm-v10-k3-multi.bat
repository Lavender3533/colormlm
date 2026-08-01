@echo off
chcp 65001 >nul
setlocal

set "ROOT=%~dp0.."
pushd "%ROOT%"
python fast16\start_fast16_runtime.py --port 8099 --ctx-size 32768 --k3-plan fast16\models\ColorLM-v10-K3-Multi.k3plan.json --runtime-alias ColorLM-v10-K3-Multi %*
set "RESULT=%ERRORLEVEL%"
if "%RESULT%"=="0" echo ColorLM v10 K3 Multi: http://127.0.0.1:8099/v1

popd
endlocal & exit /b %RESULT%
