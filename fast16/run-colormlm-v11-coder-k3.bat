@echo off
chcp 65001 >nul
setlocal

set "ROOT=%~dp0.."
pushd "%ROOT%"
python fast16\start_fast16_runtime.py --port 8100 --ctx-size 32768 --neural-bus --neural-bus-alpha 0.08 --k3-plan fast16\models\ColorLM-v10-K3-Multi.k3plan.json --runtime-alias ColorLM-v11-Coder-K3 %*
set "RESULT=%ERRORLEVEL%"
if "%RESULT%"=="0" echo ColorLM v11 Coder+K3: http://127.0.0.1:8100/v1

popd
endlocal & exit /b %RESULT%
