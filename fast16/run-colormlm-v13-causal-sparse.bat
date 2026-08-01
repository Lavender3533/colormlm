@echo off
chcp 65001 >nul
setlocal
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

set "ROOT=%~dp0.."
pushd "%ROOT%"
python fast16\start_fast16_runtime.py --port 8102 --runtime-alias ColorLM-v13-Causal-Sparse-L12 --ctx-size 32768 --k3-plan fast16\models\ColorLM-v13-Causal-Sparse-L12.k3plan.json %*
set "RESULT=%ERRORLEVEL%"
if "%RESULT%"=="0" echo ColorLM v13 Causal Sparse: http://127.0.0.1:8102/v1

popd
endlocal & exit /b %RESULT%
