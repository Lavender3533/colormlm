@echo off
chcp 65001 >nul
setlocal

set "ROOT=%~dp0.."
pushd "%ROOT%"
python fast16\start_fast16_runtime.py --alloy-plan fast16\models\ColorLM-v12-Neural-Alloy.alloyplan.json %*
set "RESULT=%ERRORLEVEL%"
if "%RESULT%"=="0" echo ColorLM v12 Neural Alloy: http://127.0.0.1:8101/v1

popd
endlocal & exit /b %RESULT%
