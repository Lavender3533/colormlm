@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0\.."
title ColorLM ZeroTrain v2 - DirectML GPU

python -m fast16.clm chat ^
  fast16/models/colormlm-zerotrain-v2.clm ^
  --gpu-graph fast16/models/colormlm-zerotrain-v2.onnx

echo.
pause
