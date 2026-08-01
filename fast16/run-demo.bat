@echo off
chcp 65001 >nul
cd /d "%~dp0\.."
title ColorLM ZeroTrain v0 Demo

echo.
echo  ColorLM ZeroTrain v0
echo  Prompt: def max
echo.

python -m fast16.clm generate ^
  fast16/models/colormlm-zerotrain-v0.clm ^
  --prompt "def max" ^
  --new-tokens 36 ^
  --steps 8

echo.
pause
