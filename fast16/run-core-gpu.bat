@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0\.."
title ColorLM ZeroTrain v3 - Vulkan GPU

python fast16\start_core.py
if errorlevel 1 goto :failed

python fast16\chat_core.py
goto :done

:failed
echo.
echo ColorLM core failed to start. See fast16\runtime\core-v3.stderr.log

:done
echo.
pause
