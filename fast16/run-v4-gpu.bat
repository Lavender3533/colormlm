@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0\.."
title ColorLM v4-SMoE - Vulkan GPU

python fast16\start_v4.py
if errorlevel 1 goto :failed

python fast16\chat_v4.py
goto :done

:failed
echo.
echo ColorLM v4 启动失败，请查看 fast16\runtime\v4-server.stderr.log

:done
echo.
pause
