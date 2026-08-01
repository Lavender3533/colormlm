@echo off
chcp 65001 >nul
where pwsh.exe >nul 2>&1
if errorlevel 1 goto windows_powershell

pwsh.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0run-polaris-v0.1-preview.ps1" %*
exit /b %ERRORLEVEL%

:windows_powershell
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0run-polaris-v0.1-preview.ps1" %*
exit /b %ERRORLEVEL%
