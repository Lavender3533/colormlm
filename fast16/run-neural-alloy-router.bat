@echo off
chcp 65001 >nul
setlocal

set "ROOT=%~dp0.."
set "MODEL=fast16\models\ColorLM-v4-SMoE.gguf"
set "ADAPTER=fast16\models\ColorLM-Neural-Alloy-Router-F16.gguf"
set "CLI=%ROOT%\build\bin\Release\llama-cli.exe"
set "ALPHA=%~1"

if "%ALPHA%"=="" set "ALPHA=1"

pushd "%ROOT%"

if not exist "%CLI%" (
    echo 找不到 llama-cli: %CLI%
    popd
    exit /b 2
)
if not exist "%ADAPTER%" (
    echo 找不到 Neural Alloy adapter: %ADAPTER%
    echo 请先运行: python fast16\research\build_neural_alloy_router.py
    popd
    exit /b 2
)

echo ColorLM Neural Alloy Router ^| Vulkan GPU ^| alpha=%ALPHA%
"%CLI%" ^
  -m "%MODEL%" ^
  --lora-scaled "%ADAPTER%:%ALPHA%" ^
  -ngl 99 ^
  --n-cpu-moe 29 ^
  -c 4096 ^
  -b 512 ^
  -ub 512 ^
  --no-mmap ^
  --flash-attn on ^
  -cnv ^
  --jinja ^
  --reasoning off ^
  --color auto ^
  --temp 0.6 ^
  --top-p 0.9 ^
  --min-p 0.05

popd
endlocal
