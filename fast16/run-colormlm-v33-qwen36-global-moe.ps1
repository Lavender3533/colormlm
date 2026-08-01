param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $ExtraArgs
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $root

& python 'fast16/start_fast16_runtime.py' `
    --server 'llama.cpp/build-v17-perf/bin/Release/llama-server.exe' `
    --model 'fast16/models/ColorLM-v33-Qwen36-Global-MoE-Pair.gguf' `
    --port 8133 `
    --runtime-alias 'ColorLM-v33-Qwen36-Global-MoE-Pair' `
    --ctx-size 4096 `
    --spec-type none `
    @ExtraArgs

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host 'ColorLM v33 research endpoint: http://127.0.0.1:8133/v1'
Write-Host 'v33 is a research candidate; v29 remains the production entry.'
