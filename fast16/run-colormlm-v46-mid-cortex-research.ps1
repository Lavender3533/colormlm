param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $ExtraArgs
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $root
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

& python 'fast16/start_fast16_runtime.py' `
    --server 'build/bin/Release/llama-server.exe' `
    --model 'fast16/models/ColorLM-v46-Qwen36-Mid-Cortex-L16-L31.gguf' `
    --port 8146 `
    --runtime-alias 'ColorLM-v46-Qwen36-Mid-Cortex-Research' `
    --ctx-size 16384 `
    --spec-type none `
    --sequence-policy-package 'fast16/research/v29_sequence_policy_head/runtime-v1' `
    --verify-sequence-policy-weights `
    --anthropic-max-tokens 1024 `
    @ExtraArgs

exit $LASTEXITCODE
