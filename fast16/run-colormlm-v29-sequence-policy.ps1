$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::InputEncoding = $utf8
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    $env:PYTHONUTF8 = '1'
    $env:PYTHONIOENCODING = 'utf-8'
    & python fast16\start_fast16_runtime.py `
        --server llama.cpp\build-v17-perf\bin\Release\llama-server.exe `
        --port 8105 `
        --ctx-size 16384 `
        --runtime-alias ColorLM-v29-Sequence-Policy `
        --neural-island-manifest fast16\research\v17_coder_island\runtime-v3\island.json `
        --neural-island-alpha 0.02 `
        --neural-island-expert-cache-slots 32 `
        --spec-type none `
        --sequence-policy-package fast16\research\v29_sequence_policy_head\runtime-v1 `
        --verify-sequence-policy-weights `
        --anthropic-max-tokens 1024 `
        @args
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
