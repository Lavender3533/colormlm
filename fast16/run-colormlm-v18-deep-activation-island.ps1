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
        --port 8106 `
        --ctx-size 16384 `
        --runtime-alias ColorLM-v18.1-Nullspace-Anchored-Island `
        --neural-island-manifest fast16\research\v18_activation_bridge\runtime-v2\island.json `
        --neural-island-alpha 0.02 `
        --neural-island-expert-cache-slots 32 `
        --anthropic-max-tokens 1024 `
        @args
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
