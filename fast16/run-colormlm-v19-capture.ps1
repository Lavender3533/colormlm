param(
    [Parameter(Mandatory = $true)]
    [string] $OutputPath,
    [int] $Port = 8106,
    [int] $MaxRecords = 256
)

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
        --server llama.cpp\build-v19-dual-head\bin\Release\llama-server.exe `
        --port $Port `
        --ctx-size 16384 `
        --runtime-alias ColorLM-v19-CNOB-Capture `
        --neural-island-manifest fast16\research\v17_coder_island\runtime-v3\island.json `
        --neural-island-alpha 0.02 `
        --neural-island-expert-cache-slots 32 `
        --neural-output-head-package fast16\research\v19_dual_head\runtime-head-v2 `
        --neural-output-head-alpha 0.03 `
        --neural-output-capture $OutputPath `
        --neural-output-capture-max-records $MaxRecords `
        --no-warmup `
        --anthropic-max-tokens 64 `
        @args
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
