param(
    [int] $Port = 8107,
    [int] $ContextSize = 16384,
    [int] $AnthropicMaxTokens = 2048,
    [double] $K3Alpha = 0.04,
    [string] $RuntimeAlias = 'ColorLM-v20-K3-Coder-Shared-Trunk',
    [string] $K3Plan = 'fast16\models\ColorLM-v13-Causal-Sparse-L12.k3plan.json'
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
        --server llama.cpp\build-v17-perf\bin\Release\llama-server.exe `
        --port $Port `
        --ctx-size $ContextSize `
        --runtime-alias $RuntimeAlias `
        --k3-plan $K3Plan `
        --k3-alpha $K3Alpha `
        --neural-island-manifest fast16\research\v17_coder_island\runtime-v3\island.json `
        --neural-island-alpha 0.02 `
        --neural-island-expert-cache-slots 32 `
        --anthropic-max-tokens $AnthropicMaxTokens `
        @args
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
