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
        --server llama.cpp\build-v16-vulkan\bin\Release\llama-server.exe `
        --port 8104 `
        --ctx-size 32768 `
        --runtime-alias ColorLM-v16-Coder-Neural-Block `
        --k3-plan fast16\models\ColorLM-v13-Causal-Sparse-L12.k3plan.json `
        --neural-block-package fast16\research\neural_blocks\qwen3_coder_next_l47\q4_0 `
        --neural-block-alpha 0.04 `
        --anthropic-max-tokens 1024 `
        @args
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
