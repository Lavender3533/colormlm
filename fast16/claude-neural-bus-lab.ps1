$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::InputEncoding = $utf8
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

$root = Split-Path -Parent $PSScriptRoot
$labHome = Join-Path $root 'fast16\runtime\claude-neural-bus-lab'
$settings = Join-Path $root 'fast16\claude-neural-bus-lab-settings.json'

Push-Location $root
try {
    $env:PYTHONUTF8 = '1'
    $env:PYTHONIOENCODING = 'utf-8'
    & python fast16\start_fast16_runtime.py --port 8102 --ctx-size 32768 --k3-plan fast16\models\ColorLM-v13-Causal-Sparse-L12.k3plan.json --runtime-alias ColorLM-v13-Causal-Sparse-L12
    if ($LASTEXITCODE -ne 0) {
        throw 'ColorLM v13 failed to start.'
    }
}
finally {
    Pop-Location
}

New-Item -ItemType Directory -Force -Path $labHome | Out-Null

$env:CLAUDE_CONFIG_DIR = Join-Path $labHome 'config'
$env:ANTHROPIC_BASE_URL = 'http://127.0.0.1:8102'
$env:ANTHROPIC_AUTH_TOKEN = ''
$env:ANTHROPIC_API_KEY = 'sk-ant-local-neural-bus-lab'
$env:ANTHROPIC_MODEL = 'ColorLM-v13-Causal-Sparse-L12'
$env:ANTHROPIC_SMALL_FAST_MODEL = 'ColorLM-v13-Causal-Sparse-L12'
$env:ANTHROPIC_DEFAULT_OPUS_MODEL = 'ColorLM-v13-Causal-Sparse-L12'
$env:ANTHROPIC_DEFAULT_SONNET_MODEL = 'ColorLM-v13-Causal-Sparse-L12'
$env:ANTHROPIC_DEFAULT_HAIKU_MODEL = 'ColorLM-v13-Causal-Sparse-L12'
$env:CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC = '1'
$env:DISABLE_TELEMETRY = '1'
$env:DISABLE_ERROR_REPORTING = '1'

# Do not inherit provider switches that could make Claude Code read cloud credentials.
'CLAUDE_CODE_USE_BEDROCK', 'CLAUDE_CODE_USE_VERTEX', 'CLAUDE_CODE_USE_FOUNDRY' | ForEach-Object {
    Remove-Item "Env:$_" -ErrorAction SilentlyContinue
}

Write-Host 'Claude Code | ColorLM v13 Lab | isolated config | 127.0.0.1:8102'
$claudeArgs = @(
    '--bare',
    '--disable-slash-commands',
    '--setting-sources', '',
    '--settings', $settings,
    '--name', 'ColorLM v13 Lab'
) + $args
& claude @claudeArgs
exit $LASTEXITCODE
