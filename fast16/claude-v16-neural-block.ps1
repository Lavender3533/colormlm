$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::InputEncoding = $utf8
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

$root = Split-Path -Parent $PSScriptRoot
$labHome = Join-Path $root 'fast16\runtime\claude-v16-neural-block'
$settings = Join-Path $root 'fast16\claude-v16-neural-block-settings.json'

Push-Location $root
try {
    & powershell -NoProfile -ExecutionPolicy Bypass -File `
        fast16\run-colormlm-v16-neural-block.ps1
    if ($LASTEXITCODE -ne 0) {
        throw 'ColorLM v16 failed to start.'
    }
}
finally {
    Pop-Location
}

New-Item -ItemType Directory -Force -Path $labHome | Out-Null
$env:CLAUDE_CONFIG_DIR = Join-Path $labHome 'config'
$env:ANTHROPIC_BASE_URL = 'http://127.0.0.1:8104'
$env:ANTHROPIC_AUTH_TOKEN = ''
$env:ANTHROPIC_API_KEY = 'sk-ant-local-v16-neural-block'
$env:ANTHROPIC_MODEL = 'ColorLM-v16-Coder-Neural-Block'
$env:ANTHROPIC_SMALL_FAST_MODEL = 'ColorLM-v16-Coder-Neural-Block'
$env:ANTHROPIC_DEFAULT_OPUS_MODEL = 'ColorLM-v16-Coder-Neural-Block'
$env:ANTHROPIC_DEFAULT_SONNET_MODEL = 'ColorLM-v16-Coder-Neural-Block'
$env:ANTHROPIC_DEFAULT_HAIKU_MODEL = 'ColorLM-v16-Coder-Neural-Block'
$env:CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC = '1'
$env:DISABLE_TELEMETRY = '1'
$env:DISABLE_ERROR_REPORTING = '1'

'CLAUDE_CODE_USE_BEDROCK', 'CLAUDE_CODE_USE_VERTEX', 'CLAUDE_CODE_USE_FOUNDRY' | ForEach-Object {
    Remove-Item "Env:$_" -ErrorAction SilentlyContinue
}

Write-Host 'Claude Code | ColorLM v16 | isolated config | 127.0.0.1:8104'
$compactSystemPrompt = @'
You are ColorLM v16, a local coding agent working in the current workspace.
Reply in the user's language. Be concise and act directly on coding requests.
Inspect relevant files before editing. Use tools only when needed.
For every tool call, provide all required parameters exactly as defined.
After a tool result, either make the next necessary tool call or give the final answer and stop.
'@
$claudeArgs = @(
    '--bare',
    '--disable-slash-commands',
    '--model', 'ColorLM-v16-Coder-Neural-Block',
    '--setting-sources', 'user',
    '--settings', $settings,
    '--system-prompt', $compactSystemPrompt,
    '--tools', 'Read,Edit,Write,Glob,Grep,Bash',
    '--name', 'ColorLM v16 Neural Block'
) + $args
& claude @claudeArgs
exit $LASTEXITCODE
