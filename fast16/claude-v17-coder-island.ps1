$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::InputEncoding = $utf8
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

$root = Split-Path -Parent $PSScriptRoot
$labHome = Join-Path $root 'fast16\runtime\claude-v17-coder-island'
$settings = Join-Path $root 'fast16\claude-v17-coder-island-settings.json'
$model = 'ColorLM-v17-Coder-Neural-Island'

& "$PSScriptRoot\start-colormlm-v17-user.ps1"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (-not (Get-Command claude -ErrorAction SilentlyContinue)) {
    throw '没有找到claude命令。请先安装Claude Code并确认claude可在终端运行。'
}

New-Item -ItemType Directory -Force -Path $labHome | Out-Null
$env:CLAUDE_CONFIG_DIR = Join-Path $labHome 'config'
$env:ANTHROPIC_BASE_URL = 'http://127.0.0.1:8105'
$env:ANTHROPIC_AUTH_TOKEN = ''
$env:ANTHROPIC_API_KEY = 'sk-ant-local-v17-coder-island'
$env:ANTHROPIC_MODEL = $model
$env:ANTHROPIC_SMALL_FAST_MODEL = $model
$env:ANTHROPIC_DEFAULT_OPUS_MODEL = $model
$env:ANTHROPIC_DEFAULT_SONNET_MODEL = $model
$env:ANTHROPIC_DEFAULT_HAIKU_MODEL = $model
$env:CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC = '1'
$env:DISABLE_TELEMETRY = '1'
$env:DISABLE_ERROR_REPORTING = '1'

'CLAUDE_CODE_USE_BEDROCK', 'CLAUDE_CODE_USE_VERTEX', 'CLAUDE_CODE_USE_FOUNDRY' | ForEach-Object {
    Remove-Item "Env:$_" -ErrorAction SilentlyContinue
}

$compactSystemPrompt = @'
You are ColorLM v17, a local coding agent working in the current workspace.
Reply in the user's language. Be concise and act directly on coding requests.
Inspect relevant files before editing. Use tools only when needed.
For every tool call, provide all required parameters exactly as defined.
After a tool result, either make the next necessary tool call or give the final answer and stop.
'@

Write-Host 'Claude Code | ColorLM v17 | isolated config | 127.0.0.1:8105'
$claudeArgs = @(
    '--bare',
    '--disable-slash-commands',
    '--model', $model,
    '--setting-sources', 'user',
    '--settings', $settings,
    '--system-prompt', $compactSystemPrompt,
    '--tools', 'Read,Edit,Write,Glob,Grep,Bash',
    '--name', 'ColorLM v17 Coder Island'
) + $args
& claude @claudeArgs
exit $LASTEXITCODE
