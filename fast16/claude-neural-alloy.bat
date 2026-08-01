@echo off
chcp 65001 >nul
setlocal

set "ROOT=%~dp0.."
pushd "%ROOT%"

python fast16\start_neural_alloy_claude.py
if errorlevel 1 goto :failed
popd

set "ANTHROPIC_BASE_URL=http://127.0.0.1:8094"
set "ANTHROPIC_AUTH_TOKEN="
set "ANTHROPIC_API_KEY=local-colorlm"
set "ANTHROPIC_MODEL=ColorLM-v6-Q3-Fused"
set "ANTHROPIC_SMALL_FAST_MODEL=ColorLM-v6-Q3-Fused"
set "ANTHROPIC_DEFAULT_OPUS_MODEL=ColorLM-v6-Q3-Fused"
set "ANTHROPIC_DEFAULT_SONNET_MODEL=ColorLM-v6-Q3-Fused"
set "ANTHROPIC_DEFAULT_HAIKU_MODEL=ColorLM-v6-Q3-Fused"
set "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1"
set "DISABLE_TELEMETRY=1"
set "DISABLE_ERROR_REPORTING=1"

echo Claude Code ^| Local Neural Alloy ^| Vulkan GPU
claude --bare --setting-sources "" --settings "%ROOT%\fast16\claude-local-settings.json" %*
set "RESULT=%ERRORLEVEL%"
exit /b %RESULT%

:failed
echo Neural Alloy 服务启动失败。
popd
exit /b 1
