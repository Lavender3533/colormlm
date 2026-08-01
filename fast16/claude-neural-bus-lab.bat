@echo off
chcp 65001 >nul
setlocal

set "ROOT=%~dp0.."
set "LAB_HOME=%ROOT%\fast16\runtime\claude-neural-bus-lab"

pushd "%ROOT%"
python fast16\start_fast16_runtime.py --port 8102 --ctx-size 32768 --k3-plan fast16\models\ColorLM-v13-Causal-Sparse-L12.k3plan.json --runtime-alias ColorLM-v13-Causal-Sparse-L12
if errorlevel 1 goto :failed
popd

if not exist "%LAB_HOME%" mkdir "%LAB_HOME%"

set "CLAUDE_CONFIG_DIR=%LAB_HOME%\config"
set "ANTHROPIC_BASE_URL=http://127.0.0.1:8102"
set "ANTHROPIC_AUTH_TOKEN="
set "ANTHROPIC_API_KEY=sk-ant-local-neural-bus-lab"
set "ANTHROPIC_MODEL=ColorLM-v13-Causal-Sparse-L12"
set "ANTHROPIC_SMALL_FAST_MODEL=ColorLM-v13-Causal-Sparse-L12"
set "ANTHROPIC_DEFAULT_OPUS_MODEL=ColorLM-v13-Causal-Sparse-L12"
set "ANTHROPIC_DEFAULT_SONNET_MODEL=ColorLM-v13-Causal-Sparse-L12"
set "ANTHROPIC_DEFAULT_HAIKU_MODEL=ColorLM-v13-Causal-Sparse-L12"
set "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1"
set "DISABLE_TELEMETRY=1"
set "DISABLE_ERROR_REPORTING=1"

echo Claude Code ^| ColorLM v13 Lab ^| isolated config ^| 127.0.0.1:8102
claude --bare --setting-sources "" --settings "%ROOT%\fast16\claude-neural-bus-lab-settings.json" --name "ColorLM v13 Lab" %*
set "RESULT=%ERRORLEVEL%"
exit /b %RESULT%

:failed
echo ColorLM v13 failed to start.
popd
exit /b 1
