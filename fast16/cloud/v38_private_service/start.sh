#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

mkdir -p -- "${LOG_DIR}" "${RUN_DIR}" "${DATA_DIR}"
source_cann
require_file "${LLAMA_SERVER}"
require_file "${OPEN_WEBUI_ENV}/bin/open-webui"
verify_exact_file "${MODEL_PATH}" "${MODEL_BYTES}" "${MODEL_SHA256}"
verify_exact_file "${POLICY_DIR}/policy.json" 1493 "${POLICY_MANIFEST_SHA256}"
verify_exact_file "${POLICY_DIR}/weights.bin" 131264 "${POLICY_WEIGHTS_SHA256}"

CTX_SIZE="${V38_CTX_SIZE:-200000}"
if [[ ! "$CTX_SIZE" =~ ^[0-9]+$ ]] || (( CTX_SIZE < 16384 || CTX_SIZE > 262144 )); then
    printf 'V38_CTX_SIZE 必须是 16384..262144 之间的整数。\n' >&2
    exit 1
fi

if [[ -z "${LLAMA_API_KEY:-}" ]]; then
    read -r -s -p '请输入私人 API 密钥（至少 24 字符，不回显）：' LLAMA_API_KEY
    printf '\n'
fi
if (( ${#LLAMA_API_KEY} < 24 )); then
    printf 'API 密钥至少需要 24 个字符。\n' >&2
    exit 1
fi

if pid_is_running "${RUN_DIR}/llama-server.pid" || pid_is_running "${RUN_DIR}/open-webui.pid"; then
    printf '服务已有进程在运行；请先执行 stop.sh。\n' >&2
    exit 1
fi

export COLORLM_SEQUENCE_POLICY_PACKAGE="${POLICY_DIR}"
export COLORLM_SEQUENCE_POLICY_MANIFEST_SHA256="${POLICY_MANIFEST_SHA256}"
export COLORLM_SEQUENCE_POLICY_VERIFY_WEIGHTS=1
export LLAMA_API_KEY

nohup "${LLAMA_SERVER}" \
    --model "${MODEL_PATH}" \
    --alias "${MODEL_ALIAS}" \
    --ctx-size "${CTX_SIZE}" \
    --parallel 1 \
    --batch-size 512 \
    --ubatch-size 512 \
    --threads 16 \
    --n-gpu-layers 99 \
    --fit off \
    --jinja \
    --no-webui \
    --host 127.0.0.1 \
    --port "${LLAMA_PORT}" \
    >"${LOG_DIR}/llama-server.log" 2>&1 &
llama_pid=$!
printf '%s\n' "$llama_pid" >"${RUN_DIR}/llama-server.pid"

ready=0
for _ in $(seq 1 450); do
    if curl --silent --fail "http://127.0.0.1:${LLAMA_PORT}/health" >/dev/null 2>&1; then
        ready=1
        break
    fi
    if ! kill -0 "$llama_pid" 2>/dev/null; then
        break
    fi
    sleep 2
done
if [[ "$ready" != 1 ]]; then
    printf 'llama-server 未就绪。日志末尾：\n' >&2
    tail -n 80 "${LOG_DIR}/llama-server.log" >&2 || true
    exit 1
fi

webui_secret_key="$(${OPEN_WEBUI_ENV}/bin/python -c 'import secrets; print(secrets.token_hex(32))')"
if [[ -f "${DATA_DIR}/webui.db" ]]; then
    enable_signup=false
else
    enable_signup=true
fi

export DATA_DIR
export WEBUI_AUTH=true
export ENABLE_SIGNUP="$enable_signup"
export WEBUI_SECRET_KEY="$webui_secret_key"
export OPENAI_API_BASE_URL="http://127.0.0.1:${LLAMA_PORT}/v1"
export OPENAI_API_BASE_URLS="http://127.0.0.1:${LLAMA_PORT}/v1"
export OPENAI_API_KEY="${LLAMA_API_KEY}"
export OPENAI_API_KEYS="${LLAMA_API_KEY}"
export ENABLE_OLLAMA_API=false
export OFFLINE_MODE=true
export HF_HUB_OFFLINE=1

nohup "${OPEN_WEBUI_ENV}/bin/open-webui" serve \
    --host 127.0.0.1 \
    --port "${WEBUI_PORT}" \
    >"${LOG_DIR}/open-webui.log" 2>&1 &
webui_pid=$!
printf '%s\n' "$webui_pid" >"${RUN_DIR}/open-webui.pid"

webui_ready=0
for _ in $(seq 1 180); do
    if curl --silent --fail "http://127.0.0.1:${WEBUI_PORT}/health" >/dev/null 2>&1 || \
       curl --silent --fail "http://127.0.0.1:${WEBUI_PORT}/" >/dev/null 2>&1; then
        webui_ready=1
        break
    fi
    if ! kill -0 "$webui_pid" 2>/dev/null; then
        break
    fi
    sleep 2
done
if [[ "$webui_ready" != 1 ]]; then
    printf 'Open WebUI 未就绪。日志末尾：\n' >&2
    tail -n 80 "${LOG_DIR}/open-webui.log" >&2 || true
    exit 1
fi

printf 'v38 与 Open WebUI 已启动，均只监听 127.0.0.1；上下文上限为 %s。\n' "${CTX_SIZE}"
printf '首次打开 Open WebUI 时创建唯一管理员账号；数据库生成后重启会自动关闭注册。\n'
printf '请执行 health-smoke.sh 完成唯一一次最短中文 smoke。\n'
