#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

if pid_is_running "${RUN_DIR}/llama-server.pid"; then
    printf 'llama-server：运行中，PID %s\n' "$(<"${RUN_DIR}/llama-server.pid")"
else
    printf 'llama-server：未运行\n'
fi
if pid_is_running "${RUN_DIR}/open-webui.pid"; then
    printf 'Open WebUI：运行中，PID %s\n' "$(<"${RUN_DIR}/open-webui.pid")"
else
    printf 'Open WebUI：未运行\n'
fi

printf 'llama health：'
curl --silent --show-error --fail "http://127.0.0.1:${LLAMA_PORT}/health" || true
printf '\nOpen WebUI health：'
curl --silent --show-error --fail "http://127.0.0.1:${WEBUI_PORT}/health" || true
printf '\n'

