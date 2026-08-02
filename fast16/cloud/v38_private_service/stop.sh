#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

stop_one() {
    local name="$1"
    local pid_file="$2"
    if ! pid_is_running "$pid_file"; then
        rm -f -- "$pid_file"
        printf '%s 未运行。\n' "$name"
        return 0
    fi
    local pid
    pid="$(<"$pid_file")"
    kill "$pid"
    for _ in $(seq 1 30); do
        if ! kill -0 "$pid" 2>/dev/null; then
            rm -f -- "$pid_file"
            printf '%s 已停止。\n' "$name"
            return 0
        fi
        sleep 1
    done
    printf '%s 在 30 秒内未退出，请先检查进程，不自动强杀。PID=%s\n' "$name" "$pid" >&2
    return 1
}

stop_one open-webui "${RUN_DIR}/open-webui.pid"
stop_one llama-server "${RUN_DIR}/llama-server.pid"

