#!/usr/bin/env bash
set -euo pipefail

umask 077

SERVICE_ROOT="${V38_SERVICE_ROOT:-/opt/atomgit/v38-private-service}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_NAME="ColorLM-v36-Qwen36-Global-Shared-Backbone.gguf"
MODEL_PATH="${V38_MODEL_PATH:-${SERVICE_ROOT}/models/${MODEL_NAME}}"
MODEL_BYTES=13613857952
MODEL_SHA256="e8e9e22f3de6844adc6ab0d1ff3b29c3721e17749741b6d6595ec2fc1fe56858"
POLICY_DIR="${SERVICE_ROOT}/runtime-v1"
POLICY_MANIFEST_SHA256="c647b614aaf9d28e2fb451df0c3e70b79cd3ae387c9407316307f9f88bbc914f"
POLICY_WEIGHTS_SHA256="5e649b4fbbaed446906f790c2074a9cedb953f72e6cb739b78a26806521654cc"
MODEL_ALIAS="ColorLM-v38-Qwen36-Shared-Sequence-Policy"
LLAMA_PORT=8138
WEBUI_PORT=3000
LLAMA_DIR="${SERVICE_ROOT}/llama.cpp"
LLAMA_BUILD_DIR="${LLAMA_DIR}/build-cann"
LLAMA_SERVER="${LLAMA_BUILD_DIR}/bin/llama-server"
OPEN_WEBUI_ENV="${SERVICE_ROOT}/venv-open-webui"
DATA_DIR="${SERVICE_ROOT}/data/open-webui"
LOG_DIR="${SERVICE_ROOT}/logs"
RUN_DIR="${SERVICE_ROOT}/run"

sha256_file() {
    sha256sum -- "$1" | awk '{print tolower($1)}'
}

require_file() {
    if [[ ! -f "$1" ]]; then
        printf '缺少文件：%s\n' "$1" >&2
        exit 1
    fi
}

verify_exact_file() {
    local path="$1"
    local expected_bytes="$2"
    local expected_sha="$3"
    require_file "$path"
    local actual_bytes
    actual_bytes="$(stat -c '%s' -- "$path")"
    if [[ "$actual_bytes" != "$expected_bytes" ]]; then
        printf '文件大小不符：%s（实际 %s，预期 %s）\n' "$path" "$actual_bytes" "$expected_bytes" >&2
        exit 1
    fi
    local actual_sha
    actual_sha="$(sha256_file "$path")"
    if [[ "$actual_sha" != "$expected_sha" ]]; then
        printf 'SHA-256 不符：%s\n' "$path" >&2
        exit 1
    fi
}

source_cann() {
    local candidate
    for candidate in \
        /usr/local/Ascend/ascend-toolkit/set_env.sh \
        /usr/local/Ascend/ascend-toolkit/latest/set_env.sh \
        /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash; do
        if [[ -f "$candidate" ]]; then
            # shellcheck disable=SC1090
            source "$candidate"
            return 0
        fi
    done
    printf '未找到 CANN set_env.sh；请确认 Ascend Toolkit 环境。\n' >&2
    return 1
}

pid_is_running() {
    local pid_file="$1"
    [[ -f "$pid_file" ]] || return 1
    local pid
    pid="$(<"$pid_file")"
    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    kill -0 "$pid" 2>/dev/null
}

