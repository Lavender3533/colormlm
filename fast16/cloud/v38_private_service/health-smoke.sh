#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

if [[ -z "${LLAMA_API_KEY:-}" ]]; then
    read -r -s -p '请输入启动时使用的私人 API 密钥（不回显）：' LLAMA_API_KEY
    printf '\n'
fi

auth_header="Authorization: Bearer ${LLAMA_API_KEY}"
printf '1/3 health\n'
curl --silent --show-error --fail "http://127.0.0.1:${LLAMA_PORT}/health"
printf '\n2/3 models\n'
models_json="$(curl --silent --show-error --fail \
    -H "$auth_header" \
    "http://127.0.0.1:${LLAMA_PORT}/v1/models")"
printf '%s\n' "$models_json" | "${OPEN_WEBUI_ENV}/bin/python" -c \
    'import json,sys; d=json.load(sys.stdin); print([x.get("id") for x in d.get("data", [])])'

printf '3/3 一次最短中文聊天 smoke\n'
response="$(curl --silent --show-error --fail \
    -H 'Content-Type: application/json' \
    -H "$auth_header" \
    -d '{"model":"ColorLM-v38-Qwen36-Shared-Sequence-Policy","messages":[{"role":"user","content":"你好，只回答：你好。"}],"temperature":0,"max_tokens":8,"stream":false}' \
    "http://127.0.0.1:${LLAMA_PORT}/v1/chat/completions")"
printf '%s\n' "$response" | "${OPEN_WEBUI_ENV}/bin/python" -c \
    'import json,sys; d=json.load(sys.stdin); print(d["choices"][0]["message"].get("content", ""))'
printf '完成：未运行旧长榜或能力评测。\n'

