#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

LLAMA_BASE_COMMIT="b46812de78f8fbcb6cf0154947e8633ebc78d9ac"
PATCH_FILE="${SCRIPT_DIR}/0001-feat-add-ColorLM-neural-bus-runtime-extensions.patch"
PATCH_SHA256="aba02e3bcc6b0ac12ecde70e34f11b34cde274f61ad1d3267eae0c5895adc08f"
OPEN_WEBUI_VERSION="${OPEN_WEBUI_VERSION:-0.11.0}"

mkdir -p -- "${SERVICE_ROOT}" "${SERVICE_ROOT}/models" "${POLICY_DIR}" "${DATA_DIR}" "${LOG_DIR}" "${RUN_DIR}"

free_bytes="$(df -Pk -- "${SERVICE_ROOT}" | awk 'NR==2 {printf "%.0f", $4 * 1024}')"
min_disk_bytes=$((25 * 1024 * 1024 * 1024))
if (( free_bytes < min_disk_bytes )); then
    printf '可用磁盘不足 25 GiB；当前约 %.2f GiB。停止安装，不量化、不换模型。\n' "$(awk -v b="$free_bytes" 'BEGIN {print b/1024/1024/1024}')" >&2
    exit 1
fi

available_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
if (( available_kib < 24 * 1024 * 1024 )); then
    printf '可用内存不足 24 GiB；停止安装，不量化、不换模型。\n' >&2
    exit 1
fi

command -v git >/dev/null
command -v cmake >/dev/null
command -v g++ >/dev/null
command -v sha256sum >/dev/null
command -v npu-smi >/dev/null
source_cann
npu-smi info

verify_exact_file "${PATCH_FILE}" 554048 "${PATCH_SHA256}"
verify_exact_file "${SCRIPT_DIR}/runtime-v1/policy.json" 1493 "${POLICY_MANIFEST_SHA256}"
verify_exact_file "${SCRIPT_DIR}/runtime-v1/weights.bin" 131264 "${POLICY_WEIGHTS_SHA256}"
install -m 600 "${SCRIPT_DIR}/runtime-v1/policy.json" "${POLICY_DIR}/policy.json"
install -m 600 "${SCRIPT_DIR}/runtime-v1/weights.bin" "${POLICY_DIR}/weights.bin"

if [[ ! -d "${LLAMA_DIR}/.git" ]]; then
    mkdir -p -- "${LLAMA_DIR}"
    git -C "${LLAMA_DIR}" init
    git -C "${LLAMA_DIR}" remote add origin https://github.com/ggml-org/llama.cpp.git
fi

git -C "${LLAMA_DIR}" fetch --depth 1 origin "${LLAMA_BASE_COMMIT}"
git -C "${LLAMA_DIR}" checkout --detach --force FETCH_HEAD
git -C "${LLAMA_DIR}" reset --hard "${LLAMA_BASE_COMMIT}"
git -C "${LLAMA_DIR}" clean -fd
git -C "${LLAMA_DIR}" apply --check "${PATCH_FILE}"
git -C "${LLAMA_DIR}" apply "${PATCH_FILE}"

cmake -S "${LLAMA_DIR}" -B "${LLAMA_BUILD_DIR}" \
    -DGGML_CANN=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_CURL=OFF
cmake --build "${LLAMA_BUILD_DIR}" --target llama-server -j "${BUILD_JOBS:-16}"
require_file "${LLAMA_SERVER}"

if [[ ! -x "${OPEN_WEBUI_ENV}/bin/python" ]]; then
    python_bin="$(command -v python3 || command -v python)"
    python_version="$(${python_bin} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    if [[ "$python_version" == "3.11" || "$python_version" == "3.12" ]]; then
        "${python_bin}" -m venv "${OPEN_WEBUI_ENV}"
    elif command -v conda >/dev/null 2>&1; then
        conda create -y -p "${OPEN_WEBUI_ENV}" python=3.11 pip
    else
        printf 'Open WebUI 需要 Python 3.11/3.12，当前为 %s，且未找到 conda。\n' "$python_version" >&2
        exit 1
    fi
fi

"${OPEN_WEBUI_ENV}/bin/python" -m pip install --upgrade pip wheel
"${OPEN_WEBUI_ENV}/bin/python" -m pip install "open-webui==${OPEN_WEBUI_VERSION}"
require_file "${OPEN_WEBUI_ENV}/bin/open-webui"

if [[ -f "${MODEL_PATH}" ]]; then
    printf '发现模型，开始一次完整大小与 SHA-256 校验……\n'
    verify_exact_file "${MODEL_PATH}" "${MODEL_BYTES}" "${MODEL_SHA256}"
else
    printf '运行时已安装。模型尚未上传到：%s\n' "${MODEL_PATH}"
fi

printf '\n安装完成。下一步：上传原始 GGUF，然后运行 bash %s/start.sh\n' "${SCRIPT_DIR}"
