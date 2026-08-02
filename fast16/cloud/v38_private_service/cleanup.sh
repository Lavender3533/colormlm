#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

expected_root="/opt/atomgit/v38-private-service"
if [[ "${SERVICE_ROOT}" != "${expected_root}" ]]; then
    printf '拒绝清理非标准目录：%s\n' "${SERVICE_ROOT}" >&2
    exit 1
fi
if [[ "$#" != 1 || "$1" != "--yes" ]]; then
    printf '这会永久删除模型、运行时、Open WebUI 数据与日志。确认时运行：bash cleanup.sh --yes\n' >&2
    exit 1
fi

bash "${SCRIPT_DIR}/stop.sh" || true
cd /opt/atomgit
rm -rf -- "${expected_root}"
printf '已删除 %s；该操作不可从脚本恢复。\n' "${expected_root}"

