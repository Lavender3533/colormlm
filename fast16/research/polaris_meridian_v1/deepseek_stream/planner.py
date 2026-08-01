"""生成 DeepSeek-V4 L39-L42 原生状态采集 dry-run 与两小时资源预算。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

try:
    from .capture_io import HEADER, HIDDEN, HC_MULT, LAYERS, TOP_K
except ImportError:
    from capture_io import HEADER, HIDDEN, HC_MULT, LAYERS, TOP_K


HERE = Path(__file__).resolve().parent
DEFAULT_SNAPSHOT = HERE / "official_snapshot.json"
MIB = 1024 * 1024
GIB = 1024 * MIB


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} 必须是 JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_metadata(snapshot: dict[str, Any], config_path: Path | None, index_path: Path | None) -> dict[str, Any]:
    result: dict[str, Any] = {"config_checked": False, "index_checked": False, "ok": True}
    artifacts = snapshot["metadata_artifacts"]
    if config_path is not None:
        expected = artifacts["config.json"]
        actual_hash = sha256_file(config_path)
        actual_size = config_path.stat().st_size
        if actual_hash != expected["sha256"] or actual_size != expected["bytes"]:
            raise ValueError("config.json 与固定 official revision 不一致")
        config = read_json(config_path)
        frozen = snapshot["config"]
        checks = {
            "hidden_size": config.get("hidden_size"),
            "num_hidden_layers": config.get("num_hidden_layers"),
            "vocab_size": config.get("vocab_size"),
            "hc_mult": config.get("hc_mult"),
            "n_routed_experts": config.get("n_routed_experts"),
            "num_experts_per_tok": config.get("num_experts_per_tok"),
        }
        for key, value in checks.items():
            if int(value) != int(frozen[key]):
                raise ValueError(f"config 字段 {key} 与 snapshot 不一致")
        result["config_checked"] = True
        result["config_sha256"] = actual_hash
    if index_path is not None:
        expected = artifacts["model.safetensors.index.json"]
        actual_hash = sha256_file(index_path)
        actual_size = index_path.stat().st_size
        if actual_hash != expected["sha256"] or actual_size != expected["bytes"]:
            raise ValueError("model.safetensors.index.json 与固定 official revision 不一致")
        index = read_json(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or len(weight_map) != expected["tensor_keys"]:
            raise ValueError("index tensor key 数不一致")
        if int(index.get("metadata", {}).get("total_size", -1)) != int(expected["metadata_total_size"]):
            raise ValueError("index metadata.total_size 不一致")
        for layer_info in snapshot["capture_layer_shards"]:
            layer = int(layer_info["layer"])
            prefix = f"layers.{layer}."
            layer_files = {name for key, name in weight_map.items() if key.startswith(prefix)}
            if layer_files != {layer_info["file"]}:
                raise ValueError(f"L{layer} shard 映射不一致: {sorted(layer_files)}")
            count = sum(key.startswith(prefix) for key in weight_map)
            if count != int(layer_info["tensor_count"]):
                raise ValueError(f"L{layer} tensor_count 不一致")
            for key in (layer_info["router_weight"], layer_info["router_bias"]):
                if weight_map.get(key) != layer_info["file"]:
                    raise ValueError(f"router tensor 映射错误: {key}")
        result["index_checked"] = True
        result["index_sha256"] = actual_hash
    return result


def capture_bytes_per_record() -> dict[str, int]:
    hidden_means = len(LAYERS) * HIDDEN * 2
    mhc_streams = len(LAYERS) * HC_MULT * HIDDEN * 2
    mhc_mixes = len(LAYERS) * 2 * 20 * 4
    router = len(LAYERS) * TOP_K * (4 + 4)
    token = 3 * 4 + 2 * 4
    chunks = len(LAYERS) * 6 + 2
    payload = hidden_means + mhc_streams + mhc_mixes + router + token
    return {
        "payload_bytes": payload,
        "header_bytes": chunks * HEADER.size,
        "cnob_bytes": payload + chunks * HEADER.size,
        "chunks": chunks,
    }


def make_plan(
    snapshot: dict[str, Any],
    *,
    task_count: int,
    observed_tokens_per_task: int,
    network_mib_s: float,
    available_disk_gib: float,
    available_ram_gib: float,
    hbm_gib: float,
    window_seconds: int,
    metadata_verification: dict[str, Any],
) -> dict[str, Any]:
    if min(task_count, observed_tokens_per_task, window_seconds) <= 0:
        raise ValueError("task/token/window 必须为正数")
    if min(network_mib_s, available_disk_gib, available_ram_gib, hbm_gib) <= 0:
        raise ValueError("网络/磁盘/RAM/HBM 预算必须为正数")
    base_bytes = int(snapshot["base_forward_files"]["file_union_bytes"])
    layer_union = sum(int(row["bytes"]) for row in snapshot["capture_layer_shards"])
    records = task_count * observed_tokens_per_task
    per_record = capture_bytes_per_record()
    cnob_bytes = records * per_record["cnob_bytes"]
    jsonl_upper = records * 2048
    output_upper = cnob_bytes + jsonl_upper + 4 * MIB
    network_seconds = base_bytes / (network_mib_s * MIB)
    reserve_seconds = 15 * 60
    estimated_download_and_reserve = network_seconds + reserve_seconds
    full_disk_fit = base_bytes <= available_disk_gib * GIB
    full_ram_fit = base_bytes <= available_ram_gib * GIB
    hbm_fit = base_bytes <= hbm_gib * GIB
    largest_stream_shard = max(int(row["bytes"]) for row in snapshot["capture_layer_shards"])
    stream_disk_fit = largest_stream_shard + 4 * GIB <= available_disk_gib * GIB
    return {
        "format": "polaris-deepseek-native-capture-plan-v1",
        "status": "dry_run_only_no_native_forward",
        "source": {
            "repo": snapshot["repo"],
            "revision": snapshot["revision"],
            "snapshot_sha256": hashlib.sha256(
                json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "metadata_verification": metadata_verification,
        },
        "capture": {
            "layers": list(LAYERS),
            "hidden_size": HIDDEN,
            "hc_mult": HC_MULT,
            "router_top_k": TOP_K,
            "task_count": task_count,
            "observed_tokens_per_task": observed_tokens_per_task,
            "record_upper_bound": records,
            "per_record": per_record,
            "cnob_upper_bound_bytes": cnob_bytes,
            "utf8_jsonl_upper_bound_bytes": jsonl_upper,
            "total_output_upper_bound_bytes": output_upper,
        },
        "weights": {
            "base_forward_file_count": snapshot["base_forward_files"]["file_count"],
            "base_forward_file_union_bytes": base_bytes,
            "tail_capture_layer_file_union_bytes": layer_union,
            "tail_only_is_not_a_valid_forward": True,
            "dspark_download_required": False,
            "largest_capture_layer_shard_bytes": largest_stream_shard,
        },
        "two_hour_budget": {
            "window_seconds": window_seconds,
            "network_mib_s_assumption": network_mib_s,
            "base_weight_transfer_seconds": network_seconds,
            "runtime_and_upload_reserve_seconds": reserve_seconds,
            "transfer_plus_reserve_seconds": estimated_download_and_reserve,
            "fits_by_network_time_only": estimated_download_and_reserve <= window_seconds,
            "warning": "即使网络时间通过，也不代表官方 CUDA/FP4 kernel 可在 Ascend 运行。",
        },
        "memory": {
            "available_disk_gib": available_disk_gib,
            "available_ram_gib": available_ram_gib,
            "hbm_gib": hbm_gib,
            "full_base_files_fit_disk": full_disk_fit,
            "full_base_files_fit_ram": full_ram_fit,
            "full_base_files_fit_hbm": hbm_fit,
            "one_shard_streaming_workspace_fit_disk": stream_disk_fit,
            "streaming_runtime_implemented": False,
        },
        "gates": [
            {"id": "metadata", "pass": bool(metadata_verification["ok"]), "meaning": "固定 config/index 契约"},
            {"id": "doctor", "pass": False, "meaning": "必须在目标 910B4 容器运行 doctor.py"},
            {"id": "adapter", "pass": False, "meaning": "必须提供并证明官方原生 runtime adapter"},
            {"id": "weights", "pass": False, "meaning": "45 个 base-forward 文件或等价远端 tensor source 必须校验"},
            {"id": "native_forward", "pass": False, "meaning": "真实 L39-L42 probe 尚未运行"},
        ],
        "next_commands": [
            "python -X utf8 doctor.py --device npu --network-mib-s <实测值>",
            "python -X utf8 selftest.py --device npu",
            "python -X utf8 run_capture.py --adapter your_adapter:run_native_capture --tasks tasks.jsonl --output-dir output",
        ],
        "claim_limit": "该计划只证明元数据、格式与资源下界；native_forward gate 为 false，不能写成已采集 DeepSeek 状态。",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--config", type=Path, help="可选：固定 revision 的 config.json，用于逐字节验证")
    parser.add_argument("--index", type=Path, help="可选：固定 revision 的 index.json，用于 tensor 映射验证")
    parser.add_argument("--task-count", type=int, default=8)
    parser.add_argument("--observed-tokens-per-task", type=int, default=64)
    parser.add_argument("--network-mib-s", type=float, default=50.0)
    parser.add_argument("--available-disk-gib", type=float, default=50.0)
    parser.add_argument("--available-ram-gib", type=float, default=1500.0)
    parser.add_argument("--hbm-gib", type=float, default=32.0)
    parser.add_argument("--window-seconds", type=int, default=7200)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    snapshot = read_json(args.snapshot)
    if snapshot.get("revision") != "7872f01b1d1fe23eabc4c98b48bffcef5a386062":
        raise ValueError("拒绝非固定 DeepSeek revision")
    verification = verify_metadata(snapshot, args.config, args.index)
    plan = make_plan(
        snapshot,
        task_count=args.task_count,
        observed_tokens_per_task=args.observed_tokens_per_task,
        network_mib_s=args.network_mib_s,
        available_disk_gib=args.available_disk_gib,
        available_ram_gib=args.available_ram_gib,
        hbm_gib=args.hbm_gib,
        window_seconds=args.window_seconds,
        metadata_verification=verification,
    )
    encoded = json.dumps(plan, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        if args.output.exists() and not args.force:
            raise FileExistsError(f"拒绝覆盖 {args.output}；使用 --force")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8", newline="\n")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
