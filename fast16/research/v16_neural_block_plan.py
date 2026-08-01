"""Build a byte-exact extraction and runtime contract for a donor neural block."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESEARCH = Path(__file__).resolve().parent
DEFAULT_CACHE = (
    RESEARCH
    / "biopsy_cache"
    / "Qwen_Qwen3-Coder-Next"
    / "master"
)
DEFAULT_OUTPUT = RESEARCH / "v16_coder_neural_block_plan.json"
DEFAULT_TRANSPORT = RESEARCH / "coder_next_to_colorlm_orthogonal_f32.npy"

DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成ColorLM v16完整神经块的精确提取与运行契约"
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--layer", type=int, default=47)
    parser.add_argument("--target-site", type=int, default=35)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--transport", type=Path, default=DEFAULT_TRANSPORT)
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="只打印审计结果，不写计划文件",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_role(name: str) -> str:
    if name.endswith("input_layernorm.weight"):
        return "input_norm"
    if name.endswith("post_attention_layernorm.weight"):
        return "post_attention_norm"
    if ".self_attn." in name:
        return "full_attention"
    if ".linear_attn." in name:
        return "gated_deltanet"
    if re.search(r"\.mlp\.experts\.\d+\.", name):
        return "routed_expert"
    if ".mlp.shared_expert." in name:
        return "shared_expert"
    if name.endswith("mlp.shared_expert_gate.weight"):
        return "shared_expert_gate"
    if name.endswith("mlp.gate.weight"):
        return "expert_router"
    return "unknown"


def load_headers(cache: Path) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for path in sorted((cache / "headers").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        shard = path.name.removesuffix(".json")
        result[shard] = payload
    return result


def build_ranges(records: list[dict]) -> list[dict]:
    by_shard: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_shard[record["source_shard"]].append(record)

    ranges: list[dict] = []
    for shard, items in sorted(by_shard.items()):
        items.sort(key=lambda item: item["absolute_start"])
        current = None
        for item in items:
            if current is not None and item["absolute_start"] == current["end_exclusive"]:
                current["end_exclusive"] = item["absolute_end"]
                current["tensor_count"] += 1
                current["last_tensor"] = item["name"]
                continue
            current = {
                "source_shard": shard,
                "start": item["absolute_start"],
                "end_exclusive": item["absolute_end"],
                "tensor_count": 1,
                "first_tensor": item["name"],
                "last_tensor": item["name"],
            }
            ranges.append(current)
        shard_ranges = [record for record in ranges if record["source_shard"] == shard]
        for ordinal, record in enumerate(shard_ranges):
            record["bytes"] = record["end_exclusive"] - record["start"]
            record["http_range"] = (
                f"bytes={record['start']}-{record['end_exclusive'] - 1}"
            )
            record["file"] = (
                f"{Path(shard).stem}.range-{record['start']}-"
                f"{record['end_exclusive'] - 1}.{ordinal}.bin"
            )
    return ranges


def q4_0_bytes(shape: list[int]) -> int:
    if not shape or shape[-1] % 32:
        return math.prod(shape) * 2
    return math.prod(shape[:-1]) * (shape[-1] // 32) * 18


def build_plan(cache: Path, layer: int, target_site: int, transport: Path) -> dict:
    index_path = cache / "model.safetensors.index.json"
    config_path = cache / "metadata" / "config.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not transport.is_file() or transport.stat().st_size != 2048 * 2048 * 4 + 128:
        raise RuntimeError(f"缺少2048x2048 F32坐标桥: {transport}")
    headers = load_headers(cache)
    prefix = f"model.layers.{layer}."
    names = sorted(name for name in index["weight_map"] if name.startswith(prefix))
    if not names:
        raise RuntimeError(f"供体索引中不存在第{layer}层")

    records: list[dict] = []
    missing_headers: list[str] = []
    for name in names:
        shard = index["weight_map"][name]
        header = headers.get(shard)
        if header is None or name not in header["tensors"]:
            missing_headers.append(shard)
            continue
        source = header["tensors"][name]
        shape = [int(value) for value in source["shape"]]
        offsets = [int(value) for value in source["data_offsets"]]
        logical_bytes = math.prod(shape) * DTYPE_BYTES[source["dtype"]]
        if offsets[1] - offsets[0] != logical_bytes:
            raise RuntimeError(f"张量字节数异常: {name}")
        data_start = 8 + int(header["header_bytes"])
        records.append(
            {
                "name": name,
                "role": tensor_role(name),
                "source_shard": shard,
                "dtype": source["dtype"],
                "shape": shape,
                "bytes": logical_bytes,
                "absolute_start": data_start + offsets[0],
                "absolute_end": data_start + offsets[1],
            }
        )
    if missing_headers:
        unique = sorted(set(missing_headers))
        raise RuntimeError("缺少供体分片头: " + ", ".join(unique))

    roles = Counter(record["role"] for record in records)
    unknown = [record["name"] for record in records if record["role"] == "unknown"]
    if unknown:
        raise RuntimeError("存在未识别的层张量: " + ", ".join(unknown))

    expert_ids = sorted(
        {
            int(match.group(1))
            for record in records
            if (match := re.search(r"\.mlp\.experts\.(\d+)\.", record["name"]))
        }
    )
    expected_experts = int(config["num_experts"])
    if expert_ids != list(range(expected_experts)):
        raise RuntimeError(
            f"专家集合不完整: {len(expert_ids)} vs {expected_experts}"
        )
    if roles["routed_expert"] != expected_experts * 3:
        raise RuntimeError("每个路由专家必须精确包含gate/up/down三个张量")

    layer_type = "full_attention" if roles["full_attention"] else "gated_deltanet"
    expected_type = (
        "full_attention"
        if (layer + 1) % int(config["full_attention_interval"]) == 0
        else "gated_deltanet"
    )
    if layer_type != expected_type:
        raise RuntimeError(f"层类型与config不一致: {layer_type} vs {expected_type}")

    total_bytes = sum(record["bytes"] for record in records)
    nonexpert_bytes = sum(
        record["bytes"] for record in records if record["role"] != "routed_expert"
    )
    routed_expert_bytes = total_bytes - nonexpert_bytes
    q4_estimate = sum(
        q4_0_bytes(record["shape"])
        if record["role"] not in {"input_norm", "post_attention_norm"}
        else record["bytes"]
        for record in records
    )
    ranges = build_ranges(records)
    ranges_total = sum(record["bytes"] for record in ranges)
    if ranges_total != total_bytes:
        raise RuntimeError(f"Range合并后字节数漂移: {ranges_total} vs {total_bytes}")
    for record in records:
        matches = [
            segment
            for segment in ranges
            if segment["source_shard"] == record["source_shard"]
            and segment["start"] <= record["absolute_start"]
            and record["absolute_end"] <= segment["end_exclusive"]
        ]
        if len(matches) != 1:
            raise RuntimeError(f"张量无法唯一映射到Range段: {record['name']}")
        segment = matches[0]
        record["segment_file"] = segment["file"]
        record["segment_offset"] = record["absolute_start"] - segment["start"]

    return {
        "format": "colorlm-neural-block-abi-v1",
        "name": f"ColorLM-v17-Coder-Neural-Block-L{layer}",
        "status": "extraction-ready-runtime-pending",
        "source": {
            "repo": "Qwen/Qwen3-Coder-Next",
            "revision": "master",
            "architecture": config["architectures"][0],
            "layer": layer,
            "layer_type": layer_type,
            "index_sha256": sha256_file(index_path),
            "config_sha256": sha256_file(config_path),
        },
        "interface": {
            "input_width": int(config["hidden_size"]),
            "output_width": int(config["hidden_size"]),
            "input_transport": "colorlm-to-coder-2048x2048",
            "output_transport": "coder-to-colorlm-2048x2048",
            "residual_contract": "h_out=h_native+alpha*energy_gate*transport_out(block(transport_in(h_native))-transport_in(h_native))",
            "hard_bypass": "alpha=0 does not load weights, allocate memory, or create graph nodes",
            "transport": {
                "method": "shared-token-orthogonal-procrustes",
                "source_to_target_file": transport.name,
                "source_to_target_sha256": sha256_file(transport),
                "shape": [2048, 2048],
                "runtime_dtype": "F16",
                "input": "T maps ColorLM column hidden states into Coder coordinates",
                "output": "transpose(T) maps Coder column hidden states back into ColorLM coordinates"
            },
        },
        "attention": {
            "type": layer_type,
            "query_heads": int(config["num_attention_heads"]),
            "kv_heads": int(config["num_key_value_heads"]),
            "head_dim": int(config["head_dim"]),
            "partial_rotary_factor": float(config["partial_rotary_factor"]),
            "rope_theta": float(config["rope_theta"]),
            "memory": (
                "independent-sidecar-kv-required"
                if layer_type == "full_attention"
                else "independent-recurrent-state-required"
            ),
        },
        "moe": {
            "expert_count": expected_experts,
            "experts_per_token": int(config["num_experts_per_tok"]),
            "expert_width": int(config["moe_intermediate_size"]),
            "shared_expert_width": int(config["shared_expert_intermediate_size"]),
            "paging": "exact-top-k-only; cache miss must wait or take the whole block no-op path",
        },
        "target": {
            "base": "ColorLM-v13-Causal-Sparse-L12",
            "site": target_site,
            "site_policy": (
                f"L{target_site} full-token residual station; donor L{layer} "
                "uses island-private state"
            ),
            "base_hidden_width": 2048,
            "base_query_heads": 16,
            "base_kv_heads": 2,
            "base_rope_theta": 10000000.0,
            "donor_rope_is_independent": True,
        },
        "budget": {
            "tensor_count": len(records),
            "bf16_total_bytes": total_bytes,
            "bf16_nonexpert_bytes": nonexpert_bytes,
            "bf16_routed_expert_bytes": routed_expert_bytes,
            "q4_0_estimated_bytes": q4_estimate,
            "q4_0_estimated_gib": q4_estimate / 1024**3,
            "range_request_count": len(ranges),
            "active_expert_bf16_bytes_per_token": (
                routed_expert_bytes // expected_experts
            )
            * int(config["num_experts_per_tok"]),
        },
        "roles": dict(sorted(roles.items())),
        "source_ranges": ranges,
        "runtime_gates": {
            "stage_1": "manifest and exact Range extraction",
            "stage_2": "independent KV memory with sequence copy/remove support",
            "stage_3": "attention + norm + shared expert, forced residual only",
            "stage_4": "exact top-10 expert paging and full MoE",
            "stage_5": "short counterfactual NLL and manual coding trial",
        },
        "tensors": records,
    }


def main() -> int:
    args = parse_args()
    plan = build_plan(args.cache, args.layer, args.target_site, args.transport)
    budget = plan["budget"]
    print(f"ABI: {plan['format']}")
    print(
        f"供体: L{args.layer} {plan['source']['layer_type']}, "
        f"张量{budget['tensor_count']}个"
    )
    print(
        f"BF16: {budget['bf16_total_bytes'] / 1024**3:.3f} GiB, "
        f"Q4_0估算: {budget['q4_0_estimated_gib']:.3f} GiB"
    )
    print(
        f"合并Range: {budget['range_request_count']}段, "
        f"每token活跃专家BF16: "
        f"{budget['active_expert_bf16_bytes_per_token'] / 1024**2:.1f} MiB"
    )
    state_kind = "sidecar KV" if plan["source"]["layer_type"] == "full_attention" else "recurrent state"
    print(f"独立状态: {state_kind} 是进入C++计算图前的硬前置条件")
    if not args.summary_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"计划: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
