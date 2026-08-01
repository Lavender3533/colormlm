"""从 ColorLM 隐藏状态 dump 离线重放 Qwen3.6 router，统计稀疏专家覆盖率。"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, os.fspath(ROOT / "llama.cpp" / "gguf-py"))

from gguf import GGUFReader  # noqa: E402


MAGIC = 0x394D4C43
HEADER = struct.Struct("<IIiI4qQ")


def parse_args() -> argparse.Namespace:
    models = ROOT / "fast16" / "models"
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="离线统计 v33 的逐层专家路由")
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=models / "ColorLM-v33-Qwen36-Global-MoE-Pair.gguf",
    )
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--output", type=Path, default=here / "route_attribution.json")
    return parser.parse_args()


def read_dump(path: Path) -> dict[int, list[np.ndarray]]:
    records: dict[int, list[np.ndarray]] = defaultdict(list)
    with path.open("rb") as stream:
        index = 0
        while header_bytes := stream.read(HEADER.size):
            index += 1
            if len(header_bytes) != HEADER.size:
                raise RuntimeError(f"记录 {index} 的头部被截断")
            magic, version, layer, tensor_type, ne0, ne1, ne2, ne3, size = (
                HEADER.unpack(header_bytes)
            )
            if magic != MAGIC or version != 1:
                raise RuntimeError(
                    f"记录 {index} 格式不兼容: magic={magic:#x}, version={version}"
                )
            if tensor_type != 0:
                raise RuntimeError(f"记录 {index} 不是 F32: type={tensor_type}")
            payload = stream.read(size)
            if len(payload) != size:
                raise RuntimeError(f"记录 {index} 的 payload 被截断")
            expected = ne0 * ne1 * ne2 * ne3 * 4
            if size != expected:
                raise RuntimeError(
                    f"记录 {index} 大小不一致: payload={size}, expected={expected}"
                )
            values = np.frombuffer(payload, dtype="<f4").reshape(
                (ne3, ne2, ne1, ne0)
            )
            records[layer].append(values.reshape((-1, ne0)).copy())
    return dict(records)


def field_int(reader: GGUFReader, name: str) -> int:
    if name not in reader.fields:
        raise KeyError(name)
    return int(reader.fields[name].contents())


def coverage_count(counter: Counter[int], fraction: float) -> int:
    target = sum(counter.values()) * fraction
    cumulative = 0
    for index, (_, count) in enumerate(counter.most_common(), 1):
        cumulative += count
        if cumulative >= target:
            return index
    return 0


def main() -> int:
    args = parse_args()
    for path in (args.dump, args.model):
        if not path.is_file():
            raise FileNotFoundError(path)

    records = read_dump(args.dump)
    reader = GGUFReader(os.fspath(args.model), "r")
    tensor_map = {tensor.name: tensor for tensor in reader.tensors}
    top_k = args.top_k or field_int(reader, "colorlmv4.expert_used_count")
    expert_count = field_int(reader, "colorlmv4.expert_count")
    if not 1 <= top_k <= expert_count:
        raise ValueError(f"top-k 无效: {top_k}")

    layer_rows = []
    global_counts: Counter[int] = Counter()
    total_hidden_tokens = 0
    for layer in sorted(records):
        name = f"blk.{layer}.ffn_gate_inp.weight"
        if name not in tensor_map:
            raise KeyError(name)
        router = np.asarray(tensor_map[name].data, dtype=np.float32)
        layer_records = records[layer]
        hidden = max(layer_records, key=lambda values: values.shape[0])
        if router.shape != (expert_count, hidden.shape[1]):
            raise RuntimeError(
                f"router 形状不兼容: {name}={router.shape}, "
                f"hidden={hidden.shape}"
            )
        logits = hidden @ router.T
        partition = np.argpartition(logits, -top_k, axis=1)[:, -top_k:]
        selected_logits = np.take_along_axis(logits, partition, axis=1)
        order = np.argsort(selected_logits, axis=1)[:, ::-1]
        selected = np.take_along_axis(partition, order, axis=1)
        counts: Counter[int] = Counter(int(item) for item in selected.flat)
        global_counts.update(counts)
        total_hidden_tokens += hidden.shape[0]

        sorted_logits = np.partition(logits, -(top_k + 1), axis=1)
        kth = sorted_logits[:, -(top_k + 1) :]
        kth.sort(axis=1)
        margin = kth[:, 1] - kth[:, 0]
        layer_rows.append(
            {
                "layer": layer,
                "records_seen": len(layer_records),
                "hidden_tokens": int(hidden.shape[0]),
                "selected_slots": int(hidden.shape[0] * top_k),
                "distinct_experts": len(counts),
                "experts_for_50pct_slots": coverage_count(counts, 0.50),
                "experts_for_75pct_slots": coverage_count(counts, 0.75),
                "experts_for_90pct_slots": coverage_count(counts, 0.90),
                "experts_for_95pct_slots": coverage_count(counts, 0.95),
                "topk_boundary_margin_mean": float(np.mean(margin)),
                "top_experts": [
                    {
                        "expert": expert,
                        "count": count,
                        "slot_share": count / (hidden.shape[0] * top_k),
                    }
                    for expert, count in counts.most_common(32)
                ],
            }
        )

    report = {
        "format": "colormlm-qwen36-route-attribution-v1",
        "model": args.model.name,
        "dump": args.dump.as_posix(),
        "top_k": top_k,
        "expert_count": expert_count,
        "layers_observed": sorted(records),
        "hidden_tokens_across_layers": total_hidden_tokens,
        "global_distinct_experts": len(global_counts),
        "global_experts_for_50pct_slots": coverage_count(global_counts, 0.50),
        "global_experts_for_75pct_slots": coverage_count(global_counts, 0.75),
        "global_experts_for_90pct_slots": coverage_count(global_counts, 0.90),
        "global_experts_for_95pct_slots": coverage_count(global_counts, 0.95),
        "global_top_experts": [
            {"expert": expert, "count": count}
            for expert, count in global_counts.most_common(64)
        ],
        "layers": layer_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"已分析 {len(layer_rows)} 层；全局 90% 路由槽需要 "
        f"{report['global_experts_for_90pct_slots']} 个专家。"
    )
    print(f"报告: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
