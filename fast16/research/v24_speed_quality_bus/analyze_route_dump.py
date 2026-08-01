"""解析v17神经岛top-k路由dump并离线比较精确缓存策略。"""

from __future__ import annotations

import argparse
import json
import struct
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


HEADER = struct.Struct("<IIiIQ")
MAGIC = 0x52454C43


def read_records(path: Path) -> list[dict[str, object]]:
    data = path.read_bytes()
    offset = 0
    records: list[dict[str, object]] = []
    while offset < len(data):
        if len(data) - offset < HEADER.size:
            raise RuntimeError("路由dump尾部不完整")
        magic, version, layer, count, step = HEADER.unpack_from(data, offset)
        offset += HEADER.size
        if magic != MAGIC or version != 1 or not 1 <= count <= 128:
            raise RuntimeError(f"路由dump头无效: magic={magic:x} version={version} count={count}")
        bytes_needed = count * 4
        if len(data) - offset < bytes_needed:
            raise RuntimeError("路由dump选中ID载荷不完整")
        selected = list(struct.unpack_from(f"<{count}i", data, offset))
        offset += bytes_needed
        records.append({"layer": layer, "step": step, "selected": selected})
    return records


def unique(values: Iterable[int]) -> list[int]:
    return list(dict.fromkeys(values))


def simulate_lru(records: list[dict[str, object]], slots: int) -> tuple[int, int]:
    mapping: dict[int, int] = {}
    reverse: list[int | None] = [None] * slots
    last: list[int] = [0] * slots
    hits = misses = tick = 0
    for record in records:
        tick += 1
        pinned: set[int] = set()
        requested = unique(record["selected"])
        for expert in requested:
            slot = mapping.get(expert)
            if slot is not None and reverse[slot] == expert:
                hits += 1
                pinned.add(slot)
                last[slot] = tick
            else:
                misses += 1
        for expert in requested:
            if expert in mapping and reverse[mapping[expert]] == expert:
                continue
            candidates = [i for i in range(slots) if i not in pinned]
            if not candidates:
                raise RuntimeError("cache容量不足以容纳当前token的distinct experts")
            empty = next((i for i in candidates if reverse[i] is None), None)
            slot = empty if empty is not None else min(candidates, key=lambda i: last[i])
            old = reverse[slot]
            if old is not None:
                mapping.pop(old, None)
            mapping[expert] = slot
            reverse[slot] = expert
            last[slot] = tick
            pinned.add(slot)
    return hits, misses


def simulate_lfu(records: list[dict[str, object]], slots: int, decay: float) -> tuple[int, int]:
    mapping: dict[int, int] = {}
    reverse: list[int | None] = [None] * slots
    score: list[float] = [0.0] * slots
    last: list[int] = [0] * slots
    hits = misses = tick = 0
    for record in records:
        tick += 1
        score = [value * decay for value in score]
        pinned: set[int] = set()
        requested = unique(record["selected"])
        for expert in requested:
            slot = mapping.get(expert)
            if slot is not None and reverse[slot] == expert:
                hits += 1
                pinned.add(slot)
                last[slot] = tick
            else:
                misses += 1
        for expert in requested:
            slot = mapping.get(expert)
            if slot is not None and reverse[slot] == expert:
                score[slot] += 1.0
                continue
            candidates = [i for i in range(slots) if i not in pinned]
            if not candidates:
                raise RuntimeError("cache容量不足以容纳当前token的distinct experts")
            empty = next((i for i in candidates if reverse[i] is None), None)
            slot = empty if empty is not None else min(candidates, key=lambda i: (score[i], last[i]))
            old = reverse[slot]
            if old is not None:
                mapping.pop(old, None)
            mapping[expert] = slot
            reverse[slot] = expert
            score[slot] = 1.0
            last[slot] = tick
            pinned.add(slot)
    return hits, misses


def simulate_belady(records: list[dict[str, object]], slots: int) -> tuple[int, int]:
    future: dict[int, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        for expert in unique(record["selected"]):
            future[expert].append(index)
    positions: dict[int, int] = defaultdict(int)
    mapping: set[int] = set()
    hits = misses = 0
    for index, record in enumerate(records):
        requested = unique(record["selected"])
        pinned: set[int] = set()
        for expert in requested:
            cursor = positions[expert]
            while cursor < len(future[expert]) and future[expert][cursor] <= index:
                cursor += 1
            positions[expert] = cursor
            if expert in mapping:
                hits += 1
                pinned.add(expert)
            else:
                misses += 1
        for expert in requested:
            if expert in mapping:
                continue
            if len(mapping) >= slots:
                candidates = mapping - pinned
                if not candidates:
                    raise RuntimeError("Belady容量不足以容纳当前token")
                victim = max(
                    candidates,
                    key=lambda item: future[item][positions[item]]
                    if positions[item] < len(future[item])
                    else 10**12,
                )
                mapping.remove(victim)
            mapping.add(expert)
    return hits, misses


def main() -> int:
    parser = argparse.ArgumentParser(description="分析神经岛专家路由dump")
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--slots", type=int, default=32)
    args = parser.parse_args()
    records = read_records(args.dump)
    if not records:
        raise RuntimeError("路由dump为空")
    by_layer: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        by_layer[int(record["layer"])].append(record)
    policies = {"lru": simulate_lru, "belady_upper_bound": simulate_belady}
    result_layers: dict[str, object] = {}
    for layer, layer_records in sorted(by_layer.items()):
        layer_result: dict[str, object] = {
            "records": len(layer_records),
            "unique_experts": len({x for r in layer_records for x in unique(r["selected"])}),
            "frequency_top10": Counter(
                x for r in layer_records for x in unique(r["selected"])
            ).most_common(10),
        }
        for name, function in policies.items():
            hits, misses = function(layer_records, args.slots)
            layer_result[name] = {
                "hits": hits,
                "misses": misses,
                "hit_rate_percent": 100.0 * hits / max(hits + misses, 1),
            }
        for decay in (0.90, 0.97, 0.99):
            hits, misses = simulate_lfu(layer_records, args.slots, decay)
            layer_result[f"decay_lfu_{decay}"] = {
                "hits": hits,
                "misses": misses,
                "hit_rate_percent": 100.0 * hits / max(hits + misses, 1),
            }
        result_layers[str(layer)] = layer_result

    report = {
        "format": "colorlm-v24-neural-island-route-analysis-v1",
        "dump": str(args.dump.resolve()),
        "records": len(records),
        "layers": result_layers,
        "decision_rule": "only implement an online policy if it beats LRU on held-out continuation and preserves exact outputs",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for layer, row in result_layers.items():
        print(
            f"L{layer}: records={row['records']} "
            f"LRU={row['lru']['hit_rate_percent']:.2f}% "
            f"LFU97={row['decay_lfu_0.97']['hit_rate_percent']:.2f}% "
            f"Belady={row['belady_upper_bound']['hit_rate_percent']:.2f}%",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
