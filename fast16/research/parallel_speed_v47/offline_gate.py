#!/usr/bin/env python3
"""用冻结记录运行v47纯CPU可行性短门；不加载模型，不调用GPU。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from cost_model import calculate


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
HEADER = struct.Struct("<IIiIQ")
MAGIC = 0x52454C43


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique(values: Iterable[int]) -> list[int]:
    return list(dict.fromkeys(values))


def read_records(path: Path) -> list[dict[str, Any]]:
    data = path.read_bytes()
    records: list[dict[str, Any]] = []
    offset = 0
    while offset < len(data):
        if len(data) - offset < HEADER.size:
            raise ValueError("路由dump尾部不完整")
        magic, version, layer, count, step = HEADER.unpack_from(data, offset)
        offset += HEADER.size
        if magic != MAGIC or version != 1 or not 1 <= count <= 128:
            raise ValueError("路由dump头无效")
        needed = count * 4
        if len(data) - offset < needed:
            raise ValueError("路由dump专家ID载荷不完整")
        selected = list(struct.unpack_from(f"<{count}i", data, offset))
        offset += needed
        records.append({"layer": layer, "step": step, "selected": unique(selected)})
    return records


def validate_manifest(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = ("base", "evidence", "draft_head", "latent_recursion", "paging", "claim_limits")
    for key in required:
        if key not in manifest:
            errors.append(f"缺少字段: {key}")
    if errors:
        return errors
    if manifest.get("format") != "colorlm-parallel-speed-v47-manifest-v1":
        errors.append("manifest format不匹配")
    if manifest.get("scope") != "offline-only":
        errors.append("scope必须为offline-only")
    draft = manifest["draft_head"]
    if not 1 <= int(draft["draft_tokens"]) <= 4:
        errors.append("draft_tokens必须在1..4")
    if not draft.get("reuse_frozen_embedding_and_lm_head"):
        errors.append("草稿头必须复用冻结embedding/lm_head")
    recursion = manifest["latent_recursion"]
    if recursion.get("k_values") != [0, 1, 2, 3, 4]:
        errors.append("动态K集合必须严格为0..4")
    if not recursion.get("hard_bypass_k0"):
        errors.append("K=0必须硬旁路")
    if not math.isclose(sum(recursion["k_distribution_assumption"]), 1.0, abs_tol=1e-9):
        errors.append("K分布之和必须为1")
    if not manifest["paging"].get("exact_miss_path"):
        errors.append("分页必须提供exact miss path")
    return errors


def validate_utf8_no_bom(root: Path) -> list[str]:
    errors: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".py", ".md", ".json"}:
            continue
        data = path.read_bytes()
        if data.startswith(b"\xef\xbb\xbf"):
            errors.append(f"含UTF-8 BOM: {path.name}")
        try:
            data.decode("utf-8")
        except UnicodeDecodeError as exc:
            errors.append(f"不是UTF-8: {path.name}: {exc}")
    return errors


@dataclass
class Metrics:
    gpu_hits: int = 0
    cpu_hits: int = 0
    cold_misses: int = 0
    gpu_uploads: int = 0
    cpu_executions: int = 0
    demand_ssd_pages: int = 0
    prefetch_ssd_pages: int = 0
    useful_prefetch_hits: int = 0
    wasted_prefetch_evictions: int = 0

    @property
    def requests(self) -> int:
        return self.gpu_hits + self.cpu_hits + self.cold_misses


@dataclass
class TieredCache:
    gpu_slots: int
    cpu_slots: int
    policy: str
    decay: float
    promotion_margin: float = 0.0
    gpu: set[int] = field(default_factory=set)
    cpu: set[int] = field(default_factory=set)
    last: dict[int, int] = field(default_factory=dict)
    score: dict[int, float] = field(default_factory=lambda: defaultdict(float))
    prefetched: set[int] = field(default_factory=set)
    tick: int = 0
    metrics: Metrics = field(default_factory=Metrics)

    def _value(self, expert: int, predicted: set[int]) -> tuple[float, int]:
        if self.policy == "lru":
            return (float(self.last.get(expert, -1)), self.last.get(expert, -1))
        bonus = 2.0 if expert in predicted and self.policy == "dali" else 0.0
        return (self.score.get(expert, 0.0) + bonus, self.last.get(expert, -1))

    def _evict_one(self, tier: set[int], pinned: set[int], predicted: set[int]) -> int | None:
        candidates = tier - pinned
        if not candidates:
            return None
        victim = min(candidates, key=lambda item: self._value(item, predicted))
        tier.remove(victim)
        if victim in self.prefetched:
            self.prefetched.remove(victim)
            self.metrics.wasted_prefetch_evictions += 1
        return victim

    def _put_cpu(self, expert: int, pinned: set[int], predicted: set[int], is_prefetch: bool) -> None:
        if expert in self.gpu or expert in self.cpu:
            return
        if len(self.cpu) >= self.cpu_slots:
            self._evict_one(self.cpu, pinned, predicted)
        if len(self.cpu) < self.cpu_slots:
            self.cpu.add(expert)
            if is_prefetch:
                self.prefetched.add(expert)

    def _promote_gpu(self, expert: int, pinned: set[int], predicted: set[int]) -> None:
        if expert in self.gpu:
            return
        if self.policy == "dali" and len(self.gpu) >= self.gpu_slots:
            candidates = self.gpu - pinned
            if candidates:
                victim = min(candidates, key=lambda item: self._value(item, predicted))
                candidate_value = self._value(expert, predicted)[0]
                victim_value = self._value(victim, predicted)[0]
                if candidate_value < victim_value + self.promotion_margin:
                    # 精确CPU路径：本token不上传，避免低复用页挤掉GPU热页。
                    self.metrics.cpu_executions += 1
                    return
        self.cpu.discard(expert)
        self.prefetched.discard(expert)
        if len(self.gpu) >= self.gpu_slots:
            victim = self._evict_one(self.gpu, pinned, predicted)
            if victim is not None:
                self._put_cpu(victim, pinned, predicted, False)
        if len(self.gpu) < self.gpu_slots:
            self.gpu.add(expert)
            self.metrics.gpu_uploads += 1

    def access(self, requested: list[int], predicted_next: list[int]) -> None:
        self.tick += 1
        if self.policy in {"lfu97", "dali"}:
            for expert in list(self.score):
                self.score[expert] *= self.decay
        requested_set = set(requested)
        predicted_set = set(predicted_next)
        for expert in requested:
            if expert in self.gpu:
                self.metrics.gpu_hits += 1
            elif expert in self.cpu:
                self.metrics.cpu_hits += 1
                if expert in self.prefetched:
                    self.prefetched.remove(expert)
                    self.metrics.useful_prefetch_hits += 1
            else:
                self.metrics.cold_misses += 1
                self.metrics.demand_ssd_pages += 1
                self._put_cpu(expert, requested_set, predicted_set, False)
            self.last[expert] = self.tick
            self.score[expert] += 1.0
        # 当前token的专家全部精确晋升；这与v24槽池语义一致。
        for expert in requested:
            self._promote_gpu(expert, requested_set, predicted_set)
        if self.policy == "dali":
            for expert in predicted_next:
                if expert in self.gpu or expert in self.cpu:
                    continue
                self._put_cpu(expert, requested_set, predicted_set, True)
                if expert in self.cpu:
                    self.metrics.prefetch_ssd_pages += 1


def train_transition(records: list[dict[str, Any]]) -> tuple[dict[int, Counter[int]], Counter[int]]:
    transition: dict[int, Counter[int]] = defaultdict(Counter)
    global_frequency: Counter[int] = Counter()
    for record in records:
        global_frequency.update(record["selected"])
    for current, following in zip(records, records[1:]):
        for source in current["selected"]:
            transition[source].update(following["selected"])
    return transition, global_frequency


def predict_next(
    current: list[int], transition: dict[int, Counter[int]], global_frequency: Counter[int], count: int
) -> list[int]:
    votes: Counter[int] = Counter()
    for source in current:
        votes.update(transition.get(source, {}))
    for expert, frequency in global_frequency.items():
        votes[expert] += 0.05 * frequency
    for expert in current:
        votes.pop(expert, None)
    return [expert for expert, _ in votes.most_common(count)]


def metrics_dict(metrics: Metrics, page_bytes: int) -> dict[str, Any]:
    requests = max(metrics.requests, 1)
    prefetch_total = metrics.useful_prefetch_hits + metrics.wasted_prefetch_evictions
    return {
        "requests": metrics.requests,
        "gpu_hits": metrics.gpu_hits,
        "cpu_warm_hits": metrics.cpu_hits,
        "cold_misses": metrics.cold_misses,
        "gpu_hit_rate": metrics.gpu_hits / requests,
        "hierarchy_hit_rate": (metrics.gpu_hits + metrics.cpu_hits) / requests,
        "gpu_uploads": metrics.gpu_uploads,
        "cpu_executions": metrics.cpu_executions,
        "demand_ssd_mib": metrics.demand_ssd_pages * page_bytes / 1024**2,
        "prefetch_ssd_mib": metrics.prefetch_ssd_pages * page_bytes / 1024**2,
        "useful_prefetch_hits": metrics.useful_prefetch_hits,
        "wasted_prefetch_evictions": metrics.wasted_prefetch_evictions,
        "useful_prefetch_precision": metrics.useful_prefetch_hits / max(prefetch_total, 1)
    }


def replay(records: list[dict[str, Any]], paging: dict[str, Any]) -> dict[str, Any]:
    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_layer[int(record["layer"])].append(record)
    fraction = float(paging["trace_train_fraction"])
    aggregate: dict[str, Metrics] = {name: Metrics() for name in ("lru", "lfu97", "dali")}
    per_layer: dict[str, Any] = {}
    for layer, layer_records in sorted(by_layer.items()):
        cut = max(1, min(len(layer_records) - 1, int(len(layer_records) * fraction)))
        train, test = layer_records[:cut], layer_records[cut:]
        transition, global_frequency = train_transition(train)
        caches = {
            name: TieredCache(
                gpu_slots=int(paging["gpu_slots_per_layer"]),
                cpu_slots=int(paging["cpu_warm_slots_per_layer"]),
                policy=name,
                decay=float(paging["score_decay"]),
                promotion_margin=float(paging.get("gpu_promotion_margin", 0.0))
            )
            for name in ("lru", "lfu97", "dali")
        }
        for index, record in enumerate(test):
            predicted = predict_next(
                record["selected"], transition, global_frequency, int(paging["prefetch_pages_per_step"])
            )
            caches["lru"].access(record["selected"], [])
            caches["lfu97"].access(record["selected"], [])
            caches["dali"].access(record["selected"], predicted if index + 1 < len(test) else [])
        layer_result: dict[str, Any] = {}
        for name, cache in caches.items():
            layer_result[name] = metrics_dict(cache.metrics, int(paging["expert_page_bytes"]))
            target = aggregate[name]
            for field_name in Metrics.__dataclass_fields__:
                setattr(target, field_name, getattr(target, field_name) + getattr(cache.metrics, field_name))
        per_layer[str(layer)] = layer_result
    aggregate_dict = {
        name: metrics_dict(metrics, int(paging["expert_page_bytes"])) for name, metrics in aggregate.items()
    }
    lru_cold = aggregate_dict["lru"]["cold_misses"]
    dali_cold = aggregate_dict["dali"]["cold_misses"]
    reduction = (lru_cold - dali_cold) / max(lru_cold, 1)
    return {
        "split": "每层前50%仅训练转移统计，后50%冷启动回放；无未来信息",
        "records": len(records),
        "per_layer": per_layer,
        "aggregate": aggregate_dict,
        "dali_cold_miss_reduction_vs_lru": reduction,
        "note": "预取字节与需求字节分别报告；命中改善不是端到端速度证明"
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="运行ColorLM v47纯CPU离线短门")
    parser.add_argument("--manifest", type=Path, default=HERE / "manifest.json")
    parser.add_argument("--contract", type=Path, default=HERE / "short_gate_contract.json")
    parser.add_argument("--output", type=Path, default=HERE / "offline_gate_report.json")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    manifest_errors = validate_manifest(manifest)

    evidence_rows: list[dict[str, Any]] = []
    for item in manifest["evidence"]:
        path = WORKSPACE / item["path"]
        actual = sha256(path) if path.is_file() else None
        evidence_rows.append({
            "id": item["id"], "path": item["path"], "exists": path.is_file(),
            "expected_sha256": item["sha256"], "actual_sha256": actual,
            "matches": actual == item["sha256"]
        })
    cost = calculate(manifest)
    route_item = next(item for item in manifest["evidence"] if item["id"] == "v24-routes")
    records = read_records(WORKSPACE / route_item["path"])
    routing = replay(records, manifest["paging"])
    gate = contract["gates"]
    a_budget = gate["A_draft_head"]["budget"]
    b_budget = gate["B_latent_recursion"]["budget"]
    c_gate = gate["C_paging"]["offline_replay"]
    gates = {
        "manifest_valid": not manifest_errors,
        "evidence_hashes": all(row["matches"] for row in evidence_rows),
        "utf8_no_bom": not validate_utf8_no_bom(HERE),
        "A_parameters": cost["draft_head"]["trainable_parameters"] <= a_budget["maximum_trainable_parameters"],
        "A_weight_bytes": cost["draft_head"]["f16_weight_mib"] <= a_budget["maximum_f16_weight_mib"],
        "A_analytical_upper_bound": cost["draft_head"]["analytical_upper_bound_speedup"] >= a_budget["minimum_analytical_upper_bound_speedup"],
        "B_parameters": cost["latent_recursion"]["trainable_parameters"] <= b_budget["maximum_trainable_parameters"],
        "B_mean_k": cost["latent_recursion"]["mean_k_assumption"] <= b_budget["maximum_mean_k"],
        "B_k0_fraction": cost["latent_recursion"]["k0_fraction_assumption"] >= b_budget["minimum_k0_fraction"],
        "C_cold_miss_reduction": routing["dali_cold_miss_reduction_vs_lru"] >= c_gate["minimum_cold_miss_reduction_vs_hierarchical_lru"],
        "C_prefetch_precision": routing["aggregate"]["dali"]["useful_prefetch_precision"] >= c_gate["minimum_useful_prefetch_precision"],
        "C_exact_miss_path": bool(manifest["paging"]["exact_miss_path"])
    }
    offline_feasible = all(gates.values())
    failed_gates = [name for name, passed in gates.items() if not passed]
    report = {
        "format": "colorlm-parallel-speed-v47-offline-report-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_only": True,
        "model_loaded": False,
        "gpu_used": False,
        "manifest_errors": manifest_errors,
        "evidence": {"all_hashes_match": all(row["matches"] for row in evidence_rows), "files": evidence_rows},
        "cost": cost,
        "routing_replay": routing,
        "decision": {
            "offline_feasible": offline_feasible,
            "runtime_promotable": False,
            "status": "needs_runtime_evidence" if offline_feasible else "reject",
            "gates": gates,
            "failed_gates": failed_gates,
            "reason": (
                "离线硬门全部通过，但A/B仍无训练权重与真实接受率/质量证据，C也只回放v17短路由"
                if offline_feasible else
                "C冷miss降幅未达到预声明5%硬门；A/B仅通过结构/预算门，仍无运行证据"
            )
        },
        "claim_limit": "本报告只证明离线工具可运行及所列解析/回放门结果；不证明论文速度、真实端到端加速或可直接集成。"
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "decision": report["decision"], "routing": routing["aggregate"]}, ensure_ascii=False, indent=2))
    return 0 if offline_feasible else 2


if __name__ == "__main__":
    raise SystemExit(main())
