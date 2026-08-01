"""Fail-closed loader for measured timing anchors and real S14 byte metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from ..assets import S14_LAYERS


class AnchorContractError(ValueError):
    """Raised when a source report or frozen measurement drifts."""


@dataclass(frozen=True)
class StageAnchors:
    model_revision: str
    layers: tuple[int, ...]
    cpu_warm_token_ms: float
    cpu_report_sha256: str
    two_token_report_sha256: str
    route_catalog_sha256: str
    gpu_evidence_sha256: str
    top6_evidence_sha256: str
    device: str
    single_expert_chain_ms: float
    single_expert_dispatches: int
    single_expert_iterations: int
    top6_routed_plus_shared_ms: float
    top6_dispatches: int
    top6_iterations: int
    wq_a_linear_ms: float
    wq_a_dispatches: int
    wq_a_iterations: int
    expert_page_bytes: int
    routed_bytes_per_layer: int
    shared_bytes_per_layer: int
    attention_weight_bytes_s14: int
    hc_weight_bytes_s14: int
    router_weight_bytes_s14: int
    shared_weight_bytes_s14: int
    final_norm_head_bytes: int
    two_token_previous_overlap_pages: int
    two_token_current_pages: int
    two_token_new_bytes: int
    two_token_expert_miss_bytes: int
    two_token_intersection_by_layer: tuple[tuple[int, int], ...]

    @property
    def layer_count(self) -> int:
        return len(self.layers)

    @property
    def six_routed_envelope_ms_per_layer(self) -> float:
        return 6 * self.single_expert_chain_ms

    @property
    def shared_and_batch_residual_ms_per_layer(self) -> float:
        return self.top6_routed_plus_shared_ms - self.six_routed_envelope_ms_per_layer

    @property
    def s14_current_moe_anchor_ms(self) -> float:
        return self.layer_count * self.top6_routed_plus_shared_ms

    @property
    def s14_wq_a_floor_ms(self) -> float:
        return self.layer_count * self.wq_a_linear_ms

    @property
    def s14_current_known_gpu_anchor_ms(self) -> float:
        return self.s14_current_moe_anchor_ms + self.s14_wq_a_floor_ms

    @property
    def s14_expert_page_count(self) -> int:
        return self.layer_count * 256

    @property
    def s14_full_expert_bank_bytes(self) -> int:
        return self.s14_expert_page_count * self.expert_page_bytes

    @property
    def s14_fixed_weight_bytes(self) -> int:
        return (
            self.attention_weight_bytes_s14
            + self.hc_weight_bytes_s14
            + self.router_weight_bytes_s14
            + self.shared_weight_bytes_s14
            + self.final_norm_head_bytes
        )

    @property
    def s14_profile_bytes(self) -> int:
        return self.s14_fixed_weight_bytes + self.s14_full_expert_bank_bytes

    @property
    def two_token_previous_only_hit_rate(self) -> float:
        return self.two_token_previous_overlap_pages / self.two_token_current_pages

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.update(
            {
                "layers": list(self.layers),
                "layer_count": self.layer_count,
                "six_routed_envelope_ms_per_layer": self.six_routed_envelope_ms_per_layer,
                "shared_and_batch_residual_ms_per_layer": self.shared_and_batch_residual_ms_per_layer,
                "s14_current_moe_anchor_ms": self.s14_current_moe_anchor_ms,
                "s14_wq_a_floor_ms": self.s14_wq_a_floor_ms,
                "s14_current_known_gpu_anchor_ms": self.s14_current_known_gpu_anchor_ms,
                "s14_expert_page_count": self.s14_expert_page_count,
                "s14_full_expert_bank_bytes": self.s14_full_expert_bank_bytes,
                "s14_fixed_weight_bytes": self.s14_fixed_weight_bytes,
                "s14_profile_bytes": self.s14_profile_bytes,
                "two_token_previous_only_hit_rate": self.two_token_previous_only_hit_rate,
                "two_token_intersection_by_layer": {
                    str(layer): count for layer, count in self.two_token_intersection_by_layer
                },
            }
        )
        return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8", errors="strict"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnchorContractError(f"无法读取 {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AnchorContractError(f"{path} 顶层必须是 object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise AnchorContractError(message)


def _sum_ranges(rows: list[dict[str, Any]]) -> int:
    return sum(int(row["bytes"]) for row in rows)


def load_real_anchors(
    asset_root: str | Path = "D:/models/Polaris-S14",
    repo_root: str | Path | None = None,
    contract_path: str | Path | None = None,
) -> StageAnchors:
    """Load real reports and enumerate the S14 route catalog.

    The delegated 1.3696 ms top-6 measurement is frozen in the tracked contract
    because its v3 evidence was concurrent work at task start.  The committed
    v2 file is still verified for the 0.157038/0.0838004 ms anchors.
    """

    package_dir = Path(__file__).resolve().parent
    contract_file = Path(contract_path) if contract_path else package_dir / "frozen_measurements.json"
    contract = _read_json(contract_file)
    root = Path(asset_root)
    if repo_root is None:
        repo = package_dir.parents[4]
    else:
        repo = Path(repo_root)

    _expect(contract.get("format") == "polaris-s14-gpu-stage-measurements-v1", "measurement contract format 漂移")
    layers = tuple(contract.get("s14_layers", ()))
    _expect(layers == S14_LAYERS, "measurement contract S14 层集漂移")

    cpu_contract = contract["cpu_warm_token"]
    cpu_path = root / cpu_contract["file"]
    _expect(_sha256(cpu_path) == cpu_contract["sha256"], "CPU warm report SHA 漂移")
    cpu_report = _read_json(cpu_path)
    _expect(cpu_report.get("format") == "polaris-s14-first-real-token-report-v1", "CPU report format 漂移")
    _expect(cpu_report.get("status") == "complete", "CPU warm token 未完成")
    _expect(tuple(cpu_report.get("registered_layers", ())) == layers, "CPU report S14 层集漂移")
    _expect(cpu_report.get("elapsed_s") == cpu_contract["elapsed_s"], "CPU warm elapsed 漂移")
    _expect(cpu_report.get("downloaded_bytes") == 0, "CPU warm report 不再是纯热缓存")

    two_contract = contract["two_token_cold_counterexample"]
    two_path = root / two_contract["file"]
    _expect(_sha256(two_path) == two_contract["sha256"], "two-token report SHA 漂移")
    two_report = _read_json(two_path)
    _expect(two_report.get("format") == two_contract["format"], "two-token report format 漂移")
    _expect(two_report.get("status") == "complete", "two-token report 未完成")
    _expect(tuple(two_report.get("registered_layers", ())) == layers, "two-token S14 层集漂移")
    committed = two_report.get("committed_tokens", ())
    _expect(len(committed) == two_contract["committed_tokens"], "two-token committed_tokens 漂移")
    tokens = two_report.get("tokens", ())
    _expect(len(tokens) == 2 and tokens[1].get("status") == "complete", "token1 未完成")
    _expect(tokens[1].get("final", {}).get("token_id") == two_contract["token1_output_token_id"], "token1 final ID 漂移")
    _expect(tokens[1].get("final", {}).get("text") == two_contract["token1_text"], "token1 final text 漂移")
    _expect(tokens[1].get("downloaded_bytes_new") == two_contract["token1_new_bytes"], "token1 新增字节漂移")
    token_routes = [
        {int(row["layer"]): set(int(value) for value in row["expert_ids"]) for row in token["layers"]}
        for token in tokens
    ]
    intersections = tuple(
        (layer, len(token_routes[0][layer] & token_routes[1][layer])) for layer in layers
    )
    expected_intersections = tuple(
        (layer, int(two_contract["intersection_by_layer"][str(layer)])) for layer in layers
    )
    _expect(intersections == expected_intersections, "two-token 逐层 top6 交集漂移")
    _expect(sum(count for _, count in intersections) == two_contract["previous_token_expert_intersection_pages"], "two-token 交集总数漂移")

    gpu_contract = contract["committed_rx5700xt_evidence"]
    gpu_path = repo / gpu_contract["file"]
    _expect(_sha256(gpu_path) == gpu_contract["sha256"], "committed RX5700XT evidence SHA 漂移")
    gpu_report = _read_json(gpu_path)
    _expect(gpu_report.get("device", {}).get("name") == gpu_contract["device"], "GPU 型号漂移")
    expert = gpu_report.get("mxfp4_expert_126_chain", {})
    wq_a = gpu_report.get("fp8_wq_a_linear", {})
    _expect(expert.get("gpu_chain_dispatch_plus_barriers_ms_mean") == gpu_contract["single_expert_chain_ms"], "single expert timing 漂移")
    _expect(len(expert.get("dispatch_sequence", ())) == gpu_contract["single_expert_dispatches"], "single expert dispatch 漂移")
    _expect(expert.get("timestamp_iterations") == gpu_contract["single_expert_iterations"], "single expert iterations 漂移")
    _expect(wq_a.get("gpu_kernel_plus_serial_barrier_ms_mean") == gpu_contract["wq_a_linear_ms"], "wq_a timing 漂移")
    _expect(wq_a.get("timestamp_iterations") == gpu_contract["wq_a_iterations"], "wq_a iterations 漂移")

    top6 = contract["delegated_current_top6_shared_evidence"]
    _expect(top6["dispatches"] == 35, "top6+shared dispatch 必须是 35")
    _expect(top6["timestamp_iterations"] == 1, "top6+shared 当前应标记为单次 timestamp")
    _expect(top6["routed_payload_bytes"] > 0, "top6 routed bytes 非法")

    catalog_path = root / "route_first_catalog.json"
    catalog = _read_json(catalog_path)
    _expect(tuple(catalog.get("selected_layers", ())) == layers, "route catalog S14 层集漂移")
    _expect(catalog.get("top_k") == 6, "route catalog top_k 不是 6")
    attention_bytes = hc_bytes = router_bytes = shared_bytes = 0
    expert_page_sizes: set[int] = set()
    for layer in layers:
        row = catalog["layers"][str(layer)]
        router_bytes += _sum_ranges(row["router"])
        shared_bytes += _sum_ranges(row["shared"])
        for tensor in row["non_expert"]:
            name = str(tensor["tensor"])
            if ".hc_attn_" in name or ".hc_ffn_" in name:
                hc_bytes += int(tensor["bytes"])
            else:
                attention_bytes += int(tensor["bytes"])
        for expert_id in range(256):
            expert_page_sizes.add(_sum_ranges(row["experts"][str(expert_id)]))
    _expect(len(expert_page_sizes) == 1, "S14 专家页字节不统一")
    page_bytes = next(iter(expert_page_sizes))
    routed_bytes = 6 * page_bytes
    _expect(routed_bytes == top6["routed_payload_bytes"], "delegated top6 bytes 与 route catalog 不符")
    shared_per_layer = shared_bytes // len(layers)
    _expect(shared_per_layer * len(layers) == shared_bytes, "shared bytes 每层不一致")

    # All retained layers have the exact L42 wq_a shape used by the GPU anchor.
    for layer in layers:
        names = {row["tensor"]: row for row in catalog["layers"][str(layer)]["non_expert"]}
        weight = names.get(f"layers.{layer}.attn.wq_a.weight")
        scale = names.get(f"layers.{layer}.attn.wq_a.scale")
        _expect(weight is not None and weight.get("shape") == [1024, 4096], f"L{layer} wq_a weight shape 漂移")
        _expect(scale is not None and scale.get("shape") == [8, 32], f"L{layer} wq_a scale shape 漂移")

    final_bytes = _sum_ranges(catalog["boundary"]["final"])
    anchors = StageAnchors(
        model_revision=contract["model_revision"],
        layers=layers,
        cpu_warm_token_ms=1000.0 * float(cpu_contract["elapsed_s"]),
        cpu_report_sha256=cpu_contract["sha256"],
        two_token_report_sha256=two_contract["sha256"],
        route_catalog_sha256=_sha256(catalog_path),
        gpu_evidence_sha256=gpu_contract["sha256"],
        top6_evidence_sha256=top6["concurrent_v3_evidence_sha256"],
        device=gpu_contract["device"],
        single_expert_chain_ms=float(gpu_contract["single_expert_chain_ms"]),
        single_expert_dispatches=int(gpu_contract["single_expert_dispatches"]),
        single_expert_iterations=int(gpu_contract["single_expert_iterations"]),
        top6_routed_plus_shared_ms=float(top6["top6_routed_plus_shared_ms"]),
        top6_dispatches=int(top6["dispatches"]),
        top6_iterations=int(top6["timestamp_iterations"]),
        wq_a_linear_ms=float(gpu_contract["wq_a_linear_ms"]),
        wq_a_dispatches=int(gpu_contract["wq_a_dispatches"]),
        wq_a_iterations=int(gpu_contract["wq_a_iterations"]),
        expert_page_bytes=page_bytes,
        routed_bytes_per_layer=routed_bytes,
        shared_bytes_per_layer=shared_per_layer,
        attention_weight_bytes_s14=attention_bytes,
        hc_weight_bytes_s14=hc_bytes,
        router_weight_bytes_s14=router_bytes,
        shared_weight_bytes_s14=shared_bytes,
        final_norm_head_bytes=final_bytes,
        two_token_previous_overlap_pages=int(two_contract["previous_token_expert_intersection_pages"]),
        two_token_current_pages=int(two_contract["current_token_expert_pages"]),
        two_token_new_bytes=int(two_contract["token1_new_bytes"]),
        two_token_expert_miss_bytes=(
            int(two_contract["current_token_expert_pages"])
            - int(two_contract["previous_token_expert_intersection_pages"])
        )
        * page_bytes,
        two_token_intersection_by_layer=intersections,
    )
    _expect(anchors.shared_and_batch_residual_ms_per_layer > 0, "top6 实测小于 6x single expert envelope")
    _expect(anchors.two_token_expert_miss_bytes + 8192 == anchors.two_token_new_bytes, "token1 新增字节不是 76 专家页 + 1 embedding row")
    return anchors
