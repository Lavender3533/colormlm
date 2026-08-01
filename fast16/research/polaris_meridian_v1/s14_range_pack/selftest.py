#!/usr/bin/env python3
"""纯 CPU/合成 metadata 自检；不访问网络、不下载权重。"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import range_pack as rp


HERE = Path(__file__).resolve().parent


def expect_error(fn, text: str) -> None:
    try:
        fn()
    except rp.ContractError as exc:
        if text not in str(exc):
            raise AssertionError(f"异常未包含 {text!r}: {exc}") from exc
    else:
        raise AssertionError(f"预期 ContractError({text})")


def synthetic_metadata(source: dict) -> tuple[dict[str, str], dict[str, dict], dict[int, set[int]]]:
    weight_map: dict[str, str] = {}
    headers: dict[str, dict] = {}
    routes = {layer: {1} for layer in source["selected_layers"]}
    file_rows = rp.source_file_contracts(source)

    def add(name: str, filename: str, begin: int, size: int) -> None:
        weight_map[name] = filename
        row = headers.setdefault(filename, {
            "format": "polaris-safetensors-header-v1",
            "repo": rp.REPO,
            "revision": rp.REVISION,
            "file": filename,
            "file_bytes": int(file_rows[filename]["bytes"]),
            "header_length": 4096,
            "data_start": 4104,
            "tensors": {},
        })
        row["tensors"][name] = {"dtype": "F16", "shape": [size // 2], "data_offsets": [begin, begin + size]}

    add("embed.weight", source["boundary_shards"]["embed.weight"]["file"], 0, 130)
    add("norm.weight", source["boundary_shards"]["norm.weight"]["file"], 0, 18)
    add("head.weight", source["boundary_shards"]["head.weight"]["file"], 64, 194)
    add("hc_head_base", source["boundary_shards"]["hc_head_base"]["file"], 320, 16)
    add("hc_head_fn", source["boundary_shards"]["hc_head_fn"]["file"], 384, 128)
    add("hc_head_scale", source["boundary_shards"]["hc_head_scale"]["file"], 576, 4)
    for layer in source["selected_layers"]:
        filename = source["layer_shards"][str(layer)]["file"]
        add(f"layers.{layer}.attn.q_proj.weight", filename, 0, 66)
        add(f"layers.{layer}.ffn.gate.weight", filename, 128, 34)
        add(f"layers.{layer}.ffn.shared_experts.w1.weight", filename, 192, 70)
        add(f"layers.{layer}.ffn.experts.0.w1.weight", filename, 320, 74)
        add(f"layers.{layer}.ffn.experts.1.w1.weight", filename, 448, 78)
    return weight_map, headers, routes


def main() -> int:
    source = rp.read_json(rp.SOURCE_CONTRACT)
    rp.validate_source_contract(source)
    checks: list[dict[str, str]] = []

    budget = rp.make_environment_budget(source, overlay_gib=50.0, ram_gib=1536.0, hbm_gib=32.0)
    assert budget["package_upper_bound_bytes"] == 52_231_273_716
    assert budget["overlay"]["raw_fit"] is True
    assert budget["overlay"]["safe_fit"] is False
    assert budget["overlay"]["decision"] == "reject_overlay_output"
    assert budget["ram_stream"]["full_package_fit"] is True
    checks.append({"id": "fixed_budget", "status": "pass"})

    rejected = rp.read_json(HERE / "route_trace.example.rejected.json")
    expect_error(lambda: rp.validate_route_trace(rejected, source["selected_layers"]), "status")
    checks.append({"id": "reject_missing_native_route_trace", "status": "pass"})

    approved = {
        "format": "polaris-s14-route-trace-v1",
        "repo": rp.REPO,
        "revision": rp.REVISION,
        "status": "approved_native_trace",
        "coverage_complete": True,
        "top_k": 6,
        "observed_tokens": 32,
        "task_groups": 4,
        "capture_manifest_sha256": "c" * 64,
        "layers": {str(layer): {"expert_ids": [1], "events": 6} for layer in source["selected_layers"]},
    }
    routes = rp.validate_route_trace(approved, source["selected_layers"])
    assert routes == {layer: {1} for layer in source["selected_layers"]}
    checks.append({"id": "accept_complete_native_route_trace", "status": "pass"})

    weight_map, headers, routes = synthetic_metadata(source)
    for header in headers.values():
        header["tensor_table_sha256"] = rp.sha256_bytes(
            json.dumps(header["tensors"], sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
    for filename, header in headers.items():
        rp.validate_header(header, filename, rp.source_file_contracts(source)[filename]["bytes"])
    plan = rp.make_plan(source, weight_map, headers, routes)
    tensors = {row["tensor"] for row in plan["entries"]}
    assert all(f"layers.{layer}.ffn.experts.1.w1.weight" in tensors for layer in source["selected_layers"])
    assert all(f"layers.{layer}.ffn.experts.0.w1.weight" not in tensors for layer in source["selected_layers"])
    assert {
        "embed.weight",
        "hc_head_base",
        "hc_head_fn",
        "hc_head_scale",
        "norm.weight",
        "head.weight",
    } <= tensors
    assert plan["status"] == "blocked_missing_integrity_locks"
    assert plan["layout"]["binary_bytes"] == plan["layout"]["payload_bytes"] + plan["layout"]["padding_bytes"]
    checks.append({"id": "exact_tensor_selection_and_layout", "status": "pass"})

    locks = {row["range_key"]: "a" * 64 for row in plan["entries"]}
    asset_locks = {name: {"bytes": 1, "sha256": "b" * 64} for name in source["tokenizer_required"]}
    ready = rp.make_plan(source, weight_map, headers, routes, locks, asset_locks)
    assert ready["status"] == "ready_to_materialize"
    assert ready["integrity"]["range_sha256_missing"] == 0
    checks.append({"id": "sha_lock_gate", "status": "pass"})

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        source_file = tmp_path / "fixture.bin"
        source_file.write_bytes(bytes(range(251)) * 100)
        # 证明用于恢复账本的 hash 在重复读取中稳定；不模拟网络，也不留下 payload。
        first = rp.sha256_file(source_file)
        second = rp.sha256_file(source_file)
        assert first == second and len(first) == 64
    checks.append({"id": "resume_hash_stability", "status": "pass"})

    report = {
        "format": "polaris-s14-range-pack-selftest-v1",
        "status": "pass",
        "network_accessed": False,
        "weights_downloaded": False,
        "model_started": False,
        "checks": checks,
        "claim_limit": "合成 metadata 自检不证明真实 DeepSeek header、route trace、runtime 或质量。",
    }
    rp.write_json(HERE / "selftest_report.json", report, force=True)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
