"""验证 L42 ``wo_a → 官方重定量 → wo_b`` 单请求 Vulkan 链。"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fast16.research.polaris_meridian_v1.l42_real_reference.l42_reference import (
    _InlineForward,
)
from fast16.research.polaris_meridian_v1.s14_range_pack import online_range

from .catalog import read_json, validate_catalog
from .fulldepth_packed_fp8_attention import (
    FullDepthPackedFp8Arena,
    PackedFp8Asset,
    PersistentFullDepthPackedFp8Attention,
    WORKER_ARG,
)
from .preflight import DEFAULT_ASSET_ROOT, DEFAULT_CATALOG
from .verify_l42_attention_replay import _standard_replays, _wo_a_replay


LAYER = 42
WO_A_INPUT_SHA256 = "eee925360c8709263a0cdfa3986c2d3ee91a38c4e4589a7220064b489ad40060"
WO_A_OUTPUT_SHA256 = "2be0aa3b4b67aae58f62a77d2a255d6240b5baf3d71f37c9084fd890741d2eb9"
REQUANTIZED_SHA256 = "94b3f7fd24ee36b8553ed513d1986ef49162c053bd6dbf62f98b9579e20ea3f0"
WO_B_OUTPUT_SHA256 = "84ce63ca9233b07bea99741f9982accac17bc65025b0098b7017acd7dab6db10"


def _sha256_array(value: np.ndarray | torch.Tensor) -> str:
    array = value.contiguous().numpy() if isinstance(value, torch.Tensor) else value
    payload = np.asarray(array, dtype="<f4").tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()


def _asset(cached: online_range.CachedRange) -> PackedFp8Asset:
    entry = cached.entry
    return PackedFp8Asset.from_mapping(
        {
            "tensor": entry["tensor"],
            "path": cached.path,
            "bytes": entry["bytes"],
            "sha256": cached.proof["observed_sha256"],
            "dtype": entry["dtype"],
            "shape": entry["shape"],
        }
    )


def verify_dynamic_attention_output_chain(
    worker_path: Path,
    standard_fixture_dir: Path,
    wo_a_fixture_dir: Path,
    *,
    asset_root: Path = DEFAULT_ASSET_ROOT,
    catalog_path: Path = DEFAULT_CATALOG,
) -> dict[str, Any]:
    worker_path = worker_path.resolve(strict=True)
    asset_root = asset_root.resolve(strict=True)
    catalog = read_json(catalog_path)
    validate_catalog(catalog)
    standards = _standard_replays(standard_fixture_dir.resolve(strict=True))
    wo_a = _wo_a_replay(wo_a_fixture_dir.resolve(strict=True))
    wo_b = standards["layers.42.attn.wo_b"]

    activation = torch.from_numpy(wo_a.input.copy()).reshape(1, 1, 8, 4096).to(
        torch.float32
    )
    requantized = _InlineForward._activation_quant(
        wo_a.output.reshape(1, 1, 8192)
    )
    frozen_boundaries = {
        "wo_a_input_sha256": _sha256_array(activation),
        "wo_a_output_sha256": _sha256_array(wo_a.output),
        "requantized_activation_sha256": _sha256_array(requantized),
        "wo_b_output_sha256": wo_b.output_sha256,
    }
    expected_boundaries = {
        "wo_a_input_sha256": WO_A_INPUT_SHA256,
        "wo_a_output_sha256": WO_A_OUTPUT_SHA256,
        "requantized_activation_sha256": REQUANTIZED_SHA256,
        "wo_b_output_sha256": WO_B_OUTPUT_SHA256,
    }
    if frozen_boundaries != expected_boundaries:
        raise AssertionError(
            f"L42 output-chain fixture 边界漂移: {frozen_boundaries}"
        )
    if _sha256_array(wo_b.input) != REQUANTIZED_SHA256:
        raise AssertionError("wo_b fixture 输入不是 wo_a 官方重定量结果")

    entries = {
        entry["tensor"]: entry
        for entry in catalog["layers"][str(LAYER)]["non_expert"]
    }
    cache = online_range.RangeCache(asset_root / "range_cache", allow_fetch=False)
    assets = {
        suffix: (
            _asset(cache.fetch(entries[f"layers.{LAYER}.attn.{suffix}.weight"])),
            _asset(cache.fetch(entries[f"layers.{LAYER}.attn.{suffix}.scale"])),
        )
        for suffix in ("wo_a", "wo_b")
    }

    started = time.perf_counter()
    runs: list[dict[str, Any]] = []
    hot_separate_seconds: list[float] = []
    hot_chain_seconds: list[float] = []
    with tempfile.TemporaryDirectory(dir=asset_root / "runtime") as directory:
        arena = FullDepthPackedFp8Arena(Path(directory) / "attention.bin", create=True)
        with PersistentFullDepthPackedFp8Attention(
            (str(worker_path), WORKER_ARG),
            arena,
            timeout_seconds=60,
        ) as worker:
            for replay_index in range(2):
                run_started = time.perf_counter()
                output, evidence = worker.execute_output_chain(
                    layer=LAYER,
                    position=replay_index,
                    activation=activation,
                    assets=assets,
                )
                observed = {
                    "wo_a_input_sha256": evidence["input_sha256"],
                    "wo_a_output_sha256": evidence["wo_a_output_sha256"],
                    "requantized_activation_sha256": evidence[
                        "requantized_activation_sha256"
                    ],
                    "wo_b_output_sha256": _sha256_array(output),
                }
                if observed != expected_boundaries:
                    raise AssertionError(
                        f"L42 output-chain 第 {replay_index} 次边界漂移: {observed}"
                    )
                expected_hit = replay_index == 1
                if any(
                    slot["gpu_slot_cache_hit"] is not expected_hit
                    for slot in evidence["slots"]
                ):
                    raise AssertionError("L42 output-chain GPU slot hit 状态漂移")
                if expected_hit and any(
                    slot["payload_uploaded_bytes"] != 0
                    for slot in evidence["slots"]
                ):
                    raise AssertionError("L42 output-chain 热重放仍上传静态 payload")
                runs.append(
                    {
                        "replay_index": replay_index,
                        "arena_epoch": evidence["arena_epoch"],
                        "boundaries": observed,
                        "gpu_slot_cache_hits": [
                            slot["gpu_slot_cache_hit"] for slot in evidence["slots"]
                        ],
                        "payload_uploaded_bytes": [
                            slot["payload_uploaded_bytes"] for slot in evidence["slots"]
                        ],
                        "activation_uploaded_bytes": [
                            slot["activation_uploaded_bytes"] for slot in evidence["slots"]
                        ],
                        "gpu_slot_cache_entries": evidence[
                            "gpu_slot_cache_entries"
                        ],
                        "elapsed_seconds": time.perf_counter() - run_started,
                    }
                )

            wo_b_activation = torch.from_numpy(requantized.copy()).to(torch.float32)
            for _ in range(8):
                separate_started = time.perf_counter()
                separate_wo_a, wo_a_evidence = worker.execute(
                    layer=LAYER,
                    position=0,
                    suffix="wo_a",
                    activation=activation,
                    weight=assets["wo_a"][0],
                    scale=assets["wo_a"][1],
                )
                separate_wo_b, wo_b_evidence = worker.execute(
                    layer=LAYER,
                    position=0,
                    suffix="wo_b",
                    activation=wo_b_activation,
                    weight=assets["wo_b"][0],
                    scale=assets["wo_b"][1],
                )
                hot_separate_seconds.append(time.perf_counter() - separate_started)
                if (
                    _sha256_array(separate_wo_a) != WO_A_OUTPUT_SHA256
                    or _sha256_array(separate_wo_b) != WO_B_OUTPUT_SHA256
                    or not wo_a_evidence["gpu_slot_cache_hit"]
                    or not wo_b_evidence["gpu_slot_cache_hit"]
                    or wo_a_evidence["payload_uploaded_bytes"] != 0
                    or wo_b_evidence["payload_uploaded_bytes"] != 0
                ):
                    raise AssertionError("L42 热独立 wo_a/wo_b 对照漂移")

                chain_started = time.perf_counter()
                chain_output, chain_evidence = worker.execute_output_chain(
                    layer=LAYER,
                    position=0,
                    activation=activation,
                    assets=assets,
                )
                hot_chain_seconds.append(time.perf_counter() - chain_started)
                if (
                    _sha256_array(chain_output) != WO_B_OUTPUT_SHA256
                    or any(
                        not slot["gpu_slot_cache_hit"]
                        or slot["payload_uploaded_bytes"] != 0
                        for slot in chain_evidence["slots"]
                    )
                ):
                    raise AssertionError("L42 热 output-chain 对照漂移")

    if [run["arena_epoch"] for run in runs] != [0, 1]:
        raise AssertionError("output-chain 必须每次整链成功后只推进一个 epoch")
    separate_median = statistics.median(hot_separate_seconds)
    chain_median = statistics.median(hot_chain_seconds)
    return {
        "format": "polaris-fulldepth43-dynamic-attention-output-chain-gate-v1",
        "status": "complete",
        "layer": LAYER,
        "request_count": len(runs),
        "projection_execution_count": len(runs) * 2,
        "frozen_boundaries": expected_boundaries,
        "runs": runs,
        "hot_adjacent_ab": {
            "rounds": len(hot_chain_seconds),
            "separate_request_seconds": hot_separate_seconds,
            "output_chain_seconds": hot_chain_seconds,
            "separate_median_seconds": separate_median,
            "output_chain_median_seconds": chain_median,
            "median_reduction_fraction": (separate_median - chain_median)
            / separate_median,
        },
        "range_proof_cache": cache.proof_cache_telemetry,
        "elapsed_seconds": time.perf_counter() - started,
        "claim_limit": (
            "只证明 L42 wo_a→官方 group-128 E4M3FN 重定量→wo_b 单请求链"
            "命中四个冻结边界并热重放 GPU slot；不单独证明完整 token 提速。"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--standard-fixture-dir", type=Path, required=True)
    parser.add_argument("--wo-a-fixture-dir", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = verify_dynamic_attention_output_chain(
        args.worker,
        args.standard_fixture_dir,
        args.wo_a_fixture_dir,
        asset_root=args.asset_root,
        catalog_path=args.catalog,
    )
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report is not None:
        args.report.write_text(encoded, encoding="utf-8", newline="\n")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
