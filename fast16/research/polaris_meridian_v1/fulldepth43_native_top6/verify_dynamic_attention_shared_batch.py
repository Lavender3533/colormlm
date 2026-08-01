"""验证 L42 两组共享输入 packed-FP8 Vulkan batch 与 GPU slot 重放。"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

from fast16.research.polaris_meridian_v1.s14_range_pack import online_range

from .catalog import read_json, validate_catalog
from .fulldepth_packed_fp8_attention import (
    FullDepthPackedFp8Arena,
    PackedFp8Asset,
    PersistentFullDepthPackedFp8Attention,
    WORKER_ARG,
)
from .preflight import DEFAULT_ASSET_ROOT, DEFAULT_CATALOG
from .verify_l42_attention_replay import _standard_replays


LAYER = 42
BATCHES = (
    ("wq_a", "wkv"),
    ("wq_b", "indexer.wq_b"),
)


def _sha256_tensor(tensor: torch.Tensor) -> str:
    payload = tensor.contiguous().numpy().astype("<f4").tobytes()
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


def verify_dynamic_attention_shared_batch(
    worker_path: Path,
    standard_fixture_dir: Path,
    *,
    asset_root: Path = DEFAULT_ASSET_ROOT,
    catalog_path: Path = DEFAULT_CATALOG,
) -> dict[str, Any]:
    worker_path = worker_path.resolve(strict=True)
    asset_root = asset_root.resolve(strict=True)
    catalog = read_json(catalog_path)
    validate_catalog(catalog)
    records = _standard_replays(standard_fixture_dir.resolve(strict=True))
    entries = {
        entry["tensor"]: entry
        for entry in catalog["layers"][str(LAYER)]["non_expert"]
    }
    cache = online_range.RangeCache(asset_root / "range_cache", allow_fetch=False)

    activations: dict[tuple[str, str], torch.Tensor] = {}
    assets: dict[
        tuple[str, str], dict[str, tuple[PackedFp8Asset, PackedFp8Asset]]
    ] = {}
    for suffixes in BATCHES:
        first_name = f"layers.{LAYER}.attn.{suffixes[0]}"
        second_name = f"layers.{LAYER}.attn.{suffixes[1]}"
        first_record = records[first_name]
        second_record = records[second_name]
        if (
            first_record.input.shape != second_record.input.shape
            or not bool((first_record.input == second_record.input).all())
        ):
            raise AssertionError(f"{suffixes} fixture 不是严格共享输入")
        activations[suffixes] = torch.from_numpy(first_record.input.copy()).reshape(
            1, 1, first_record.input.size
        ).to(torch.float32)
        assets[suffixes] = {
            suffix: (
                _asset(cache.fetch(entries[f"layers.{LAYER}.attn.{suffix}.weight"])),
                _asset(cache.fetch(entries[f"layers.{LAYER}.attn.{suffix}.scale"])),
            )
            for suffix in suffixes
        }

    started = time.perf_counter()
    runs: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(dir=asset_root / "runtime") as directory:
        arena = FullDepthPackedFp8Arena(Path(directory) / "attention.bin", create=True)
        with PersistentFullDepthPackedFp8Attention(
            (str(worker_path), WORKER_ARG),
            arena,
            timeout_seconds=60,
        ) as worker:
            for replay_index in range(2):
                for suffixes in BATCHES:
                    run_started = time.perf_counter()
                    outputs, evidence = worker.execute_shared_batch(
                        layer=LAYER,
                        position=replay_index,
                        suffixes=suffixes,
                        activation=activations[suffixes],
                        assets=assets[suffixes],
                    )
                    output_evidence = evidence["outputs"]
                    observed: list[dict[str, Any]] = []
                    for suffix, item in zip(suffixes, output_evidence, strict=True):
                        name = f"layers.{LAYER}.attn.{suffix}"
                        output_sha256 = _sha256_tensor(outputs[suffix])
                        if output_sha256 != records[name].output_sha256:
                            raise AssertionError(
                                f"{name} batch 输出漂移: expected="
                                f"{records[name].output_sha256}, actual={output_sha256}"
                            )
                        expected_hit = replay_index == 1
                        if item["gpu_slot_cache_hit"] is not expected_hit:
                            raise AssertionError(f"{name} GPU slot hit 状态漂移")
                        if expected_hit and item["payload_uploaded_bytes"] != 0:
                            raise AssertionError(f"{name} 重放仍上传静态 payload")
                        observed.append(
                            {
                                "projection": name,
                                "output_sha256": output_sha256,
                                "gpu_slot_cache_hit": item["gpu_slot_cache_hit"],
                                "payload_uploaded_bytes": item["payload_uploaded_bytes"],
                            }
                        )
                    runs.append(
                        {
                            "replay_index": replay_index,
                            "arena_epoch": evidence["arena_epoch"],
                            "suffixes": list(suffixes),
                            "input_sha256": evidence["input_sha256"],
                            "activation_uploaded_bytes": evidence[
                                "activation_uploaded_bytes"
                            ],
                            "gpu_slot_cache_entries": evidence[
                                "gpu_slot_cache_entries"
                            ],
                            "outputs": observed,
                            "elapsed_seconds": time.perf_counter() - run_started,
                        }
                    )

    if [run["arena_epoch"] for run in runs] != [0, 1, 2, 3]:
        raise AssertionError("共享 batch 必须整批成功后只推进一个 epoch")
    return {
        "format": "polaris-fulldepth43-dynamic-attention-shared-batch-gate-v1",
        "status": "complete",
        "layer": LAYER,
        "batch_count": len(runs),
        "projection_execution_count": sum(len(run["outputs"]) for run in runs),
        "runs": runs,
        "range_proof_cache": cache.proof_cache_telemetry,
        "elapsed_seconds": time.perf_counter() - started,
        "claim_limit": (
            "只证明 L42 两组共享输入双投影 batch 命中冻结输出 SHA，"
            "且同进程重放命中 GPU slot；不证明完整 token 提速。"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--standard-fixture-dir", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = verify_dynamic_attention_shared_batch(
        args.worker,
        args.standard_fixture_dir,
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
