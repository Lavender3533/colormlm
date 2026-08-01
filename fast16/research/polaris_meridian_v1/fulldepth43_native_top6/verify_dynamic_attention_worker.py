"""用同一通用 worker 连续闭合 L42 六条 packed-FP8 attention 投影。"""

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
from .verify_l42_attention_replay import _standard_replays, _wo_a_replay


ORDER = (
    "layers.42.attn.wq_a",
    "layers.42.attn.wq_b",
    "layers.42.attn.wkv",
    "layers.42.attn.indexer.wq_b",
    "layers.42.attn.wo_a",
    "layers.42.attn.wo_b",
)


def _sha256_tensor(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().numpy().astype("<f4").tobytes()).hexdigest()


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


def verify_dynamic_attention_worker(
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
    records = _standard_replays(standard_fixture_dir.resolve(strict=True))
    records["layers.42.attn.wo_a"] = _wo_a_replay(wo_a_fixture_dir.resolve(strict=True))
    entries = {
        entry["tensor"]: entry
        for entry in catalog["layers"]["42"]["non_expert"]
    }
    cache = online_range.RangeCache(asset_root / "range_cache", allow_fetch=False)

    started = time.perf_counter()
    projections: list[dict[str, Any]] = []
    cache_replay: dict[str, Any] | None = None
    with tempfile.TemporaryDirectory(dir=asset_root / "runtime") as directory:
        arena = FullDepthPackedFp8Arena(Path(directory) / "attention.bin", create=True)
        with PersistentFullDepthPackedFp8Attention(
            (str(worker_path), WORKER_ARG),
            arena,
            timeout_seconds=60,
        ) as worker:
            for name in ORDER:
                record = records[name]
                suffix = name.removeprefix("layers.42.attn.")
                shape = (1, 1, 8, 4096) if suffix == "wo_a" else (1, 1, record.input.size)
                activation = torch.from_numpy(record.input.copy()).reshape(shape).to(torch.float32)
                weight = _asset(cache.fetch(entries[name + ".weight"]))
                scale = _asset(cache.fetch(entries[name + ".scale"]))
                projection_started = time.perf_counter()
                output, evidence = worker.execute(
                    layer=42,
                    position=0,
                    suffix=suffix,
                    activation=activation,
                    weight=weight,
                    scale=scale,
                )
                output_sha256 = _sha256_tensor(output)
                if output_sha256 != record.output_sha256:
                    raise AssertionError(
                        f"{name} 动态 worker 输出漂移: "
                        f"expected={record.output_sha256}, actual={output_sha256}"
                    )
                projections.append(
                    {
                        "projection": name,
                        "arena_epoch": evidence["arena_epoch"],
                        "input_sha256": evidence["input_sha256"],
                        "output_sha256": output_sha256,
                        "elapsed_seconds": time.perf_counter() - projection_started,
                    }
                )
            replay_name = ORDER[0]
            replay_record = records[replay_name]
            replay_activation = torch.from_numpy(replay_record.input.copy()).reshape(
                1, 1, replay_record.input.size
            ).to(torch.float32)
            replay_output, replay_evidence = worker.execute(
                layer=42,
                position=1,
                suffix=replay_name.removeprefix("layers.42.attn."),
                activation=replay_activation,
                weight=_asset(cache.fetch(entries[replay_name + ".weight"])),
                scale=_asset(cache.fetch(entries[replay_name + ".scale"])),
            )
            replay_sha256 = _sha256_tensor(replay_output)
            if (
                replay_sha256 != replay_record.output_sha256
                or replay_evidence["gpu_slot_cache_hit"] is not True
                or replay_evidence["payload_uploaded_bytes"] != 0
            ):
                raise AssertionError("L42 wq_a 真实 GPU slot 复用门失败")
            cache_replay = {
                "projection": replay_name,
                "arena_epoch": replay_evidence["arena_epoch"],
                "output_sha256": replay_sha256,
                "gpu_slot_cache_hit": True,
                "payload_uploaded_bytes": 0,
                "gpu_slot_cache_entries": replay_evidence["gpu_slot_cache_entries"],
                "gpu_slot_resident_bytes": replay_evidence["gpu_slot_resident_bytes"],
            }
    if [row["arena_epoch"] for row in projections] != list(range(6)):
        raise AssertionError("动态 worker arena epoch 不连续")
    return {
        "format": "polaris-fulldepth43-dynamic-attention-worker-gate-v1",
        "status": "complete",
        "layer": 42,
        "projection_count": len(projections),
        "request_count": len(projections) + 1,
        "projections": projections,
        "gpu_slot_cache_replay": cache_replay,
        "range_proof_cache": cache.proof_cache_telemetry,
        "elapsed_seconds": time.perf_counter() - started,
        "claim_limit": "只证明同一通用 worker 连续执行六条 L42 attention 投影并命中冻结 SHA。",
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
    report = verify_dynamic_attention_worker(
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
