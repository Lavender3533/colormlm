#!/usr/bin/env python3
"""冻结 43 层真实 router 权重与已有 FFN capture 的只读回放清单。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
RANGE_PACK = ROOT / "fast16/research/polaris_meridian_v1/s14_range_pack"
sys.path.insert(0, str(RANGE_PACK))

import online_range  # noqa: E402

FORMAT = "polaris-fulldepth43-router-replay-manifest-v1"
REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def select_tensor(items: list[dict[str, Any]], name: str) -> dict[str, Any]:
    matches = [item for item in items if item.get("tensor") == name]
    if len(matches) != 1:
        raise RuntimeError(f"{name} 必须唯一，实际 {len(matches)}")
    return matches[0]


def cached_payload(
    cache: online_range.RangeCache, entry: dict[str, Any]
) -> tuple[Path, str]:
    identity = cache._identity(entry)
    _, payload, _, meta_path = cache._paths(identity)
    if not payload.is_file() or payload.stat().st_size != int(entry["bytes"]):
        raise RuntimeError(f"缓存缺失或长度漂移：{entry['tensor']}")
    if not meta_path.is_file():
        raise RuntimeError(f"缓存 proof 缺失：{entry['tensor']}")
    meta = read_json(meta_path)
    expected = meta.get("observed_sha256")
    observed = sha256_file(payload)
    if observed != expected:
        raise RuntimeError(f"缓存 SHA-256 漂移：{entry['tensor']}")
    return payload.resolve(), observed


def build(args: argparse.Namespace) -> dict[str, Any]:
    catalog_path = args.catalog.resolve()
    capture_root = args.capture_root.resolve()
    catalog = read_json(catalog_path)
    profile = catalog.get("profile")
    if (
        catalog.get("revision") != REVISION
        or not isinstance(profile, dict)
        or profile.get("id") != "fulldepth43_native_top6"
    ):
        raise RuntimeError("catalog revision/profile 漂移")
    cache = online_range.RangeCache(args.cache_dir.resolve())
    rows: list[dict[str, Any]] = []

    for layer in range(43):
        layer_catalog = catalog["layers"].get(str(layer))
        if not isinstance(layer_catalog, dict):
            raise RuntimeError(f"catalog 缺少 L{layer}")
        weight_entry = select_tensor(
            layer_catalog["router"], f"layers.{layer}.ffn.gate.weight"
        )
        if weight_entry.get("dtype") != "BF16" or weight_entry.get("shape") != [256, 4096]:
            raise RuntimeError(f"L{layer} router weight ABI 漂移")
        weight_path, weight_sha = cached_payload(cache, weight_entry)

        capture_dir = capture_root / f"layer-{layer:02d}"
        bridge_path = capture_dir / "bridge_manifest.json"
        bridge = read_json(bridge_path)
        if bridge.get("layer") != layer or bridge.get("position") != args.position:
            raise RuntimeError(f"L{layer} bridge layer/position 漂移")
        input_meta = bridge.get("input", {})
        input_path = (capture_dir / str(input_meta.get("file"))).resolve()
        if input_meta.get("shape") != [1, 1, 4096] or input_meta.get("bytes") != 16_384:
            raise RuntimeError(f"L{layer} capture input ABI 漂移")
        input_sha = sha256_file(input_path)
        if input_sha != input_meta.get("f32_le_sha256"):
            raise RuntimeError(f"L{layer} capture input SHA-256 漂移")

        rows.append(
            {
                "layer": layer,
                "weight_path": str(weight_path),
                "weight_bytes": weight_entry["bytes"],
                "weight_sha256": weight_sha,
                "input_path": str(input_path),
                "input_bytes": input_meta["bytes"],
                "input_sha256": input_sha,
                "observed_route_source": bridge.get("route_source"),
                "observed_expert_ids": bridge.get("expert_ids"),
            }
        )

    return {
        "format": FORMAT,
        "revision": REVISION,
        "profile": "fulldepth43_native_top6",
        "position": args.position,
        "catalog_path": str(catalog_path),
        "catalog_sha256": sha256_file(catalog_path),
        "capture_root": str(capture_root),
        "layers": rows,
        "summary": {
            "layer_count": len(rows),
            "router_weight_bytes": sum(row["weight_bytes"] for row in rows),
            "input_bytes": sum(row["input_bytes"] for row in rows),
        },
        "input_semantics": (
            "已有 capture 是 MoE activation-quant 后的 F32 输入，不是 router 原始 RMSNorm BF16 输入；"
            "本清单只允许 GPU/CPU 同输入数值门，observed_expert_ids 仅作观测，不作正式路由晋级。"
        ),
        "claim_limit": (
            "真实固定 revision 的 43 层 router 权重回放清单；不证明正式路由一致、完整 token 或质量。"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("D:/models/Polaris-S14/fulldepth43_native_top6_catalog.json"),
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("D:/models/Polaris-S14/range_cache")
    )
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--position", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
