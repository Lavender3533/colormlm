"""FullDepth43/native-top6 离线 payload/cache preflight。"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from fast16.research.polaris_meridian_v1.s14_range_pack import online_range

from .catalog import build_catalog, read_json, validate_catalog, write_json
from .profile import FULLDEPTH43_NATIVE_TOP6, ExecutionProfile


FORMAT = "polaris-fulldepth43-native-top6-preflight-v1"
DEFAULT_ASSET_ROOT = Path("D:/models/Polaris-S14")
DEFAULT_CATALOG = DEFAULT_ASSET_ROOT / "fulldepth43_native_top6_catalog.json"
DEFAULT_REPORT = Path(__file__).resolve().parent / "preflight_report.json"


class PreflightError(RuntimeError):
    pass


def _static_entries(catalog: Mapping[str, Any], token_id: int) -> list[dict[str, Any]]:
    entries = [online_range.embedding_row_entry(catalog, token_id)]
    entries.extend(catalog["boundary"]["final"])
    for layer in catalog["profile"]["layers"]:
        row = catalog["layers"][str(layer)]
        entries.extend(row["non_expert"])
        entries.extend(row["router"])
        entries.extend(row["shared"])
    return entries


def _cache_candidate(cache: online_range.RangeCache, entry: Mapping[str, Any]) -> dict[str, Any]:
    identity = cache._identity(entry)
    key, payload, partial, meta_path = cache._paths(identity)
    result: dict[str, Any] = {
        "tensor": entry["tensor"],
        "layer": entry.get("layer"),
        "kind": entry["kind"],
        "range_key": entry["range_key"],
        "bytes": int(entry["bytes"]),
        "cache_key": key,
        "payload": str(payload),
        "metadata": str(meta_path),
        "candidate_ready": False,
        "reason": "missing_payload",
    }
    if partial.exists():
        result["partial_bytes"] = partial.stat().st_size
    if not payload.is_file():
        return result
    if payload.stat().st_size != int(entry["bytes"]):
        result["reason"] = "payload_size_mismatch"
        return result
    if not meta_path.is_file():
        result["reason"] = "missing_metadata"
        return result
    try:
        meta = read_json(meta_path)
    except Exception as error:
        result["reason"] = f"invalid_metadata:{type(error).__name__}"
        return result
    observed = meta.get("observed_sha256")
    if (
        meta.get("format") != online_range.CACHE_META_FORMAT
        or meta.get("cache_key") != key
        or meta.get("identity") != identity
        or meta.get("bytes") != int(entry["bytes"])
        or not isinstance(observed, str)
        or len(observed) != 64
    ):
        result["reason"] = "metadata_contract_mismatch"
        return result
    result.update(
        candidate_ready=True,
        reason="metadata_and_size_match_payload_rehash_deferred_to_executor",
        hash_authority=meta.get("hash_authority"),
        authoritative=meta.get("authoritative") is True,
        observed_sha256=observed,
    )
    return result


def _dynamic_budget(catalog: Mapping[str, Any], profile: ExecutionProfile) -> dict[str, Any]:
    per_layer: list[dict[str, Any]] = []
    uniform = True
    total = 0
    for layer in profile.layers:
        experts = catalog["layers"][str(layer)]["experts"]
        sizes = [sum(int(item["bytes"]) for item in experts[str(expert)]) for expert in range(256)]
        layer_uniform = len(set(sizes)) == 1
        uniform = uniform and layer_uniform
        if not layer_uniform:
            bytes_for_top6 = sum(sorted(sizes, reverse=True)[: profile.top_k])
        else:
            bytes_for_top6 = sizes[0] * profile.top_k
        total += bytes_for_top6
        per_layer.append(
            {
                "layer": layer,
                "expert_payload_bytes_min": min(sizes),
                "expert_payload_bytes_max": max(sizes),
                "expert_schema_uniform": layer_uniform,
                "native_top6_cold_bytes": bytes_for_top6,
            }
        )
    return {
        "route_known_pre_attention": False,
        "expert_payload_size_uniform_within_each_layer": uniform,
        "native_top6_cold_bytes_per_token": total,
        "per_layer": per_layer,
        "policy": "only current-layer native route may select six expert pages",
    }


def run_preflight(
    *,
    asset_root: Path,
    catalog: Mapping[str, Any],
    token_id: int = 0,
    profile: ExecutionProfile = FULLDEPTH43_NATIVE_TOP6,
) -> dict[str, Any]:
    profile.validate()
    validate_catalog(catalog)
    if isinstance(token_id, bool) or not isinstance(token_id, int) or not 0 <= token_id < 129_280:
        raise PreflightError("token_id 必须在 0..129279")
    root = asset_root.resolve()
    cache = online_range.RangeCache(root / "range_cache", allow_fetch=False, download_budget_bytes=0)
    rows = [_cache_candidate(cache, entry) for entry in _static_entries(catalog, token_id)]
    missing = [row for row in rows if not row["candidate_ready"]]
    ready = [row for row in rows if row["candidate_ready"]]
    missing_by_layer: dict[str, dict[str, int]] = defaultdict(lambda: {"ranges": 0, "bytes": 0})
    for row in missing:
        key = "boundary" if row["layer"] is None else str(row["layer"])
        missing_by_layer[key]["ranges"] += 1
        missing_by_layer[key]["bytes"] += int(row["bytes"])
    dynamic = _dynamic_budget(catalog, profile)
    usage = shutil.disk_usage(root)
    static_missing_bytes = sum(int(row["bytes"]) for row in missing)
    cold_upper = static_missing_bytes + int(dynamic["native_top6_cold_bytes_per_token"])
    blockers = []
    if missing:
        blockers.append("static_prerequisite_ranges_missing")
    if not dynamic["expert_payload_size_uniform_within_each_layer"]:
        blockers.append("expert_payload_sizes_not_uniform_budget_is_worst_case")
    if usage.free < cold_upper:
        blockers.append("insufficient_free_storage_for_cold_upper_bound")
    return {
        "format": FORMAT,
        "status": "ready" if not blockers else "blocked",
        "repo": profile.repo,
        "revision": profile.revision,
        "profile": profile.as_dict(),
        "token_id_for_embedding_row": token_id,
        "download_authorized": False,
        "catalog": {
            "format": catalog["format"],
            "range_count": catalog["summary"]["range_count"],
            "range_bytes": catalog["summary"]["range_bytes"],
            "headers": catalog["headers"]["count"],
        },
        "static_prerequisites": {
            "range_count": len(rows),
            "bytes": sum(int(row["bytes"]) for row in rows),
            "candidate_ready_ranges": len(ready),
            "candidate_ready_bytes": sum(int(row["bytes"]) for row in ready),
            "missing_ranges": len(missing),
            "missing_bytes": static_missing_bytes,
            "missing_by_layer": dict(sorted(missing_by_layer.items())),
            "missing": missing,
            "integrity_level": "size_and_cache_metadata_only; executor rehashes every consumed page",
        },
        "dynamic_routed_experts": dynamic,
        "cold_execution_upper_bound": {
            "static_missing_bytes": static_missing_bytes,
            "one_token_native_top6_bytes": dynamic["native_top6_cold_bytes_per_token"],
            "total_bytes": cold_upper,
        },
        "storage": {
            "root": str(root),
            "free_bytes": usage.free,
            "total_bytes": usage.total,
            "cold_upper_bound_fits": usage.free >= cold_upper,
        },
        "blocking_conditions": blockers,
        "native_token_executed": False,
        "fake_token_emitted": False,
        "claim_limit": "preflight/catalog evidence only; no model forward, token, quality, or speed claim",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--token-id", type=int, default=0)
    parser.add_argument("--rebuild-catalog", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.rebuild_catalog or not args.catalog.is_file():
            catalog = build_catalog(asset_root=args.asset_root)
            write_json(args.catalog, catalog)
        else:
            catalog = read_json(args.catalog)
        report = run_preflight(asset_root=args.asset_root, catalog=catalog, token_id=args.token_id)
        write_json(args.report, report)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "report": str(args.report.resolve()),
                    "catalog": str(args.catalog.resolve()),
                    "missing_ranges": report["static_prerequisites"]["missing_ranges"],
                    "missing_bytes": report["static_prerequisites"]["missing_bytes"],
                    "one_token_native_top6_bytes": report["dynamic_routed_experts"][
                        "native_top6_cold_bytes_per_token"
                    ],
                    "blocking_conditions": report["blocking_conditions"],
                },
                ensure_ascii=False,
            )
        )
        return 0 if report["status"] == "ready" else 2
    except Exception as error:
        print(
            json.dumps(
                {"status": "invalid", "type": type(error).__name__, "message": str(error)},
                ensure_ascii=False,
            )
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
