"""审计现有 K3 真实胶囊和证据边界，不启动模型。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .runtime import DensePortal, SUPPORTED_CAPSULE_FORMATS, ContractError


AUDIT_FORMAT = "polaris-k3-counterfactual-asset-audit-v1"
REQUIRED_F16 = ("b_in", "gate", "up", "down", "norm", "b_out")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"{path} 必须是 JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def audit_runtime(runtime_dir: Path) -> dict[str, Any]:
    manifest_path = runtime_dir / "capsule.json"
    manifest = _read_json(manifest_path)
    if manifest.get("format") not in SUPPORTED_CAPSULE_FORMATS:
        raise ContractError(f"{manifest_path} format 不支持")
    dimensions = manifest.get("dimensions")
    files = manifest.get("runtime_files")
    if not isinstance(dimensions, dict) or not isinstance(files, dict):
        raise ContractError(f"{manifest_path} 缺少 dimensions/runtime_files")
    checks: list[dict[str, Any]] = []
    verified_bytes = 0
    for name, spec in sorted(files.items()):
        if not isinstance(spec, dict):
            raise ContractError(f"{manifest_path} runtime_files.{name} 异常")
        path = runtime_dir / str(spec.get("file", ""))
        shape = tuple(int(value) for value in spec.get("shape", []))
        declared_bytes = int(spec.get("bytes", 0))
        if not path.is_file() or declared_bytes <= 0 or path.stat().st_size != declared_bytes:
            raise ContractError(f"{path} 缺失或字节数异常")
        if spec.get("dtype") == "float16-le" and declared_bytes != math.prod(shape) * 2:
            raise ContractError(f"{path} F16 形状/字节数不一致")
        digest = _sha256(path)
        if digest != spec.get("sha256"):
            raise ContractError(f"{path} SHA-256 不匹配")
        checks.append(
            {
                "name": name,
                "file": path.name,
                "bytes": declared_bytes,
                "sha256": digest,
                "verified": True,
            }
        )
        verified_bytes += declared_bytes
    missing_reference = sorted(set(REQUIRED_F16) - set(files))
    return {
        "manifest": str(manifest_path),
        "format": manifest["format"],
        "layer": int(manifest.get("layer", -1)),
        "expert": int(manifest.get("expert", -1)),
        "dimensions": dimensions,
        "runtime_total_bytes": int(manifest.get("runtime_total_bytes", 0)),
        "verified_file_bytes": verified_bytes,
        "files": checks,
        "f16_reference_complete": not missing_reference,
        "missing_f16_reference": missing_reference,
        "bridge_source": str(files.get("b_in", {}).get("source_npy", "")),
    }


def audit_router(router_dir: Path) -> dict[str, Any]:
    manifest_path = router_dir / "router.json"
    manifest = _read_json(manifest_path)
    mapped = manifest.get("mapped_row")
    if not isinstance(mapped, dict):
        raise ContractError(f"{manifest_path} 缺少 mapped_row")
    path = router_dir / str(mapped.get("file", ""))
    if not path.is_file() or _sha256(path) != mapped.get("sha256"):
        raise ContractError(f"{path} 缺失或 SHA-256 不匹配")
    array = np.load(path, allow_pickle=False)
    if array.dtype != np.dtype("float32") or array.shape != (2048,):
        raise ContractError(f"{path} 必须是 float32[2048]")
    return {
        "manifest": str(manifest_path),
        "layer": int(manifest.get("layer", -1)),
        "expert": int(manifest.get("expert", -1)),
        "mapped_shape": list(array.shape),
        "mapped_sha256": mapped["sha256"],
        "capability_label": False,
        "warning": manifest.get("warning"),
    }


def audit_portal(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {
            "present": False,
            "approved": False,
            "full_width": 4096,
            "bus_width": 2048,
            "reason": "no approved 4096<->2048 FullDepth portal manifest",
        }
    manifest = _read_json(path)
    evidence = manifest.get("evidence", {})
    portal = DensePortal.from_manifest(path)
    return {
        "present": True,
        "approved": True,
        "manifest": str(path),
        "format": manifest.get("format"),
        "evidence": evidence,
        "full_width": portal.full_width,
        "bus_width": portal.bus_width,
        "matrices_verified": True,
    }


def run_audit(research_root: Path, *, portal_manifest: Path | None = None) -> dict[str, Any]:
    capsules_root = research_root / "neural_bus_capsules"
    l28_root = capsules_root / "kimi_k3_l28_e780_real"
    l92_root = capsules_root / "kimi_k3_l92_e291_real"
    runtimes = [
        audit_runtime(l28_root / "runtime_v4_hybrid"),
        audit_runtime(l92_root / "runtime"),
    ]
    routers = [audit_router(l28_root / "router"), audit_router(l92_root / "router")]

    transport = _read_json(research_root / "kimi_k3_coordinate_transport_report.json")
    precision = _read_json(research_root / "v20_k3_precision_summary.json")
    frontend_manifest = _read_json(research_root / "parallel_frontend_v47" / "MANIFEST.json")
    frontend_selftest = _read_json(research_root / "parallel_frontend_v47" / "SELFTEST_REPORT.json")
    after = float(transport["router_best_cosine"]["after_transport"]["mean"])
    shuffled = float(transport["router_best_cosine"]["shuffled_donor_axes"]["mean"])
    portal = audit_portal(portal_manifest)
    all_assets_verified = bool(
        all(runtime["f16_reference_complete"] for runtime in runtimes)
        and all(router["mapped_shape"] == [2048] for router in routers)
    )
    gate_enabled = bool(
        all_assets_verified
        and portal["approved"]
        and precision.get("status") != "rejected_for_capability"
    )
    return {
        "format": AUDIT_FORMAT,
        "research_root": str(research_root.resolve()),
        "real_capsules": runtimes,
        "router_priors": routers,
        "coordinate_transport": {
            "format": transport.get("format"),
            "embedding_test_cosine_mean": transport["embedding_cosine"]["test_after"]["mean"],
            "router_cosine_after_mean": after,
            "router_cosine_shuffled_mean": shuffled,
            "router_cosine_margin_over_shuffled": after - shuffled,
            "capability_evidence": False,
        },
        "v20_evidence": {
            "status": precision.get("status"),
            "all_current_loto_robust": precision.get("all_current_loto_robust"),
            "decision": precision.get("decision"),
        },
        "parallel_frontend_v47": {
            "manifest_format": frontend_manifest.get("schema_version"),
            "selftest_ok": frontend_selftest.get("ok") is True,
            "selftest_count": frontend_selftest.get("tests"),
            "role": "frozen offline HTML evaluator only",
            "online_hidden_gate": False,
            "donor_weights": False,
        },
        "full_depth_portal": portal,
        "all_real_assets_verified": all_assets_verified,
        "gate_enabled": gate_enabled,
        "default_path": "no_op",
        "status": "assets_verified_gate_disabled" if all_assets_verified and not gate_enabled else "ready",
        "claim_limit": "real weights verified; no FullDepth portal, K3 frontend capability, quality, or speed claim",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-root", type=Path, required=True)
    parser.add_argument("--portal-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_audit(args.research_root, portal_manifest=args.portal_manifest)
    _write_json(args.output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "all_real_assets_verified": report["all_real_assets_verified"],
                "gate_enabled": report["gate_enabled"],
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
