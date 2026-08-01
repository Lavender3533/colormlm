"""真实 L42 单层单 token 参考的短自检。"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from fast16.research.polaris_meridian_v1.l42_real_reference.l42_reference import (
    DEFAULT_ASSET_ROOT,
    FROZEN_OUTPUT_SHA256,
    LAYER,
    REPO,
    REVISION,
    ROUTE_IDS,
    ROUTE_WEIGHTS,
    run_reference,
    validate_assets,
)


ROOT = Path(__file__).resolve().parent


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} 顶层不是对象")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _must_reject(name: str, action: Callable[[], object]) -> dict[str, str]:
    try:
        action()
    except (FileNotFoundError, ValueError) as exc:
        return {"name": name, "status": "passed", "rejected_by": type(exc).__name__, "message": str(exc)}
    raise AssertionError(f"负向自检未拒绝: {name}")


def run_selftest(asset_root: str | Path = DEFAULT_ASSET_ROOT) -> dict[str, Any]:
    root = Path(asset_root).resolve()
    contract = _load(ROOT / "source_contract.json")
    if (
        contract.get("format") != "polaris-l42-real-reference-source-contract-v1"
        or contract.get("repo") != REPO
        or contract.get("revision") != REVISION
        or contract.get("layer") != LAYER
        or contract.get("frozen_f32_le_sha256") != FROZEN_OUTPUT_SHA256
        or contract.get("native_route", {}).get("expert_ids") != ROUTE_IDS
        or contract.get("native_route", {}).get("route_weights") != ROUTE_WEIGHTS
    ):
        raise AssertionError("source_contract.json 与可执行冻结合同不一致")
    report = run_reference(root, verify_hashes=True)
    actual = {key: report[key]["f32_le_sha256"] for key in FROZEN_OUTPUT_SHA256}
    if actual != FROZEN_OUTPUT_SHA256:
        raise AssertionError(f"真实 L42 冻结指纹漂移: expected={FROZEN_OUTPUT_SHA256}, actual={actual}")

    negatives: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory(prefix="polaris_l42_negative_") as temp_name:
        temp = Path(temp_name)
        base = _load(root / "l42_base_cache_manifest.json")
        route = _load(root / "l42_real_layer_route_manifest.json")

        missing = copy.deepcopy(base)
        missing["entries"][0]["path"] = str(root / "range_cache" / "definitely_missing_l42_payload.bin")
        missing_path = temp / "missing.json"
        _write(missing_path, missing)
        negatives.append(
            _must_reject(
                "missing_payload",
                lambda: validate_assets(root, manifest_paths={"base": missing_path}),
            )
        )

        bad_hash = copy.deepcopy(base)
        bad_hash["entries"][0]["sha256"] = "0" * 64
        bad_hash_path = temp / "bad_hash.json"
        _write(bad_hash_path, bad_hash)
        negatives.append(
            _must_reject(
                "sha256_drift",
                lambda: validate_assets(root, manifest_paths={"base": bad_hash_path}),
            )
        )

        bad_revision = copy.deepcopy(base)
        bad_revision["revision"] = "drifted-revision"
        bad_revision_path = temp / "bad_revision.json"
        _write(bad_revision_path, bad_revision)
        negatives.append(
            _must_reject(
                "revision_drift",
                lambda: validate_assets(root, manifest_paths={"base": bad_revision_path}),
            )
        )

        missing_route = copy.deepcopy(route)
        missing_route["entries"].pop()
        missing_route["entry_count"] -= 1
        missing_route["bytes"] = sum(entry["bytes"] for entry in missing_route["entries"])
        missing_route_path = temp / "missing_route.json"
        _write(missing_route_path, missing_route)
        negatives.append(
            _must_reject(
                "route_tensor_missing",
                lambda: validate_assets(root, manifest_paths={"route": missing_route_path}),
            )
        )

    return {
        "format": "polaris-l42-real-reference-selftest-v1",
        "status": "passed",
        "positive": {
            "source_contract": "passed",
            "f32_le_sha256": actual,
            "route_ids": report["expert_ids"],
            "route_weights": report["route_weights"],
            "integrity": report["integrity"],
        },
        "negative_tests": negatives,
        "negative_passed": len(negatives),
        "negative_total": 4,
        "claim_limit": "仅证明真实 L42 单层单 token 参考可复现；不是 S14/43 层首 token",
    }


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--report", type=Path, default=ROOT / "SELFTEST_REPORT.json")
    args = parser.parse_args()
    result = run_selftest(args.asset_root)
    _write(args.report, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
