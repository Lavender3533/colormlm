"""零网络验证 Python 合同、两个 profile 与可选 L42/E0 ABI 样本。"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from s14_contract import (
    CONTRACTS,
    PROFILES,
    read_json,
    validate_abi_manifest,
    validate_capability_manifest,
    validate_shared_contract,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rust-bin", type=Path)
    args = parser.parse_args()
    checks: list[dict[str, object]] = []

    shared = read_json(CONTRACTS / "s14_contract.json")
    validate_shared_contract(shared)
    checks.append({"id": "shared_utf8_contract", "pass": True})

    for filename, profile in (
        ("current_vulkan_capabilities.json", "s14_top6"),
        ("current_fulldepth_top1_capabilities.json", "full_depth_top1"),
    ):
        manifest = read_json(CONTRACTS / filename)
        missing = validate_capability_manifest(manifest)
        checks.append(
            {
                "id": f"{profile}_hard_refusal",
                "pass": bool(missing)
                and manifest["selected_layers"] == PROFILES[profile]["layers"]
                and manifest["experts_per_token"] == PROFILES[profile]["topk"],
            }
        )

    abi_path = Path("D:/models/Polaris-S14/abi_samples/l42_e0/manifest.json")
    if abi_path.exists():
        abi = validate_abi_manifest(read_json(abi_path))
        abi_pass = abi["routing_authority"] is False
        status = "pass"
    else:
        abi_pass = True
        status = "skip_external_sample_absent"
    checks.append({"id": "l42_e0_abi_not_route", "pass": abi_pass, "status": status})

    if args.rust_bin:
        proc = subprocess.run(
            [str(args.rust_bin), "contract"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        rust_contract = json.loads(proc.stdout) if proc.returncode == 0 else {}
        checks.append(
            {
                "id": "rust_python_contract_roundtrip",
                "pass": rust_contract == shared,
            }
        )

    ok = all(bool(check["pass"]) for check in checks)
    report = {
        "format": "polaris-local-s14-python-selftest-v1",
        "ok": ok,
        "checks": checks,
        "native_forward_executed": False,
        "token_emitted": False,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
