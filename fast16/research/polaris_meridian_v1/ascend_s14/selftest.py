"""零网络、零设备地验证 Ascend S14 adapter 合同与生命周期。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from adapter import (
    ContractError,
    NativeForwardUnavailable,
    OfficialAscendLayerExecutor,
    RangePackLayerSource,
    S14_LAYERS,
    make_dry_run,
    read_json,
    run_synthetic,
    validate_matrix,
)
from doctor import make_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path(__file__).with_name("support_matrix.json"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    matrix = read_json(args.matrix)
    checks: list[dict[str, object]] = []

    validate_matrix(matrix)
    checks.append({"id": "frozen_identity", "pass": True})

    dry_run = make_dry_run(matrix)
    checks.append(
        {
            "id": "dry_run_no_io",
            "pass": dry_run["weight_io_performed"] is False and dry_run["device_touched"] is False,
        }
    )

    synthetic = run_synthetic()
    lifecycle_ok = (
        synthetic["ok"]
        and synthetic["layers"] == list(S14_LAYERS)
        and synthetic["max_live_layers"] == 1
        and synthetic["max_live_payloads"] == 1
        and synthetic["remaining_live_payloads"] == 0
        and len(synthetic["events"]) == len(S14_LAYERS) * 3
    )
    checks.append({"id": "single_layer_lifecycle", "pass": lifecycle_ok})

    class PartialFailureProvider:
        def __init__(self) -> None:
            self.live = 0

        def iter_verified_tensors(self, layer_id: int):
            del layer_id
            self.live += 1
            yield "already.loaded", object(), {"verified": True}
            self.live += 1
            yield "bad.integrity", object(), {"verified": False}

        def release_tensor(self, name: str, payload: object) -> None:
            del name, payload
            self.live -= 1

    partial = PartialFailureProvider()
    partial_refused = False
    try:
        RangePackLayerSource(partial).load_layer(S14_LAYERS[0])
    except ContractError:
        partial_refused = True
    checks.append(
        {
            "id": "partial_load_failure_releases_payloads",
            "pass": partial_refused and partial.live == 0,
        }
    )

    native_refused = False
    try:
        OfficialAscendLayerExecutor(matrix)
    except NativeForwardUnavailable:
        native_refused = True
    checks.append({"id": "native_hard_refusal", "pass": native_refused})

    offline_doctor = make_report(matrix, do_probe=False, required_device="Ascend910B4")
    checks.append(
        {
            "id": "doctor_does_not_touch_device",
            "pass": offline_doctor["device"]["probed"] is False
            and offline_doctor["native_forward_ready"] is False,
        }
    )

    ok = all(bool(row["pass"]) for row in checks)
    report = {
        "format": "polaris-ascend-s14-selftest-v1",
        "ok": ok,
        "checks": checks,
        "synthetic_summary": {
            "layers": synthetic["layers"],
            "max_live_layers": synthetic["max_live_layers"],
            "event_count": len(synthetic["events"]),
            "native_forward_executed": False,
        },
        "claim_limit": "selftest 只验证代码合同，未加载 DeepSeek 权重、未触碰 NPU。",
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8", newline="\n")
    print(encoded, end="")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
