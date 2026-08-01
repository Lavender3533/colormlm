"""在 CPU/Ascend 上自检 planner、mHC/router 探针与 CNOB v2；不产生模型证据。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

try:
    from .capture_io import CaptureWriter, pack_bf16_from_f32, sha256_json, validate_capture
    from .instrumentation import OfficialDeepSeekStepProbe
    from .planner import DEFAULT_SNAPSHOT, make_plan, read_json, verify_metadata
except ImportError:
    from capture_io import CaptureWriter, pack_bf16_from_f32, sha256_json, validate_capture
    from instrumentation import OfficialDeepSeekStepProbe
    from planner import DEFAULT_SNAPSHOT, make_plan, read_json, verify_metadata


def select_device(preference: str) -> tuple[Any, dict[str, Any]]:
    import torch

    if preference in {"auto", "npu"}:
        try:
            import torch_npu

            if not bool(torch_npu.npu.is_available()):
                raise RuntimeError("NPU unavailable")
            device = torch.device("npu:0")
            probe = torch.ones((8, 8), device=device, dtype=torch.float32) + 1
            torch_npu.npu.synchronize()
            if float(probe.mean().cpu()) != 2.0:
                raise RuntimeError("NPU probe result mismatch")
            return device, {
                "requested": preference,
                "selected": str(device),
                "fallback_used": False,
                "device_name": str(torch_npu.npu.get_device_name(0)),
            }
        except Exception as exc:
            if preference == "npu":
                raise
            fallback = f"{type(exc).__name__}: {exc}"
    else:
        fallback = None
    return torch.device("cpu"), {
        "requested": preference,
        "selected": "cpu",
        "fallback_used": preference == "auto",
        "fallback_reason": fallback,
        "device_name": None,
    }


def synchronize(device: Any) -> None:
    if str(device).startswith("npu"):
        import torch_npu

        torch_npu.npu.synchronize()


def make_fake_model(device: Any) -> Any:
    import torch
    from torch import nn

    class FakeGate(nn.Module):
        def forward(self, x: Any, input_ids: Any = None):
            rows = x.reshape(-1, x.shape[-1]).shape[0]
            weights = torch.full((rows, 6), 1.0 / 6.0, device=x.device, dtype=torch.float32)
            indices = torch.arange(6, device=x.device, dtype=torch.int32).view(1, 6).repeat(rows, 1)
            return weights, indices

    class FakeFfn(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = FakeGate()

    class FakeBlock(nn.Module):
        def __init__(self, layer_id: int):
            super().__init__()
            self.layer_id = layer_id
            self.hc_attn_fn = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
            self.hc_ffn_fn = nn.Parameter(torch.tensor([2.0], dtype=torch.float32))
            self.hc_attn_scale = nn.Parameter(torch.ones(1, dtype=torch.float32))
            self.hc_ffn_scale = nn.Parameter(torch.ones(1, dtype=torch.float32))
            self.hc_attn_base = nn.Parameter(torch.zeros(1, dtype=torch.float32))
            self.hc_ffn_base = nn.Parameter(torch.zeros(1, dtype=torch.float32))
            self.ffn = FakeFfn()

        def hc_pre(self, x: Any, hc_fn: Any, hc_scale: Any, hc_base: Any):
            batch, sequence, hc, _ = x.shape
            post = torch.full((batch, sequence, hc), 0.25, device=x.device, dtype=torch.float32)
            comb = torch.eye(hc, device=x.device, dtype=torch.float32).view(1, 1, hc, hc).repeat(batch, sequence, 1, 1)
            return x.mean(dim=2), post, comb

        def forward(self, x: Any):
            self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
            self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
            self.ffn.gate(x.mean(dim=2), None)
            return x + torch.tensor(self.layer_id / 1000.0, device=x.device, dtype=x.dtype)

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Identity() for _ in range(43)])
            for layer in (39, 40, 41, 42):
                self.layers[layer] = FakeBlock(layer)

    return FakeModel().to(device)


def run_selftest(device: Any, temporary: Path) -> dict[str, Any]:
    import torch

    checks: list[dict[str, Any]] = []
    started = time.perf_counter()
    snapshot = read_json(DEFAULT_SNAPSHOT)
    verification = verify_metadata(snapshot, None, None)
    plan = make_plan(
        snapshot,
        task_count=2,
        observed_tokens_per_task=2,
        network_mib_s=50.0,
        available_disk_gib=50.0,
        available_ram_gib=1500.0,
        hbm_gib=32.0,
        window_seconds=7200,
        metadata_verification=verification,
    )
    assert plan["status"] == "dry_run_only_no_native_forward"
    assert plan["gates"][-1]["pass"] is False
    checks.append({"name": "planner_keeps_native_gate_false", "passed": True})

    if pack_bf16_from_f32([1.0]) != b"\x80\x3f":
        raise AssertionError("BF16 little-endian encoding error")
    checks.append({"name": "bf16_rounding_and_endianness", "passed": True})

    cnob = temporary / "fixture.cnob"
    sidecar = temporary / "fixture.jsonl"
    writer = CaptureWriter(cnob, sidecar)
    model = make_fake_model(device)
    with OfficialDeepSeekStepProbe(model) as probe:
        for record in range(2):
            probe.begin_step()
            hidden = torch.full((1, 1, 4, 4096), 0.125 + record, device=device, dtype=torch.bfloat16)
            for layer in (39, 40, 41, 42):
                hidden = model.layers[layer](hidden)
            synchronize(device)
            layers = probe.materialize()
            target_id = 11 if record == 0 else -1
            token = {
                "sequence_id": "合成-管线自检",
                "phase": "prefill" if record == 0 else "decode",
                "token_position": record,
                "input_token_id": 10 + record,
                "input_token_text": "你" if record == 0 else "好",
                "target_token_id": target_id,
                "target_token_text": "好" if target_id >= 0 else "",
                "predicted_token_id": 11 + record,
                "predicted_token_text": "好" if record == 0 else "！",
                "target_logprob": -0.25 if target_id >= 0 else None,
                "target_nll": 0.25 if target_id >= 0 else None,
                "prefix_sha256": sha256_json([10, 11][: record + 1]),
                "synthetic": True,
                "native_forward_completed": False,
            }
            writer.write_step(layers, token)
    writer.finish()
    checks.append({"name": "official_probe_interface_fixture", "passed": True})

    validation = validate_capture(cnob, sidecar, require_real=False)
    if validation["records"] != 2 or validation["cnob_chunks"] != 52:
        raise AssertionError("capture record/chunk count error")
    if "合成-管线自检" not in sidecar.read_text(encoding="utf-8"):
        raise AssertionError("UTF-8 sidecar roundtrip error")
    checks.append({"name": "cnob_and_utf8_roundtrip", "passed": True})

    try:
        validate_capture(cnob, sidecar, require_real=True)
    except ValueError:
        checks.append({"name": "synthetic_rejected_by_real_gate", "passed": True})
    else:
        raise AssertionError("real gate accepted synthetic fixture")

    truncated = temporary / "truncated.cnob"
    raw = cnob.read_bytes()
    truncated.write_bytes(raw[:-1])
    try:
        validate_capture(truncated, sidecar, require_real=False)
    except ValueError:
        checks.append({"name": "truncation_detected", "passed": True})
    else:
        raise AssertionError("truncated CNOB was accepted")

    return {
        "format": "polaris-deepseek-stream-selftest-v1",
        "ok": all(row["passed"] for row in checks),
        "evidence_status": "synthetic_contract_fixture_not_model_evidence",
        "checks": checks,
        "capture": validation,
        "plan_summary": {
            "base_forward_file_union_bytes": plan["weights"]["base_forward_file_union_bytes"],
            "capture_bytes_per_record": plan["capture"]["per_record"]["cnob_bytes"],
            "native_forward_gate": plan["gates"][-1]["pass"],
        },
        "elapsed_seconds": time.perf_counter() - started,
        "claim_limit": "合成 fixture 只验证接口、NPU tensor 搬运和文件校验；没有运行 DeepSeek 权重。",
    }


def main() -> int:
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "npu", "cpu"), default="auto")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    device, selection = select_device(args.device)
    with tempfile.TemporaryDirectory(prefix="polaris-dsv4-stream-selftest-") as value:
        result = run_selftest(device, Path(value))
    result["device"] = selection
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        if args.output.exists() and not args.force:
            raise FileExistsError(f"拒绝覆盖 {args.output}；使用 --force")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8", newline="\n")
    print(encoded, end="")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
