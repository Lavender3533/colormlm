"""用固化 FullDepth43 capture 验证 RX 5700 XT 单层逐 BF16 位回写。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from fast16.research.polaris_meridian_v1.l42_real_reference.l42_reference import (
    _InlineForward,
)

from .vulkan_writeback import PersistentVulkanWriteback, verify_exact_bf16_writeback


DEFAULT_CAPTURE = (
    Path(__file__).resolve().parents[4]
    / "scheduler/ssd_inference/evidence/fulldepth43_vulkan_bridge_capture"
)
DEFAULT_REPORT = (
    Path(__file__).resolve().parents[4]
    / "scheduler/ssd_inference/evidence/fulldepth43_vulkan_writeback_rx5700xt.json"
)


def _f32_sha(tensor: torch.Tensor) -> str:
    payload = tensor.float().contiguous().numpy().astype("<f4", copy=False).tobytes()
    return hashlib.sha256(payload).hexdigest()


def cpu_official_moe_reference(manifest_path: Path) -> torch.Tensor:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8", errors="strict"))
    payloads = manifest["payloads"]
    entries = {item["tensor"]: {**item, "path": item["path"]} for item in payloads}
    specs = {item["tensor"]: (item["dtype"], tuple(item["shape"])) for item in payloads}
    helper = _InlineForward(SimpleNamespace(entries=entries, specs=specs))
    ffn_input = np.fromfile(
        manifest_path.parent / manifest["input"]["file"], dtype="<f4"
    ).reshape(1, 1, 4096)
    moe = torch.zeros((1, 1, 4096), dtype=torch.float32)
    for expert_id, route_weight in zip(
        manifest["expert_ids"], manifest["route_weights"], strict=True
    ):
        prefix = f"layers.{manifest['layer']}.ffn.experts.{expert_id}"
        gate = torch.from_numpy(helper._linear_fp4(ffn_input, prefix + ".w1")).to(
            torch.bfloat16
        )
        up = torch.from_numpy(helper._linear_fp4(ffn_input, prefix + ".w3")).to(
            torch.bfloat16
        )
        hidden = F.silu(gate.float().clamp(max=10)) * up.float().clamp(-10, 10)
        weighted = (hidden * route_weight).to(torch.bfloat16)
        down = torch.from_numpy(
            helper._linear_fp4(weighted.float().numpy(), prefix + ".w2")
        ).to(torch.bfloat16)
        moe += down.float()

    prefix = f"layers.{manifest['layer']}.ffn.shared_experts"
    gate = torch.from_numpy(helper._linear_fp8(ffn_input, prefix + ".w1")).to(
        torch.bfloat16
    )
    up = torch.from_numpy(helper._linear_fp8(ffn_input, prefix + ".w3")).to(
        torch.bfloat16
    )
    hidden = (F.silu(gate.float().clamp(max=10)) * up.float().clamp(-10, 10)).to(
        torch.bfloat16
    )
    down = torch.from_numpy(
        helper._linear_fp8(hidden.float().numpy(), prefix + ".w2")
    ).to(torch.bfloat16)
    moe += down.float()
    return moe.to(torch.bfloat16)


def write_json(path: Path, document: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def verify(worker: Path, capture: Path, report: Path) -> dict[str, Any]:
    started = time.perf_counter()
    worker = worker.resolve(strict=True)
    capture = capture.resolve(strict=True)
    source_manifest = capture / "bridge_manifest.json"
    source_input = capture / "ffn_input_activation_quant.f32le.bin"
    with tempfile.TemporaryDirectory(prefix="polaris-vulkan-writeback-") as directory:
        scratch = Path(directory)
        shutil.copy2(source_manifest, scratch / source_manifest.name)
        shutil.copy2(source_input, scratch / source_input.name)
        manifest = scratch / source_manifest.name
        with PersistentVulkanWriteback(
            (str(worker), "--fulldepth43-writeback-worker"), timeout_seconds=30
        ) as bridge:
            vulkan_branch, worker_evidence = bridge.execute(manifest)
        cpu_started = time.perf_counter()
        cpu_branch = cpu_official_moe_reference(manifest)
        cpu_seconds = time.perf_counter() - cpu_started
        comparison = verify_exact_bf16_writeback(cpu_branch, vulkan_branch)
        repository_root = Path(__file__).resolve().parents[4]
        try:
            source_capture = capture.relative_to(repository_root).as_posix()
        except ValueError:
            source_capture = str(capture)
        document = {
            "format": "polaris-fulldepth43-vulkan-writeback-evidence-v1",
            "status": "exact_bf16_single_layer_writeback_verified",
            "source_capture": source_capture,
            "source_manifest_sha256": hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
            "worker": worker_evidence,
            "cpu_reference": {
                "semantics": "official Python FullDepth Expert.forward boundaries",
                "elapsed_seconds": cpu_seconds,
                "output_f32_le_sha256": _f32_sha(cpu_branch),
            },
            "vulkan_output_f32_le_sha256": _f32_sha(vulkan_branch),
            "comparison": comparison,
            "execution_seconds": time.perf_counter() - started,
            "state_transition": (
                "executor accepts this branch only after exact BF16 equality, then feeds the "
                "Vulkan tensor itself into hc_post; mismatch poisons the worker and blocks token commit"
            ),
            "expansion_status": "single_real_layer_writeback_only",
            "claim_limit": (
                "Proves one real FullDepth43 L42 MoE branch writeback. Does not prove all 43 "
                "layers on GPU, a complete GPU token, token/s, Kimi capability, or quality."
            ),
        }
    write_json(report, document)
    return document


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    document = verify(args.worker, args.capture, args.report)
    print(json.dumps(document, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
