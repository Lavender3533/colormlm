from __future__ import annotations

import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import torch

from fast16.research.polaris_meridian_v1.fulldepth43_native_top6.vulkan_writeback import (
    PersistentVulkanWriteback,
    VulkanWritebackError,
    verify_exact_bf16_writeback,
)


FAKE_WORKER = r"""
import hashlib
import json
import sys
from pathlib import Path

protocol = "polaris-fulldepth43-vulkan-writeback-v1"
print(json.dumps({
    "protocol": protocol,
    "op": "hello",
    "ready": True,
    "device": "fixture",
    "persistent_context": True,
    "official_boundary_graph": True,
}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    manifest = Path(request["manifest"]).resolve()
    output = manifest.parent / "vulkan_moe_branch.bf16le.bin"
    payload = bytes(8192)
    output.write_bytes(payload)
    print(json.dumps({
        "protocol": protocol,
        "request_id": request["request_id"],
        "ok": True,
        "device": "fixture",
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "layer": 42,
        "position": 0,
        "input_token_id": 0,
        "output": {
            "path": str(output),
            "dtype": "bf16_le",
            "shape": [1, 1, 4096],
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
        "gpu_kernel_ms": 1.0,
        "wall_ms": 2.0,
        "boundaries": ["fixture"],
        "expansion_status": "single_real_layer_writeback_only",
        "claim_limit": "fixture",
    }), flush=True)
"""


class VulkanWritebackTests(unittest.TestCase):
    def test_persistent_protocol_reads_hash_verified_bf16(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "fake_worker.py"
            script.write_text(textwrap.dedent(FAKE_WORKER), encoding="utf-8", newline="\n")
            manifest = root / "bridge_manifest.json"
            manifest.write_text("{}\n", encoding="utf-8", newline="\n")
            with PersistentVulkanWriteback(
                (sys.executable, "-X", "utf8", str(script)),
                timeout_seconds=5,
            ) as worker:
                tensor, evidence = worker.execute(manifest)
                self.assertEqual(tuple(tensor.shape), (1, 1, 4096))
                self.assertEqual(tensor.dtype, torch.bfloat16)
                self.assertTrue(torch.equal(tensor, torch.zeros_like(tensor)))
                self.assertEqual(worker.counter, 1)
                self.assertTrue(evidence["persistent_context"])

    def test_exact_comparison_rejects_one_bf16_bit(self) -> None:
        cpu = torch.zeros((1, 1, 4096), dtype=torch.bfloat16)
        gpu = cpu.clone()
        self.assertTrue(verify_exact_bf16_writeback(cpu, gpu)["exact_bf16_equal"])
        gpu[..., 7] = 1
        with self.assertRaises(VulkanWritebackError):
            verify_exact_bf16_writeback(cpu, gpu)


if __name__ == "__main__":
    unittest.main()
