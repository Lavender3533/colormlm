from __future__ import annotations

import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import torch

from fast16.research.polaris_meridian_v1.fulldepth43_native_top6.vulkan_final_head import (
    PersistentVulkanFinalHead,
    VulkanFinalHeadError,
)


FAKE_WORKER = r"""
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

protocol = "polaris-s14-bf16-head-worker-v1"
head_sha = "1" * 64
print(json.dumps({
    "protocol": protocol,
    "status": "ready",
    "gpu": "fixture",
    "head_path": "fixture",
    "head_bytes": 129280 * 4096 * 2,
    "head_sha256": head_sha,
    "upload_wall_ms": 1.0,
}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    path = Path(request["input_path"])
    payload = path.read_bytes()
    assert len(payload) == request["input_bytes"]
    assert hashlib.sha256(payload).hexdigest() == request["input_sha256"]
    values = struct.unpack("<" + "f" * (len(payload) // 4), payload)
    assert all(math.isfinite(value) for value in values)
    batch = request["batch"]
    diagnostics = request["diagnostics"]
    print(json.dumps({
        "protocol": protocol,
        "status": "ok",
        "request_id": request["request_id"],
        "batch": batch,
        "input_path": str(path.resolve()),
        "input_bytes": len(payload),
        "input_sha256": request["input_sha256"],
        "head_sha256": head_sha,
        "argmax_token_ids": list(range(10, 10 + batch)),
        "max_logits": [2.5 + index for index in range(batch)],
        "top10": [[{"token_id": 10, "logit": 2.5}]] * batch if diagnostics else None,
        "logits": {
            "shape": [batch, 129280],
            "l2": 1.0,
            "mean": 0.0,
            "maxabs": 2.5,
            "f32_le_sha256": "2" * 64,
        } if diagnostics else None,
        "input_ready_wall_ms": 0.1,
        "kernel_wall_ms": 1.5,
        "postprocess_wall_ms": 0.2,
        "worker_wall_ms": 1.8,
        "equivalent_head_tokens_per_second": batch * 1000.0 / 1.5,
    }), flush=True)
"""


class VulkanFinalHeadTests(unittest.TestCase):
    def _worker(self, root: Path) -> PersistentVulkanFinalHead:
        script = root / "fake_head_worker.py"
        script.write_text(textwrap.dedent(FAKE_WORKER), encoding="utf-8", newline="\n")
        return PersistentVulkanFinalHead(
            (sys.executable, "-X", "utf8", str(script)),
            timeout_seconds=5,
        )

    def test_persistent_worker_validates_and_deletes_hidden_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self._worker(root) as worker:
                hidden = torch.linspace(-1, 1, 4096, dtype=torch.float32).reshape(1, 1, 4096)
                token_ids, evidence = worker.execute(hidden, root, diagnostics=True)
                self.assertEqual(token_ids, [10])
                self.assertEqual(evidence["batch"], 1)
                self.assertEqual(evidence["gpu_head_argmax_ms"], 1.5)
                self.assertTrue(evidence["persistent_context"])
                self.assertEqual(list(root.glob("head-input-*.bin")), [])

    def test_k4_is_one_request_and_invalid_hidden_fails_before_worker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self._worker(root) as worker:
                token_ids, evidence = worker.execute(torch.zeros((4, 4096)), root)
                self.assertEqual(token_ids, [10, 11, 12, 13])
                self.assertEqual(evidence["batch"], 4)
                self.assertEqual(worker.counter, 1)
                with self.assertRaises(VulkanFinalHeadError):
                    worker.execute(torch.zeros((2, 4096)), root)
                self.assertFalse(worker.poisoned)


if __name__ == "__main__":
    unittest.main()
