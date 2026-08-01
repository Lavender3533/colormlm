from __future__ import annotations

import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

import torch

from fast16.research.polaris_meridian_v1.fulldepth43_native_top6.vulkan_final_head import (
    HEAD_BYTES,
    FinalHiddenNormalizer,
    FullDepthVulkanFinalHead,
    PersistentVulkanFinalHead,
    VulkanFinalHeadError,
)
from fast16.research.polaris_meridian_v1.s14_range_pack.online_range import CachedRange


FAKE_WORKER = r"""
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

protocol = "polaris-s14-bf16-head-worker-v2"
head_sha = "1" * 64
print(json.dumps({
    "protocol": protocol,
    "status": "ready",
    "gpu": "fixture",
    "head_path": "fixture",
    "head_bytes": 129280 * 4096 * 2,
    "head_sha256": head_sha,
    "upload_wall_ms": 1.0,
    "position_contract": "first_any_then_strict_increment",
    "production_response": "argmax_only_no_top10_or_logits",
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
        "position": request["position"],
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
            expected_head_sha256="1" * 64,
            timeout_seconds=5,
        )

    def test_persistent_worker_validates_and_deletes_hidden_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self._worker(root) as worker:
                hidden = torch.linspace(-1, 1, 4096, dtype=torch.float32).reshape(1, 1, 4096)
                token_ids, evidence = worker.execute(
                    hidden,
                    root,
                    position=1,
                    diagnostics=True,
                )
                self.assertEqual(token_ids, [10])
                self.assertEqual(evidence["position"], 1)
                self.assertEqual(evidence["batch"], 1)
                self.assertEqual(evidence["gpu_head_argmax_ms"], 1.5)
                self.assertTrue(evidence["persistent_context"])
                self.assertEqual(list(root.glob("head-input-*.bin")), [])

    def test_k4_is_one_request_and_invalid_hidden_fails_before_worker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self._worker(root) as worker:
                token_ids, evidence = worker.execute(
                    torch.zeros((4, 4096)),
                    root,
                    position=1,
                )
                self.assertEqual(token_ids, [10, 11, 12, 13])
                self.assertEqual(evidence["batch"], 4)
                self.assertIsNone(evidence["top10"])
                self.assertIsNone(evidence["logits"])
                self.assertEqual(worker.counter, 1)
                worker.execute(torch.zeros((1, 4096)), root, position=2)
                self.assertEqual(worker.last_position, 2)
                with self.assertRaisesRegex(VulkanFinalHeadError, "严格递增"):
                    worker.execute(torch.zeros((1, 4096)), root, position=4)
                with self.assertRaises(VulkanFinalHeadError):
                    worker.execute(torch.zeros((2, 4096)), root, position=3)
                self.assertFalse(worker.poisoned)

    def test_normalizer_never_maps_cpu_head_and_returns_one_f32_hidden(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "range_cache"
            cache.mkdir()
            specs = {
                "hc_head_base": ("F32", [4], torch.zeros(4, dtype=torch.float32).numpy().tobytes()),
                "hc_head_fn": (
                    "F32",
                    [4, 4 * 4096],
                    torch.zeros((4, 4 * 4096), dtype=torch.float32).numpy().tobytes(),
                ),
                "hc_head_scale": (
                    "F32",
                    [1],
                    torch.ones(1, dtype=torch.float32).numpy().tobytes(),
                ),
                "norm.weight": (
                    "BF16",
                    [4096],
                    torch.ones(4096, dtype=torch.bfloat16)
                    .view(torch.uint16)
                    .numpy()
                    .tobytes(),
                ),
            }
            ranges: list[CachedRange] = []
            for index, (name, (dtype, shape, payload)) in enumerate(specs.items()):
                path = cache / f"small-{index}.bin"
                path.write_bytes(payload)
                ranges.append(
                    CachedRange(
                        entry={
                            "tensor": name,
                            "dtype": dtype,
                            "shape": shape,
                            "bytes": len(payload),
                        },
                        path=path,
                        proof={"observed_sha256": f"{index + 2:064x}", "hash_authority": "fixture"},
                        cache_hit=True,
                    )
                )
            head = cache / "head.bin"
            with head.open("wb") as stream:
                stream.seek(HEAD_BYTES - 1)
                stream.write(b"\0")
            ranges.append(
                CachedRange(
                    entry={
                        "tensor": "head.weight",
                        "dtype": "BF16",
                        "shape": [129280, 4096],
                        "bytes": HEAD_BYTES,
                    },
                    path=head,
                    proof={"observed_sha256": "1" * 64, "hash_authority": "fixture"},
                    cache_hit=True,
                )
            )
            normalizer = FinalHiddenNormalizer(ranges, cache)
            state = torch.linspace(-1, 1, 4 * 4096, dtype=torch.float32).to(
                torch.bfloat16
            ).reshape(1, 1, 4, 4096)
            with mock.patch.object(torch, "from_file", side_effect=AssertionError("CPU head mmap")):
                normalized, evidence = normalizer.normalize(state)
            self.assertEqual(tuple(normalized.shape), (1, 4096))
            self.assertEqual(normalized.dtype, torch.float32)
            self.assertTrue(torch.isfinite(normalized).all().item())
            self.assertEqual(evidence["cpu_scope"], "hc_reduce_and_rmsnorm_only")

    def test_full_depth_pipeline_position1_smoke_uses_gpu_argmax_without_logits(self) -> None:
        class FakeNormalizer:
            head_path = Path("head.bin")

            def normalize(self, state: torch.Tensor):
                self.state = state
                return torch.zeros((1, 4096), dtype=torch.float32), {
                    "hc_pre": [0.1, 0.2, 0.3, 0.4],
                    "normalized": {"shape": [1, 1, 4096]},
                    "integrity": {"payload_files": 5},
                    "cpu_scope": "hc_reduce_and_rmsnorm_only",
                }

        class FakeWorker:
            hello = {"status": "ready"}

            def execute(self, hidden, scratch, *, position, diagnostics):
                self.call = (hidden, scratch, position, diagnostics)
                return [3648], {
                    "position": position,
                    "top10": None,
                    "logits": None,
                    "gpu_head_argmax_ms": 7.5,
                }

        pipeline = object.__new__(FullDepthVulkanFinalHead)
        pipeline.normalizer = FakeNormalizer()
        pipeline.worker = FakeWorker()
        pipeline.scratch_dir = Path("scratch")
        pipeline.validate_cpu_once = False
        pipeline.cpu_validation_completed = False
        pipeline.head_chunk_size = 4096
        state = torch.zeros((1, 1, 4, 4096), dtype=torch.bfloat16)
        final = pipeline.forward(state, position=1)
        self.assertEqual(final["token_id"], 3648)
        self.assertEqual(final["position"], 1)
        self.assertEqual(final["backend"], "persistent_vulkan_bf16_head_device_argmax")
        self.assertIsNone(final["top10"])
        self.assertIsNone(final["logits"])
        self.assertFalse(final["production_full_logits_returned"])
        self.assertEqual(pipeline.worker.call[2:], (1, False))


if __name__ == "__main__":
    unittest.main()
