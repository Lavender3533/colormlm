from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import torch

from fast16.research.polaris_meridian_v1.fulldepth43_native_top6 import (
    fulldepth_packed_fp8_attention as attention,
)


FAKE_WORKER = r"""
import hashlib
import json
import sys
from pathlib import Path

protocol = "polaris-fulldepth43-packed-fp8-attention-v1"
revision = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
profile = "fulldepth43_native_top6"
catalog_sha = "ca619984d4a46ad1a3701d2b4035766ea40c3a3dbedd3a474ce1df7aad4d0049"
mode = sys.argv[-2]
payload_root = str(Path(sys.argv[-1]).resolve(strict=True))
hello = {
    "protocol": protocol,
    "op": "hello",
    "ready": True,
    "revision": revision,
    "profile": profile,
    "layers": {"min": 0, "max": 42},
    "position_max": 1_048_575,
    "projections": ["wq_a", "wkv", "wq_b", "indexer.wq_b", "wo_b", "wo_a"],
    "kernels": ["standard", "grouped_wo_a"],
    "arena_transport": "shared_binary_file",
    "catalog_sha256": catalog_sha,
    "payload_root": payload_root,
    "payload_cache_bytes": 256 * 1024 * 1024,
    "output_rounding": "bf16_rne_then_f32_le",
}
if mode == "bad_hello_key":
    hello["unexpected"] = True
print(json.dumps(hello), flush=True)
if mode == "bad_hello_key":
    raise SystemExit(2)

expected_epoch = 0
for line in sys.stdin:
    request = json.loads(line)
    if mode == "poison":
        print(json.dumps({
            "protocol": protocol,
            "request_id": request["request_id"],
            "ok": False,
            "error": "injected poison",
            "poisoned": True,
        }), flush=True)
        raise SystemExit(3)
    if request["arena_epoch"] != expected_epoch:
        raise SystemExit(4)
    output = bytes(request["output"]["bytes"])
    arena = Path(request["output"]["path"])
    with arena.open("r+b") as stream:
        stream.seek(request["output"]["offset"])
        stream.write(output)
        stream.flush()
    output_sha = hashlib.sha256(output).hexdigest()
    if mode == "bad_output_sha":
        output_sha = "f" * 64
    response = {
        "protocol": protocol,
        "request_id": request["request_id"],
        "ok": True,
        "revision": revision,
        "profile": profile,
        "layer": request["layer"],
        "position": request["position"],
        "arena_epoch": request["arena_epoch"],
        "projection": request["projection"],
        "input": request["input"],
        "output_written": request["output"],
        "input_sha256": request["input_sha256"],
        "output_sha256": output_sha,
        "weight_sha256": request["weight"]["sha256"],
        "scale_sha256": request["scale"]["sha256"],
        "catalog_sha256": catalog_sha,
        "payload_hash_verified": True,
        "gpu_slot_cache_hit": expected_epoch > 0,
        "gpu_slot_cache_entries": 1,
        "gpu_slot_resident_bytes": request["weight"]["bytes"] + request["scale"]["bytes"],
        "payload_uploaded_bytes": (
            0 if expected_epoch > 0
            else request["weight"]["bytes"] + request["scale"]["bytes"]
        ),
        "activation_uploaded_bytes": request["input"]["bytes"],
        "numeric_mode": (
            "grouped_packed_fp8_e4m3_ue8m0_bf16_input_output"
            if request["projection"]["kernel"] == "grouped_wo_a"
            else "packed_fp8_e4m3_ue8m0_bf16_output"
        ),
        "output_rounding": "bf16_rne_then_f32_le",
    }
    if mode == "bad_response_key":
        response["unexpected"] = True
    print(json.dumps(response), flush=True)
    expected_epoch += 1
    if mode in {"bad_output_sha", "bad_response_key"}:
        raise SystemExit(5)
"""


class CleanPersistentClient(attention.PersistentFullDepthPackedFp8Attention):
    """仅为测试补齐 pipe/thread 回收，协议行为保持父类实现。"""

    def _cleanup_process(self, process: object | None) -> None:
        if process is None:
            return
        for stream_name in ("stdin",):
            stream = getattr(process, stream_name, None)
            if stream is not None and not stream.closed:
                stream.close()
        if process.poll() is None:
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        for thread in self._threads:
            thread.join(timeout=1)
        self._threads.clear()
        for stream_name in ("stdout", "stderr"):
            stream = getattr(process, stream_name, None)
            if stream is not None and not stream.closed:
                stream.close()

    def _start(self) -> None:
        try:
            super()._start()
        except BaseException:
            process, self.process = self.process, None
            self._cleanup_process(process)
            raise

    def _fail(self, message: str) -> None:
        self.poisoned = True
        process, self.process = self.process, None
        self._cleanup_process(process)
        raise attention.FullDepthPackedFp8Error(message)

    def close(self) -> None:
        process, self.process = self.process, None
        self._cleanup_process(process)


class FullDepthPackedFp8AttentionTests(unittest.TestCase):
    @staticmethod
    def _write_worker(root: Path) -> Path:
        path = root / "fake_fulldepth_fp8_worker.py"
        path.write_text(textwrap.dedent(FAKE_WORKER), encoding="utf-8", newline="\n")
        return path

    @staticmethod
    def _asset(
        cache: Path,
        tensor: str,
        filename: str,
        byte_count: int,
        sha256: str,
        dtype: str,
        shape: tuple[int, ...],
    ) -> attention.PackedFp8Asset:
        path = cache / filename
        with path.open("wb") as stream:
            stream.truncate(byte_count)
        return attention.PackedFp8Asset.from_mapping(
            {
                "tensor": tensor,
                "path": path,
                "bytes": byte_count,
                "sha256": sha256,
                "dtype": dtype,
                "shape": shape,
            }
        )

    def _fixture(
        self, root: Path
    ) -> tuple[
        Path,
        attention.FullDepthPackedFp8Arena,
        attention.PackedFp8Asset,
        attention.PackedFp8Asset,
    ]:
        cache = root / "range_cache"
        cache.mkdir()
        worker = self._write_worker(root)
        arena = attention.FullDepthPackedFp8Arena(root / "arena.bin", create=True)
        weight = self._asset(
            cache,
            "layers.42.attn.wq_a.weight",
            "weight.bin",
            4_194_304,
            "1" * 64,
            "F8_E4M3",
            (1024, 4096),
        )
        scale = self._asset(
            cache,
            "layers.42.attn.wq_a.scale",
            "scale.bin",
            256,
            "2" * 64,
            "F8_E8M0",
            (8, 32),
        )
        return worker, arena, weight, scale

    @staticmethod
    def _command(worker: Path, mode: str, payload_root: Path) -> tuple[str, ...]:
        return (
            sys.executable,
            "-X",
            "utf8",
            str(worker),
            attention.WORKER_ARG,
            mode,
            str(payload_root),
        )

    def test_valid_hello_consecutive_epoch_and_output_sha(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script, arena, weight, scale = self._fixture(root)
            activation = torch.zeros((1, 1, 4096), dtype=torch.float32)
            with CleanPersistentClient(
                self._command(script, "success", root / "range_cache"),
                arena,
                timeout_seconds=5,
            ) as worker:
                first, first_evidence = worker.execute(
                    layer=42,
                    position=0,
                    suffix="wq_a",
                    activation=activation,
                    weight=weight,
                    scale=scale,
                )
                second, second_evidence = worker.execute(
                    layer=42,
                    position=1,
                    suffix="wq_a",
                    activation=activation,
                    weight=weight,
                    scale=scale,
                )
                expected_sha = hashlib.sha256(bytes(4096)).hexdigest()
                self.assertEqual(worker.epoch, 2)
                self.assertEqual(first_evidence["arena_epoch"], 0)
                self.assertEqual(second_evidence["arena_epoch"], 1)
                self.assertEqual(first_evidence["output_sha256"], expected_sha)
                self.assertEqual(second_evidence["output_sha256"], expected_sha)
                self.assertEqual(tuple(first.shape), (1, 1, 1024))
                self.assertTrue(torch.equal(first, torch.zeros_like(first)))
                self.assertTrue(torch.equal(second, first))

    def test_unknown_hello_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script, arena, _, _ = self._fixture(root)
            with self.assertRaisesRegex(attention.FullDepthPackedFp8Error, "hello key 漂移"):
                CleanPersistentClient(
                    self._command(script, "bad_hello_key", root / "range_cache"),
                    arena,
                    timeout_seconds=5,
                )

    def test_unknown_response_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script, arena, weight, scale = self._fixture(root)
            worker = CleanPersistentClient(
                self._command(script, "bad_response_key", root / "range_cache"),
                arena,
                timeout_seconds=5,
            )
            try:
                with self.assertRaisesRegex(
                    attention.FullDepthPackedFp8Error, "response key 漂移"
                ):
                    worker.execute(
                        layer=42,
                        position=0,
                        suffix="wq_a",
                        activation=torch.zeros((1, 1, 4096), dtype=torch.float32),
                        weight=weight,
                        scale=scale,
                    )
            finally:
                worker.close()

    def test_wrong_output_sha_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script, arena, weight, scale = self._fixture(root)
            worker = CleanPersistentClient(
                self._command(script, "bad_output_sha", root / "range_cache"),
                arena,
                timeout_seconds=5,
            )
            try:
                with self.assertRaisesRegex(
                    attention.FullDepthPackedFp8Error, "output 字节/SHA 漂移"
                ):
                    worker.execute(
                        layer=42,
                        position=0,
                        suffix="wq_a",
                        activation=torch.zeros((1, 1, 4096), dtype=torch.float32),
                        weight=weight,
                        scale=scale,
                    )
            finally:
                worker.close()

    def test_worker_rejection_poisons_client(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script, arena, weight, scale = self._fixture(root)
            worker = CleanPersistentClient(
                self._command(script, "poison", root / "range_cache"),
                arena,
                timeout_seconds=5,
            )
            with self.assertRaisesRegex(attention.FullDepthPackedFp8Error, "injected poison"):
                worker.execute(
                    layer=42,
                    position=0,
                    suffix="wq_a",
                    activation=torch.zeros((1, 1, 4096), dtype=torch.float32),
                    weight=weight,
                    scale=scale,
                )
            self.assertTrue(worker.poisoned)
            self.assertIsNone(worker.process)

    def test_arena_rejects_shape_and_non_bf16_grouped_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            arena = attention.FullDepthPackedFp8Arena(
                Path(directory) / "arena.bin", create=True
            )
            with self.assertRaisesRegex(attention.FullDepthPackedFp8Error, "shape"):
                arena.prepare(
                    torch.zeros((1, 1, 4095), dtype=torch.float32),
                    attention.projection_spec(42, "wq_a"),
                )
            grouped = torch.zeros((1, 1, 8, 4096), dtype=torch.float32)
            grouped[..., 0] = 0.1
            with self.assertRaisesRegex(attention.FullDepthPackedFp8Error, "BF16-carrying"):
                arena.prepare(grouped, attention.projection_spec(42, "wo_a"))


if __name__ == "__main__":
    unittest.main()
