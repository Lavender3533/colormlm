from __future__ import annotations

import hashlib
import sys
import tempfile
import textwrap
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch

from fast16.research.polaris_meridian_v1.fulldepth43_native_top6 import (
    packed_fp8_projection as projection,
)


FAKE_WORKER = r"""
import hashlib
import json
import sys
from pathlib import Path

protocol = "polaris-fulldepth43-packed-fp8-projection-v1"
revision = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
profile = "fulldepth43_native_top6"
projection = {
    "name": "layers.42.attn.wq_a",
    "n": 1024,
    "k": 4096,
    "activation_contract": "cpu_e4m3fn_quant_dequant_f32",
    "output_rounding": "bf16_rne_then_f32_le",
}
mode = sys.argv[-1]
input_sha = sys.argv[-3]
output_sha = sys.argv[-2]
hello_projection = dict(projection)
if mode == "bad_hello":
    hello_projection["n"] = 2048
print(json.dumps({
    "protocol": protocol,
    "op": "hello",
    "ready": True,
    "revision": revision,
    "profile": profile,
    "layer": 42,
    "position": 0,
    "projection": hello_projection,
    "arena_transport": "shared_binary_file",
    "weight_resident": True,
    "weight_sha256": "1efcea39938dfadc143c41813bc32327a9bb5369b2b612feac76d9dfb8001ce7",
    "scale_sha256": "dfb4085717aa527f8affa5a1640c5f806867c5ba6e0301d170f387be8b6660cf",
    "catalog_sha256": "ca619984d4a46ad1a3701d2b4035766ea40c3a3dbedd3a474ce1df7aad4d0049",
    "input_sha256": input_sha,
    "output_sha256": output_sha,
    "numeric_mode": "packed_fp8_e4m3_ue8m0_exact_audit",
}), flush=True)

for line in sys.stdin:
    request = json.loads(line)
    arena = Path(request["input"]["path"])
    with arena.open("rb") as stream:
        stream.seek(request["input"]["offset"])
        payload = stream.read(request["input"]["bytes"])
    if hashlib.sha256(payload).hexdigest() != input_sha:
        print(json.dumps({
            "protocol": protocol,
            "request_id": request["request_id"],
            "ok": False,
            "error": "input hash drift",
            "poisoned": True,
        }), flush=True)
        raise SystemExit(2)

    output_payload = bytes(4096)
    with arena.open("r+b") as stream:
        stream.seek(request["output"]["offset"])
        if mode == "partial_output":
            stream.write(output_payload[:2048])
        else:
            stream.write(output_payload)
        stream.flush()
    response = {
        "protocol": protocol,
        "request_id": request["request_id"],
        "ok": True,
        "revision": "bad" if mode == "wrong_identity" else revision,
        "profile": profile,
        "layer": 42,
        "position": 0,
        "arena_epoch": request["arena_epoch"],
        "projection": "layers.42.attn.wq_a",
        "input": request["input"],
        "output_written": request["output"],
        "input_sha256": input_sha,
        "output_sha256": output_sha,
        "weight_sha256": "1efcea39938dfadc143c41813bc32327a9bb5369b2b612feac76d9dfb8001ce7",
        "scale_sha256": "dfb4085717aa527f8affa5a1640c5f806867c5ba6e0301d170f387be8b6660cf",
        "catalog_sha256": "ca619984d4a46ad1a3701d2b4035766ea40c3a3dbedd3a474ce1df7aad4d0049",
        "weight_resident": True,
        "static_uploaded_bytes": 4194560,
        "request_uploaded_bytes": 16384,
        "numeric_mode": "packed_fp8_e4m3_ue8m0_exact_audit",
        "output_rounding": "bf16_rne_then_f32_le",
    }
    print(json.dumps(response), flush=True)
"""


class PackedFp8ProjectionTests(unittest.TestCase):
    @staticmethod
    def _fixture_hashes() -> tuple[str, str]:
        return (
            hashlib.sha256(bytes(projection.INPUT_BYTES)).hexdigest(),
            hashlib.sha256(bytes(projection.OUTPUT_BYTES)).hexdigest(),
        )

    @staticmethod
    def _write_worker(root: Path) -> Path:
        script = root / "fake_packed_fp8_worker.py"
        script.write_text(textwrap.dedent(FAKE_WORKER), encoding="utf-8", newline="\n")
        return script

    def _patch_fixture_hashes(self, input_sha: str, output_sha: str) -> ExitStack:
        stack = ExitStack()
        stack.enter_context(patch.object(projection, "INPUT_SHA256", input_sha))
        stack.enter_context(patch.object(projection, "OUTPUT_SHA256", output_sha))
        return stack

    @staticmethod
    def _command(script: Path, input_sha: str, output_sha: str, mode: str) -> tuple[str, ...]:
        return (
            sys.executable,
            "-X",
            "utf8",
            str(script),
            projection.WORKER_ARG,
            input_sha,
            output_sha,
            mode,
        )

    def test_persistent_worker_uses_arena_and_advances_epoch(self) -> None:
        input_sha, output_sha = self._fixture_hashes()
        with tempfile.TemporaryDirectory() as directory, self._patch_fixture_hashes(
            input_sha, output_sha
        ):
            root = Path(directory)
            script = self._write_worker(root)
            arena = projection.PackedFp8Arena(root / "projection.bin", create=True)
            activation = torch.zeros(projection.INPUT_SHAPE, dtype=torch.float32)
            with projection.PersistentPackedFp8Projection(
                self._command(script, input_sha, output_sha, "success"),
                arena,
                timeout_seconds=5,
            ) as worker:
                first, first_evidence = worker.execute(activation)
                second, second_evidence = worker.execute(activation)
                self.assertEqual(worker.counter, 2)
                self.assertEqual(worker.arena_epoch, 2)
                self.assertEqual(first_evidence["arena_epoch"], 0)
                self.assertEqual(second_evidence["arena_epoch"], 1)
                self.assertEqual(tuple(first.shape), projection.OUTPUT_SHAPE)
                self.assertEqual(first.dtype, torch.float32)
                self.assertTrue(torch.equal(first, torch.zeros_like(first)))
                self.assertTrue(torch.equal(second, first))
                self.assertEqual(
                    hashlib.sha256(
                        arena._read_range(arena.output_offset, projection.OUTPUT_BYTES)
                    ).hexdigest(),
                    output_sha,
                )

    def test_partial_output_cannot_pass_full_coverage_sha(self) -> None:
        input_sha, output_sha = self._fixture_hashes()
        with tempfile.TemporaryDirectory() as directory, self._patch_fixture_hashes(
            input_sha, output_sha
        ):
            root = Path(directory)
            script = self._write_worker(root)
            arena = projection.PackedFp8Arena(root / "projection.bin", create=True)
            worker = projection.PersistentPackedFp8Projection(
                self._command(script, input_sha, output_sha, "partial_output"),
                arena,
                timeout_seconds=5,
            )
            with self.assertRaisesRegex(
                projection.PackedFp8ProjectionError, "output SHA-256"
            ):
                worker.execute(torch.zeros(projection.INPUT_SHAPE, dtype=torch.float32))
            self.assertTrue(worker.poisoned)
            self.assertIsNone(worker.process)

    def test_response_identity_drift_poisons_worker(self) -> None:
        input_sha, output_sha = self._fixture_hashes()
        with tempfile.TemporaryDirectory() as directory, self._patch_fixture_hashes(
            input_sha, output_sha
        ):
            root = Path(directory)
            script = self._write_worker(root)
            arena = projection.PackedFp8Arena(root / "projection.bin", create=True)
            worker = projection.PersistentPackedFp8Projection(
                self._command(script, input_sha, output_sha, "wrong_identity"),
                arena,
                timeout_seconds=5,
            )
            with self.assertRaisesRegex(
                projection.PackedFp8ProjectionError, "response 身份/SHA"
            ):
                worker.execute(torch.zeros(projection.INPUT_SHAPE, dtype=torch.float32))
            self.assertTrue(worker.poisoned)

    def test_hello_projection_drift_is_rejected(self) -> None:
        input_sha, output_sha = self._fixture_hashes()
        with tempfile.TemporaryDirectory() as directory, self._patch_fixture_hashes(
            input_sha, output_sha
        ):
            root = Path(directory)
            script = self._write_worker(root)
            arena = projection.PackedFp8Arena(root / "projection.bin", create=True)
            with self.assertRaisesRegex(
                projection.PackedFp8ProjectionError, "hello 身份"
            ):
                projection.PersistentPackedFp8Projection(
                    self._command(script, input_sha, output_sha, "bad_hello"),
                    arena,
                    timeout_seconds=5,
                )

    def test_input_sha_drift_fails_before_request(self) -> None:
        input_sha, output_sha = self._fixture_hashes()
        with tempfile.TemporaryDirectory() as directory, self._patch_fixture_hashes(
            input_sha, output_sha
        ):
            root = Path(directory)
            script = self._write_worker(root)
            arena = projection.PackedFp8Arena(root / "projection.bin", create=True)
            worker = projection.PersistentPackedFp8Projection(
                self._command(script, input_sha, output_sha, "success"),
                arena,
                timeout_seconds=5,
            )
            with self.assertRaisesRegex(
                projection.PackedFp8ProjectionError, "frozen input SHA-256"
            ):
                worker.execute(torch.ones(projection.INPUT_SHAPE, dtype=torch.float32))
            self.assertTrue(worker.poisoned)

    def test_arena_rejects_overlap_and_declared_size_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(
                projection.PackedFp8ProjectionError, "不得重叠"
            ):
                projection.PackedFp8Arena(
                    root / "overlap.bin",
                    input_offset=0,
                    output_offset=4,
                    create=True,
                )
            path = root / "existing.bin"
            path.write_bytes(bytes(projection.INPUT_BYTES + projection.OUTPUT_BYTES))
            with self.assertRaisesRegex(
                projection.PackedFp8ProjectionError, "声明不一致"
            ):
                projection.PackedFp8Arena(
                    path,
                    arena_bytes=projection.INPUT_BYTES + projection.OUTPUT_BYTES + 4,
                )


if __name__ == "__main__":
    unittest.main()
