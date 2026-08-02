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
    if request["op"] == "execute_fp8_attention_output_chain":
        output = bytes(request["output"]["bytes"])
        arena = Path(request["output"]["path"])
        with arena.open("r+b") as stream:
            stream.seek(request["output"]["offset"])
            stream.write(output)
            stream.flush()
        output_sha = hashlib.sha256(output).hexdigest()
        intermediate_sha = hashlib.sha256(bytes(8192 * 4)).hexdigest()
        slots = []
        for index, item in enumerate(request["projections"]):
            slots.append({
                "projection": item["projection"],
                "weight_sha256": item["weight"]["sha256"],
                "scale_sha256": item["scale"]["sha256"],
                "payload_hash_verified": True,
                "gpu_slot_cache_hit": expected_epoch > 0,
                "gpu_slot_cache_entries": index + 1,
                "gpu_slot_resident_bytes": sum(
                    stage["weight"]["bytes"] + stage["scale"]["bytes"]
                    for stage in request["projections"][: index + 1]
                ),
                "payload_uploaded_bytes": (
                    0 if expected_epoch > 0
                    else item["weight"]["bytes"] + item["scale"]["bytes"]
                ),
                "activation_uploaded_bytes": 8 * 4096 * 4 if index == 0 else 8192 * 4,
                "numeric_mode": (
                    "grouped_packed_fp8_e4m3_ue8m0_bf16_input_output"
                    if index == 0
                    else "packed_fp8_e4m3_ue8m0_bf16_output"
                ),
                "output_rounding": "bf16_rne_then_f32_le",
            })
        if mode == "chain_bad_slot_order":
            slots.reverse()
        response = {
            "protocol": protocol,
            "request_id": request["request_id"],
            "ok": True,
            "revision": revision,
            "profile": profile,
            "layer": request["layer"],
            "position": request["position"],
            "arena_epoch": request["arena_epoch"],
            "input": request["input"],
            "output_written": request["output"],
            "input_sha256": request["input_sha256"],
            "wo_a_output_sha256": intermediate_sha,
            "requantized_activation_sha256": intermediate_sha,
            "output_sha256": output_sha,
            "requantization": request["requantization"],
            "slots": slots,
            "catalog_sha256": catalog_sha,
            "gpu_slot_cache_entries": 2,
            "numeric_mode": "grouped_wo_a_then_e4m3fn_group128_then_wo_b",
            "output_rounding": "bf16_rne_then_f32_le",
        }
        if mode == "chain_bad_requant_sha":
            response["requantized_activation_sha256"] = "not-a-sha"
        print(json.dumps(response), flush=True)
        expected_epoch += 1
        if mode in {"chain_bad_slot_order", "chain_bad_requant_sha"}:
            raise SystemExit(7)
        continue
    if request["op"] == "execute_fp8_attention_shared_batch":
        outputs = []
        for item in request["projections"]:
            output = bytes(item["output"]["bytes"])
            arena = Path(item["output"]["path"])
            with arena.open("r+b") as stream:
                stream.seek(item["output"]["offset"])
                stream.write(output)
                stream.flush()
            output_sha = hashlib.sha256(output).hexdigest()
            outputs.append({
                "projection": item["projection"],
                "output_written": item["output"],
                "output_sha256": output_sha,
                "weight_sha256": item["weight"]["sha256"],
                "scale_sha256": item["scale"]["sha256"],
                "payload_hash_verified": True,
                "gpu_slot_cache_hit": expected_epoch > 0,
                "gpu_slot_resident_bytes": (
                    item["weight"]["bytes"] + item["scale"]["bytes"]
                ),
                "payload_uploaded_bytes": (
                    0 if expected_epoch > 0
                    else item["weight"]["bytes"] + item["scale"]["bytes"]
                ),
                "numeric_mode": "packed_fp8_e4m3_ue8m0_bf16_output",
                "output_rounding": "bf16_rne_then_f32_le",
            })
        if mode == "batch_bad_order":
            outputs.reverse()
        elif mode == "batch_missing_output":
            outputs.pop()
        elif mode == "batch_bad_output_sha":
            outputs[-1]["output_sha256"] = "f" * 64
        response = {
            "protocol": protocol,
            "request_id": request["request_id"],
            "ok": True,
            "revision": revision,
            "profile": profile,
            "layer": request["layer"],
            "position": request["position"],
            "arena_epoch": request["arena_epoch"],
            "input": request["input"],
            "input_sha256": request["input_sha256"],
            "outputs": outputs,
            "catalog_sha256": catalog_sha,
            "gpu_slot_cache_entries": len(outputs),
            "activation_uploaded_bytes": request["input"]["bytes"],
        }
        print(json.dumps(response), flush=True)
        expected_epoch += 1
        if mode in {
            "batch_bad_order", "batch_missing_output", "batch_bad_output_sha"
        }:
            raise SystemExit(6)
        continue
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

    def _batch_assets(
        self,
        cache: Path,
        layer: int,
        suffixes: tuple[str, str],
    ) -> dict[
        str,
        tuple[attention.PackedFp8Asset, attention.PackedFp8Asset],
    ]:
        assets = {}
        for index, suffix in enumerate(suffixes, start=1):
            spec = attention.projection_spec(layer, suffix)
            n, k = int(spec["n"]), int(spec["k"])
            weight = self._asset(
                cache,
                f"{spec['name']}.weight",
                f"batch-{index}-weight.bin",
                n * k,
                f"{index + 2:x}" * 64,
                "F8_E4M3",
                (n, k),
            )
            scale = self._asset(
                cache,
                f"{spec['name']}.scale",
                f"batch-{index}-scale.bin",
                (n // 128) * (k // 128),
                f"{index + 4:x}" * 64,
                "F8_E8M0",
                (n // 128, k // 128),
            )
            assets[suffix] = (weight, scale)
        return assets

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

    def test_shared_batch_is_one_epoch_and_preserves_output_order_and_sha(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script, arena, _, _ = self._fixture(root)
            suffixes = ("wq_a", "wkv")
            assets = self._batch_assets(root / "range_cache", 42, suffixes)
            activation = torch.zeros((1, 1, 4096), dtype=torch.float32)
            with CleanPersistentClient(
                self._command(script, "success", root / "range_cache"),
                arena,
                timeout_seconds=5,
            ) as worker:
                outputs, evidence = worker.execute_shared_batch(
                    layer=42,
                    position=0,
                    suffixes=suffixes,
                    activation=activation,
                    assets=assets,
                )
                self.assertEqual(worker.epoch, 1)
                self.assertEqual(evidence["arena_epoch"], 0)
                self.assertEqual(tuple(outputs), suffixes)
                self.assertEqual(tuple(item["projection"]["name"] for item in evidence["outputs"]), (
                    "layers.42.attn.wq_a",
                    "layers.42.attn.wkv",
                ))
                for suffix, output, item in zip(suffixes, outputs.values(), evidence["outputs"]):
                    spec = attention.projection_spec(42, suffix)
                    expected_sha = hashlib.sha256(bytes(int(spec["n"]) * 4)).hexdigest()
                    self.assertEqual(item["output_sha256"], expected_sha)
                    self.assertEqual(tuple(output.shape), (1, 1, int(spec["n"])))
                    self.assertTrue(torch.equal(output, torch.zeros_like(output)))

    def test_shared_batch_output_arena_ranges_do_not_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script, arena, _, _ = self._fixture(root)
            suffixes = ("wq_b", "indexer.wq_b")
            assets = self._batch_assets(root / "range_cache", 42, suffixes)
            with CleanPersistentClient(
                self._command(script, "success", root / "range_cache"),
                arena,
                timeout_seconds=5,
            ) as worker:
                _, evidence = worker.execute_shared_batch(
                    layer=42,
                    position=0,
                    suffixes=suffixes,
                    activation=torch.zeros((1, 1, 1024), dtype=torch.float32),
                    assets=assets,
                )
            views = [evidence["input"]] + [
                item["output_written"] for item in evidence["outputs"]
            ]
            ranges = sorted(
                (int(view["offset"]), int(view["offset"]) + int(view["bytes"]))
                for view in views
            )
            self.assertTrue(all(left[1] <= right[0] for left, right in zip(ranges, ranges[1:])))

    def test_shared_batch_rejects_unapproved_and_duplicate_suffixes(self) -> None:
        cases = (
            (("wq_a", "wq_b"), "未批准|组合"),
            (("wq_a", "wq_a"), "重复|suffix"),
        )
        for suffixes, message in cases:
            with self.subTest(suffixes=suffixes), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                script, arena, _, _ = self._fixture(root)
                assets = self._batch_assets(root / "range_cache", 42, suffixes)
                worker = CleanPersistentClient(
                    self._command(script, "success", root / "range_cache"),
                    arena,
                    timeout_seconds=5,
                )
                try:
                    with self.assertRaisesRegex(attention.FullDepthPackedFp8Error, message):
                        worker.execute_shared_batch(
                            layer=42,
                            position=0,
                            suffixes=suffixes,
                            activation=torch.zeros((1, 1, 4096), dtype=torch.float32),
                            assets=assets,
                        )
                    self.assertTrue(worker.poisoned)
                finally:
                    worker.close()

    def test_shared_batch_response_drift_poisons_client(self) -> None:
        cases = (
            ("batch_bad_order", "顺序|身份|合同"),
            ("batch_missing_output", "缺项|数量|合同"),
            ("batch_bad_output_sha", "output 字节/SHA 漂移"),
        )
        for mode, message in cases:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                script, arena, _, _ = self._fixture(root)
                suffixes = ("wq_a", "wkv")
                assets = self._batch_assets(root / "range_cache", 42, suffixes)
                worker = CleanPersistentClient(
                    self._command(script, mode, root / "range_cache"),
                    arena,
                    timeout_seconds=5,
                )
                try:
                    with self.assertRaisesRegex(attention.FullDepthPackedFp8Error, message):
                        worker.execute_shared_batch(
                            layer=42,
                            position=0,
                            suffixes=suffixes,
                            activation=torch.zeros((1, 1, 4096), dtype=torch.float32),
                            assets=assets,
                        )
                    self.assertTrue(worker.poisoned)
                    self.assertEqual(worker.epoch, 0)
                finally:
                    worker.close()

    def test_output_chain_is_one_epoch_and_preserves_two_slot_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script, arena, _, _ = self._fixture(root)
            assets = self._batch_assets(root / "range_cache", 42, ("wo_a", "wo_b"))
            activation = torch.zeros((1, 1, 8, 4096), dtype=torch.float32)
            with CleanPersistentClient(
                self._command(script, "success", root / "range_cache"),
                arena,
                timeout_seconds=5,
            ) as worker:
                output, evidence = worker.execute_output_chain(
                    layer=42,
                    position=0,
                    activation=activation,
                    assets=assets,
                )
                self.assertEqual(worker.epoch, 1)
                self.assertEqual(evidence["arena_epoch"], 0)
                self.assertEqual(
                    [slot["projection"]["name"] for slot in evidence["slots"]],
                    ["layers.42.attn.wo_a", "layers.42.attn.wo_b"],
                )
                self.assertEqual(tuple(output.shape), (1, 1, 4096))
                self.assertTrue(torch.equal(output, torch.zeros_like(output)))
                input_end = evidence["input"]["offset"] + evidence["input"]["bytes"]
                self.assertLessEqual(input_end, evidence["output_written"]["offset"])

    def test_output_chain_rejects_asset_and_response_identity_drift(self) -> None:
        for mode, message in (
            ("chain_bad_slot_order", "slot 顺序|身份合同"),
            ("chain_bad_requant_sha", "顶层身份/SHA 合同"),
        ):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                script, arena, _, _ = self._fixture(root)
                assets = self._batch_assets(
                    root / "range_cache", 42, ("wo_a", "wo_b")
                )
                worker = CleanPersistentClient(
                    self._command(script, mode, root / "range_cache"),
                    arena,
                    timeout_seconds=5,
                )
                try:
                    with self.assertRaisesRegex(
                        attention.FullDepthPackedFp8Error, message
                    ):
                        worker.execute_output_chain(
                            layer=42,
                            position=0,
                            activation=torch.zeros(
                                (1, 1, 8, 4096), dtype=torch.float32
                            ),
                            assets=assets,
                        )
                    self.assertTrue(worker.poisoned)
                    self.assertEqual(worker.epoch, 0)
                finally:
                    worker.close()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script, arena, _, _ = self._fixture(root)
            assets = self._batch_assets(root / "range_cache", 42, ("wo_a", "wo_b"))
            assets.pop("wo_b")
            worker = CleanPersistentClient(
                self._command(script, "success", root / "range_cache"),
                arena,
                timeout_seconds=5,
            )
            try:
                with self.assertRaisesRegex(
                    attention.FullDepthPackedFp8Error, "assets 必须严格"
                ):
                    worker.execute_output_chain(
                        layer=42,
                        position=0,
                        activation=torch.zeros((1, 1, 8, 4096), dtype=torch.float32),
                        assets=assets,
                    )
                self.assertTrue(worker.poisoned)
                self.assertEqual(worker.epoch, 0)
            finally:
                worker.close()

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
