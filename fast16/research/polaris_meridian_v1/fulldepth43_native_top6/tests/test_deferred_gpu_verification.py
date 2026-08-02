from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from fast16.research.polaris_meridian_v1.fulldepth43_native_top6.executor import (
    ExecutionConfig,
    FullDepthError,
    _gpu_verifier_receipt_closure,
    _gpu_verification_owner,
)
from fast16.research.polaris_meridian_v1.l42_real_reference.l42_reference import (
    _InlineForward,
)
from fast16.research.polaris_meridian_v1.s14_first_real_token.executor import (
    NativeLayerReference,
)
from fast16.research.polaris_meridian_v1.s14_range_pack import online_range


PAYLOAD = b"\x10\x20\x30\x40"
OWNER = "vulkan_attention_worker"


class FakeResponse:
    def __init__(self, payload: bytes, *, start: int, end: int, total: int) -> None:
        self._payload = payload
        self._cursor = 0
        self.status = 206
        self.headers = {
            "Content-Range": f"bytes {start}-{end}/{total}",
            "Content-Length": str(len(payload)),
        }

    def read(self, size: int = -1) -> bytes:
        if self._cursor >= len(self._payload):
            return b""
        end = len(self._payload) if size < 0 else min(self._cursor + size, len(self._payload))
        result = self._payload[self._cursor:end]
        self._cursor = end
        return result

    def geturl(self) -> str:
        return "https://fixture.invalid/fixture.safetensors"

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        return None


class FakeTransport:
    def __init__(self) -> None:
        self.requests: list[tuple[int, int]] = []

    def open_range(self, url: str, start: int, end: int, timeout: float) -> FakeResponse:
        del url, timeout
        self.requests.append((start, end))
        offset = start - 8
        payload = PAYLOAD[offset : offset + end - start + 1]
        return FakeResponse(payload, start=start, end=end, total=64)


def gpu_entry() -> dict[str, Any]:
    return {
        "tensor": "layers.0.attn.wq_a.weight",
        "kind": "non_expert",
        "layer": 0,
        "file": "fixture.safetensors",
        "file_bytes": 64,
        "header_tensor_table_sha256": hashlib.sha256(b"fixture-table").hexdigest(),
        "start": 8,
        "end": 11,
        "bytes": 4,
        "dtype": "F8_E4M3",
        "shape": [4],
        "range_key": "fixture.safetensors:8-11",
    }


class DeferredGpuVerificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "range-cache"
        self.transport = FakeTransport()
        self.entry = gpu_entry()
        self.assertEqual(_gpu_verification_owner(self.entry), OWNER)

    def cache(
        self,
        *,
        allow_fetch: bool,
        deferred: bool,
    ) -> online_range.RangeCache:
        return online_range.RangeCache(
            self.root,
            endpoint="https://fixture.invalid",
            transport=self.transport,
            allow_fetch=allow_fetch,
            download_budget_bytes=64 if allow_fetch else 0,
            cache_budget_bytes=64,
            chunk_bytes=2,
            deferred_verifier=_gpu_verification_owner if deferred else None,
        )

    def seed_verified_hit(self) -> online_range.CachedRange:
        result = self.cache(allow_fetch=True, deferred=False).fetch(self.entry)
        self.assertFalse(result.cache_hit)
        self.assertTrue(result.content_verified)
        self.assertEqual(result.path.read_bytes(), PAYLOAD)
        return result

    def test_cache_hit_defers_content_sha_to_gpu_without_python_hash(self) -> None:
        seeded = self.seed_verified_hit()
        cache = self.cache(allow_fetch=False, deferred=True)

        with mock.patch.object(
            online_range.rp,
            "sha256_file",
            side_effect=AssertionError("延迟命中页禁止 Python 内容 SHA"),
        ) as sha256_file:
            result = cache.fetch(self.entry)

        sha256_file.assert_not_called()
        self.assertTrue(result.cache_hit)
        self.assertFalse(result.content_verified)
        self.assertEqual(result.verification_owner, OWNER)
        self.assertEqual(result.path, seeded.path)
        self.assertEqual(cache.proof_cache_telemetry["full_hashes"], 0)
        self.assertEqual(cache.proof_cache_telemetry["deferred"], 1)
        self.assertEqual(cache.proof_cache_telemetry["bytes_deferred"], len(PAYLOAD))

    def test_cache_miss_still_performs_complete_python_sha_before_publish(self) -> None:
        cache = self.cache(allow_fetch=True, deferred=True)
        original = online_range.rp.sha256_file

        with mock.patch.object(
            online_range.rp,
            "sha256_file",
            wraps=original,
        ) as sha256_file:
            result = cache.fetch(self.entry)

        self.assertEqual(sha256_file.call_count, 1)
        self.assertFalse(result.cache_hit)
        self.assertTrue(result.content_verified)
        self.assertEqual(result.verification_owner, "python_range_cache")
        self.assertEqual(result.proof["observed_sha256"], hashlib.sha256(PAYLOAD).hexdigest())
        self.assertEqual(cache.proof_cache_telemetry["full_hashes"], 1)
        self.assertEqual(cache.proof_cache_telemetry["deferred"], 0)

    def test_bad_metadata_is_rejected_before_any_content_hash(self) -> None:
        seeded = self.seed_verified_hit()
        cache = self.cache(allow_fetch=False, deferred=True)
        identity = cache._identity(self.entry)
        _, _, _, metadata_path = cache._paths(identity)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["cache_key"] = "0" * 64
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

        with mock.patch.object(
            online_range.rp,
            "sha256_file",
            side_effect=AssertionError("坏 metadata 必须先于内容 SHA 拒绝"),
        ) as sha256_file:
            with self.assertRaisesRegex(online_range.CacheEntryCorruption, "metadata format/key"):
                cache.fetch(self.entry)

        sha256_file.assert_not_called()
        self.assertFalse(seeded.path.exists(), "坏 metadata 与 payload 必须被隔离")

    def test_same_size_tamper_is_never_marked_python_verified(self) -> None:
        seeded = self.seed_verified_hit()
        before = seeded.path.stat()
        seeded.path.write_bytes(b"\x99\x88\x77\x66")
        os.utime(seeded.path, ns=(before.st_atime_ns, before.st_mtime_ns))
        cache = self.cache(allow_fetch=False, deferred=True)

        with mock.patch.object(
            online_range.rp,
            "sha256_file",
            side_effect=AssertionError("GPU 单一验证者模式不应由 Python 内容 SHA"),
        ) as sha256_file:
            result = cache.fetch(self.entry)

        sha256_file.assert_not_called()
        self.assertTrue(result.cache_hit)
        self.assertFalse(result.content_verified)
        self.assertEqual(result.verification_owner, OWNER)
        self.assertNotEqual(
            hashlib.sha256(result.path.read_bytes()).hexdigest(),
            result.proof["observed_sha256"],
            "测试必须真正制造同大小、同 mtime 的内容漂移",
        )

    def test_all_cpu_numeric_loaders_reject_deferred_pages(self) -> None:
        entries = {
            name: {
                "path": str(self.root / f"{name}.bin"),
                "content_verified": False,
                "verification_owner": OWNER,
            }
            for name in (
                "plain",
                "fp8.weight",
                "fp8.scale",
                "fp4.weight",
                "fp4.scale",
                "indices",
            )
        }
        specs = {
            "plain": ("F32", (1,)),
            "fp8.weight": ("F8_E4M3", (1,)),
            "fp8.scale": ("F8_E8M0", (1,)),
            "fp4.weight": ("I8", (1,)),
            "fp4.scale": ("F8_E8M0", (1,)),
            "indices": ("I64", (1,)),
        }
        inline = _InlineForward(SimpleNamespace(entries=entries, specs=specs))
        native = object.__new__(NativeLayerReference)
        native.bundle = SimpleNamespace(entries=entries, specs=specs)

        calls = (
            (inline._load_tensor, ("plain",)),
            (inline._weight_fp8, ("fp8",)),
            (inline._weight_fp4, ("fp4",)),
            (native._load_i64, ("indices",)),
        )
        for function, arguments in calls:
            with self.subTest(loader=function.__name__):
                with self.assertRaisesRegex(
                    ValueError,
                    r"vulkan_attention_worker.*禁止CPU数值路径读取",
                ):
                    function(*arguments)

    def test_gpu_verifier_ownership_requires_exclusive_full_gpu_path(self) -> None:
        worker = Path(self.tmp.name) / "worker.exe"
        worker.write_bytes(b"fixture")
        valid = ExecutionConfig(
            range_gpu_verifier_ownership=True,
            vulkan_bridge_capture=Path(self.tmp.name) / "captures",
            vulkan_writeback_worker=worker,
            vulkan_writeback_all_layers=True,
            vulkan_writeback_verify_cpu=False,
            vulkan_writeback_cpu_fallback=False,
            vulkan_writeback_fast_production=True,
            vulkan_attention_worker=worker,
            vulkan_attention_verify_cpu=False,
        )
        valid.validate()

        invalid_variants = (
            {"allow_fetch": True, "download_budget_bytes": 1},
            {"vulkan_attention_worker": None},
            {"vulkan_attention_verify_cpu": True},
            {"vulkan_writeback_all_layers": False},
            {"vulkan_writeback_fast_production": False},
            {"vulkan_writeback_verify_cpu": True},
            {"vulkan_writeback_cpu_fallback": True},
        )
        for changes in invalid_variants:
            with self.subTest(changes=changes):
                with self.assertRaises(FullDepthError):
                    ExecutionConfig(**{**valid.__dict__, **changes}).validate()

    def test_python_deferred_ownership_closes_to_rust_receipts(self) -> None:
        receipt = {
            "verification_owner": "rust_vulkan_worker",
            "verified_before_compute": True,
            "verified_count": 2,
            "verified_bytes": 12,
        }
        report = {
            "tokens": [
                {
                    "position": 0,
                    "layers": [
                        {
                            "layer": 0,
                            "vulkan_attention": [receipt],
                            "vulkan_writeback": {
                                "payload_verification": {
                                    **receipt,
                                    "verified_count": 42,
                                    "verified_bytes": 84,
                                }
                            },
                        }
                    ],
                }
            ]
        }
        telemetry = {
            "deferred_by_owner": {
                "vulkan_attention_worker": {"ranges": 2, "bytes": 12},
                "vulkan_moe_worker": {"ranges": 42, "bytes": 84},
            }
        }
        closed = _gpu_verifier_receipt_closure(report, telemetry)
        self.assertTrue(closed["closed"])
        telemetry["deferred_by_owner"]["vulkan_moe_worker"]["bytes"] = 85
        self.assertFalse(_gpu_verifier_receipt_closure(report, telemetry)["closed"])


if __name__ == "__main__":
    unittest.main()
