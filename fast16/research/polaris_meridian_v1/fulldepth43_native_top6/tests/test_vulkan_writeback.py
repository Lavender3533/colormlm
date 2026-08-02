from __future__ import annotations

import sys
import hashlib
import json
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

def verification_receipt(payloads):
    payloads = sorted(payloads, key=lambda payload: payload["tensor"])
    digest = hashlib.sha256()
    digest.update(b"polaris-rust-vulkan-payload-identity-v1\0")
    for item in payloads:
        tensor = item["tensor"].encode("utf-8")
        digest.update(len(tensor).to_bytes(8, "little"))
        digest.update(tensor)
        digest.update(item["bytes"].to_bytes(8, "little"))
        digest.update(item["sha256"].encode("ascii"))
    return {
        "verification_owner": "rust_vulkan_worker",
        "verified_count": len(payloads),
        "verified_bytes": sum(item["bytes"] for item in payloads),
        "payload_identity_sha256": digest.hexdigest(),
        "payload_identity_contract": "sha256(v1_nul || sorted(length_le64(tensor),tensor,bytes_le64,expected_sha256_ascii))",
        "verified_before_compute": True,
        "verification_scope": "all_listed_payloads_before_corresponding_gpu_compute",
    }
print(json.dumps({
    "protocol": protocol,
    "op": "hello",
    "ready": True,
    "device": "fixture",
    "persistent_context": True,
    "official_boundary_graph": True,
    "inline_manifest_json": True,
    "batch_payload_verification": True,
    "batch_payload_verification_concurrency_limit": 8,
}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    if "manifest_json" in request:
        assert "manifest" not in request
        manifest_bytes = request["manifest_json"].encode("utf-8")
        assert hashlib.sha256(manifest_bytes).hexdigest() == request["manifest_sha256"]
        manifest_document = json.loads(request["manifest_json"])
        capture_root = Path(request["capture_root"]).resolve()
        manifest_sha256 = request["manifest_sha256"]
        manifest_transport = "inline_json"
    else:
        manifest = Path(request["manifest"]).resolve()
        manifest_bytes = manifest.read_bytes()
        manifest_document = json.loads(manifest_bytes)
        capture_root = manifest.parent
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        manifest_transport = "capture_file"
    output = capture_root / "vulkan_moe_branch.bf16le.bin"
    payload = bytes(8192)
    output.write_bytes(payload)
    batch_enabled = request.get("batch_verify_payloads") is True
    payload_count = manifest_document["payload_count"]
    print(json.dumps({
        "protocol": protocol,
        "request_id": request["request_id"],
        "ok": True,
        "device": "fixture",
        "manifest_sha256": manifest_sha256,
        "manifest_transport": manifest_transport,
        "layer": manifest_document["layer"],
        "position": manifest_document["position"],
        "input_token_id": manifest_document["input_token_id"],
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
        "batch_payload_verification": {
            "enabled": batch_enabled,
            "batch_entries": payload_count if batch_enabled else 0,
            "batch_hits": 0,
            "batch_misses": payload_count if batch_enabled else 0,
            "batch_disk_bytes_read": 3 if batch_enabled else 0,
            "concurrency_limit": 8,
            "followup_cached_loader_hits": payload_count if batch_enabled else 0,
            "all_verified_before_compute": batch_enabled,
        },
        **verification_receipt(manifest_document["payloads"]),
    }), flush=True)
"""


class VulkanWritebackTests(unittest.TestCase):
    @staticmethod
    def _manifest_document(*, position: int, layer: int, token_id: int) -> dict[str, object]:
        return {
            "position": position,
            "layer": layer,
            "input_token_id": token_id,
            "payload_count": 2,
            "payloads": [
                {
                    "tensor": f"layers.{layer}.fixture.scale",
                    "bytes": 1,
                    "sha256": "1" * 64,
                },
                {
                    "tensor": f"layers.{layer}.fixture.weight",
                    "bytes": 2,
                    "sha256": "2" * 64,
                },
            ],
        }

    @staticmethod
    def _write_manifest(root: Path, *, position: int, layer: int, token_id: int) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        manifest = root / "bridge_manifest.json"
        manifest.write_text(
            json.dumps(
                VulkanWritebackTests._manifest_document(
                    position=position,
                    layer=layer,
                    token_id=token_id,
                ),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return manifest

    @staticmethod
    def _canonical_manifest_sha256(document: dict[str, object]) -> str:
        encoded = json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def test_persistent_protocol_reads_hash_verified_bf16(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "fake_worker.py"
            script.write_text(textwrap.dedent(FAKE_WORKER), encoding="utf-8", newline="\n")
            manifest = self._write_manifest(root, position=0, layer=42, token_id=0)
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
                self.assertFalse(evidence["batch_payload_verification"]["enabled"])

    def test_explicit_batch_verification_receipt_closes_before_compute(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "fake_worker.py"
            script.write_text(textwrap.dedent(FAKE_WORKER), encoding="utf-8", newline="\n")
            manifest = self._write_manifest(root, position=0, layer=42, token_id=0)
            with PersistentVulkanWriteback(
                (sys.executable, "-X", "utf8", str(script)),
                timeout_seconds=5,
                batch_verify_payloads=True,
            ) as worker:
                _, evidence = worker.execute(manifest)
            receipt = evidence["batch_payload_verification"]
            self.assertTrue(receipt["enabled"])
            self.assertEqual(receipt["batch_entries"], 2)
            self.assertEqual(receipt["batch_misses"], 2)
            self.assertEqual(receipt["followup_cached_loader_hits"], 2)
            self.assertTrue(receipt["all_verified_before_compute"])

    def test_persistent_protocol_accepts_next_token_after_layer_42(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "fake_worker.py"
            script.write_text(textwrap.dedent(FAKE_WORKER), encoding="utf-8", newline="\n")
            first = self._write_manifest(
                root / "position-000000" / "layer-42",
                position=0,
                layer=42,
                token_id=0,
            )
            second = self._write_manifest(
                root / "position-000001" / "layer-00",
                position=1,
                layer=0,
                token_id=5,
            )
            with PersistentVulkanWriteback(
                (sys.executable, "-X", "utf8", str(script)), timeout_seconds=5
            ) as worker:
                worker.execute(first)
                worker.execute(second)
                self.assertEqual(worker.counter, 2)

    def test_inline_manifest_is_sent_directly_without_bridge_manifest_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "fake_worker.py"
            script.write_text(textwrap.dedent(FAKE_WORKER), encoding="utf-8", newline="\n")
            capture_root = root / "live-layer"
            capture_root.mkdir()
            document = self._manifest_document(position=0, layer=42, token_id=0)

            with PersistentVulkanWriteback(
                (sys.executable, "-X", "utf8", str(script)), timeout_seconds=5
            ) as worker:
                tensor, evidence = worker.execute(document, capture_root=capture_root)

            self.assertFalse((capture_root / "bridge_manifest.json").exists())
            self.assertEqual(tuple(tensor.shape), (1, 1, 4096))
            self.assertEqual(evidence["manifest_transport"], "inline_json")
            self.assertEqual(
                evidence["manifest_sha256"],
                self._canonical_manifest_sha256(document),
            )

    def test_worker_requires_inline_manifest_capability_at_startup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "fake_worker.py"
            script.write_text(
                textwrap.dedent(FAKE_WORKER).replace(
                    '"inline_manifest_json": True,',
                    '"inline_manifest_json": False,',
                    1,
                ),
                encoding="utf-8",
                newline="\n",
            )
            with self.assertRaisesRegex(VulkanWritebackError, "hello 合同漂移"):
                PersistentVulkanWriteback(
                    (sys.executable, "-X", "utf8", str(script)), timeout_seconds=5
                )

    def test_inline_manifest_rejects_response_transport_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "fake_worker.py"
            script.write_text(
                textwrap.dedent(FAKE_WORKER).replace(
                    '"manifest_transport": manifest_transport,',
                    '"manifest_transport": "capture_file",',
                    1,
                ),
                encoding="utf-8",
                newline="\n",
            )
            capture_root = root / "live-layer"
            capture_root.mkdir()
            document = self._manifest_document(position=0, layer=42, token_id=0)

            with PersistentVulkanWriteback(
                (sys.executable, "-X", "utf8", str(script)), timeout_seconds=5
            ) as worker:
                with self.assertRaisesRegex(
                    VulkanWritebackError,
                    "response 身份/SHA 漂移",
                ):
                    worker.execute(document, capture_root=capture_root)

    def test_inline_manifest_sequence_is_shared_with_legacy_state_machine(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "fake_worker.py"
            script.write_text(textwrap.dedent(FAKE_WORKER), encoding="utf-8", newline="\n")
            first_root = root / "position-000000" / "layer-42"
            first_root.mkdir(parents=True)
            second_root = root / "position-000001" / "layer-00"
            second_root.mkdir(parents=True)
            legacy_root = root / "position-000001" / "layer-01"
            legacy = self._write_manifest(
                legacy_root,
                position=1,
                layer=1,
                token_id=5,
            )

            with PersistentVulkanWriteback(
                (sys.executable, "-X", "utf8", str(script)), timeout_seconds=5
            ) as worker:
                worker.execute(
                    self._manifest_document(position=0, layer=42, token_id=0),
                    capture_root=first_root,
                )
                worker.execute(
                    self._manifest_document(position=1, layer=0, token_id=5),
                    capture_root=second_root,
                )
                worker.execute(legacy)
                self.assertEqual(worker.counter, 3)

    def test_inline_manifest_rejects_sequence_skip_before_worker_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "fake_worker.py"
            script.write_text(textwrap.dedent(FAKE_WORKER), encoding="utf-8", newline="\n")
            first_root = root / "first"
            first_root.mkdir()
            skipped_root = root / "skipped"
            skipped_root.mkdir()

            with PersistentVulkanWriteback(
                (sys.executable, "-X", "utf8", str(script)), timeout_seconds=5
            ) as worker:
                worker.execute(
                    self._manifest_document(position=0, layer=42, token_id=0),
                    capture_root=first_root,
                )
                with self.assertRaisesRegex(VulkanWritebackError, "请求序列漂移"):
                    worker.execute(
                        self._manifest_document(position=2, layer=0, token_id=5),
                        capture_root=skipped_root,
                    )

    def test_inline_manifest_rejects_invalid_identity_before_compute(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "fake_worker.py"
            script.write_text(textwrap.dedent(FAKE_WORKER), encoding="utf-8", newline="\n")
            capture_root = root / "invalid"
            capture_root.mkdir()
            invalid = self._manifest_document(position=0, layer=0, token_id=0)
            invalid["position"] = True

            with PersistentVulkanWriteback(
                (sys.executable, "-X", "utf8", str(script)), timeout_seconds=5
            ) as worker:
                with self.assertRaisesRegex(
                    VulkanWritebackError,
                    "position/layer/token 合同漂移",
                ):
                    worker.execute(invalid, capture_root=capture_root)
            self.assertFalse((capture_root / "vulkan_moe_branch.bf16le.bin").exists())

    def test_persistent_protocol_rejects_position_skip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "fake_worker.py"
            script.write_text(textwrap.dedent(FAKE_WORKER), encoding="utf-8", newline="\n")
            first = self._write_manifest(root / "first", position=0, layer=42, token_id=0)
            skipped = self._write_manifest(root / "skipped", position=2, layer=0, token_id=5)
            with PersistentVulkanWriteback(
                (sys.executable, "-X", "utf8", str(script)), timeout_seconds=5
            ) as worker:
                worker.execute(first)
                with self.assertRaisesRegex(VulkanWritebackError, "请求序列漂移"):
                    worker.execute(skipped)

    def test_exact_comparison_rejects_one_bf16_bit(self) -> None:
        cpu = torch.zeros((1, 1, 4096), dtype=torch.bfloat16)
        gpu = cpu.clone()
        self.assertTrue(verify_exact_bf16_writeback(cpu, gpu)["exact_bf16_equal"])
        gpu[..., 7] = 1
        with self.assertRaises(VulkanWritebackError):
            verify_exact_bf16_writeback(cpu, gpu)


if __name__ == "__main__":
    unittest.main()
