from __future__ import annotations

import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import torch

from fast16.research.polaris_meridian_v1.fulldepth43_native_top6.vulkan_writeback import (
    PROTOCOL as WRITEBACK_PROTOCOL,
)

from ..cached_layer_replay import (
    CachedLayerReplayError,
    PersistentCachedLayerReplay,
)


FAKE_WORKER = r'''
import hashlib
import json
import struct
import sys
from pathlib import Path

PROTOCOL = "polaris-fulldepth43-vulkan-writeback-v1"
OUTPUT_FILE = "vulkan_moe_block_branches.bf16le.bin"
scenario = sys.argv[1]

hello = {
    "protocol": PROTOCOL,
    "op": "hello",
    "ready": True,
    "causal_block_layer_replay": True,
    "causal_block_sizes": [4, 8],
    "batch_payload_verification": True,
}
if scenario == "hello":
    hello["causal_block_sizes"] = [4]
print(json.dumps(hello, separators=(",", ":")), flush=True)

for line in sys.stdin:
    request = json.loads(line)
    if set(request) != {
        "protocol", "op", "request_id", "manifests", "batch_verify_payloads"
    } or request["batch_verify_payloads"] is not True:
        print(json.dumps({
            "protocol": PROTOCOL,
            "request_id": request.get("request_id"),
            "ok": False,
            "error": "request keys drift",
        }, separators=(",", ":")), flush=True)
        continue

    paths = [Path(value) for value in request["manifests"]]
    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    positions = [item["position"] for item in manifests]
    layer = manifests[0]["layer"]
    manifest_sha256s = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    outputs = []
    rows = []
    for index, (path, position) in enumerate(zip(paths, positions)):
        bits = struct.unpack("<I", struct.pack("<f", float(position + 1)))[0] >> 16
        payload = struct.pack("<H", bits) + bytes(8190)
        if scenario == "nonfinite" and index == 0:
            payload = struct.pack("<H", 0x7F80) + bytes(8190)
        rows.append(payload)
    output_path = paths[0].parent / OUTPUT_FILE
    if scenario == "path_escape":
        output_path = paths[0].parent.parent / "escaped" / OUTPUT_FILE
        output_path.parent.mkdir(exist_ok=True)
    if scenario == "filename":
        output_path = paths[0].parent / "wrong-name.bin"
    combined = b"".join(rows)
    if scenario == "truncated":
        combined = combined[:-2]
    output_path.write_bytes(combined)
    for index, (path, position, payload) in enumerate(zip(paths, positions, rows)):
        output_sha256 = hashlib.sha256(payload).hexdigest()
        outputs.append({
            "position": position,
            "input_token_id": manifests[index]["input_token_id"],
            "manifest_sha256": manifest_sha256s[index],
            "expert_ids": manifests[index]["expert_ids"],
            "output": {
                "path": str(output_path.resolve()),
                "offset": index * 8192,
                "dtype": "bf16_le",
                "shape": [1, 1, 4096],
                "bytes": 8192,
                "sha256": output_sha256,
            },
        })

    response = {
        "protocol": PROTOCOL,
        "request_id": request["request_id"],
        "ok": True,
        "mode": "causal_block_layer_replay",
        "block_size": len(paths),
        "layer": layer,
        "positions": positions,
        "outputs": outputs,
        "speed_eligible_verifier": False,
    }
    if scenario == "worker_error":
        response["ok"] = False
        response["error"] = "injected failure"
    elif scenario == "protocol":
        response["protocol"] = "wrong"
    elif scenario == "request_id":
        response["request_id"] += "-wrong"
    elif scenario == "mode":
        response["mode"] = "wrong"
    elif scenario == "block_size":
        response["block_size"] += 1
    elif scenario == "layer":
        response["layer"] += 1
    elif scenario == "positions":
        response["positions"] = list(reversed(positions))
    elif scenario == "speed":
        response["speed_eligible_verifier"] = True
    elif scenario == "manifest_missing":
        del response["outputs"][0]["manifest_sha256"]
    elif scenario == "manifest_order":
        response["outputs"][0]["manifest_sha256"] = manifest_sha256s[-1]
    elif scenario == "manifest_bad":
        response["outputs"][0]["manifest_sha256"] = "0" * 64
    elif scenario == "outputs_count":
        response["outputs"] = outputs[:-1]
    elif scenario == "output_position":
        response["outputs"][0]["position"] += 1
    elif scenario == "input_token":
        response["outputs"][0]["input_token_id"] += 1
    elif scenario == "experts":
        response["outputs"][0]["expert_ids"] = list(reversed(
            response["outputs"][0]["expert_ids"]
        ))
    elif scenario == "offset":
        response["outputs"][1]["output"]["offset"] = 0
    elif scenario == "dtype":
        response["outputs"][0]["output"]["dtype"] = "bf16"
    elif scenario == "shape":
        response["outputs"][0]["output"]["shape"] = [1, 4096]
    elif scenario == "bytes":
        response["outputs"][0]["output"]["bytes"] = 8190
    elif scenario == "sha":
        response["outputs"][0]["output"]["sha256"] = "0" * 64

    encoded = json.dumps(response, separators=(",", ":"))
    if scenario == "response_duplicate":
        encoded = encoded[:-1] + ',"ok":true}'
    elif scenario == "response_nan":
        encoded = encoded[:-1] + ',"invalid":NaN}'
    elif scenario == "response_oversize":
        response["padding"] = "x" * 65536
        encoded = json.dumps(response, separators=(",", ":"))
    print(encoded, flush=True)
'''


class CachedLayerReplayTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.script = self.root / "fake_cached_replay_worker.py"
        self.script.write_text(
            textwrap.dedent(FAKE_WORKER),
            encoding="utf-8",
            newline="\n",
        )
        self._manifest_generation = 0

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _manifests(
        self,
        count: int,
        *,
        layer: int = 7,
        start: int = 0,
    ) -> tuple[Path, ...]:
        self._manifest_generation += 1
        generation = self._manifest_generation
        paths = []
        for index in range(count):
            capture = self.root / f"capture-{generation}-{count}-{index}"
            capture.mkdir(exist_ok=True)
            path = capture / "bridge_manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "layer": layer,
                        "position": start + index,
                        "input_token_id": 1000 + start + index,
                        "expert_ids": [0, 1, 2, 3, 4, 5],
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                encoding="utf-8",
                newline="\n",
            )
            paths.append(path.resolve())
        return tuple(paths)

    def _worker(self, scenario: str = "success") -> PersistentCachedLayerReplay:
        return PersistentCachedLayerReplay(
            (sys.executable, "-X", "utf8", str(self.script), scenario),
            timeout_seconds=5,
        )

    def _assert_poisoned(self, scenario: str, pattern: str) -> None:
        manifests = self._manifests(4)
        worker = self._worker(scenario)
        process = worker.process
        with self.assertRaisesRegex(CachedLayerReplayError, pattern):
            worker.execute(manifests)
        self.assertTrue(worker.poisoned)
        self.assertIsNone(worker.process)
        self.assertIsNotNone(process)
        self.assertIsNotNone(process.poll())
        with self.assertRaisesRegex(CachedLayerReplayError, "poisoned"):
            worker.execute(manifests)

    def test_k4_success_and_evidence(self) -> None:
        manifests = self._manifests(4, layer=7, start=10)
        with self._worker() as worker:
            tensors, evidence = worker.execute(manifests)

        self.assertEqual(len(tensors), 4)
        self.assertTrue(all(tensor.dtype == torch.bfloat16 for tensor in tensors))
        self.assertTrue(all(tuple(tensor.shape) == (1, 1, 4096) for tensor in tensors))
        self.assertEqual(
            [float(tensor[0, 0, 0].float().item()) for tensor in tensors],
            [11.0, 12.0, 13.0, 14.0],
        )
        self.assertEqual(evidence["protocol"], WRITEBACK_PROTOCOL)
        self.assertEqual(evidence["block_size"], 4)
        self.assertEqual(evidence["layer"], 7)
        self.assertEqual(evidence["positions"], [10, 11, 12, 13])
        self.assertIs(evidence["speed_eligible_verifier"], False)

    def test_k8_success_preserves_request_order(self) -> None:
        manifests = self._manifests(8, layer=42, start=20)
        with self._worker() as worker:
            tensors, evidence = worker.execute(manifests)

        self.assertEqual(evidence["positions"], list(range(20, 28)))
        self.assertEqual(
            [float(tensor[0, 0, 0].float().item()) for tensor in tensors],
            [float(value) for value in range(21, 29)],
        )
        self.assertEqual(
            evidence["output_paths"],
            [
                str(manifests[0].parent / "vulkan_moe_block_branches.bf16le.bin")
            ]
            * 8,
        )

    def test_rejects_invalid_manifest_count(self) -> None:
        for count in (1, 5):
            with self.subTest(count=count):
                worker = self._worker()
                with self.assertRaisesRegex(CachedLayerReplayError, "数量只能"):
                    worker.execute(self._manifests(count))
                self.assertTrue(worker.poisoned)

    def test_rejects_relative_missing_path_and_wrong_filename(self) -> None:
        valid = list(self._manifests(4))
        cases = []
        relative = list(valid)
        relative[0] = Path("relative/bridge_manifest.json")
        cases.append((relative, "绝对路径"))
        missing = list(valid)
        missing[0] = (self.root / "missing" / "bridge_manifest.json").resolve()
        cases.append((missing, "不存在或不可访问"))
        wrong = self.root / "capture-wrong" / "wrong.json"
        wrong.parent.mkdir()
        wrong.write_text('{"layer":7,"position":0}', encoding="utf-8")
        wrong_paths = list(valid)
        wrong_paths[0] = wrong.resolve()
        cases.append((wrong_paths, "文件名"))

        for paths, pattern in cases:
            with self.subTest(pattern=pattern):
                worker = self._worker()
                with self.assertRaisesRegex(CachedLayerReplayError, pattern):
                    worker.execute(paths)
                self.assertTrue(worker.poisoned)

    def test_rejects_mixed_layers_duplicate_and_skipped_positions(self) -> None:
        cases = []
        mixed = list(self._manifests(4))
        mixed_document = json.loads(mixed[-1].read_text(encoding="utf-8"))
        mixed_document["layer"] = 8
        mixed[-1].write_text(json.dumps(mixed_document), encoding="utf-8")
        cases.append((mixed, "同一层"))

        duplicate = list(self._manifests(4, start=10))
        duplicate_document = json.loads(duplicate[2].read_text(encoding="utf-8"))
        duplicate_document["position"] = 11
        duplicate[2].write_text(json.dumps(duplicate_document), encoding="utf-8")
        cases.append((duplicate, "严格连续"))

        skipped = list(self._manifests(4, start=20))
        skipped_document = json.loads(skipped[2].read_text(encoding="utf-8"))
        skipped_document["position"] = 23
        skipped[2].write_text(json.dumps(skipped_document), encoding="utf-8")
        cases.append((skipped, "严格连续"))

        for paths, pattern in cases:
            with self.subTest(pattern=pattern):
                worker = self._worker()
                with self.assertRaisesRegex(CachedLayerReplayError, pattern):
                    worker.execute(paths)
                self.assertTrue(worker.poisoned)

    def test_rejects_response_identity_drifts(self) -> None:
        for scenario in (
            "protocol",
            "request_id",
            "mode",
            "block_size",
            "layer",
            "positions",
        ):
            with self.subTest(scenario=scenario):
                self._assert_poisoned(scenario, "身份/block/layer/positions")

    def test_rejects_hello_drift(self) -> None:
        with self.assertRaisesRegex(CachedLayerReplayError, "hello 合同漂移"):
            self._worker("hello")

    def test_rejects_non_strict_or_oversized_response_json(self) -> None:
        scenarios = {
            "response_duplicate": "重复 key",
            "response_nan": "非有限常量",
            "response_oversize": "超过 64 KiB",
        }
        for scenario, pattern in scenarios.items():
            with self.subTest(scenario=scenario):
                self._assert_poisoned(scenario, pattern)

    def test_rejects_speed_eligible_true(self) -> None:
        self._assert_poisoned("speed", "speed eligible")

    def test_rejects_manifest_sha_missing_reordered_or_wrong(self) -> None:
        for scenario in ("manifest_missing", "manifest_order", "manifest_bad"):
            with self.subTest(scenario=scenario):
                self._assert_poisoned(scenario, "身份/manifest SHA")

    def test_rejects_output_metadata_drifts(self) -> None:
        scenarios = {
            "outputs_count": "数量漂移",
            "output_position": "身份/manifest SHA",
            "input_token": "身份/manifest SHA",
            "experts": "身份/manifest SHA",
            "offset": "offset/dtype/shape/bytes",
            "dtype": "offset/dtype/shape/bytes",
            "shape": "offset/dtype/shape/bytes",
            "bytes": "offset/dtype/shape/bytes",
        }
        for scenario, pattern in scenarios.items():
            with self.subTest(scenario=scenario):
                self._assert_poisoned(scenario, pattern)

    def test_rejects_output_escape_and_filename_drift(self) -> None:
        self._assert_poisoned("path_escape", "capture 边界")
        self._assert_poisoned("filename", "文件名漂移")

    def test_rejects_output_sha_truncation_and_nonfinite_bf16(self) -> None:
        scenarios = {
            "sha": "SHA-256",
            "truncated": "字节数漂移",
            "nonfinite": "非有限 BF16",
        }
        for scenario, pattern in scenarios.items():
            with self.subTest(scenario=scenario):
                self._assert_poisoned(scenario, pattern)

    def test_worker_error_poisons_and_blocks_reuse(self) -> None:
        self._assert_poisoned("worker_error", "拒绝请求")

    def test_close_reaps_worker_and_is_idempotent(self) -> None:
        worker = self._worker()
        process = worker.process
        self.assertIsNotNone(process)
        worker.close()
        worker.close()

        self.assertTrue(worker.closed)
        self.assertFalse(worker.poisoned)
        self.assertIsNone(worker.process)
        self.assertIsNotNone(process.poll())


if __name__ == "__main__":
    unittest.main()
