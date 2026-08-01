from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pack_capture import HEADER, MAGIC, pack_capture


class PackCaptureTest(unittest.TestCase):
    def test_pack_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = root / "capture.bin"
            with records.open("wb") as output:
                for kind, array in (
                    (1, np.arange(10, dtype=np.float32).reshape(2, 5)),
                    (2, np.arange(6, dtype=np.float32).reshape(2, 3)),
                    (3, np.arange(8, dtype=np.float32).reshape(2, 4)),
                ):
                    output.write(
                        HEADER.pack(MAGIC, 1, kind, 0, 0, 0, array.shape[1], 2, 1, 1, array.nbytes)
                    )
                    output.write(array.tobytes())
            metadata = root / "metadata.jsonl"
            metadata.write_text(
                '\n'.join(
                    json.dumps(row)
                    for row in (
                        {"target_token_id": 1, "task_id": "a", "sample_id": "a0"},
                        {"target_token_id": 2, "task_id": "b", "sample_id": "b0"},
                    )
                ) + '\n',
                encoding="utf-8",
            )
            package = root / "head"
            package.mkdir()
            base_ids = np.asarray([0, 1, 3, 4], dtype="<i8")
            (package / "weights.bin").write_bytes(base_ids.tobytes())
            (package / "head.json").write_text(
                json.dumps(
                    {
                        "mapping": {"source_map_sha256": "0" * 64},
                        "weights": {"file": "weights.bin"},
                        "tensors": [
                            {
                                "name": "mapping.base_ids",
                                "dtype": "I64",
                                "ggml_shape": [4],
                                "offset": 0,
                                "bytes": 32,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            capture = root / "capture.npz"
            manifest = root / "capture.json"
            report = pack_capture(records, metadata, package, capture, manifest)
            self.assertEqual(report["rows"], 2)
            with np.load(capture, allow_pickle=False) as data:
                self.assertEqual(data["base_logits"].shape, (2, 5))
                self.assertTrue(np.array_equal(data["donor_0_base_ids"], base_ids))
                self.assertEqual(data["task_ids"].tolist(), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
