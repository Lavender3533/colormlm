from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fast16.memory_bridge import CompiledMemoryRetriever


class MemoryBridgeTests(unittest.TestCase):
    def test_bm25_prefers_matching_technical_record(self) -> None:
        records = [
            {"key": "GQA", "value": "GQA 通过多个查询头共享 KV 头来减少 KV Cache。"},
            {"key": "Java", "value": "Java 整数加法使用加号。"},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "records.jsonl"
            path.write_text(
                "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
                encoding="utf-8",
            )
            retriever = CompiledMemoryRetriever([path])
            hit = retriever.search("什么是 GQA，它怎样减少 KV Cache？")
            self.assertIsNotNone(hit)
            self.assertEqual(hit.key, "GQA")


if __name__ == "__main__":
    unittest.main()
