from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fast16.clm.format import ClmReader, pack_checkpoint
from fast16.clm.doccompile import compile_documents
from fast16.clm.model import ZeroTrainModel


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "colormlm" / "data" / "v3_final.pt"
MEMORY = ROOT / "fast16" / "data" / "bootstrap_memory.jsonl"
MEMORY_V1 = ROOT / "fast16" / "data" / "bootstrap_memory_v1.jsonl"


class ClmRoundTripTests(unittest.TestCase):
    def test_document_compiler_emits_utf8_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "知识.md"
            output = Path(temp_dir) / "memory.jsonl"
            source.write_text("# GQA\n分组查询注意力可以减少 KV Cache。\n", encoding="utf-8")
            summary = compile_documents([source], output, max_value_bytes=128)
            content = output.read_text(encoding="utf-8")
            self.assertEqual(summary["files"], 1)
            self.assertGreater(summary["records"], 0)
            self.assertIn("GQA", content)
            self.assertIn("KV Cache", content)
            self.assertEqual(content.count('"key":"什么是GQA"'), 1)

    def test_pack_read_and_forward(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test.clm"
            pack_checkpoint(
                CHECKPOINT,
                model_path,
                memory_paths=[MEMORY],
                tokenizer_mode="character",
            )

            with ClmReader(model_path) as reader:
                self.assertEqual(reader.metadata["architecture"], "colormlm_zerotrain_v0")
                self.assertEqual(reader.verify_tensors(), [])
                self.assertTrue(reader.has_tensor("memory.keys"))
                self.assertGreater(reader.summary()["tensor_count"], 60)

            model = ZeroTrainModel.from_clm(model_path)
            result = model.generate("def max", new_tokens=8, refinement_steps=4)
            self.assertTrue(result.text.startswith("def max"))
            self.assertGreater(result.memory_records, 0)
            self.assertGreater(result.steps, 0)

            recalled = model.generate("def max", new_tokens=36, refinement_steps=8)
            self.assertEqual(
                recalled.text,
                "def max(a,b):\n  if a>b: return a\n  return b",
            )

    def test_utf8_byte_transplant_and_chinese_memory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test-v1.clm"
            pack_checkpoint(CHECKPOINT, model_path, memory_paths=[MEMORY_V1])
            model = ZeroTrainModel.from_clm(model_path)

            self.assertEqual(model.tokenizer["type"], "utf8_byte")
            self.assertEqual(model.token_count, 258)
            result = model.generate("你好", refinement_steps=8)
            self.assertEqual(result.text, "你好！我是 ColorLM ZeroTrain。")


if __name__ == "__main__":
    unittest.main()
