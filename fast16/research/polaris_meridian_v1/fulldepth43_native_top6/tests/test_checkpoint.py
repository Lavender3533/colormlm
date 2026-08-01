from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from fast16.research.polaris_meridian_v1.fulldepth43_native_top6 import checkpoint
from fast16.research.polaris_meridian_v1.fulldepth43_native_top6 import executor as fd43
from fast16.research.polaris_meridian_v1.s14_chat_encoding.forced_prefill import (
    S14_TOKENIZER_SHA256,
)
from fast16.research.polaris_meridian_v1.s14_first_real_token import executor as s14
from fast16.research.polaris_meridian_v1.speculative_full_verifier.fulldepth_runtime_bridge import (
    FullDepthDecoderStateBridge,
    build_cpu_causal_block_reference_backend,
)
from fast16.research.polaris_meridian_v1.speculative_full_verifier.runtime_controller import (
    decide_acceptance,
)
from fast16.research.polaris_meridian_v1.speculative_full_verifier.verifier import (
    VerificationRequest,
)


def layer_states(position: int) -> dict[int, s14.LayerRuntimeState]:
    compressed4 = (position + 1) // 4
    compressed128 = (position + 1) // 128
    result: dict[int, s14.LayerRuntimeState] = {}
    for layer in range(43):
        ratio = fd43.FULLDEPTH43_NATIVE_TOP6.ratio_for(layer)
        compressor = None
        if ratio == 4:
            compressor = s14.CompressorRemainderState(
                ratio=4,
                overlap=True,
                main_kv_state=torch.full((1, 8, 1024), layer + position, dtype=torch.float32),
                main_score_state=torch.full((1, 8, 1024), layer - position, dtype=torch.float32),
                main_compressed_kv=torch.full(
                    (1, compressed4, 512), layer, dtype=torch.bfloat16
                ),
                indexer_kv_state=torch.full((1, 8, 256), position, dtype=torch.float32),
                indexer_score_state=torch.full((1, 8, 256), -position, dtype=torch.float32),
                indexer_compressed_kv=torch.full(
                    (1, compressed4, 128), layer + 1, dtype=torch.bfloat16
                ),
            )
        elif ratio == 128:
            compressor = s14.CompressorRemainderState(
                ratio=128,
                overlap=False,
                main_kv_state=torch.full((1, 128, 512), layer + position, dtype=torch.float32),
                main_score_state=torch.full((1, 128, 512), layer - position, dtype=torch.float32),
                main_compressed_kv=torch.full(
                    (1, compressed128, 512), layer, dtype=torch.bfloat16
                ),
                indexer_kv_state=None,
                indexer_score_state=None,
                indexer_compressed_kv=None,
            )
        result[layer] = s14.LayerRuntimeState(
            layer=layer,
            position=position,
            window_kv=torch.full(
                (1, s14.WINDOW_SIZE, 512), layer + position, dtype=torch.bfloat16
            ),
            compressor=compressor,
        )
    return result


def one_token_state() -> fd43.DecoderState:
    decoder = fd43.DecoderState(input_token_id=s14.BOS_TOKEN_ID)
    decoder.commit(
        output_token_id=91,
        next_states=layer_states(0),
        profile=fd43.FULLDEPTH43_NATIVE_TOP6,
    )
    return decoder


def five_token_forced_state() -> fd43.DecoderState:
    queue = s14.ForcedTokenQueue(
        token_ids=(0, 10, 11, 12, 13),
        cursor=0,
        artifact_sha256="a" * 64,
    )
    decoder = fd43.DecoderState(input_token_id=0, forced_queue=queue)
    for position in range(5):
        decoder.commit(
            output_token_id=100 + position,
            next_states=layer_states(position),
            profile=fd43.FULLDEPTH43_NATIVE_TOP6,
        )
    return decoder


def assert_tensor_state_equal(
    test: unittest.TestCase, left: fd43.DecoderState, right: fd43.DecoderState
) -> None:
    test.assertEqual(left.position, right.position)
    test.assertEqual(left.input_token_id, right.input_token_id)
    test.assertEqual(left.committed_tokens, right.committed_tokens)
    test.assertEqual(left.forced_queue, right.forced_queue)
    test.assertEqual(set(left.layer_states), set(right.layer_states))
    for layer in left.layer_states:
        a = left.layer_states[layer]
        b = right.layer_states[layer]
        test.assertEqual((a.layer, a.position), (b.layer, b.position))
        test.assertTrue(torch.equal(a.window_kv, b.window_kv))
        test.assertEqual(a.compressor is None, b.compressor is None)
        if a.compressor is None:
            continue
        test.assertEqual((a.compressor.ratio, a.compressor.overlap), (b.compressor.ratio, b.compressor.overlap))
        for field in checkpoint._TENSOR_FIELDS:
            x = getattr(a.compressor, field)
            y = getattr(b.compressor, field)
            test.assertEqual(x is None, y is None)
            if x is not None:
                test.assertTrue(torch.equal(x, y), f"L{layer}.{field}")


class ExactStateWorker:
    def __call__(self, position, input_token_id, previous):
        routes = {
            layer: tuple((position * 13 + layer + offset) % 256 for offset in range(6))
            for layer in range(43)
        }
        return fd43.FullDepthTokenComputation(
            predicted_token_id=123,
            next_layer_states=layer_states(position),
            top6_by_layer=routes,
            value={"position": position, "input_token_id": input_token_id},
        )


class FakeExecutorWorker:
    def __init__(self, config, catalog, cache, *, profile, progress_callback):
        self.progress_callback = progress_callback
        self.stage = "idle"
        self.current_layer = None
        self.writeback_layers = []
        self.writeback_fallbacks = []
        self.writeback_hello = None

    def start(self):
        return None

    def close(self):
        return None

    def __call__(self, position, input_token_id, previous):
        self.stage = f"position_{position}_ready_to_commit"
        token_report = {
            "position": position,
            "input_token_id": input_token_id,
            "completed_layers": list(range(43)),
            "layers": [],
            "final": {"token_id": 123},
            "state_committed": False,
        }
        self.progress_callback(token_report)
        return fd43.FullDepthTokenComputation(
            predicted_token_id=123,
            next_layer_states=layer_states(position),
            top6_by_layer={layer: tuple(range(6)) for layer in range(43)},
            value={"token_report": token_report, "final": token_report["final"]},
        )


def ready_preflight() -> dict:
    return {
        "status": "ready",
        "cold_execution_upper_bound": {"total_bytes": 0},
        "storage": {"cold_upper_bound_fits": True},
    }


class DecoderCheckpointTests(unittest.TestCase):
    def test_execution_config_rejects_conflicting_resume_inputs(self) -> None:
        with self.assertRaisesRegex(fd43.FullDepthError, "forced cursor"):
            fd43.ExecutionConfig(
                forced_prefill_path=Path("forced.json"),
                resume_checkpoint_path=Path("state.json"),
            ).validate()
        with self.assertRaisesRegex(fd43.FullDepthError, "execution report"):
            fd43.ExecutionConfig(
                report_path=Path("same.json"),
                checkpoint_path=Path("same.json"),
            ).validate()

    def test_zero_position_round_trip_without_tensor_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            decoder = fd43.DecoderState(input_token_id=0)
            saved = checkpoint.save_decoder_checkpoint(decoder, path)
            restored, loaded = checkpoint.load_decoder_checkpoint(path)

        self.assertEqual(saved["payload_bytes"], 0)
        self.assertEqual(loaded["tensor_count"], 0)
        assert_tensor_state_equal(self, decoder, restored)

    def test_round_trip_then_k1_matches_uninterrupted_serial_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            uninterrupted = one_token_state()
            checkpoint.save_decoder_checkpoint(
                uninterrupted,
                path,
                provenance={"test": "K1 连续恢复"},
            )
            resumed, evidence = checkpoint.load_decoder_checkpoint(path)

            direct = FullDepthDecoderStateBridge(
                uninterrupted,
                ExactStateWorker(),
                context_token_ids=(s14.BOS_TOKEN_ID, 91),
                tokenizer_fingerprint=S14_TOKENIZER_SHA256,
            )
            backend = build_cpu_causal_block_reference_backend(
                resumed,
                ExactStateWorker(),
                context_token_ids=(s14.BOS_TOKEN_ID, 91),
                tokenizer_fingerprint=S14_TOKENIZER_SHA256,
            )
            direct.step(forced_next_token_id=123)
            transaction = backend.begin_causal_block(
                VerificationRequest(
                    (s14.BOS_TOKEN_ID, 91),
                    (123,),
                    S14_TOKENIZER_SHA256,
                )
            )
            decision = decide_acceptance(
                (123,), transaction.verification.predicted_token_ids
            )
            transaction.prepare_commit(decision)
            transaction.commit()

        self.assertEqual(evidence["position"], 1)
        self.assertEqual(transaction.metrics.forward_calls, 1)
        assert_tensor_state_equal(self, direct.decoder, backend.runtime.decoder)

    def test_five_token_forced_cursor_and_provenance_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "five.json"
            decoder = five_token_forced_state()
            checkpoint.save_decoder_checkpoint(
                decoder,
                path,
                provenance={"source": "synthetic-five-token-prefix", "approved": False},
            )
            restored, evidence = checkpoint.load_decoder_checkpoint(path)

        self.assertEqual(restored.position, 5)
        self.assertIsNotNone(restored.forced_queue)
        self.assertEqual(restored.forced_queue.cursor, 5)
        self.assertFalse(restored.forced_queue.active)
        self.assertEqual(evidence["provenance"]["approved"], False)
        assert_tensor_state_equal(self, decoder, restored)

        stripped = decoder.clone()
        stripped.forced_queue = None
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(checkpoint.CheckpointError, "token queue provenance"):
                checkpoint.save_decoder_checkpoint(
                    stripped, Path(directory) / "stripped.json"
                )

    def test_payload_corruption_and_truncation_are_rejected(self) -> None:
        for mutation in ("flip", "truncate"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "state.json"
                saved = checkpoint.save_decoder_checkpoint(one_token_state(), path)
                payload = Path(saved["payload"])
                raw = bytearray(payload.read_bytes())
                if mutation == "flip":
                    raw[len(raw) // 2] ^= 1
                else:
                    del raw[-1]
                payload.write_bytes(raw)
                with self.assertRaises(checkpoint.CheckpointError):
                    checkpoint.load_decoder_checkpoint(path)

    def test_model_and_manifest_mismatch_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            checkpoint.save_decoder_checkpoint(fd43.DecoderState(), path)
            document = json.loads(path.read_text(encoding="utf-8"))
            document["model"]["revision"] = "wrong"
            path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(checkpoint.CheckpointError, "model/revision"):
                checkpoint.load_decoder_checkpoint(path)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            checkpoint.save_decoder_checkpoint(fd43.DecoderState(), path)
            document = json.loads(path.read_text(encoding="utf-8"))
            document["decoder"]["input_token_id"] = 7
            path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(checkpoint.CheckpointError, "manifest 元数据"):
                checkpoint.load_decoder_checkpoint(path)

    def test_wrong_tokenizer_and_partial_layer_state_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            checkpoint.save_decoder_checkpoint(fd43.DecoderState(), path)
            with self.assertRaisesRegex(checkpoint.CheckpointError, "model/revision"):
                checkpoint.load_decoder_checkpoint(path, tokenizer_sha256="b" * 64)

        partial = fd43.DecoderState(
            position=1,
            input_token_id=91,
            layer_states={0: layer_states(0)[0]},
            committed_tokens=[
                {"position": 0, "input_token_id": s14.BOS_TOKEN_ID, "output_token_id": 91}
            ],
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(checkpoint.CheckpointError, "43 层"):
                checkpoint.save_decoder_checkpoint(partial, Path(directory) / "bad.json")

    def test_atomic_replacement_keeps_only_current_content_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            first = one_token_state()
            checkpoint.save_decoder_checkpoint(first, path)
            second = first.clone()
            second.commit(
                output_token_id=92,
                next_states=layer_states(1),
                profile=fd43.FULLDEPTH43_NATIVE_TOP6,
            )
            checkpoint.save_decoder_checkpoint(second, path)
            restored, _ = checkpoint.load_decoder_checkpoint(path)
            payloads = list(Path(directory).glob("state.*.state.bin"))

        self.assertEqual(len(payloads), 1)
        assert_tensor_state_equal(self, second, restored)

    def test_existing_preview_report_cannot_masquerade_as_checkpoint(self) -> None:
        report = Path(__file__).resolve().parents[1] / "first_preview_real_report.json"
        self.assertTrue(report.is_file())
        with self.assertRaisesRegex(checkpoint.CheckpointError, "拒绝非 FullDepth43"):
            checkpoint.load_decoder_checkpoint(report)

    def test_execute_resumes_valid_state_and_atomically_advances_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_path = root / "state.json"
            report_path = root / "report.json"
            checkpoint.save_decoder_checkpoint(
                one_token_state(),
                state_path,
                provenance={"catalog_sha256": checkpoint.catalog_fingerprint({})},
            )
            config = fd43.ExecutionConfig(
                asset_root=root,
                catalog_path=root / "catalog.json",
                report_path=report_path,
                resume_checkpoint_path=state_path,
                checkpoint_path=state_path,
            )
            with (
                mock.patch.object(fd43, "_load_or_build_catalog", return_value={}),
                mock.patch.object(fd43, "run_preflight", return_value=ready_preflight()),
                mock.patch.object(fd43.online_range, "RangeCache", return_value=object()),
                mock.patch.object(fd43, "FullDepthTokenWorker", FakeExecutorWorker),
            ):
                report = fd43.execute(config)
            restored, _ = checkpoint.load_decoder_checkpoint(state_path)

        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["resume_checkpoint"]["position"], 1)
        self.assertEqual(report["checkpoint"]["position"], 2)
        self.assertEqual(restored.position, 2)
        self.assertEqual(restored.input_token_id, 123)
        self.assertEqual(len(restored.committed_tokens), 2)

    def test_checkpoint_publish_failure_does_not_commit_candidate_decoder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = fd43.ExecutionConfig(
                asset_root=root,
                catalog_path=root / "catalog.json",
                report_path=root / "report.json",
                checkpoint_path=root / "state.json",
            )
            with (
                mock.patch.object(fd43, "_load_or_build_catalog", return_value={}),
                mock.patch.object(fd43, "run_preflight", return_value=ready_preflight()),
                mock.patch.object(fd43.online_range, "RangeCache", return_value=object()),
                mock.patch.object(fd43, "FullDepthTokenWorker", FakeExecutorWorker),
                mock.patch.object(
                    fd43.decoder_checkpoint,
                    "save_decoder_checkpoint",
                    side_effect=checkpoint.CheckpointError("injected checkpoint failure"),
                ),
            ):
                report = fd43.execute(config)

        self.assertEqual(report["status"], "blocked")
        self.assertFalse(report["native_token_executed"])
        self.assertEqual(report["committed_tokens"], [])
        self.assertIn("injected checkpoint failure", report["error"]["message"])

    def test_execute_rejects_checkpoint_from_different_catalog_before_worker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_path = root / "state.json"
            checkpoint.save_decoder_checkpoint(
                one_token_state(),
                state_path,
                provenance={"catalog_sha256": checkpoint.catalog_fingerprint({"wrong": True})},
            )
            config = fd43.ExecutionConfig(
                asset_root=root,
                catalog_path=root / "catalog.json",
                report_path=root / "report.json",
                resume_checkpoint_path=state_path,
            )
            with (
                mock.patch.object(fd43, "_load_or_build_catalog", return_value={}),
                mock.patch.object(fd43, "run_preflight", return_value=ready_preflight()),
                mock.patch.object(fd43.online_range, "RangeCache", return_value=object()),
                mock.patch.object(
                    fd43, "FullDepthTokenWorker", side_effect=FakeExecutorWorker
                ) as worker,
            ):
                report = fd43.execute(config)

        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["error"]["stage"], "checkpoint_restore")
        self.assertIn("catalog provenance", report["error"]["message"])
        worker.assert_not_called()


if __name__ == "__main__":
    unittest.main()
