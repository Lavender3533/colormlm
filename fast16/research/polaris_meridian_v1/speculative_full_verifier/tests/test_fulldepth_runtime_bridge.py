from __future__ import annotations

import unittest

import torch

from fast16.research.polaris_meridian_v1.fulldepth43_native_top6 import executor as fd43
from fast16.research.polaris_meridian_v1.s14_first_real_token import executor as s14

from ..cpu_causal_block import CpuCausalBlockReferenceBackend
from ..fulldepth_runtime_bridge import (
    FullDepthDecoderStateBridge,
    FullDepthTokenComputation,
)
from ..runtime_controller import decide_acceptance
from ..verifier import VerificationRequest
from ..verifier import VerifierContractError


FINGERPRINT = "f" * 64


class TinyFullDepthWorker:
    def __init__(self, predictions: tuple[int, ...]) -> None:
        self.predictions = predictions

    def __call__(self, position, input_token_id, previous):
        states = {}
        for layer in range(43):
            old = previous.get(layer)
            old_kv = (
                torch.empty((0,), dtype=torch.bfloat16)
                if old is None
                else old.window_kv.flatten()
            )
            old_remainder = (
                torch.empty((0,), dtype=torch.bfloat16)
                if old is None or old.compressor is None
                else old.compressor.main_kv_state.flatten()
            )
            marker = torch.tensor(
                [position * 100 + layer], dtype=torch.bfloat16
            )
            remainder = torch.cat((old_remainder, marker))
            compressor = s14.CompressorRemainderState(
                ratio=4,
                overlap=False,
                main_kv_state=remainder,
                main_score_state=remainder.clone(),
                main_compressed_kv=remainder.clone(),
                indexer_kv_state=None,
                indexer_score_state=None,
                indexer_compressed_kv=None,
            )
            states[layer] = s14.LayerRuntimeState(
                layer=layer,
                position=position,
                window_kv=torch.cat((old_kv, marker)),
                compressor=compressor,
            )
        routes = {
            layer: tuple((position * 11 + layer + index) % 256 for index in range(6))
            for layer in range(43)
        }
        return FullDepthTokenComputation(
            predicted_token_id=self.predictions[position],
            next_layer_states=states,
            top6_by_layer=routes,
            value={"position": position, "input_token_id": input_token_id},
        )


def make_bridge(predictions: tuple[int, ...]) -> FullDepthDecoderStateBridge:
    return FullDepthDecoderStateBridge(
        fd43.DecoderState(input_token_id=90),
        TinyFullDepthWorker(predictions),
        context_token_ids=(90,),
        tokenizer_fingerprint=FINGERPRINT,
    )


def assert_bridge_equal(test: unittest.TestCase, left, right) -> None:
    test.assertEqual(left.context_token_ids, right.context_token_ids)
    test.assertEqual(left.decoder.position, right.decoder.position)
    test.assertEqual(left.decoder.input_token_id, right.decoder.input_token_id)
    test.assertEqual(left.decoder.committed_tokens, right.decoder.committed_tokens)
    test.assertEqual(set(left.decoder.layer_states), set(right.decoder.layer_states))
    for layer in range(43):
        a = left.decoder.layer_states[layer]
        b = right.decoder.layer_states[layer]
        test.assertEqual(a.layer, b.layer)
        test.assertEqual(a.position, b.position)
        test.assertTrue(torch.equal(a.window_kv, b.window_kv))
        test.assertTrue(
            torch.equal(
                a.compressor.main_kv_state,
                b.compressor.main_kv_state,
            )
        )


class FullDepthRuntimeBridgeTest(unittest.TestCase):
    def test_forged_earlier_context_is_rejected_before_any_worker_call(self):
        decoder = fd43.DecoderState(input_token_id=90)
        with self.assertRaisesRegex(VerifierContractError, "context 长度"):
            FullDepthDecoderStateBridge(
                decoder,
                TinyFullDepthWorker((1,)),
                context_token_ids=(777, 90),
                tokenizer_fingerprint=FINGERPRINT,
            )

    def test_nonzero_position_requires_exact_committed_input_chain(self):
        bridge = make_bridge((1, 2))
        bridge.step(forced_next_token_id=1)
        # 真实非零 position 历史可以重新建桥。
        FullDepthDecoderStateBridge(
            bridge.decoder,
            TinyFullDepthWorker((1, 2)),
            context_token_ids=bridge.context_token_ids,
            tokenizer_fingerprint=FINGERPRINT,
        )
        bridge.decoder.committed_tokens[0]["input_token_id"] = 777

        with self.assertRaisesRegex(VerifierContractError, "committed input"):
            FullDepthDecoderStateBridge(
                bridge.decoder,
                TinyFullDepthWorker((1, 2)),
                context_token_ids=(90, 1),
                tokenizer_fingerprint=FINGERPRINT,
            )

    def test_nonzero_position_rejects_layer_state_position_drift(self):
        bridge = make_bridge((1, 2))
        bridge.step(forced_next_token_id=1)
        original = bridge.decoder.layer_states[42]
        bridge.decoder.layer_states[42] = s14.LayerRuntimeState(
            layer=42,
            position=99,
            window_kv=original.window_kv,
            compressor=original.compressor,
        )

        with self.assertRaisesRegex(VerifierContractError, "KV/compressor position"):
            FullDepthDecoderStateBridge(
                bridge.decoder,
                TinyFullDepthWorker((1, 2)),
                context_token_ids=(90, 1),
                tokenizer_fingerprint=FINGERPRINT,
            )

    def test_real_decoder_state_k4_matches_serial_teacher_force(self):
        draft = (1, 2, 3, 4)
        block_bridge = make_bridge(draft)
        serial_bridge = make_bridge(draft)
        backend = CpuCausalBlockReferenceBackend(block_bridge)
        request = VerificationRequest((90,), draft, FINGERPRINT)

        transaction = backend.begin_causal_block(request)
        for token_id in draft:
            serial_bridge.step(forced_next_token_id=token_id)
        decision = decide_acceptance(
            draft, transaction.verification.predicted_token_ids
        )
        transaction.prepare_commit(decision)
        transaction.commit()

        assert_bridge_equal(self, block_bridge, serial_bridge)

    def test_real_decoder_state_mismatch_drops_future_layer_deltas(self):
        draft = (1, 2, 3, 4)
        bridge = make_bridge((1, 2, 9, 777))
        transaction = CpuCausalBlockReferenceBackend(bridge).begin_causal_block(
            VerificationRequest((90,), draft, FINGERPRINT)
        )
        decision = decide_acceptance(
            draft, transaction.verification.predicted_token_ids
        )

        transaction.prepare_commit(decision)
        transaction.commit()

        self.assertEqual(bridge.context_token_ids, (90, 1, 2, 9))
        self.assertEqual(bridge.decoder.position, 3)
        for layer, state in bridge.decoder.layer_states.items():
            self.assertEqual(state.position, 2)
            expected = torch.tensor(
                [layer, 100 + layer, 200 + layer], dtype=torch.bfloat16
            )
            self.assertTrue(torch.equal(state.window_kv, expected))
            self.assertTrue(
                torch.equal(state.compressor.main_kv_state, expected)
            )


if __name__ == "__main__":
    unittest.main()
