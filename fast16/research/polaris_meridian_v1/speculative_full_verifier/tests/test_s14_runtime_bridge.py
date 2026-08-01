from __future__ import annotations

from dataclasses import replace
import unittest

from fast16.research.polaris_meridian_v1.s14_first_real_token import executor as s14

from ..s14_runtime_bridge import S14DecoderRuntimeBridge
from ..verifier import S14_LAYERS, VerifierContractError


class FakeS14Runtime:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.snapshot = s14.DecoderSnapshot(input_token_id=90)

    def run_token(self, worker):
        token_id = next(self.outputs)
        record = {
            "position": self.snapshot.position,
            "input_token_id": self.snapshot.input_token_id,
            "output_token_id": token_id,
        }
        self.snapshot = replace(
            self.snapshot,
            position=self.snapshot.position + 1,
            input_token_id=token_id,
            committed_tokens=self.snapshot.committed_tokens + (record,),
        )
        return {
            "layers": [
                {"layer": layer, "expert_ids": [0, 1, 2, 3, 4, 5]}
                for layer in S14_LAYERS
            ]
        }


class S14RuntimeBridgeTest(unittest.TestCase):
    def test_real_decoder_snapshot_shape_is_adapted_without_token_conversion(self):
        runtime = FakeS14Runtime([1, 2])
        bridge = S14DecoderRuntimeBridge(
            runtime,
            lambda *args: None,
            context_token_ids=[90],
            tokenizer_fingerprint="a" * 64,
        )
        base = bridge.capture_state()

        first = bridge.step()
        second = bridge.step()

        self.assertEqual(first.predicted_token_id, 1)
        self.assertEqual(second.predicted_token_id, 2)
        self.assertEqual(set(first.top6_by_layer), set(S14_LAYERS))
        self.assertEqual(bridge.context_token_ids, (90, 1, 2))
        bridge.restore_state(base)
        self.assertEqual(bridge.context_token_ids, (90,))
        self.assertEqual(runtime.snapshot.position, 0)

    def test_replace_pending_fallback_repairs_snapshot_and_commit_ledger(self):
        runtime = FakeS14Runtime([1])
        bridge = S14DecoderRuntimeBridge(
            runtime,
            lambda *args: None,
            context_token_ids=[90],
            tokenizer_fingerprint="a" * 64,
        )
        bridge.step()
        patched = bridge.replace_pending_token(bridge.capture_state(), 9)
        bridge.restore_state(patched)
        self.assertEqual(bridge.context_token_ids, (90, 9))
        self.assertEqual(runtime.snapshot.input_token_id, 9)
        self.assertEqual(runtime.snapshot.committed_tokens[-1]["output_token_id"], 9)

    def test_active_forced_prefill_is_rejected(self):
        runtime = FakeS14Runtime([1])
        runtime.snapshot = replace(
            runtime.snapshot,
            input_token_id=0,
            forced_queue=s14.ForcedTokenQueue((0, 1), 0, "b" * 64),
        )
        with self.assertRaises(VerifierContractError):
            S14DecoderRuntimeBridge(
                runtime,
                lambda *args: None,
                context_token_ids=[0],
                tokenizer_fingerprint="a" * 64,
            )


if __name__ == "__main__":
    unittest.main()
