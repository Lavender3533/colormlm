from __future__ import annotations

from dataclasses import dataclass
import unittest

from ..cache_replay import RouteBlock
from ..runtime_controller import (
    BATCHED_CAUSAL_MODE,
    AtomicSpeculativeController,
    NativeBatchedVerifierBackend,
    NativeTokenStep,
    SerialSnapshotDraftBackend,
    SerialSnapshotVerifierBackend,
    TransactionMetrics,
)
from ..verifier import (
    DRAFT_PROFILE,
    S14_LAYERS,
    VERIFIER_PROFILE,
    FullDepthVerification,
    VerifierContractError,
)


def route_map(layers) -> dict[int, tuple[int, ...]]:
    return {layer: (0, 1, 2, 3, 4, 5) for layer in layers}


def native_routes(block_size: int) -> RouteBlock:
    return RouteBlock.from_rows(
        "native-batched",
        block_size,
        (
            {
                "token_offset": token,
                "layer": layer,
                "expert_ids": [0, 1, 2, 3, 4, 5],
            }
            for token in range(block_size)
            for layer in range(43)
        ),
    )


@dataclass(frozen=True)
class MemoryState:
    context: tuple[int, ...]
    cursor: int


class MemorySnapshotRuntime:
    vocab_size = 1000
    tokenizer_fingerprint = "same-deepseek-tokenizer"

    def __init__(self, profile: str, depth: int, context: list[int], script: list[int]):
        self.profile = profile
        self.depth = depth
        self.state = MemoryState(tuple(context), 0)
        self.script = tuple(script)
        self.fail_non_base_restore_once = False
        self.base_context = tuple(context)

    @property
    def context_token_ids(self) -> tuple[int, ...]:
        return self.state.context

    def capture_state(self) -> MemoryState:
        return self.state

    def restore_state(self, state: MemoryState) -> None:
        if self.fail_non_base_restore_once and state.context != self.base_context:
            self.fail_non_base_restore_once = False
            raise RuntimeError("injected commit failure")
        self.state = state

    def replace_pending_token(self, state: MemoryState, token_id: int) -> MemoryState:
        if not state.context:
            raise AssertionError("missing pending token")
        return MemoryState(state.context[:-1] + (token_id,), state.cursor)

    def step(self, *, forced_next_token_id: int | None = None) -> NativeTokenStep:
        if self.state.cursor >= len(self.script):
            raise RuntimeError("script exhausted")
        prediction = self.script[self.state.cursor]
        next_token = prediction if forced_next_token_id is None else forced_next_token_id
        self.state = MemoryState(
            self.state.context + (next_token,), self.state.cursor + 1
        )
        layers = S14_LAYERS if self.profile == DRAFT_PROFILE else range(self.depth)
        return NativeTokenStep(prediction, route_map(layers))


def serial_controller(
    draft_tokens: list[int], target_tokens: list[int]
) -> tuple[AtomicSpeculativeController, MemorySnapshotRuntime, MemorySnapshotRuntime]:
    draft = MemorySnapshotRuntime(DRAFT_PROFILE, 14, [90], draft_tokens)
    target = MemorySnapshotRuntime(VERIFIER_PROFILE, 43, [90], target_tokens)
    controller = AtomicSpeculativeController(
        [90], SerialSnapshotDraftBackend(draft), SerialSnapshotVerifierBackend(target)
    )
    return controller, draft, target


class RuntimeControllerTest(unittest.TestCase):
    def test_k4_mismatch_commits_prefix_and_native_fallback_to_both_runtimes(self):
        controller, draft, target = serial_controller([1, 2, 3, 4], [1, 2, 9, 777])

        result = controller.run_round(4)

        self.assertEqual(result.decision.accepted_prefix, (1, 2))
        self.assertEqual(result.decision.fallback_token_id, 9)
        self.assertEqual(result.decision.rejected_draft_suffix, (3, 4))
        self.assertEqual(tuple(controller.context_token_ids), (90, 1, 2, 9))
        self.assertEqual(draft.context_token_ids, (90, 1, 2, 9))
        self.assertEqual(target.context_token_ids, (90, 1, 2, 9))
        self.assertEqual(result.verifier_metrics.forward_calls, 4)
        self.assertFalse(result.speed_eligible_verifier)

    def test_k8_full_match_commits_all_tokens(self):
        tokens = list(range(1, 9))
        controller, draft, target = serial_controller(tokens, tokens)

        result = controller.run_round(8)

        self.assertEqual(result.decision.accepted_length, 8)
        self.assertIsNone(result.decision.fallback_token_id)
        self.assertEqual(tuple(controller.context_token_ids), (90, *tokens))
        self.assertEqual(draft.context_token_ids, target.context_token_ids)

    def test_target_failure_rolls_back_draft_and_target(self):
        controller, draft, target = serial_controller([1, 2, 3, 4], [1, 2])

        with self.assertRaises(RuntimeError):
            controller.run_round(4)

        self.assertEqual(tuple(controller.context_token_ids), (90,))
        self.assertEqual(draft.context_token_ids, (90,))
        self.assertEqual(target.context_token_ids, (90,))

    def test_commit_failure_rolls_back_both_runtimes_atomically(self):
        controller, draft, target = serial_controller([1, 2, 3, 4], [1, 2, 3, 4])
        target.fail_non_base_restore_once = True

        with self.assertRaisesRegex(RuntimeError, "injected commit failure"):
            controller.run_round(4)

        self.assertEqual(tuple(controller.context_token_ids), (90,))
        self.assertEqual(draft.context_token_ids, (90,))
        self.assertEqual(target.context_token_ids, (90,))

    def test_only_k4_or_k8_are_accepted(self):
        controller, _, _ = serial_controller([1, 2, 3, 4], [1, 2, 3, 4])
        with self.assertRaises(VerifierContractError):
            controller.run_round(2)


class BatchedTransaction:
    def __init__(self, runtime: "BatchedRuntime", predictions: tuple[int, ...]):
        self.runtime = runtime
        self.base = runtime.context_token_ids
        self.request = runtime.last_request
        self.verification = FullDepthVerification(
            predictions, native_routes(len(predictions))
        )
        self.metrics = TransactionMetrics(BATCHED_CAUSAL_MODE, 1, 0.001)
        self.prepared: tuple[int, ...] | None = None

    def prepare_commit(self, decision) -> None:
        self.prepared = self.base + decision.committed_token_ids

    def commit(self) -> None:
        if self.prepared is None:
            raise AssertionError("not prepared")
        self.runtime.context = self.prepared

    def rollback(self) -> None:
        self.runtime.context = self.base


class BatchedRuntime:
    tokenizer_fingerprint = "same-deepseek-tokenizer"
    vocab_size = 1000

    def __init__(self, context: tuple[int, ...], predictions: tuple[int, ...]):
        self.context = context
        self.predictions = predictions
        self.calls = 0
        self.last_request = None

    @property
    def context_token_ids(self) -> tuple[int, ...]:
        return self.context

    def begin_causal_block(self, request):
        self.calls += 1
        self.last_request = request
        return BatchedTransaction(self, self.predictions)


class NativeBatchedBoundaryTest(unittest.TestCase):
    def test_one_call_batched_boundary_is_speed_eligible(self):
        draft_runtime = MemorySnapshotRuntime(
            DRAFT_PROFILE, 14, [90], [1, 2, 3, 4]
        )
        batched_runtime = BatchedRuntime((90,), (1, 2, 3, 4))
        controller = AtomicSpeculativeController(
            [90],
            SerialSnapshotDraftBackend(draft_runtime),
            NativeBatchedVerifierBackend(batched_runtime),
        )

        result = controller.run_round(4)

        self.assertTrue(result.speed_eligible_verifier)
        self.assertEqual(batched_runtime.calls, 1)
        self.assertEqual(batched_runtime.context_token_ids, (90, 1, 2, 3, 4))

    def test_batched_boundary_rejects_fake_serial_metrics(self):
        class FakeSerialBatchedRuntime(BatchedRuntime):
            def begin_causal_block(self, request):
                transaction = super().begin_causal_block(request)
                transaction.metrics = TransactionMetrics("serial_reference", 4, 0.1)
                return transaction

        runtime = FakeSerialBatchedRuntime((90,), (1, 2, 3, 4))
        backend = NativeBatchedVerifierBackend(runtime)
        from ..verifier import VerificationRequest

        with self.assertRaises(VerifierContractError):
            backend.begin_block(
                VerificationRequest((90,), (1, 2, 3, 4), runtime.tokenizer_fingerprint)
            )
        self.assertEqual(runtime.context_token_ids, (90,))


if __name__ == "__main__":
    unittest.main()
