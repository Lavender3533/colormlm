from __future__ import annotations

from dataclasses import dataclass
import unittest

from ..cpu_causal_block import (
    CPU_CAUSAL_BLOCK_REFERENCE_MODE,
    CpuCausalBlockReferenceBackend,
)
from ..runtime_controller import (
    AtomicSpeculativeController,
    NativeBatchedVerifierBackend,
    NativeTokenStep,
    SerialSnapshotDraftBackend,
    decide_acceptance,
)
from ..verifier import (
    DRAFT_PROFILE,
    S14_LAYERS,
    VERIFIER_PROFILE,
    VerificationRequest,
    VerifierContractError,
)


TOKENIZER_FINGERPRINT = "deepseek-fixture-tokenizer"


@dataclass(frozen=True)
class FixtureLayerState:
    layer: int
    position: int
    window_kv: tuple[tuple[int, int, int], ...]
    compressor_remainder: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class FixtureDecoderState:
    context: tuple[int, ...]
    cursor: int
    layer_states: tuple[FixtureLayerState, ...]


class FixtureFullDepthRuntime:
    """离线 FullDepth43 状态机。

    window/compressor 用可读 tuple 代替大 tensor，但遵守与真实
    executor 相同的「每 token 全 43 层同一 position」提交语义。
    """

    profile = VERIFIER_PROFILE
    depth = 43
    vocab_size = 1000
    tokenizer_fingerprint = TOKENIZER_FINGERPRINT

    def __init__(
        self,
        predictions: tuple[int, ...],
        *,
        context: tuple[int, ...] = (90,),
        fail_at: int | None = None,
        omit_route_layer_at: int | None = None,
    ) -> None:
        self.predictions = predictions
        self.state = FixtureDecoderState(context, 0, ())
        self.fail_at = fail_at
        self.omit_route_layer_at = omit_route_layer_at
        self.step_calls = 0

    @property
    def context_token_ids(self) -> tuple[int, ...]:
        return self.state.context

    def capture_state(self) -> FixtureDecoderState:
        return self.state

    def restore_state(self, state: FixtureDecoderState) -> None:
        self.state = state

    def replace_pending_token(
        self, state: FixtureDecoderState, token_id: int
    ) -> FixtureDecoderState:
        if not state.context:
            raise VerifierContractError("没有 pending token 可替换")
        return FixtureDecoderState(
            state.context[:-1] + (token_id,),
            state.cursor,
            state.layer_states,
        )

    def step(self, *, forced_next_token_id: int | None = None) -> NativeTokenStep:
        offset = self.state.cursor
        self.step_calls += 1
        if self.fail_at == offset:
            raise RuntimeError("injected CPU layer failure")
        if offset >= len(self.predictions):
            raise RuntimeError("fixture predictions exhausted")
        if forced_next_token_id is None:
            forced_next_token_id = self.predictions[offset]

        input_token_id = self.state.context[-1]
        previous = {state.layer: state for state in self.state.layer_states}
        next_layers = []
        for layer in range(43):
            old = previous.get(layer)
            old_window = () if old is None else old.window_kv
            old_compressor = () if old is None else old.compressor_remainder
            # window 只保留四个位置，模拟环形 KV；compressor 保留
            # 全部 remainder 事件，方便检测 mismatch 后的泄漏。
            window = (*old_window, (offset, input_token_id, layer))[-4:]
            compressor = (*old_compressor, (offset, input_token_id))
            next_layers.append(
                FixtureLayerState(layer, offset, window, compressor)
            )
        self.state = FixtureDecoderState(
            self.state.context + (forced_next_token_id,),
            offset + 1,
            tuple(next_layers),
        )

        routes = {
            layer: tuple((offset * 17 + layer * 7 + index) % 256 for index in range(6))
            for layer in range(43)
        }
        if self.omit_route_layer_at == offset:
            routes.pop(42)
        return NativeTokenStep(self.predictions[offset], routes)


class FixtureDraftRuntime(FixtureFullDepthRuntime):
    profile = DRAFT_PROFILE
    depth = len(S14_LAYERS)

    def step(self, *, forced_next_token_id: int | None = None) -> NativeTokenStep:
        step = super().step(forced_next_token_id=forced_next_token_id)
        return NativeTokenStep(
            step.predicted_token_id,
            {layer: step.top6_by_layer[layer] for layer in S14_LAYERS},
        )


class CommitFailingDraftRuntime(FixtureDraftRuntime):
    def __init__(self, predictions: tuple[int, ...]) -> None:
        super().__init__(predictions)
        self.fail_non_base_restore_once = True

    def restore_state(self, state: FixtureDecoderState) -> None:
        if self.fail_non_base_restore_once and state.context != (90,):
            self.fail_non_base_restore_once = False
            raise RuntimeError("injected draft commit failure")
        super().restore_state(state)


def request(drafts: tuple[int, ...], context: tuple[int, ...] = (90,)) -> VerificationRequest:
    return VerificationRequest(context, drafts, TOKENIZER_FINGERPRINT)


def commit_native_decision(transaction, drafts: tuple[int, ...]) -> None:
    transaction.prepare_commit(
        decide_acceptance(drafts, transaction.verification.predicted_token_ids)
    )
    transaction.commit()


class CpuCausalBlockEquivalenceTest(unittest.TestCase):
    def test_k1_block_is_exactly_equal_to_one_serial_token(self):
        draft = (7,)
        block_runtime = FixtureFullDepthRuntime((7,))
        serial_runtime = FixtureFullDepthRuntime((7,))
        backend = CpuCausalBlockReferenceBackend(block_runtime)

        transaction = backend.begin_causal_block(request(draft))
        self.assertEqual(block_runtime.context_token_ids, (90,))
        commit_native_decision(transaction, draft)
        serial_runtime.step(forced_next_token_id=7)

        self.assertEqual(block_runtime.state, serial_runtime.state)
        self.assertEqual(transaction.audit.block_api_calls, 1)
        self.assertEqual(transaction.audit.model_forward_calls, 1)
        self.assertEqual(transaction.audit.checkpoint_count, 1)
        self.assertEqual(transaction.audit.route_rows, 43)

    def test_k4_block_matches_strict_serial_k_state_and_routes(self):
        draft = (1, 2, 3, 4)
        block_runtime = FixtureFullDepthRuntime(draft)
        serial_runtime = FixtureFullDepthRuntime(draft)
        backend = CpuCausalBlockReferenceBackend(block_runtime)

        transaction = backend.begin_causal_block(request(draft))
        for token_id in draft:
            serial_runtime.step(forced_next_token_id=token_id)
        commit_native_decision(transaction, draft)

        self.assertEqual(block_runtime.state, serial_runtime.state)
        self.assertEqual(backend.block_api_calls, 1)
        self.assertEqual(transaction.metrics.mode, CPU_CAUSAL_BLOCK_REFERENCE_MODE)
        self.assertEqual(transaction.metrics.forward_calls, 4)
        self.assertEqual(transaction.verification.native_routes.block_size, 4)
        self.assertEqual(
            len(transaction.verification.native_routes.routes_by_token), 4
        )
        self.assertTrue(
            all(len(token_routes) == 43 for token_routes in transaction.verification.native_routes.routes_by_token)
        )

    def test_k8_saves_every_native_route_and_checkpoint(self):
        draft = tuple(range(1, 9))
        runtime = FixtureFullDepthRuntime(draft)
        serial_runtime = FixtureFullDepthRuntime(draft)
        transaction = CpuCausalBlockReferenceBackend(runtime).begin_causal_block(
            request(draft)
        )

        self.assertEqual(transaction.audit.checkpoint_count, 8)
        self.assertEqual(transaction.audit.route_rows, 8 * 43)
        for token_routes in transaction.verification.native_routes.routes_by_token:
            self.assertEqual(len(token_routes), 43)
            self.assertTrue(all(len(experts) == 6 for experts in token_routes))
        for token_id in draft:
            serial_runtime.step(forced_next_token_id=token_id)
        commit_native_decision(transaction, draft)
        self.assertEqual(runtime.state, serial_runtime.state)

    def test_backend_plugs_into_atomic_controller_but_is_not_speed_eligible(self):
        draft = (1, 2, 3, 4)
        draft_runtime = FixtureDraftRuntime(draft)
        target_runtime = FixtureFullDepthRuntime(draft)
        controller = AtomicSpeculativeController(
            (90,),
            SerialSnapshotDraftBackend(draft_runtime),
            CpuCausalBlockReferenceBackend(target_runtime),
        )

        result = controller.run_round(4)

        self.assertEqual(tuple(controller.context_token_ids), (90, 1, 2, 3, 4))
        self.assertEqual(draft_runtime.context_token_ids, target_runtime.context_token_ids)
        self.assertEqual(result.verifier_metrics.forward_calls, 4)
        self.assertFalse(result.speed_eligible_verifier)

    def test_target_committed_state_is_compensated_if_draft_commit_fails(self):
        draft = (1, 2, 3, 4)
        draft_runtime = CommitFailingDraftRuntime(draft)
        target_runtime = FixtureFullDepthRuntime(draft)
        controller = AtomicSpeculativeController(
            (90,),
            SerialSnapshotDraftBackend(draft_runtime),
            CpuCausalBlockReferenceBackend(target_runtime),
        )

        with self.assertRaisesRegex(RuntimeError, "draft commit failure"):
            controller.run_round(4)

        self.assertEqual(draft_runtime.context_token_ids, (90,))
        self.assertEqual(target_runtime.context_token_ids, (90,))
        self.assertEqual(tuple(controller.context_token_ids), (90,))


class CpuCausalBlockTransactionTest(unittest.TestCase):
    def test_mismatch_trims_later_kv_and_compressor_checkpoints(self):
        draft = (1, 2, 3, 4)
        runtime = FixtureFullDepthRuntime((1, 2, 9, 777))
        transaction = CpuCausalBlockReferenceBackend(runtime).begin_causal_block(
            request(draft)
        )
        decision = decide_acceptance(
            draft, transaction.verification.predicted_token_ids
        )

        transaction.prepare_commit(decision)
        transaction.commit()

        self.assertEqual(runtime.context_token_ids, (90, 1, 2, 9))
        self.assertEqual(runtime.state.cursor, 3)
        self.assertEqual(decision.mismatch_index, 2)
        for state in runtime.state.layer_states:
            self.assertEqual(state.position, 2)
            self.assertEqual(tuple(row[0] for row in state.window_kv), (0, 1, 2))
            self.assertEqual(
                tuple(row[0] for row in state.compressor_remainder), (0, 1, 2)
            )
            self.assertNotIn(3, tuple(row[0] for row in state.compressor_remainder))

    def test_forged_acceptance_decision_is_rejected_without_state_leak(self):
        draft = (1, 2, 3, 4)
        runtime = FixtureFullDepthRuntime((1, 2, 9, 4))
        transaction = CpuCausalBlockReferenceBackend(runtime).begin_causal_block(
            request(draft)
        )
        forged = decide_acceptance(draft, draft)

        with self.assertRaisesRegex(VerifierContractError, "提交决策"):
            transaction.prepare_commit(forged)
        self.assertEqual(runtime.context_token_ids, (90,))
        transaction.rollback()

    def test_mid_block_failure_restores_exact_base_state(self):
        runtime = FixtureFullDepthRuntime((1, 2, 3, 4), fail_at=2)
        base = runtime.state
        backend = CpuCausalBlockReferenceBackend(runtime)

        with self.assertRaisesRegex(RuntimeError, "injected CPU layer failure"):
            backend.begin_causal_block(request((1, 2, 3, 4)))

        self.assertEqual(runtime.state, base)
        self.assertEqual(backend.block_api_calls, 1)

    def test_incomplete_k_by_43_route_fails_closed_and_restores_state(self):
        runtime = FixtureFullDepthRuntime(
            (1, 2, 3, 4), omit_route_layer_at=1
        )
        base = runtime.state

        with self.assertRaisesRegex(VerifierContractError, "K×43×6"):
            CpuCausalBlockReferenceBackend(runtime).begin_causal_block(
                request((1, 2, 3, 4))
            )

        self.assertEqual(runtime.state, base)

    def test_reference_backend_cannot_claim_native_batched_speed_mode(self):
        runtime = FixtureFullDepthRuntime((1, 2, 3, 4))
        reference = CpuCausalBlockReferenceBackend(runtime)
        production_boundary = NativeBatchedVerifierBackend(reference)

        with self.assertRaisesRegex(VerifierContractError, "batched_causal"):
            production_boundary.begin_block(request((1, 2, 3, 4)))

        self.assertEqual(runtime.context_token_ids, (90,))

    def test_only_k1_k4_k8_are_accepted_by_reference(self):
        runtime = FixtureFullDepthRuntime((1, 2))
        backend = CpuCausalBlockReferenceBackend(runtime)
        with self.assertRaisesRegex(VerifierContractError, "K="):
            backend.begin_causal_block(request((1, 2)))
        self.assertEqual(runtime.step_calls, 0)

    def test_stale_commit_cannot_overwrite_external_runtime_advance(self):
        runtime = FixtureFullDepthRuntime((1,))
        backend = CpuCausalBlockReferenceBackend(runtime)
        transaction = backend.begin_causal_block(request((1,)))
        transaction.prepare_commit(
            decide_acceptance((1,), transaction.verification.predicted_token_ids)
        )
        runtime.step(forced_next_token_id=999)

        with self.assertRaisesRegex(VerifierContractError, "外部推进"):
            transaction.commit()

        self.assertEqual(runtime.context_token_ids, (90, 999))
        transaction.rollback()  # stale 事务的补充 rollback 必须是 no-op
        self.assertEqual(runtime.context_token_ids, (90, 999))

    def test_stale_open_rollback_cannot_overwrite_external_runtime_advance(self):
        runtime = FixtureFullDepthRuntime((1,))
        backend = CpuCausalBlockReferenceBackend(runtime)
        transaction = backend.begin_causal_block(request((1,)))
        runtime.step(forced_next_token_id=999)

        with self.assertRaisesRegex(VerifierContractError, "外部推进"):
            transaction.rollback()

        self.assertEqual(runtime.context_token_ids, (90, 999))

    def test_old_committed_transaction_cannot_rollback_newer_commit(self):
        runtime = FixtureFullDepthRuntime((1, 2))
        backend = CpuCausalBlockReferenceBackend(runtime)
        old = backend.begin_causal_block(request((1,)))
        commit_native_decision(old, (1,))
        newer_request = request((2,), context=(90, 1))
        newer = backend.begin_causal_block(newer_request)
        commit_native_decision(newer, (2,))

        with self.assertRaisesRegex(VerifierContractError, "过期.*rollback"):
            old.rollback()

        self.assertEqual(runtime.context_token_ids, (90, 1, 2))

    def test_backend_rejects_overlapping_open_transactions(self):
        runtime = FixtureFullDepthRuntime((1,))
        backend = CpuCausalBlockReferenceBackend(runtime)
        transaction = backend.begin_causal_block(request((1,)))

        with self.assertRaisesRegex(VerifierContractError, "未结束"):
            backend.begin_causal_block(request((1,)))

        transaction.rollback()


if __name__ == "__main__":
    unittest.main()
