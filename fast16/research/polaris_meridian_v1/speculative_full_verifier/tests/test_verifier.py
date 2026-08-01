from __future__ import annotations

import unittest

from ..cache_replay import RouteBlock
from ..verifier import (
    DRAFT_PROFILE,
    S14_LAYERS,
    DraftStep,
    FullDepthVerification,
    SessionState,
    SpeculativeSession,
    VerifierContractError,
)


class StubTokenizer:
    vocab_size = 1000
    fingerprint = "stub-deepseek-tokenizer"

    def encode(self, text: str) -> tuple[int, ...]:
        return tuple(ord(char) % self.vocab_size for char in text)

    def validate_token_ids(self, token_ids):
        result = tuple(token_ids)
        if any(not isinstance(token, int) or isinstance(token, bool) or not 0 <= token < self.vocab_size for token in result):
            raise VerifierContractError("bad token")
        return result


def draft_steps(token_ids: list[int]) -> tuple[DraftStep, ...]:
    routes = {layer: tuple(range(6)) for layer in S14_LAYERS}
    return tuple(DraftStep(token_id, routes) for token_id in token_ids)


def native_routes(block_size: int) -> RouteBlock:
    rows = [
        {"token_offset": token, "layer": layer, "expert_ids": [0, 1, 2, 3, 4, 5]}
        for token in range(block_size)
        for layer in range(43)
    ]
    return RouteBlock.from_rows("verify", block_size, rows)


class ScriptedDraft:
    profile = DRAFT_PROFILE

    def __init__(self, token_ids: list[int]):
        self.steps = draft_steps(token_ids)
        self.calls = 0

    def generate(self, context_token_ids, block_size):
        self.calls += 1
        if block_size != len(self.steps):
            raise AssertionError("wrong block")
        return self.steps


class ScriptedVerifier:
    def __init__(self, predictions: list[int]):
        self.predictions = predictions
        self.calls = 0
        self.requests = []

    def verify_causal_block(self, request):
        self.calls += 1
        self.requests.append(request)
        return FullDepthVerification(tuple(self.predictions), native_routes(len(self.predictions)))


class VerifierStateMachineTest(unittest.TestCase):
    def test_mismatch_commits_only_longest_prefix_and_native_fallback(self):
        session = SpeculativeSession(StubTokenizer(), [90])
        draft = ScriptedDraft([1, 2, 3, 4])
        # Predictions after index 2 are conditional on a now-rejected draft path
        # and must never leak into committed context.
        verifier = ScriptedVerifier([1, 2, 9, 777])

        result = session.run_round(draft, verifier, 4)

        self.assertEqual(result.accepted_prefix, (1, 2))
        self.assertEqual(result.fallback_token_id, 9)
        self.assertEqual(result.rejected_draft_suffix, (3, 4))
        self.assertEqual(result.committed_token_ids, (1, 2, 9))
        self.assertEqual(result.mismatch_index, 2)
        self.assertEqual(session.context_token_ids, [90, 1, 2, 9])
        self.assertEqual(session.state, SessionState.READY)
        self.assertEqual(draft.calls, 1)
        self.assertEqual(verifier.calls, 1)
        self.assertEqual(result.native_routes.block_size, 4)
        self.assertTrue(verifier.requests[0].causal)
        self.assertEqual(verifier.requests[0].depth, 43)
        self.assertEqual(verifier.requests[0].top_k, 6)

    def test_first_mismatch_commits_no_draft_token(self):
        session = SpeculativeSession(StubTokenizer(), [90])
        result = session.run_round(ScriptedDraft([1, 2]), ScriptedVerifier([8, 2]), 2)
        self.assertEqual(result.accepted_prefix, ())
        self.assertEqual(result.committed_token_ids, (8,))
        self.assertEqual(session.context_token_ids, [90, 8])

    def test_full_match_has_no_unverified_bonus_token(self):
        session = SpeculativeSession(StubTokenizer(), [90])
        result = session.run_round(ScriptedDraft([1, 2, 3]), ScriptedVerifier([1, 2, 3]), 3)
        self.assertEqual(result.accepted_prefix, (1, 2, 3))
        self.assertIsNone(result.fallback_token_id)
        self.assertEqual(result.committed_token_ids, (1, 2, 3))

    def test_explicit_transitions_reject_out_of_order_calls(self):
        session = SpeculativeSession(StubTokenizer(), [])
        with self.assertRaises(VerifierContractError):
            session.make_verification_request()
        session.start_round(1)
        self.assertEqual(session.state, SessionState.DRAFTING)
        with self.assertRaises(VerifierContractError):
            session.start_round(1)
        session.submit_draft(draft_steps([1]))
        self.assertEqual(session.state, SessionState.AWAITING_VERIFICATION)
        session.make_verification_request()
        self.assertEqual(session.state, SessionState.VERIFYING)

    def test_malformed_s14_route_fails_closed_without_context_change(self):
        class BadDraft:
            def generate(self, context_token_ids, block_size):
                return [DraftStep(1, {0: (0, 1, 2, 3, 4, 5)})]

        session = SpeculativeSession(StubTokenizer(), [90])
        with self.assertRaises(VerifierContractError):
            session.run_round(BadDraft(), ScriptedVerifier([1]), 1)
        self.assertEqual(session.context_token_ids, [90])
        self.assertEqual(session.state, SessionState.READY)

    def test_malformed_verification_fails_closed_without_context_change(self):
        session = SpeculativeSession(StubTokenizer(), [90])
        with self.assertRaises(VerifierContractError):
            session.run_round(ScriptedDraft([1, 2]), ScriptedVerifier([1]), 2)
        self.assertEqual(session.context_token_ids, [90])
        self.assertEqual(session.state, SessionState.READY)

    def test_incomplete_native_route_block_fails_closed(self):
        class WrongRouteVerifier:
            def verify_causal_block(self, request):
                return FullDepthVerification((1, 2), native_routes(1))

        session = SpeculativeSession(StubTokenizer(), [90])
        with self.assertRaises(VerifierContractError):
            session.run_round(ScriptedDraft([1, 2]), WrongRouteVerifier(), 2)
        self.assertEqual(session.context_token_ids, [90])
        self.assertEqual(session.state, SessionState.READY)


if __name__ == "__main__":
    unittest.main()
