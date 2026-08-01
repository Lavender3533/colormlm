"""连续 token runtime 的零网络最小验收。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from fast16.research.polaris_meridian_v1.s14_first_real_token.executor import (
    DEFAULT_ASSET_ROOT,
    REVISION,
    REGISTERED_LAYERS,
    VOCAB_SIZE,
    ContractError,
    DecoderRuntime,
    DecoderSnapshot,
    ForcedTokenQueue,
    LayerRuntimeState,
    NativeLayerReference,
    TensorStore,
    TokenComputation,
    _advance_compressor_buffers,
    _advance_window_kv,
    _apply_position_rope,
    _cpu_fp4_activation_quant,
    _cpu_hadamard_rotate,
    _deterministic_compressed_topk_indices,
    _load_forced_prefill,
    _precompute_position_freqs,
    _read_json,
    _window_topk_indices,
)
from fast16.research.polaris_meridian_v1.s14_range_pack import online_range


def _catalog_and_cache() -> tuple[dict, online_range.RangeCache]:
    root = DEFAULT_ASSET_ROOT.resolve()
    catalog = _read_json(root / "route_first_catalog.json")
    return catalog, online_range.RangeCache(root / "range_cache", allow_fetch=False)


def test_position1_rope_and_inverse_match_official_complex_multiply() -> None:
    freqs = _precompute_position_freqs(4, seqlen=2)[1:2]
    original = torch.linspace(-1.0, 1.0, 64, dtype=torch.float32).view(1, 1, 1, 64)
    expected_complex = torch.view_as_complex(original.unflatten(-1, (-1, 2)))
    expected = torch.view_as_real(expected_complex * freqs.view(1, 1, 1, 32)).flatten(-2)

    actual = original.clone()
    _apply_position_rope(actual, freqs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    _apply_position_rope(actual, freqs, inverse=True)
    torch.testing.assert_close(actual, original, rtol=1e-6, atol=1e-6)


def test_position1_window_writes_before_attention_and_reads_p0_plus_p1() -> None:
    p0 = torch.arange(512, dtype=torch.float32).to(torch.bfloat16).view(1, 1, 512)
    committed = _advance_window_kv(p0, position=0, previous=None)
    committed_before = committed.clone()
    p1 = (-torch.arange(512, dtype=torch.float32)).to(torch.bfloat16).view(1, 1, 512)
    next_window = _advance_window_kv(p1, position=1, previous=committed)

    torch.testing.assert_close(committed, committed_before, rtol=0, atol=0)
    torch.testing.assert_close(next_window[:, 0:1], p0, rtol=0, atol=0)
    torch.testing.assert_close(next_window[:, 1:2], p1, rtol=0, atol=0)
    indices = _window_topk_indices(1)
    assert tuple(indices.shape) == (1, 1, 128)
    assert indices[0, 0, :2].tolist() == [0, 1]
    assert bool((indices[0, 0, 2:] == -1).all().item())


def test_ratio4_p0_to_p7_emits_boundary_blocks_and_migrates_overlap() -> None:
    kv_state = None
    score_state = None
    first_block = None
    for position in range(4):
        current = torch.full((1, 1, 8), float(position + 1))
        score = torch.zeros_like(current)
        kv_state, score_state, block, row = _advance_compressor_buffers(
            kv_state,
            score_state,
            current,
            score,
            position=position,
            ratio=4,
            overlap=True,
        )
        assert row == position + 4
        if position < 3:
            assert block is None
        else:
            first_block = block
    assert first_block is not None
    assert tuple(first_block.shape) == (1, 1, 4)
    torch.testing.assert_close(first_block, torch.full((1, 1, 4), 2.5))
    for row, value in enumerate((1.0, 2.0, 3.0, 4.0)):
        torch.testing.assert_close(kv_state[:, row], torch.full((1, 8), value))

    second_block = None
    for position in range(4, 8):
        current = torch.full((1, 1, 8), float(position + 1))
        score = torch.zeros_like(current)
        kv_state, score_state, block, row = _advance_compressor_buffers(
            kv_state,
            score_state,
            current,
            score,
            position=position,
            ratio=4,
            overlap=True,
        )
        assert row == position % 4 + 4
        if position == 7:
            second_block = block
    assert second_block is not None
    assert tuple(second_block.shape) == (1, 1, 4)
    torch.testing.assert_close(second_block, torch.full((1, 1, 4), 4.5))
    for row, value in enumerate((5.0, 6.0, 7.0, 8.0)):
        torch.testing.assert_close(kv_state[:, row], torch.full((1, 8), value))


def test_ratio128_only_emits_at_p127() -> None:
    kv_state = None
    score_state = None
    block = None
    for position in range(128):
        current = torch.full((1, 1, 4), float(position + 1))
        score = torch.zeros_like(current)
        kv_state, score_state, current_block, row = _advance_compressor_buffers(
            kv_state,
            score_state,
            current,
            score,
            position=position,
            ratio=128,
            overlap=False,
        )
        assert row == position
        if position < 127:
            assert current_block is None
        else:
            block = current_block
    assert block is not None
    torch.testing.assert_close(block, torch.full((1, 1, 4), 64.5))


def test_window_ring_and_compressed_offset_at_p127_p128() -> None:
    window = None
    for position in range(129):
        current = torch.full((1, 1, 512), float(position), dtype=torch.bfloat16)
        window = _advance_window_kv(current, position=position, previous=window)
    assert window is not None
    assert _window_topk_indices(127)[0, 0].tolist() == list(range(128))
    assert _window_topk_indices(128)[0, 0].tolist() == [*range(1, 128), 0]
    assert float(window[0, 0, 0]) == 128.0
    assert float(window[0, 1, 0]) == 1.0
    assert _deterministic_compressed_topk_indices(128, 127, 1)[0, 0].tolist() == [128]
    assert _deterministic_compressed_topk_indices(128, 128, 1)[0, 0].tolist() == [128]


def test_hadamard_and_fp4_small_cpu_fixtures() -> None:
    source = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]], dtype=torch.bfloat16)
    rotated = _cpu_hadamard_rotate(source)
    expected = torch.tensor([[[5.0, -1.0, -2.0, 0.0]]], dtype=torch.bfloat16)
    torch.testing.assert_close(rotated, expected, rtol=0, atol=0)

    fp4_source = torch.tensor(
        [[[float(value) / 8 for value in range(-16, 16)]]], dtype=torch.bfloat16
    )
    quantized = _cpu_fp4_activation_quant(fp4_source)
    assert quantized.dtype == torch.bfloat16
    assert bool(torch.isfinite(quantized).all().item())
    assert float((quantized.float() - fp4_source.float()).abs().max()) <= 0.5


def test_hash_route_uses_current_token_row_from_physical_i64() -> None:
    token_id = 108967
    catalog, cache = _catalog_and_cache()
    session = online_range.RouteFirstSession(catalog, cache)
    session.prepare_embedding_row(0)
    prerequisites = session.prepare_layer(0, token_id)
    store = TensorStore(DEFAULT_ASSET_ROOT.resolve() / "range_cache")
    store.add_ranges((*prerequisites.non_expert, *prerequisites.router))
    kernel = NativeLayerReference(0, store)
    physical = kernel._load_i64("layers.0.ffn.gate.tid2eid")

    route_ids, weights, source = kernel._route(
        torch.zeros((1, 1, 4096), dtype=torch.bfloat16),
        token_id=token_id,
    )
    assert route_ids == [int(value) for value in physical[token_id].tolist()]
    assert source == "current_token_tid2eid_physical_i64"
    assert sum(weights) == pytest.approx(1.5, abs=2e-7)


def test_decoder_runtime_rolls_back_state_and_token_id_on_failure() -> None:
    catalog, cache = _catalog_and_cache()
    committed_states = {
        layer: LayerRuntimeState(
            layer=layer,
            position=0,
            window_kv=torch.zeros((1, 128, 512), dtype=torch.bfloat16),
            compressor=None,
        )
        for layer in REGISTERED_LAYERS
    }
    snapshot = DecoderSnapshot(
        position=1,
        input_token_id=108967,
        layer_states=committed_states,
        committed_tokens=(
            {"position": 0, "input_token_id": 0, "output_token_id": 108967},
        ),
    )
    runtime = DecoderRuntime(catalog=catalog, cache=cache, snapshot=snapshot)

    def fail(position, input_token_id, previous_states, session):
        assert position == 1 and input_token_id == 108967
        assert session.phase is online_range.SessionPhase.INIT
        previous_states[0].window_kv.fill_(7)
        raise RuntimeError("injected token failure")

    with pytest.raises(RuntimeError, match="injected token failure"):
        runtime.run_token(fail)
    assert runtime.snapshot is snapshot
    assert runtime.position == 1 and runtime.input_token_id == 108967
    assert bool((runtime.layer_states[0].window_kv == 0).all().item())


def _next_states(position: int) -> dict[int, LayerRuntimeState]:
    return {
        layer: LayerRuntimeState(
            layer=layer,
            position=position,
            window_kv=torch.full(
                (1, 128, 512), float(position), dtype=torch.bfloat16
            ),
            compressor=None,
        )
        for layer in REGISTERED_LAYERS
    }


def test_forced_prefill_queue_wins_until_exhausted_then_argmax_takes_over() -> None:
    catalog, cache = _catalog_and_cache()
    queue = ForcedTokenQueue(
        token_ids=(0, 11, 22),
        cursor=0,
        artifact_sha256="a" * 64,
    )
    runtime = DecoderRuntime(
        catalog=catalog,
        cache=cache,
        snapshot=DecoderSnapshot(forced_queue=queue),
    )

    def succeed(position, input_token_id, previous_states, session):
        del previous_states, session
        return TokenComputation(
            output_token_id=99,
            next_layer_states=_next_states(position),
            value={"input": input_token_id},
        )

    runtime.run_token(succeed)
    assert (runtime.position, runtime.input_token_id) == (1, 11)
    assert runtime.snapshot.forced_queue is not None
    assert runtime.snapshot.forced_queue.cursor == 1
    runtime.run_token(succeed)
    assert (runtime.position, runtime.input_token_id) == (2, 22)
    assert runtime.snapshot.forced_queue.cursor == 2
    runtime.run_token(succeed)
    assert (runtime.position, runtime.input_token_id) == (3, 99)
    assert runtime.snapshot.forced_queue.cursor == 3
    assert not runtime.snapshot.forced_queue.active


def test_non_forced_committed_token_format_remains_compatible() -> None:
    catalog, cache = _catalog_and_cache()
    runtime = DecoderRuntime(catalog=catalog, cache=cache)

    runtime.run_token(
        lambda position, input_token_id, previous_states, session: TokenComputation(
            output_token_id=99,
            next_layer_states=_next_states(position),
            value={},
        )
    )
    assert runtime.snapshot.committed_tokens == (
        {"position": 0, "input_token_id": 0, "output_token_id": 99},
    )


def test_forced_prefill_failure_at_p1_rolls_back_cursor_and_all_state() -> None:
    catalog, cache = _catalog_and_cache()
    queue = ForcedTokenQueue(
        token_ids=(0, 11, 22),
        cursor=0,
        artifact_sha256="b" * 64,
    )
    runtime = DecoderRuntime(
        catalog=catalog,
        cache=cache,
        snapshot=DecoderSnapshot(forced_queue=queue),
    )

    runtime.run_token(
        lambda position, input_token_id, previous_states, session: TokenComputation(
            output_token_id=77,
            next_layer_states=_next_states(position),
            value={},
        )
    )
    committed = runtime.snapshot

    def fail(position, input_token_id, previous_states, session):
        del session
        assert (position, input_token_id) == (1, 11)
        previous_states[0].window_kv.fill_(9)
        raise RuntimeError("injected forced p1 failure")

    with pytest.raises(RuntimeError, match="injected forced p1 failure"):
        runtime.run_token(fail)
    assert runtime.snapshot is committed
    assert runtime.snapshot.forced_queue is not None
    assert runtime.snapshot.forced_queue.cursor == 1
    assert (runtime.position, runtime.input_token_id) == (1, 11)
    assert bool((runtime.layer_states[0].window_kv == 0).all().item())


def test_forced_prefill_artifact_contract_and_hash(tmp_path: Path) -> None:
    token_ids = [0, 11, 22]
    token_hash = hashlib.sha256(
        json.dumps(token_ids, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    artifact = {
        "format": "polaris-s14-forced-prefill-v1",
        "chat_encoding": {"revision": REVISION},
        "tokenizer": {
            "profile": "s14",
            "vocab_size": VOCAB_SIZE,
            "bos_token_id": 0,
        },
        "token_ids": token_ids,
        "token_count": len(token_ids),
        "token_ids_sha256": token_hash,
        "decoder_consumption": {
            "mode": "sequential_forced_prefill",
            "position_base": 0,
            "position_count": len(token_ids),
            "position_rule": "token_ids[position]",
            "polaris_s14_compatible": True,
        },
    }
    path = tmp_path / "forced.json"
    path.write_text(json.dumps(artifact, ensure_ascii=False), encoding="utf-8")
    queue = _load_forced_prefill(path)
    assert queue.token_ids == (0, 11, 22)
    assert queue.cursor == 0
    assert queue.artifact_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()

    artifact["decoder_consumption"]["polaris_s14_compatible"] = False
    path.write_text(json.dumps(artifact, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ContractError, match="decoder_consumption"):
        _load_forced_prefill(path)


def test_new_files_are_utf8_without_bom() -> None:
    here = Path(__file__).resolve().parents[1]
    for path in here.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".md", ".json"}:
            payload = path.read_bytes()
            assert not payload.startswith(b"\xef\xbb\xbf"), path
            payload.decode("utf-8", errors="strict")
