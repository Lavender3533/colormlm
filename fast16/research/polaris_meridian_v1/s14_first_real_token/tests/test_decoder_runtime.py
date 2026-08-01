"""连续 token runtime 的零网络最小验收。"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from fast16.research.polaris_meridian_v1.s14_first_real_token.executor import (
    DEFAULT_ASSET_ROOT,
    REGISTERED_LAYERS,
    DecoderRuntime,
    DecoderSnapshot,
    LayerRuntimeState,
    NativeLayerReference,
    TensorStore,
    _advance_window_kv,
    _apply_position_rope,
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


def test_new_files_are_utf8_without_bom() -> None:
    here = Path(__file__).resolve().parents[1]
    for path in here.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".md", ".json"}:
            payload = path.read_bytes()
            assert not payload.startswith(b"\xef\xbb\xbf"), path
            payload.decode("utf-8", errors="strict")
