from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import pytest
import torch

from fast16.research.polaris_meridian_v1.local_s14_primitives import (
    FP4_E2M1_VALUES,
    decode_fp8_e4m3fn,
    decode_mxfp4,
    decode_scaled_fp8_e4m3fn,
    decode_ue8m0,
    fp4_linear,
    fp8_linear,
    fp8_weight_linear,
    hc_post,
    hc_pre,
    hc_split_sinkhorn,
    sparse_attention,
    unpack_mxfp4_e2m1,
)
from fast16.research.polaris_meridian_v1.local_s14_primitives.abi_sample import (
    EXPECTED_EXPERT_REGRESSION,
    inspect_external_abi_sample,
    run_external_expert_regression,
)


ROOT = Path(__file__).resolve().parents[1]


def test_e2m1_all_codes_and_i8_container_order() -> None:
    packed = torch.tensor([0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE], dtype=torch.uint8)
    actual = unpack_mxfp4_e2m1(packed)
    expected = torch.tensor(FP4_E2M1_VALUES)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.signbit(actual[8])
    torch.testing.assert_close(
        unpack_mxfp4_e2m1(torch.tensor([0x21, 0xF7], dtype=torch.uint8)),
        torch.tensor([0.5, 1.0, 6.0, -6.0]),
    )


def test_ue8m0_edges_and_nan_rejection() -> None:
    raw = torch.tensor([0, 1, 120, 121, 126, 127, 128, 254], dtype=torch.uint8)
    expected = torch.tensor([2.0 ** (int(code) - 127) for code in raw], dtype=torch.float32)
    torch.testing.assert_close(decode_ue8m0(raw), expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="0xFF"):
        decode_ue8m0(torch.tensor([255], dtype=torch.uint8))
    assert torch.isnan(decode_ue8m0(torch.tensor([255], dtype=torch.uint8), reject_nan=False)).item()


def test_mxfp4_decode_and_chunked_linear() -> None:
    packed = torch.arange(3 * 64, dtype=torch.uint8).reshape(3, 64)
    scales = torch.tensor([[126, 127, 128, 129], [127, 128, 126, 127], [128, 126, 127, 128]], dtype=torch.uint8)
    weight = decode_mxfp4(packed, scales)
    x = torch.sin(torch.arange(2 * 128, dtype=torch.float32).reshape(2, 128) * 0.013)
    activation_scale = torch.tensor([[127], [128]], dtype=torch.uint8)
    expected = (x * decode_ue8m0(activation_scale).repeat_interleave(128, -1)) @ weight.T
    actual = fp4_linear(
        x,
        packed,
        scales,
        activation_scale=activation_scale,
        output_chunk_size=2,
    )
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)


def test_mxfp4_negative_shapes_and_codes() -> None:
    with pytest.raises(ValueError, match="形状不匹配"):
        decode_mxfp4(torch.zeros(1, 16, dtype=torch.uint8), torch.zeros(1, 2, dtype=torch.uint8))
    with pytest.raises(ValueError, match="activation_group_size"):
        fp4_linear(torch.zeros(64), torch.zeros(1, 32, dtype=torch.uint8), torch.zeros(1, 2, dtype=torch.uint8))
    with pytest.raises(ValueError, match="weight_scale"):
        fp4_linear(torch.zeros(128), torch.zeros(1, 64, dtype=torch.uint8), torch.zeros(1, 3, dtype=torch.uint8))
    with pytest.raises(TypeError, match="packed"):
        unpack_mxfp4_e2m1(torch.zeros(1, dtype=torch.int16))


def test_e4m3_bit_level_decoder_matches_known_values_and_torch() -> None:
    raw = torch.tensor([0x00, 0x01, 0x07, 0x08, 0x38, 0x7E, 0x80, 0xB8, 0xFE], dtype=torch.uint8)
    expected = torch.tensor([0.0, 2**-9, 7 * 2**-9, 2**-6, 1.0, 448.0, -0.0, -1.0, -448.0])
    actual = decode_fp8_e4m3fn(raw)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if hasattr(torch, "float8_e4m3fn"):
        torch.testing.assert_close(actual, raw.view(torch.float8_e4m3fn).float(), rtol=0, atol=0)
    with pytest.raises(ValueError, match="NaN"):
        decode_fp8_e4m3fn(torch.tensor([0x7F, 0xFF], dtype=torch.uint8))


def test_scaled_fp8_and_both_chunked_linear_paths() -> None:
    torch.manual_seed(7)
    raw_x = torch.randint(0, 255, (2, 130), dtype=torch.uint8)
    raw_w = torch.randint(0, 255, (129, 130), dtype=torch.uint8)
    raw_x[raw_x == 0x7F] = 0x7E
    raw_w[raw_w == 0x7F] = 0x7E
    scale_a = torch.tensor([[127, 128], [126, 127]], dtype=torch.uint8)
    scale_w = torch.tensor([[127, 128], [126, 127]], dtype=torch.uint8)
    decoded_x = decode_scaled_fp8_e4m3fn(raw_x, scale_a)
    decoded_w = decode_fp8_e4m3fn(raw_w) * decode_ue8m0(scale_w).repeat_interleave(128, 0).repeat_interleave(128, 1)[:129, :130]
    expected = decoded_x @ decoded_w.T
    actual = fp8_linear(raw_x, scale_a, raw_w, scale_w, output_chunk_size=37)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    logical_x = torch.sin(torch.arange(260, dtype=torch.float32).reshape(2, 130) * 0.01)
    actual_weight_only = fp8_weight_linear(logical_x, raw_w, scale_w, output_chunk_size=41)
    torch.testing.assert_close(actual_weight_only, logical_x @ decoded_w.T, rtol=2e-5, atol=2e-5)


def test_fp8_negative_shapes() -> None:
    raw_x = torch.zeros(1, 128, dtype=torch.uint8)
    raw_w = torch.zeros(2, 128, dtype=torch.uint8)
    with pytest.raises(ValueError, match="activation_scale"):
        fp8_linear(raw_x, torch.zeros(1, 2, dtype=torch.uint8), raw_w, torch.zeros(1, 1, dtype=torch.uint8))
    with pytest.raises(ValueError, match="weight_scale"):
        fp8_linear(raw_x, torch.zeros(1, 1, dtype=torch.uint8), raw_w, torch.zeros(2, 1, dtype=torch.uint8))
    with pytest.raises(TypeError, match="uint8"):
        decode_fp8_e4m3fn(torch.zeros(1, dtype=torch.int8))


def test_hc_split_sinkhorn_and_state_transition() -> None:
    mixes = torch.linspace(-1000, 1000, 24, dtype=torch.float32).reshape(1, 1, 24)
    scale = torch.tensor([0.5, -0.25, 0.01])
    base = torch.linspace(-1, 1, 24)
    pre, post, comb = hc_split_sinkhorn(mixes, scale, base, sinkhorn_iters=20)
    assert torch.isfinite(pre).all() and torch.isfinite(post).all() and torch.isfinite(comb).all()
    torch.testing.assert_close(comb.sum(dim=-2), torch.ones(1, 1, 4), rtol=2e-5, atol=2e-5)

    x = torch.sin(torch.arange(32, dtype=torch.float32) * 0.1).reshape(1, 1, 4, 8)
    fn = torch.linspace(-0.2, 0.2, 24 * 32).reshape(24, 32)
    reduced, post2, comb2 = hc_pre(x, fn, scale, base)
    branch = torch.cos(torch.arange(8, dtype=torch.float32)).reshape(1, 1, 8)
    merged = hc_post(branch, x, post2, comb2)
    assert reduced.shape == (1, 1, 8) and merged.shape == (1, 1, 4, 8)
    assert torch.isfinite(reduced).all() and torch.isfinite(merged).all()


def test_hc_negative_contracts() -> None:
    with pytest.raises(ValueError, match="最后一维"):
        hc_split_sinkhorn(torch.zeros(1, 23), torch.ones(3), torch.zeros(24))
    with pytest.raises(ValueError, match="至少 1"):
        hc_split_sinkhorn(torch.zeros(1, 24), torch.ones(3), torch.zeros(24), sinkhorn_iters=0)
    with pytest.raises(ValueError, match="hc_fn"):
        hc_pre(torch.zeros(1, 1, 4, 8), torch.zeros(24, 31), torch.ones(3), torch.zeros(24))


def test_sparse_attention_sink_mask_duplicate_and_stability() -> None:
    q = torch.tensor([[[[1.0]]]])
    kv = torch.tensor([[[1.0], [2.0], [3.0]]])
    sink = torch.tensor([0.0])
    indices = torch.tensor([[[0, 2]]], dtype=torch.int32)
    actual = sparse_attention(q, kv, sink, indices, softmax_scale=1.0)
    expected = (math.exp(1) * 1 + math.exp(3) * 3) / (math.exp(1) + math.exp(3) + 1)
    torch.testing.assert_close(actual, torch.tensor([[[[expected]]]]))

    duplicate = sparse_attention(q, kv, sink, torch.tensor([[[2, 2, -1]]], dtype=torch.int64), softmax_scale=1.0)
    duplicate_expected = (2 * math.exp(3) * 3) / (2 * math.exp(3) + 1)
    torch.testing.assert_close(duplicate, torch.tensor([[[[duplicate_expected]]]]))
    all_padding = sparse_attention(q * 1e20, kv * 1e10, sink + 1e20, torch.tensor([[[-1, -1]]]), softmax_scale=1.0)
    assert torch.equal(all_padding, torch.zeros_like(all_padding))


@pytest.mark.parametrize("bad", [-2, 3])
def test_sparse_attention_rejects_invalid_index(bad: int) -> None:
    with pytest.raises(IndexError, match="越界"):
        sparse_attention(
            torch.zeros(1, 1, 1, 2),
            torch.zeros(1, 3, 2),
            torch.zeros(1),
            torch.tensor([[[bad]]], dtype=torch.int32),
        )


def test_source_contract_vulkan_abi_and_utf8() -> None:
    source = json.loads((ROOT / "source_audit.json").read_text(encoding="utf-8", errors="strict"))
    abi = json.loads((ROOT / "vulkan_abi_v1.json").read_text(encoding="utf-8", errors="strict"))
    assert source["revision"] == "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
    assert source["source_files"]["inference/kernel.py"]["sha256"] == "59b325083d7103975cba025bd0d60ea343bb82d8fff53088afb7c04bd380c0c2"
    assert abi["kernels"]["fp8_linear"]["logical_shapes"]["weight_scale"] == "[ceil(N/128),ceil(K/128)]"
    for path in ROOT.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".md", ".json"}:
            path.read_text(encoding="utf-8", errors="strict")


def _sample_path(env_name: str, fallback: str) -> Path:
    return Path(os.environ.get(env_name, fallback))


def test_optional_real_l42_e0_expert_sample() -> None:
    path = _sample_path(
        "POLARIS_S14_EXPERT_ABI_SAMPLE",
        "D:/models/Polaris-S14/abi_samples/l42_e0/manifest.json",
    )
    if not path.is_file():
        pytest.skip("外部 L42/E0 ABI 样本不存在；默认 CI 不依赖它")
    sanity = inspect_external_abi_sample(path)
    assert sanity["tofu_hashes_checked"] == 8
    assert [item["scale_code"] for item in sanity["expert_slices"]] == [121, 120, 121]
    regression = run_external_expert_regression(path, verify_hashes=False)
    assert regression["numpy_frozen_baseline"] == EXPECTED_EXPERT_REGRESSION
    assert regression["pytorch_chunked_linear_parity"]["allclose"]


def test_optional_real_fp8_hc_sample_sanity() -> None:
    path = _sample_path(
        "POLARIS_S14_FP8_HC_ABI_SAMPLE",
        "D:/models/Polaris-S14/abi_samples/l42_fp8_hc/manifest.json",
    )
    if not path.is_file():
        pytest.skip("外部 L42 FP8/HC ABI 样本不存在；默认 CI 不依赖它")
    manifest = json.loads(path.read_text(encoding="utf-8", errors="strict"))
    assert manifest["revision"] == "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
    assert manifest["payload_bytes"] == 5_767_532
    entries = {entry["name"]: entry for entry in manifest["entries"]}
    for entry in entries.values():
        payload = Path(entry["path"]).read_bytes()
        assert len(payload) == entry["bytes"]
        assert hashlib.sha256(payload).hexdigest() == entry["sha256_tofu"]

    weight_entry = entries["layers.42.attn.wq_a.weight"]
    scale_entry = entries["layers.42.attn.wq_a.scale"]
    raw_weight = torch.frombuffer(bytearray(Path(weight_entry["path"]).read_bytes()), dtype=torch.uint8).reshape(weight_entry["shape"])
    raw_scale = torch.frombuffer(bytearray(Path(scale_entry["path"]).read_bytes()), dtype=torch.uint8).reshape(scale_entry["shape"])
    decoded_weight = decode_fp8_e4m3fn(raw_weight) * decode_ue8m0(raw_scale).repeat_interleave(128, 0).repeat_interleave(128, 1)
    x = torch.sin(torch.arange(4096, dtype=torch.float32) * 0.013)
    y = decoded_weight @ x
    assert torch.isfinite(y).all()
    assert hashlib.sha256(y.numpy().astype("<f4", copy=False).tobytes()).hexdigest() == "bba6c0b77cc1fb1f9e45b6d4a8cf067eaf6d70b8822baa903c0ee28f8ac1fe87"
    chunked = fp8_weight_linear(x, raw_weight, raw_scale)
    torch.testing.assert_close(chunked, y, rtol=1e-5, atol=2e-5)

    def load_f32(name: str) -> torch.Tensor:
        entry = entries[name]
        return torch.frombuffer(bytearray(Path(entry["path"]).read_bytes()), dtype=torch.float32).reshape(entry["shape"])

    hc_x = torch.sin(torch.arange(4 * 4096, dtype=torch.float32) * 0.007).reshape(1, 1, 4, 4096)
    reduced, post, comb = hc_pre(
        hc_x,
        load_f32("layers.42.hc_attn_fn"),
        load_f32("layers.42.hc_attn_scale"),
        load_f32("layers.42.hc_attn_base"),
    )
    branch = torch.cos(torch.arange(4096, dtype=torch.float32) * 0.009).reshape(1, 1, 4096)
    merged = hc_post(branch, hc_x, post, comb)
    assert torch.isfinite(reduced).all() and torch.isfinite(merged).all()
    torch.testing.assert_close(torch.linalg.vector_norm(merged), torch.tensor(122.05479431152344), rtol=2e-6, atol=2e-6)
