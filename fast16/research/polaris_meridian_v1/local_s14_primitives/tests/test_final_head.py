from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from fast16.research.polaris_meridian_v1.local_s14_primitives.final_head import (
    OFFICIAL_REVISION,
    bf16_checkpoint_head_logits,
    hc_head_reduce,
    native_final_logits,
    official_rms_norm,
)


ROOT = Path(__file__).resolve().parents[1]


def fixtures() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(731)
    x = torch.randn(2, 3, 4, 8, dtype=torch.float32).to(torch.bfloat16)
    hc_fn = torch.randn(4, 32, dtype=torch.float32) * 0.1
    hc_scale = torch.tensor([0.7], dtype=torch.float32)
    hc_base = torch.linspace(-0.3, 0.3, 4, dtype=torch.float32)
    norm_weight = torch.linspace(0.8, 1.2, 8, dtype=torch.float32)
    head_weight = torch.randn(13, 8, dtype=torch.float32).to(torch.bfloat16)
    return x, hc_fn, hc_scale, hc_base, norm_weight, head_weight


def test_hc_head_matches_independent_official_formula() -> None:
    x, hc_fn, scale, base, _, _ = fixtures()
    actual, pre = hc_head_reduce(x, hc_fn, scale, base)
    flat = x.flatten(2).float()
    rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-6)
    expected_pre = torch.sigmoid(F.linear(flat, hc_fn) * rsqrt * scale + base) + 1e-6
    expected = torch.sum(expected_pre.unsqueeze(-1) * flat.view(x.shape), dim=2).to(x.dtype)
    torch.testing.assert_close(pre, expected_pre, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.dtype == torch.bfloat16 and pre.dtype == torch.float32


def test_rms_norm_matches_official_fp32_order_and_restores_dtype() -> None:
    x, _, _, _, norm_weight, _ = fixtures()
    reduced = x[:, :, 0]
    actual = official_rms_norm(reduced, norm_weight)
    value = reduced.float()
    expected = (norm_weight * value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)).to(
        reduced.dtype
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.dtype == reduced.dtype


@pytest.mark.parametrize("full_logits", [False, True])
def test_bf16_checkpoint_head_uses_official_fp32_linear(full_logits: bool) -> None:
    x, _, _, _, _, head_weight = fixtures()
    hidden = x[:, :, 0]
    actual = bf16_checkpoint_head_logits(
        hidden,
        head_weight,
        full_logits=full_logits,
        output_chunk_size=5,
    )
    selected = hidden if full_logits else hidden[:, -1]
    expected = F.linear(selected.float(), head_weight.float())
    # 分块会让 BLAS 选择不同的 M 维 kernel；FP32 末位不要求跨 kernel bitwise 一致。
    torch.testing.assert_close(actual, expected, rtol=2e-7, atol=3e-7)
    assert actual.dtype == torch.float32


def test_complete_native_final_path_matches_explicit_composition() -> None:
    x, hc_fn, scale, base, norm_weight, head_weight = fixtures()
    logits, normalized, pre = native_final_logits(
        x,
        hc_fn,
        scale,
        base,
        norm_weight,
        head_weight,
        full_logits=True,
        output_chunk_size=4,
        enforce_official_shape=False,
    )
    reduced, expected_pre = hc_head_reduce(x, hc_fn, scale, base)
    expected_norm = official_rms_norm(reduced, norm_weight)
    expected_logits = F.linear(expected_norm.float(), head_weight.float())
    torch.testing.assert_close(pre, expected_pre, rtol=0, atol=0)
    torch.testing.assert_close(normalized, expected_norm, rtol=0, atol=0)
    torch.testing.assert_close(logits, expected_logits, rtol=2e-7, atol=3e-7)


def test_negative_dtype_and_shapes() -> None:
    x, hc_fn, scale, base, norm_weight, head_weight = fixtures()
    with pytest.raises(TypeError, match="bfloat16"):
        native_final_logits(
            x.float(), hc_fn, scale, base, norm_weight, head_weight, enforce_official_shape=False
        )
    with pytest.raises(ValueError, match=r"\[4,4096\]"):
        native_final_logits(x, hc_fn, scale, base, norm_weight, head_weight)
    with pytest.raises(TypeError, match="head_weight"):
        bf16_checkpoint_head_logits(x[:, :, 0], head_weight.float())
    with pytest.raises(ValueError, match="hc_head_fn"):
        hc_head_reduce(x, hc_fn[:, :-1], scale, base)
    with pytest.raises(ValueError, match="hc_head_scale"):
        hc_head_reduce(x, hc_fn, torch.ones(4), base)
    with pytest.raises(ValueError, match="norm_weight"):
        official_rms_norm(x[:, :, 0], norm_weight[:-1])


def test_source_audit_is_frozen_and_explicit_about_logit_dtype() -> None:
    audit = json.loads((ROOT / "final_head_source_audit.json").read_text(encoding="utf-8"))
    assert audit["revision"] == OFFICIAL_REVISION
    assert audit["source_file"]["sha256"] == "c0c19e6c9fa439bac7fbb1c5bc1868232dfd5aa2f439a548d0e33dcc2a9edd3f"
    assert audit["official_path"]["checkpoint_head_dtype"] == "BF16"
    assert audit["official_path"]["logits_dtype"] == "FP32"
    assert audit["weights_downloaded"] is False
