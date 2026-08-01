"""零网络的小型确定性自检入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from fast16.research.polaris_meridian_v1.local_s14_primitives import (
    decode_fp8_e4m3fn,
    decode_ue8m0,
    hc_split_sinkhorn,
    native_final_logits,
    sparse_attention,
    unpack_mxfp4_e2m1,
)


def run_selftest() -> dict:
    checks = {}
    checks["e2m1_order"] = torch.equal(
        unpack_mxfp4_e2m1(torch.tensor([0x21, 0xF7], dtype=torch.uint8)),
        torch.tensor([0.5, 1.0, 6.0, -6.0]),
    )
    checks["ue8m0_known_bits"] = torch.equal(
        decode_ue8m0(torch.tensor([120, 121, 127], dtype=torch.uint8)),
        torch.tensor([0.0078125, 0.015625, 1.0]),
    )
    checks["e4m3_known_bits"] = torch.equal(
        decode_fp8_e4m3fn(torch.tensor([0x01, 0x38, 0x7E], dtype=torch.uint8)),
        torch.tensor([2**-9, 1.0, 448.0]),
    )
    _, _, comb = hc_split_sinkhorn(torch.zeros(1, 1, 24), torch.ones(3), torch.zeros(24))
    checks["hc_finite_column_normalized"] = bool(
        torch.isfinite(comb).all() and torch.allclose(comb.sum(-2), torch.ones(1, 1, 4), rtol=2e-5, atol=2e-5)
    )
    attention = sparse_attention(
        torch.ones(1, 1, 1, 1),
        torch.tensor([[[1.0], [2.0]]]),
        torch.zeros(1),
        torch.tensor([[[-1, -1]]]),
    )
    checks["sparse_all_padding_stable_zero"] = torch.equal(attention, torch.zeros_like(attention))
    try:
        sparse_attention(torch.ones(1, 1, 1, 1), torch.ones(1, 2, 1), torch.zeros(1), torch.tensor([[[-2]]]))
    except IndexError:
        checks["invalid_index_rejected"] = True
    else:
        checks["invalid_index_rejected"] = False
    logits, normalized, pre = native_final_logits(
        torch.ones(1, 1, 4, 2, dtype=torch.bfloat16),
        torch.zeros(4, 8),
        torch.ones(1),
        torch.zeros(4),
        torch.ones(2),
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 1.0]], dtype=torch.bfloat16),
        output_chunk_size=2,
        enforce_official_shape=False,
    )
    checks["final_hc_norm_bf16_head_path"] = bool(
        logits.dtype == torch.float32
        and normalized.dtype == torch.bfloat16
        and pre.dtype == torch.float32
        and tuple(logits.shape) == (1, 3)
        and torch.isfinite(logits).all()
    )
    return {
        "format": "polaris-local-s14-primitives-selftest-v1",
        "ok": all(checks.values()),
        "checks": checks,
        "network_accessed": False,
        "weights_downloaded": False,
        "evidence_status": "synthetic_reference_semantics_not_native_s14_forward",
        "claim_limit": "自检只证明小型参考语义，不证明首 token、速度或质量。",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_selftest()
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8", errors="strict")
    print(encoded, end="")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
