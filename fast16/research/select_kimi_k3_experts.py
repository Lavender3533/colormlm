"""Select Kimi K3 experts from real ColorLM hidden-state traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESEARCH = Path(__file__).resolve().parent
K3_EXTRACTED = (
    RESEARCH
    / "biopsy_cache"
    / "moonshotai_Kimi-K3"
    / "master"
    / "extracted"
)
HEADER = struct.Struct("<IIiI4qQ")
MAGIC = 0x394D4C43
K3_WIDTH = 7168
COLORLM_WIDTH = 2048
DEFAULT_LAYER_MAP = {12: 28, 28: 65, 39: 92}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用ColorLM真实隐藏态和K3原生gate选择v9专家库"
    )
    parser.add_argument("--hidden-dump", type=Path, required=True)
    parser.add_argument(
        "--transport",
        type=Path,
        default=RESEARCH / "kimi_k3_to_colorlm_semiorthogonal_f32.npy",
    )
    parser.add_argument("--experts-per-layer", type=int, default=24)
    parser.add_argument("--native-top-k", type=int, default=16)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESEARCH / "kimi_k3_v9_expert_selection.json",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def bf16(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    expected = math.prod(shape) * 2
    if not path.is_file() or path.stat().st_size != expected:
        raise RuntimeError(f"BF16张量大小不匹配: {path}")
    bits = np.fromfile(path, dtype="<u2").astype(np.uint32) << 16
    return np.ascontiguousarray(bits.view(np.float32).reshape(shape))


def read_hidden_dump(path: Path) -> dict[int, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"隐藏态记录不存在: {path}")
    records: dict[int, list[np.ndarray]] = {}
    with path.open("rb") as stream:
        while raw_header := stream.read(HEADER.size):
            if len(raw_header) != HEADER.size:
                raise RuntimeError("隐藏态记录头被截断")
            magic, version, layer, tensor_type, ne0, ne1, ne2, ne3, payload_bytes = (
                HEADER.unpack(raw_header)
            )
            if magic != MAGIC or version != 1 or tensor_type != 0:
                raise RuntimeError(
                    f"隐藏态记录契约不匹配: magic={magic:x}, version={version}, "
                    f"type={tensor_type}"
                )
            expected = ne0 * ne1 * ne2 * ne3 * 4
            if ne0 != COLORLM_WIDTH or payload_bytes != expected:
                raise RuntimeError(
                    f"隐藏态形状异常: layer={layer}, ne={[ne0, ne1, ne2, ne3]}"
                )
            payload = stream.read(payload_bytes)
            if len(payload) != payload_bytes:
                raise RuntimeError("隐藏态记录数据被截断")
            values = np.frombuffer(payload, dtype="<f4").reshape(-1, ne0).copy()
            if not np.isfinite(values).all():
                raise RuntimeError(f"第{layer}层隐藏态含NaN/Inf")
            records.setdefault(layer, []).append(values)
    if not records:
        raise RuntimeError("隐藏态记录为空")
    return {layer: np.concatenate(parts, axis=0) for layer, parts in records.items()}


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-values))


def select_for_layer(
    color_states: np.ndarray,
    transport: np.ndarray,
    gate: np.ndarray,
    correction_bias: np.ndarray,
    experts_per_layer: int,
    native_top_k: int,
) -> tuple[list[int], dict[str, object]]:
    rms = np.sqrt(np.mean(np.square(color_states), axis=1, keepdims=True))
    normalized = color_states / np.maximum(rms, 1e-6)
    donor_states = normalized @ transport.T
    donor_states *= np.float32(math.sqrt(K3_WIDTH / COLORLM_WIDTH))

    logits = donor_states @ gate.T
    probs = sigmoid(logits)
    selection_scores = probs + correction_bias[None, :]
    top = np.argpartition(selection_scores, -native_top_k, axis=1)[:, -native_top_k:]
    top_scores = np.take_along_axis(selection_scores, top, axis=1)
    order = np.argsort(top_scores, axis=1)[:, ::-1]
    top = np.take_along_axis(top, order, axis=1)
    selected_probs = np.take_along_axis(probs, top, axis=1)
    selected_probs /= np.maximum(selected_probs.sum(axis=1, keepdims=True), 1e-12)

    counts = np.bincount(top.ravel(), minlength=gate.shape[0])
    top1_counts = np.bincount(top[:, 0], minlength=gate.shape[0])
    weight_sum = np.bincount(
        top.ravel(), weights=selected_probs.ravel(), minlength=gate.shape[0]
    )
    # Frequency carries most of the decision. Top-1 wins and normalized K3
    # mixture mass break ties without introducing text or expert-id priors.
    rank_score = counts + 2.0 * top1_counts + native_top_k * weight_sum
    selected = np.argsort(rank_score)[::-1][:experts_per_layer].astype(int).tolist()
    covered = np.any(np.isin(top, np.asarray(selected, dtype=np.int64)), axis=1)

    details = []
    for expert in selected:
        details.append(
            {
                "expert": expert,
                "top16_count": int(counts[expert]),
                "top1_count": int(top1_counts[expert]),
                "normalized_weight_mass": float(weight_sum[expert]),
                "rank_score": float(rank_score[expert]),
            }
        )
    return selected, {
        "tokens": int(color_states.shape[0]),
        "state_rms_mean": float(rms.mean()),
        "native_top_k": native_top_k,
        "selected_experts": details,
        "token_coverage": float(np.mean(covered)),
    }


def main() -> int:
    args = parse_args()
    if not 1 <= args.experts_per_layer <= 896:
        raise ValueError("--experts-per-layer必须在1..896")
    if not 1 <= args.native_top_k <= 896:
        raise ValueError("--native-top-k必须在1..896")
    transport = np.asarray(np.load(args.transport, allow_pickle=False), dtype=np.float32)
    if transport.shape != (K3_WIDTH, COLORLM_WIDTH):
        raise RuntimeError(f"坐标运输形状错误: {transport.shape}")
    traces = read_hidden_dump(args.hidden_dump)

    stations = []
    for color_layer, k3_layer in DEFAULT_LAYER_MAP.items():
        states = traces.get(color_layer)
        if states is None:
            raise RuntimeError(f"隐藏态记录缺少ColorLM第{color_layer}层")
        prefix = f"language_model.model.layers.{k3_layer}.block_sparse_moe"
        gate_path = K3_EXTRACTED / f"{prefix}.gate.weight.bin"
        bias_path = K3_EXTRACTED / f"{prefix}.gate.e_score_correction_bias.bin"
        gate = bf16(gate_path, (896, K3_WIDTH))
        if not bias_path.is_file() or bias_path.stat().st_size != 896 * 4:
            raise RuntimeError(f"K3修正偏置大小不匹配: {bias_path}")
        bias = np.fromfile(bias_path, dtype="<f4")
        selected, evidence = select_for_layer(
            states,
            transport,
            gate,
            bias,
            args.experts_per_layer,
            args.native_top_k,
        )
        stations.append(
            {
                "colorlm_layer": color_layer,
                "k3_layer": k3_layer,
                "experts": selected,
                "evidence": evidence,
                "gate": {
                    "file": str(gate_path.relative_to(ROOT)),
                    "sha256": sha256_file(gate_path),
                },
                "correction_bias": {
                    "file": str(bias_path.relative_to(ROOT)),
                    "sha256": sha256_file(bias_path),
                },
            }
        )

    report = {
        "format": "colorlm-kimi-k3-hidden-route-selection-v1",
        "selection_input": "real ColorLM attn_post_norm hidden states",
        "prohibited_inputs": ["text keywords", "host task classifier", "expert id prior"],
        "hidden_dump": {
            "file": str(args.hidden_dump),
            "sha256": sha256_file(args.hidden_dump),
            "records_by_layer": {
                str(layer): int(values.shape[0]) for layer, values in traces.items()
            },
        },
        "transport": {
            "file": str(args.transport.relative_to(ROOT)),
            "sha256": sha256_file(args.transport),
            "shape": list(transport.shape),
        },
        "experts_per_layer": args.experts_per_layer,
        "stations": stations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
