"""Build two transported Coder-Next capsules without creating a full GGUF."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESEARCH = Path(__file__).resolve().parent
EXTRACTED = (
    RESEARCH
    / "biopsy_cache"
    / "Qwen_Qwen3-Coder-Next"
    / "master"
    / "extracted"
)
METADATA = EXTRACTED.parent / "metadata"
sys.path.insert(0, os.fspath(ROOT / "llama.cpp" / "gguf-py"))

from gguf import GGUFReader  # noqa: E402
from gguf.constants import GGMLQuantizationType  # noqa: E402
from gguf.quants import dequantize, quantize  # noqa: E402


WIDTH = 2048
FF_WIDTH = 512
DONOR_LAYER = 47
EXPERTS = {
    471: {"target_expert": 201, "name": "coder_next_l47_e471_v2_q4_0"},
    0: {"target_expert": 255, "name": "coder_next_l47_e0_v2_q4_0"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="直接从本地BF16活检切片构建Neural Bus v2双胶囊"
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=ROOT / "fast16" / "models" / "ColorLM-v6-Q3Router-Fused-A1.gguf",
    )
    parser.add_argument(
        "--transport",
        type=Path,
        default=RESEARCH / "coder_next_to_colorlm_orthogonal_f32.npy",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=RESEARCH / "neural_bus_capsules",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=RESEARCH / "neural_bus_v2_capsule_build_report.json",
    )
    parser.add_argument("--base-layer", type=int, default=39)
    parser.add_argument("--train-count", type=int, default=3072)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def bf16_matrix(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    expected = int(np.prod(shape, dtype=np.int64)) * 2
    if not path.is_file() or path.stat().st_size != expected:
        raise RuntimeError(f"BF16切片大小不匹配: {path}")
    bits = np.fromfile(path, dtype="<u2").astype(np.uint32) << 16
    return np.ascontiguousarray(bits.view(np.float32).reshape(shape))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def importance_energy(gate: np.ndarray, up: np.ndarray, down: np.ndarray) -> float:
    gate_rms = np.sqrt(np.mean(np.square(gate), axis=1, dtype=np.float64))
    up_rms = np.sqrt(np.mean(np.square(up), axis=1, dtype=np.float64))
    down_rms = np.sqrt(np.mean(np.square(down), axis=0, dtype=np.float64))
    return float(np.linalg.norm(gate_rms * up_rms * down_rms))


def normalized_rows(values: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norm, 1e-12)


def best_binary_bias(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    order = np.argsort(scores)
    sorted_scores = scores[order]
    sorted_labels = labels[order].astype(np.int64)
    positives_total = int(np.sum(sorted_labels))
    negatives_before = np.concatenate(([0], np.cumsum(1 - sorted_labels)))
    positives_before = np.concatenate(([0], np.cumsum(sorted_labels)))
    correct = negatives_before + (positives_total - positives_before)
    split = int(np.argmax(correct))
    if split == 0:
        threshold = float(sorted_scores[0] - 1e-6)
    elif split == len(sorted_scores):
        threshold = float(sorted_scores[-1] + 1e-6)
    else:
        threshold = float((sorted_scores[split - 1] + sorted_scores[split]) * 0.5)
    return -threshold, float(correct[split] / len(sorted_scores))


def best_f1_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, dict[str, float]]:
    candidates = np.unique(scores)
    best_threshold = float(candidates[-1] + 1e-6)
    best = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    positives = int(np.sum(labels))
    for threshold in candidates:
        predicted = scores >= threshold
        true_positive = int(np.sum(predicted & labels))
        selected = int(np.sum(predicted))
        precision = true_positive / max(selected, 1)
        recall = true_positive / max(positives, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        if f1 > best["f1"]:
            best_threshold = float(threshold)
            best = {"precision": precision, "recall": recall, "f1": f1}
    return best_threshold, best


def binary_metrics(predicted: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    true_positive = int(np.sum(predicted & labels))
    selected = int(np.sum(predicted))
    positives = int(np.sum(labels))
    precision = true_positive / max(selected, 1)
    recall = true_positive / max(positives, 1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "predicted_active_rate": float(np.mean(predicted)),
        "true_active_rate": float(np.mean(labels)),
    }


def matched_token_ids(base: GGUFReader, donor_rows: int) -> tuple[np.ndarray, np.ndarray]:
    base_ids: dict[str, int] = {}
    for token_id, token in enumerate(base.fields["tokenizer.ggml.tokens"].contents()):
        base_ids.setdefault(token, token_id)
    tokenizer = json.loads((METADATA / "tokenizer.json").read_text(encoding="utf-8"))
    donor_tokens = {
        int(token_id): token
        for token, token_id in tokenizer["model"]["vocab"].items()
    }
    donor_tokens.update(
        {
            int(item["id"]): item["content"]
            for item in tokenizer.get("added_tokens", [])
        }
    )
    pairs = [
        (donor_id, base_ids[token])
        for donor_id, token in sorted(donor_tokens.items())
        if donor_id < donor_rows and token in base_ids
    ]
    return (
        np.asarray([pair[0] for pair in pairs], dtype=np.int64),
        np.asarray([pair[1] for pair in pairs], dtype=np.int64),
    )


def main() -> int:
    args = parse_args()
    if not args.base.is_file() or not args.transport.is_file():
        raise FileNotFoundError("缺少ColorLM基座或坐标运输矩阵")

    transport = np.asarray(np.load(args.transport, allow_pickle=False), dtype=np.float32)
    if transport.shape != (WIDTH, WIDTH):
        raise RuntimeError(f"运输矩阵形状错误: {transport.shape}")

    print("读取ColorLM基座索引与参考专家", flush=True)
    base = GGUFReader(os.fspath(args.base), "r")
    tensors = {tensor.name: tensor for tensor in base.tensors}
    prefix = f"blk.{args.base_layer}."
    gate_tensor = tensors[prefix + "ffn_gate_exps.weight"]
    up_tensor = tensors[prefix + "ffn_up_exps.weight"]
    down_tensor = tensors[prefix + "ffn_down_exps.weight"]
    base_router = np.asarray(tensors[prefix + "ffn_gate_inp.weight"].data, dtype=np.float32)

    donor_router_path = EXTRACTED / f"model.layers.{DONOR_LAYER}.mlp.gate.weight.bin"
    donor_router = bf16_matrix(donor_router_path, (512, WIDTH))
    base_router_rms = float(np.median(np.sqrt(np.mean(np.square(base_router), axis=1))))
    donor_router_rms = float(np.median(np.sqrt(np.mean(np.square(donor_router), axis=1))))
    common_router_scale = base_router_rms / max(donor_router_rms, 1e-12)

    manifests: dict[str, dict[str, object]] = {}
    route_rows: dict[int, np.ndarray] = {}
    for expert, spec in EXPERTS.items():
        output = args.output_root / str(spec["name"])
        known_files = [
            output / "gate.q4_0",
            output / "up.q4_0",
            output / "down.q4_0",
            output / "router.f32",
            output / "capsule.json",
        ]
        if any(path.exists() for path in known_files) and not args.force:
            raise FileExistsError(f"胶囊已存在，使用--force重建: {output}")
        output.mkdir(parents=True, exist_ok=True)
        if args.force:
            for path in known_files:
                path.unlink(missing_ok=True)

        stem = f"model.layers.{DONOR_LAYER}.mlp.experts.{expert}"
        raw_paths = {
            "gate": EXTRACTED / f"{stem}.gate_proj.weight.bin",
            "up": EXTRACTED / f"{stem}.up_proj.weight.bin",
            "down": EXTRACTED / f"{stem}.down_proj.weight.bin",
        }
        donor_gate = bf16_matrix(raw_paths["gate"], (FF_WIDTH, WIDTH))
        donor_up = bf16_matrix(raw_paths["up"], (FF_WIDTH, WIDTH))
        donor_down = bf16_matrix(raw_paths["down"], (WIDTH, FF_WIDTH))

        transported_gate = np.ascontiguousarray(donor_gate @ transport)
        transported_up = np.ascontiguousarray(donor_up @ transport)
        transported_down = np.ascontiguousarray(transport.T @ donor_down)

        target = int(spec["target_expert"])
        base_gate = np.asarray(
            dequantize(gate_tensor.data[target : target + 1], gate_tensor.tensor_type),
            dtype=np.float32,
        )[0]
        base_up = np.asarray(
            dequantize(up_tensor.data[target : target + 1], up_tensor.tensor_type),
            dtype=np.float32,
        )[0]
        base_down = np.asarray(
            dequantize(down_tensor.data[target : target + 1], down_tensor.tensor_type),
            dtype=np.float32,
        )[0]
        base_energy = importance_energy(base_gate, base_up, base_down)
        donor_energy = importance_energy(
            transported_gate, transported_up, transported_down
        )
        beta = float(np.clip(base_energy / max(donor_energy, 1e-12), 0.25, 4.0))
        transported_down *= np.float32(beta)

        arrays = {
            "gate": quantize(transported_gate, GGMLQuantizationType.Q4_0),
            "up": quantize(transported_up, GGMLQuantizationType.Q4_0),
            "down": quantize(transported_down, GGMLQuantizationType.Q4_0),
        }
        tensor_records: dict[str, dict[str, object]] = {}
        for role, values in arrays.items():
            payload = np.ascontiguousarray(values).tobytes()
            path = output / f"{role}.q4_0"
            path.write_bytes(payload)
            tensor_records[role] = {
                "file": path.name,
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
            }

        route = np.ascontiguousarray(
            (donor_router[expert] @ transport) * np.float32(common_router_scale),
            dtype="<f4",
        )
        route_rows[expert] = route
        route_payload = route.tobytes()
        (output / "router.f32").write_bytes(route_payload)
        manifest = {
            "format": "colorlm-neural-bus-capsule-v2",
            "donor": f"Qwen/Qwen3-Coder-Next layer {DONOR_LAYER} expert {expert}",
            "source_expert": expert,
            "reference_colorlm_expert": target,
            "transport": "shared-token-orthogonal-procrustes, baked into weights",
            "dtype": "Q4_0",
            "input_width": WIDTH,
            "intermediate_width": FF_WIDTH,
            "output_width": WIDTH,
            "expert_beta": beta,
            "base_energy": base_energy,
            "donor_energy_before_scale": donor_energy,
            "router": {
                "file": "router.f32",
                "dtype": "F32",
                "shape": [WIDTH, 1],
                "common_scale": common_router_scale,
                "bytes": len(route_payload),
                "sha256": sha256_bytes(route_payload),
            },
            "tensors": tensor_records,
            "source_sha256": {
                path.name: sha256_file(path) for path in raw_paths.values()
            },
        }
        (output / "capsule.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifests[str(expert)] = manifest
        print(f"完成专家{expert}胶囊: {output}", flush=True)

    donor_embedding_path = EXTRACTED / "model.embed_tokens.weight.axis0-0-4096.bin"
    donor_ids, base_ids = matched_token_ids(base, 4096)
    if not 0 < args.train_count < len(donor_ids):
        raise ValueError(f"train-count越界: {args.train_count} vs {len(donor_ids)}")
    donor_embedding = normalized_rows(
        bf16_matrix(donor_embedding_path, (4096, WIDTH))[donor_ids]
    )
    base_embedding_tensor = tensors["token_embd.weight"]
    base_embedding = normalized_rows(
        np.asarray(
            dequantize(
                base_embedding_tensor.data[base_ids], base_embedding_tensor.tensor_type
            ),
            dtype=np.float32,
        )
    )
    donor_diff = donor_embedding @ (donor_router[471] - donor_router[0])
    target_diff = base_embedding @ (route_rows[471] - route_rows[0])
    labels = donor_diff >= 0.0
    rng = np.random.default_rng(args.seed)
    permutation = rng.permutation(len(labels))
    train = permutation[: args.train_count]
    test = permutation[args.train_count :]
    fitted_bias, train_accuracy = best_binary_bias(target_diff[train], labels[train])
    zero_bias_accuracy = float(np.mean((target_diff[test] >= 0.0) == labels[test]))
    fitted_bias_accuracy = float(
        np.mean((target_diff[test] + np.float32(fitted_bias) >= 0.0) == labels[test])
    )
    recommended_bias = fitted_bias if fitted_bias_accuracy > zero_bias_accuracy else 0.0

    # A third no-op route prevents two isolated experts from being forced onto
    # tokens where neither belonged to Coder-Next's original top-10 set.
    donor_logits = donor_embedding @ donor_router.T
    donor_top10_floor = np.partition(donor_logits, -10, axis=1)[:, -10]
    donor_candidate_max = np.maximum(donor_logits[:, 471], donor_logits[:, 0])
    active_labels = donor_candidate_max >= donor_top10_floor
    target_candidate_max = np.maximum(
        base_embedding @ route_rows[471], base_embedding @ route_rows[0]
    )
    noop_threshold, train_noop = best_f1_threshold(
        target_candidate_max[train], active_labels[train]
    )
    test_noop = binary_metrics(
        target_candidate_max[test] >= noop_threshold, active_labels[test]
    )

    report = {
        "format": "colorlm-neural-bus-v2-capsule-build",
        "base": args.base.name,
        "transport": {
            "file": args.transport.name,
            "sha256": sha256_file(args.transport),
        },
        "capsules": manifests,
        "router": {
            "type": "transported Coder-Next hidden-state top-1",
            "parameters": 2 * WIDTH + 1,
            "primary_expert": 471,
            "secondary_expert": 0,
            "common_scale": common_router_scale,
            "calibration": "shared-token embedding decision transfer",
            "matched_tokens": len(labels),
            "train_tokens": len(train),
            "test_tokens": len(test),
            "fitted_bias": fitted_bias,
            "recommended_bias": recommended_bias,
            "train_accuracy_fitted_bias": train_accuracy,
            "test_accuracy_zero_bias": zero_bias_accuracy,
            "test_accuracy_fitted_bias": fitted_bias_accuracy,
            "bias_decision": (
                "use fitted bias"
                if recommended_bias != 0.0
                else "reject fitted bias because held-out accuracy regressed"
            ),
            "no_op": {
                "donor_top_k": 10,
                "threshold": noop_threshold,
                "train": train_noop,
                "test": test_noop,
            },
        },
    }
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["router"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
