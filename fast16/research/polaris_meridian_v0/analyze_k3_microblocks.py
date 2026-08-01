"""分析真实 Kimi K3 专家切片的神经元微块稀疏可行性。

该工具不下载模型、不训练参数，也不启动推理服务。它使用已有 K3 F16
运行包和真实 ColorLM 隐藏态，比较完整专家与 64 神经元微块的输出方向。

两种选择器：

* ``activation_oracle``：按真实 SiTU 激活能量选块，仅作可行性上界；
* ``analytic_proxy``：对每块的 ``G^T G + U^T U`` 做随机低秩分解，
  只通过隐藏态点积选块，可作为无训练在线路由器。
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CAPSULE = (
    ROOT
    / "fast16/research/neural_bus_capsules/kimi_k3_l28_e41_real/runtime_v2"
)
DEFAULT_HIDDEN = ROOT / "fast16/research/v14_live/l12_feature.hidden.npz"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "k3_l28_e41_microblock_report.json"

SITU_BETA = 4.0
SITU_LINEAR_BETA = 25.0


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层必须为对象：{path}")
    return value


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_raw_f16(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    expected = math.prod(shape) * np.dtype("<f2").itemsize
    actual = path.stat().st_size
    if actual != expected:
        raise ValueError(f"{path} 字节数异常：{actual}，预期 {expected}")
    return np.memmap(path, dtype="<f2", mode="r", shape=shape)


def situ(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    gate_branch = SITU_BETA * np.tanh(gate / SITU_BETA)
    gate_branch *= 1.0 / (1.0 + np.exp(-np.clip(gate, -80.0, 80.0)))
    up_branch = SITU_LINEAR_BETA * np.tanh(up / SITU_LINEAR_BETA)
    return gate_branch * up_branch


def topk_mask(scores: np.ndarray, k: int, block_size: int) -> np.ndarray:
    samples, blocks = scores.shape
    if k >= blocks:
        return np.ones((samples, blocks * block_size), dtype=np.float32)
    chosen = np.argpartition(scores, blocks - k, axis=1)[:, -k:]
    block_mask = np.zeros((samples, blocks), dtype=np.float32)
    block_mask[np.arange(samples)[:, None], chosen] = 1.0
    return np.repeat(block_mask, block_size, axis=1)


def randomized_block_proxy(
    gate: np.ndarray,
    up: np.ndarray,
    block_size: int,
    rank: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """近似每块 ``G^T G + U^T U`` 的主方向和特征值。"""
    if rank < 1:
        raise ValueError("proxy rank 必须大于 0")
    blocks = gate.shape[0] // block_size
    input_width = gate.shape[1]
    directions = np.empty((blocks, rank, input_width), dtype=np.float32)
    eigenvalues = np.empty((blocks, rank), dtype=np.float32)
    rng = np.random.default_rng(seed)
    sketch_rank = min(2 * block_size, max(rank + 4, rank * 2))

    for block in range(blocks):
        start = block * block_size
        end = start + block_size
        stacked = np.concatenate(
            (
                np.asarray(gate[start:end], dtype=np.float32),
                np.asarray(up[start:end], dtype=np.float32),
            ),
            axis=0,
        )
        omega = rng.standard_normal((input_width, sketch_rank), dtype=np.float32)
        sketch = stacked @ omega
        basis, _ = np.linalg.qr(sketch, mode="reduced")
        compressed = basis.T @ stacked
        _, singular, right = np.linalg.svd(compressed, full_matrices=False)
        keep = min(rank, right.shape[0])
        directions[block, :keep] = right[:keep]
        eigenvalues[block, :keep] = np.square(singular[:keep])
        if keep < rank:
            directions[block, keep:] = 0.0
            eigenvalues[block, keep:] = 0.0
    return directions, eigenvalues


def proxy_scores(
    latent: np.ndarray,
    directions: np.ndarray,
    eigenvalues: np.ndarray,
) -> np.ndarray:
    # latent: [样本, 3584]；directions: [块, 秩, 3584]
    projection = np.einsum("sd,brd->sbr", latent, directions, optimize=True)
    return np.sum(np.square(projection) * eigenvalues[None, :, :], axis=2)


def selected_output(
    activation: np.ndarray,
    mask: np.ndarray,
    down: np.ndarray,
    norm: np.ndarray,
    b_out: np.ndarray,
    eps: float,
) -> np.ndarray:
    latent_output = (activation * mask) @ np.asarray(down, dtype=np.float32).T
    rms = np.sqrt(np.mean(np.square(latent_output), axis=1, keepdims=True) + eps)
    normalized = (latent_output / rms) * norm[None, :]
    return normalized @ np.asarray(b_out, dtype=np.float32).T


def output_contribution_scores(
    activation: np.ndarray,
    down: np.ndarray,
    norm: np.ndarray,
    b_out: np.ndarray,
    block_size: int,
) -> np.ndarray:
    """计算每个块对 RMSNorm 分子输出的真实贡献能量，作为离线上界。"""
    weighted_output = np.asarray(b_out, dtype=np.float32) * norm[None, :]
    output_columns = weighted_output @ np.asarray(down, dtype=np.float32)
    blocks = activation.shape[1] // block_size
    scores = np.empty((activation.shape[0], blocks), dtype=np.float32)
    for block in range(blocks):
        start = block * block_size
        end = start + block_size
        contribution = activation[:, start:end] @ output_columns[:, start:end].T
        scores[:, block] = np.sum(np.square(contribution), axis=1)
    return scores


def fidelity(reference: np.ndarray, candidate: np.ndarray) -> dict:
    dot = np.sum(reference * candidate, axis=1)
    reference_norm = np.linalg.norm(reference, axis=1)
    candidate_norm = np.linalg.norm(candidate, axis=1)
    cosine = dot / np.maximum(reference_norm * candidate_norm, 1.0e-20)
    error = np.linalg.norm(candidate - reference, axis=1)
    nrmse = error / np.maximum(reference_norm, 1.0e-20)
    norm_ratio = candidate_norm / np.maximum(reference_norm, 1.0e-20)
    return {
        "cosine_mean": float(np.mean(cosine)),
        "cosine_median": float(np.median(cosine)),
        "cosine_p10": float(np.quantile(cosine, 0.10)),
        "nrmse_mean": float(np.mean(nrmse)),
        "nrmse_median": float(np.median(nrmse)),
        "norm_ratio_mean": float(np.mean(norm_ratio)),
    }


def set_overlap(left: np.ndarray, right: np.ndarray, k: int) -> float:
    blocks = left.shape[1]
    if k >= blocks:
        return 1.0
    left_top = np.argpartition(left, blocks - k, axis=1)[:, -k:]
    right_top = np.argpartition(right, blocks - k, axis=1)[:, -k:]
    values = []
    for left_row, right_row in zip(left_top, right_top, strict=True):
        values.append(len(set(left_row.tolist()) & set(right_row.tolist())) / k)
    return float(np.mean(values))


def active_budget(
    block_size: int,
    active_blocks: int,
    block_count: int,
    colorlm_width: int,
    latent_width: int,
    proxy_rank: int,
) -> dict:
    active_neurons = block_size * active_blocks
    # 输入桥分别折进 gate/up；norm+b_out 折进 down 的输出侧。
    params_per_neuron = 3 * colorlm_width + latent_width
    active_params = active_neurons * params_per_neuron
    router_params = block_count * proxy_rank * colorlm_width
    dynamic_f16_bytes = active_params * 2
    return {
        "active_blocks": active_blocks,
        "active_neurons": active_neurons,
        "fraction_of_expert": active_blocks / block_count,
        "folded_dynamic_parameters": active_params,
        "folded_dynamic_f16_mib_per_token": dynamic_f16_bytes / 1024**2,
        "analytic_router_parameters": router_params,
        "analytic_router_f16_mib_resident": router_params * 2 / 1024**2,
        "bandwidth_mib_s_at_20_tps": dynamic_f16_bytes * 20 / 1024**2,
        "bandwidth_mib_s_at_50_tps": dynamic_f16_bytes * 50 / 1024**2,
    }


def build_markdown(report: dict) -> str:
    lines = [
        "# Polaris Meridian v0：K3 微块扫描",
        "",
        f"- 胶囊：`{report['inputs']['capsule']}`",
        f"- 隐藏态：`{report['inputs']['hidden_states']}`",
        f"- 样本：{report['inputs']['samples']} 条真实隐藏态",
        f"- 切分：{report['configuration']['block_count']} × "
        f"{report['configuration']['block_size']} 神经元",
        f"- 解析路由秩：{report['configuration']['proxy_rank']}",
        "",
        "| 路由 | top 块 | 激活能量覆盖 | cosine 均值 | cosine P10 | NRMSE 均值 | 动态 F16/Token |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["results"]:
        lines.append(
            f"| {row['selector']} | {row['top_blocks']} | "
            f"{row['activation_energy_coverage']:.4f} | "
            f"{row['fidelity']['cosine_mean']:.4f} | "
            f"{row['fidelity']['cosine_p10']:.4f} | "
            f"{row['fidelity']['nrmse_mean']:.4f} | "
            f"{row['budget']['folded_dynamic_f16_mib_per_token']:.2f} MiB |"
        )
    lines.extend(
        [
            "",
            "## 判定",
            "",
            f"**{report['decision']['status']}**：{report['decision']['reason']}",
            "",
            "该结果只回答‘微块能否重构一颗真实 K3 专家的输出方向’，不等于整模能力提升。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capsule", type=Path, default=DEFAULT_CAPSULE)
    parser.add_argument("--hidden", type=Path, default=DEFAULT_HIDDEN)
    parser.add_argument("--hidden-key", default="layer_12")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--proxy-rank", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260801)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    manifest = read_json(args.capsule / "capsule.json")
    dimensions = manifest["dimensions"]
    colorlm_width = int(dimensions["colorlm"])
    latent_width = int(dimensions["latent"])
    intermediate_width = int(dimensions["intermediate"])
    if intermediate_width % args.block_size:
        raise ValueError("intermediate width 必须能被 block size 整除")
    block_count = intermediate_width // args.block_size

    hidden_archive = np.load(args.hidden, allow_pickle=False)
    hidden = np.asarray(hidden_archive[args.hidden_key], dtype=np.float32)
    if hidden.ndim != 2 or hidden.shape[1] != colorlm_width:
        raise ValueError(f"隐藏态形状异常：{hidden.shape}")

    b_in = load_raw_f16(args.capsule / "b_in.f16", (latent_width, colorlm_width))
    gate = load_raw_f16(args.capsule / "gate.f16", (intermediate_width, latent_width))
    up = load_raw_f16(args.capsule / "up.f16", (intermediate_width, latent_width))
    down = load_raw_f16(args.capsule / "down.f16", (latent_width, intermediate_width))
    norm = np.asarray(
        load_raw_f16(args.capsule / "norm.f16", (latent_width,)),
        dtype=np.float32,
    )
    b_out = load_raw_f16(args.capsule / "b_out.f16", (colorlm_width, latent_width))
    eps = float(manifest.get("rms_norm_eps", 1.0e-5))

    latent = hidden @ np.asarray(b_in, dtype=np.float32).T
    gate_value = latent @ np.asarray(gate, dtype=np.float32).T
    up_value = latent @ np.asarray(up, dtype=np.float32).T
    activation = situ(gate_value, up_value)
    full_mask = np.ones_like(activation, dtype=np.float32)
    reference = selected_output(activation, full_mask, down, norm, b_out, eps)

    activation_scores = np.sum(
        np.square(activation.reshape(hidden.shape[0], block_count, args.block_size)),
        axis=2,
    )
    linear_scores = np.sum(
        (
            np.square(gate_value) + np.square(up_value)
        ).reshape(hidden.shape[0], block_count, args.block_size),
        axis=2,
    )
    directions, eigenvalues = randomized_block_proxy(
        gate, up, args.block_size, args.proxy_rank, args.seed
    )
    analytic_scores = proxy_scores(latent, directions, eigenvalues)
    contribution_scores = output_contribution_scores(
        activation, down, norm, b_out, args.block_size
    )

    total_activation_energy = np.sum(activation_scores, axis=1)
    results = []
    top_values = [value for value in (1, 2, 4, 8, 16, block_count) if value <= block_count]
    for selector, scores in (
        ("activation_oracle", activation_scores),
        ("output_contribution_oracle", contribution_scores),
        ("analytic_proxy", analytic_scores),
    ):
        for top_blocks in top_values:
            mask = topk_mask(scores, top_blocks, args.block_size)
            candidate = selected_output(activation, mask, down, norm, b_out, eps)
            selected_energy = np.sum(
                activation_scores
                * mask[:, :: args.block_size],
                axis=1,
            )
            results.append(
                {
                    "selector": selector,
                    "top_blocks": top_blocks,
                    "activation_energy_coverage": float(
                        np.mean(
                            selected_energy / np.maximum(total_activation_energy, 1.0e-20)
                        )
                    ),
                    "fidelity": fidelity(reference, candidate),
                    "budget": active_budget(
                        args.block_size,
                        top_blocks,
                        block_count,
                        colorlm_width,
                        latent_width,
                        args.proxy_rank,
                    ),
                    "overlap_with_activation_oracle": set_overlap(
                        scores, activation_scores, top_blocks
                    ),
                    "overlap_with_linear_oracle": set_overlap(
                        scores, linear_scores, top_blocks
                    ),
                }
            )

    proxy_top4 = next(
        row
        for row in results
        if row["selector"] == "analytic_proxy" and row["top_blocks"] == 4
    )
    oracle_top4 = next(
        row
        for row in results
        if row["selector"] == "activation_oracle" and row["top_blocks"] == 4
    )
    if proxy_top4["fidelity"]["cosine_mean"] >= 0.90:
        status = "进入 C++ 动态微 MoE"
        reason = "top-4 解析路由已保留至少 0.90 的平均输出余弦方向。"
    elif oracle_top4["fidelity"]["cosine_mean"] >= 0.90:
        status = "继续改进解析路由"
        reason = "微块本身可行，但当前无训练解析代理未追上激活神谕。"
    else:
        status = "停止 top-4，检查 top-8/16"
        reason = "即使激活神谕的 top-4 也无法稳定重构完整专家方向。"

    report = {
        "format": "polaris-k3-microblock-analysis-v1",
        "inputs": {
            "capsule": str(args.capsule.resolve()),
            "hidden_states": str(args.hidden.resolve()),
            "hidden_key": args.hidden_key,
            "samples": int(hidden.shape[0]),
            "source_repo": manifest.get("repo"),
            "source_revision": manifest.get("revision"),
            "source_layer": manifest.get("layer"),
            "source_expert": manifest.get("expert"),
        },
        "configuration": {
            "block_size": args.block_size,
            "block_count": block_count,
            "proxy_rank": args.proxy_rank,
            "seed": args.seed,
            "activation": manifest.get("activation"),
            "rms_norm_eps": eps,
        },
        "folding_contract": {
            "gate_eff": "gate_block @ b_in",
            "up_eff": "up_block @ b_in",
            "down_partial": "down_block @ activation_block，仅用于 RMS 分母",
            "output_eff": "b_out @ diag(norm) @ down_block",
            "exact_selected_output": "output_eff @ activation / sqrt(mean(down_partial^2)+eps)",
            "training_required": False,
        },
        "results": results,
        "self_check": {
            "all_blocks_activation_oracle": fidelity(
                reference,
                selected_output(activation, full_mask, down, norm, b_out, eps),
            )
        },
        "decision": {"status": status, "reason": reason},
        "elapsed_seconds": time.perf_counter() - started,
        "scope_warning": "微块重构成功不等于模型能力提升，仍需短 NLL 与真实生成门。",
    }
    write_json(args.output, report)
    markdown_path = args.output.with_suffix(".md")
    markdown_path.write_text(build_markdown(report), encoding="utf-8")
    print(json.dumps(report["decision"], ensure_ascii=False))
    print(f"JSON: {args.output}")
    print(f"Markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
