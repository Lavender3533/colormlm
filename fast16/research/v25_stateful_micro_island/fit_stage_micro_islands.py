"""逐供体层评估低秩微岛可压缩性，严格按独立请求留出。"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

from fit_stateful_micro_island import (
    fit_candidate,
    group_sequences,
    prepare_basis,
    public_row,
    split_sequences,
)


HEADER = struct.Struct("<IIiIQQQ")
MAGIC = 0x534C4349


def read_stage_pairs(
    path: Path,
) -> dict[int, list[tuple[np.ndarray, np.ndarray]]]:
    data = path.read_bytes()
    offset = 0
    pending: dict[int, list[np.ndarray]] = {}
    result: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    while offset < len(data):
        if len(data) - offset < HEADER.size:
            raise RuntimeError("分层dump头部不完整")
        magic, version, layer, kind, ne0, ne1, payload_bytes = HEADER.unpack_from(
            data, offset
        )
        offset += HEADER.size
        if magic != MAGIC or version != 1 or kind not in (0, 1):
            raise RuntimeError("分层dump头部无效")
        expected = ne0 * ne1 * 4
        if payload_bytes != expected or len(data) - offset < payload_bytes:
            raise RuntimeError("分层dump载荷长度不匹配")
        array = np.frombuffer(
            data, dtype="<f4", count=ne0 * ne1, offset=offset
        ).reshape((ne1, ne0))
        array = np.array(array, dtype=np.float32, copy=True)
        offset += payload_bytes
        if kind == 0:
            pending.setdefault(layer, []).append(array)
        else:
            queue = pending.get(layer)
            if not queue:
                raise RuntimeError(f"L{layer}残差没有对应输入")
            input_array = queue.pop(0)
            if input_array.shape != array.shape:
                raise RuntimeError(f"L{layer}输入/残差shape不匹配")
            result.setdefault(layer, []).append((input_array, array))
    if any(queue for queue in pending.values()):
        raise RuntimeError("存在没有残差的分层输入记录")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="逐层拟合v25微岛")
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--expected-sequences", type=int, default=8)
    parser.add_argument("--drop-leading-sequences", type=int, default=1)
    parser.add_argument("--ridge", type=float, default=1e-3)
    args = parser.parse_args()

    by_layer = read_stage_pairs(args.dump)
    if sorted(by_layer) != [44, 45, 46, 47]:
        raise RuntimeError(f"供体层集合错误: {sorted(by_layer)}")
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "format": "colorlm-v25-stage-micro-island-fit-v1",
        "dump": str(args.dump.resolve()),
        "architecture": "每层独立q=U(h-mean); z=rho*z+q; delta=[q,z]V+y_mean",
        "selection": "仅在validation选择rank/rho；test只打开一次",
        "layers": {},
    }
    passed_layers: list[int] = []
    for layer in sorted(by_layer):
        all_sequences = group_sequences(by_layer[layer])
        dropped = all_sequences[: args.drop_leading_sequences]
        sequences = all_sequences[args.drop_leading_sequences :]
        if len(sequences) != args.expected_sequences:
            raise RuntimeError(
                f"L{layer}请求数不匹配: actual={len(sequences)}, "
                f"expected={args.expected_sequences}"
            )
        train, validation, test = split_sequences(sequences)
        prepared_validation = prepare_basis(train, 128)
        candidates: list[dict[str, Any]] = []
        for rank in (32, 64, 128):
            for rho in (0.0, 0.5, 0.8):
                candidates.append(
                    fit_candidate(
                        train,
                        validation,
                        rank,
                        rho,
                        args.ridge,
                        prepared_validation,
                    )
                )
        selected = min(
            candidates,
            key=lambda row: (
                row["metrics"]["relative_rmse"],
                -row["metrics"]["cosine_median"],
                row["weights_bytes_f16"],
            ),
        )
        train_validation = train + validation
        prepared_test = prepare_basis(train_validation, selected["requested_rank"])
        test_candidate = fit_candidate(
            train_validation,
            test,
            selected["requested_rank"],
            selected["rho"],
            args.ridge,
            prepared_test,
        )
        per_sequence_passes = sum(
            row["relative_rmse"] < 1.0 and row["cosine_median"] > 0.0
            for row in test_candidate["per_sequence"]
        )
        metrics = test_candidate["metrics"]
        passed = bool(
            metrics["relative_rmse"] <= 0.85
            and metrics["cosine_median"] >= 0.40
            and per_sequence_passes == len(test)
        )
        if passed:
            passed_layers.append(layer)
        artifact = args.artifact_dir / f"stage_l{layer}.npz"
        np.savez_compressed(
            artifact,
            basis=test_candidate["basis"],
            output_projection=test_candidate["output_projection"],
            x_mean=test_candidate["x_mean"],
            y_mean=test_candidate["y_mean"],
            rho=np.asarray([test_candidate["rho"]], dtype=np.float32),
        )
        report["layers"][str(layer)] = {
            "dropped_leading_sequence_tokens": [len(row[0]) for row in dropped],
            "split_tokens": {
                "train": sum(len(row[0]) for row in train),
                "validation": sum(len(row[0]) for row in validation),
                "test": sum(len(row[0]) for row in test),
            },
            "validation_candidates": [public_row(row) for row in candidates],
            "selected": {
                "requested_rank": selected["requested_rank"],
                "rho": selected["rho"],
            },
            "test": public_row(test_candidate),
            "gate_passed": passed,
            "artifact": str(artifact.resolve()),
        }
        print(
            f"L{layer}: rank={test_candidate['rank']} rho={test_candidate['rho']:.2f} "
            f"cos={metrics['cosine_median']:.4f} "
            f"rel_rmse={metrics['relative_rmse']:.4f} pass={passed}",
            flush=True,
        )
    report["decision"] = {
        "passed_layers": passed_layers,
        "all_layers_passed": len(passed_layers) == 4,
        "next_action": (
            "仅为通过层实现默认关闭的替换候选，其余层保留原计算"
            if passed_layers
            else "线性分站仍不可压缩；停止低秩线性路线"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
