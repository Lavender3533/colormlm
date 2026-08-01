"""构建 v36：Qwen3.6 全深度 router/shared expert + ColorLM routed expert bank。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RESEARCH = ROOT / "fast16" / "research"
sys.path.insert(0, os.fspath(ROOT / "llama.cpp" / "gguf-py"))
sys.path.insert(0, os.fspath(RESEARCH))

from gguf import GGUFReader, GGUFWriter  # noqa: E402

from build_heterogeneous_slices import copy_metadata  # noqa: E402
from v31_qwen36_expert_pair.audit_qwen36_pair import (  # noqa: E402
    TOKEN_IDENTITY_FIELDS,
    tokenizer_manifest,
)


LAYER_COUNT = 40
MIN_FREE_BYTES = 25 * 1024**3
SHARED_BACKBONE_SUFFIXES = (
    "post_attention_norm.weight",
    "ffn_gate_inp.weight",
    "ffn_gate_shexp.weight",
    "ffn_up_shexp.weight",
    "ffn_down_shexp.weight",
    "ffn_gate_inp_shexp.weight",
)


def parse_args() -> argparse.Namespace:
    models = ROOT / "fast16" / "models"
    parser = argparse.ArgumentParser(description="构建 v36 Qwen3.6 共享 MoE 主干消融")
    parser.add_argument(
        "--base",
        type=Path,
        default=models / "ColorLM-v6-Q3Router-Fused-A1.gguf",
    )
    parser.add_argument(
        "--donor",
        type=Path,
        default=models / "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=models / "ColorLM-v36-Qwen36-Global-Shared-Backbone.gguf",
    )
    return parser.parse_args()


def add_tensor_info(writer: GGUFWriter, name: str, tensor) -> None:
    writer.add_tensor_info(
        name,
        tensor.data.shape,
        tensor.data.dtype,
        tensor.data.nbytes,
        raw_dtype=tensor.tensor_type,
    )


def main() -> int:
    args = parse_args()
    for path in (args.base, args.donor):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.resolve() in {args.base.resolve(), args.donor.resolve()}:
        raise ValueError("输出不能覆盖基座或供体")
    if args.output.exists():
        raise FileExistsError(f"输出已存在: {args.output}")

    partial = args.output.with_suffix(args.output.suffix + ".partial")
    base = GGUFReader(os.fspath(args.base), "r")
    donor = GGUFReader(os.fspath(args.donor), "r")
    base_tokenizer = tokenizer_manifest(base)
    donor_tokenizer = tokenizer_manifest(donor)
    mismatches = [
        key
        for key in TOKEN_IDENTITY_FIELDS
        if base_tokenizer.get(key) != donor_tokenizer.get(key)
    ]
    if mismatches:
        raise RuntimeError(f"token ID 身份不一致: {mismatches}")

    base_map = {tensor.name: tensor for tensor in base.tensors}
    donor_map = {tensor.name: tensor for tensor in donor.tensors}
    replacement_names = {
        f"blk.{layer}.{suffix}"
        for layer in range(LAYER_COUNT)
        for suffix in SHARED_BACKBONE_SUFFIXES
    }
    missing = sorted(
        (replacement_names - set(base_map)) | (replacement_names - set(donor_map))
    )
    if missing:
        raise RuntimeError(f"共享主干张量缺失: {missing[:4]}")

    replacements = {}
    manifest = []
    base_payload = 0
    donor_payload = 0
    for name in sorted(replacement_names):
        target = base_map[name]
        source = donor_map[name]
        if tuple(target.shape) != tuple(source.shape):
            raise RuntimeError(f"张量形状不兼容: {name}")
        replacements[name] = source
        base_payload += int(target.data.nbytes)
        donor_payload += int(source.data.nbytes)
        manifest.append(
            {
                "name": name,
                "shape": [int(value) for value in source.shape],
                "base_type": target.tensor_type.name,
                "donor_type": source.tensor_type.name,
                "base_bytes": int(target.data.nbytes),
                "donor_bytes": int(source.data.nbytes),
            }
        )

    estimated_output = args.base.stat().st_size - base_payload + donor_payload
    free_bytes = shutil.disk_usage(args.output.parent).free
    remaining = free_bytes - estimated_output
    print(
        f"选中 {len(replacements)} 个张量，预计输出 {estimated_output / 1024**3:.3f} GiB，"
        f"构建后剩余 {remaining / 1024**3:.3f} GiB",
        flush=True,
    )
    if remaining < MIN_FREE_BYTES:
        raise RuntimeError("构建会突破25GiB磁盘安全线")

    writer = GGUFWriter(
        partial,
        arch=str(base.fields["general.architecture"].contents()),
    )
    copy_metadata(
        writer,
        base,
        {"general.name": "ColorLM-v36-Qwen36-Global-Shared-Backbone"},
    )
    writer.add_string("colorlm.qwen36_shared_backbone.format", "ablation-v1")
    writer.add_string("colorlm.qwen36_shared_backbone.donor", args.donor.name)
    writer.add_uint32("colorlm.qwen36_shared_backbone.layer_count", LAYER_COUNT)
    writer.add_uint32(
        "colorlm.qwen36_shared_backbone.tensor_count", len(replacements)
    )

    selected = []
    for tensor in base.tensors:
        source = replacements.get(tensor.name, tensor)
        selected.append(source)
        add_tensor_info(writer, tensor.name, source)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    try:
        for index, source in enumerate(selected, 1):
            writer.write_tensor_data(source.data)
            if index % 80 == 0 or index == len(selected):
                print(f"[{index}/{len(selected)}]", flush=True)
        writer.close()
        partial.replace(args.output)
    except BaseException:
        writer.close()
        raise

    report = {
        "format": "colormlm-qwen36-shared-backbone-ablation-v1",
        "model": args.output.name,
        "base": args.base.name,
        "donor": args.donor.name,
        "layer_count": LAYER_COUNT,
        "replacement_tensor_count": len(replacements),
        "hypothesis": (
            "若 v36 保留 v33 的方向性增益，则增益不依赖 Qwen3.6 routed expert bank；"
            "若回到 v6，则 routed expert bank 是必要条件。"
        ),
        "preserved_base_modules": [
            "token_embedding",
            "gated_deltanet_or_attention",
            "routed_expert_bank",
            "final_norm",
            "output_head",
        ],
        "base_replaced_payload_bytes": base_payload,
        "donor_payload_bytes": donor_payload,
        "output_bytes": args.output.stat().st_size,
        "tensors": manifest,
    }
    report_path = args.output.with_suffix(args.output.suffix + ".json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"v36 消融原型: {args.output}")
    print(f"构建报告: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
