"""把完整 Qwen3.6 的连续中层皮层运输到 v36；保留 v36 入口、末层与输出头。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RESEARCH = ROOT / "fast16/research"
sys.path.insert(0, os.fspath(ROOT / "llama.cpp/gguf-py"))
sys.path.insert(0, os.fspath(RESEARCH))

from gguf import GGUFReader, GGUFWriter  # noqa: E402

from build_heterogeneous_slices import copy_metadata  # noqa: E402
from v31_qwen36_expert_pair.audit_qwen36_pair import (  # noqa: E402
    TOKEN_IDENTITY_FIELDS,
    tokenizer_manifest,
)


MIN_FREE_AFTER_BUILD = 20 * 1024**3


def parse_args() -> argparse.Namespace:
    models = ROOT / "fast16/models"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        type=Path,
        default=models / "ColorLM-v36-Qwen36-Global-Shared-Backbone.gguf",
    )
    parser.add_argument(
        "--donor",
        type=Path,
        default=models / "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=models / "ColorLM-v46-Qwen36-Mid-Cortex-L16-L31.gguf",
    )
    parser.add_argument("--first-layer", type=int, default=16)
    parser.add_argument("--last-layer", type=int, default=31)
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
    if not 0 <= args.first_layer <= args.last_layer < 40:
        raise ValueError("层范围必须落在[0,39]")
    if args.output.resolve() in {args.base.resolve(), args.donor.resolve()}:
        raise ValueError("输出不能覆盖基座或供体")
    for path in (args.base, args.donor):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists():
        raise FileExistsError(args.output)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    if partial.exists():
        print(f"发现上次中断的临时文件，将原位截断重建: {partial}", flush=True)

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
        raise RuntimeError(f"token ID身份不一致: {mismatches}")

    base_map = {tensor.name: tensor for tensor in base.tensors}
    donor_map = {tensor.name: tensor for tensor in donor.tensors}
    prefixes = tuple(
        f"blk.{layer}." for layer in range(args.first_layer, args.last_layer + 1)
    )
    replacement_names = {name for name in base_map if name.startswith(prefixes)}
    if not replacement_names:
        raise RuntimeError("没有选中连续中层张量")
    missing = replacement_names - set(donor_map)
    if missing:
        raise RuntimeError(f"供体缺少张量: {sorted(missing)[:8]}")

    replacements = {}
    tensor_records = []
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
        tensor_records.append(
            {
                "name": name,
                "shape": [int(value) for value in source.shape],
                "base_type": int(target.tensor_type),
                "donor_type": int(source.tensor_type),
                "base_bytes": int(target.data.nbytes),
                "donor_bytes": int(source.data.nbytes),
            }
        )

    estimated_output = args.base.stat().st_size - base_payload + donor_payload
    free_bytes = shutil.disk_usage(args.output.parent).free
    if free_bytes - estimated_output < MIN_FREE_AFTER_BUILD:
        raise RuntimeError(
            "构建会突破20GiB磁盘安全线: "
            f"free={free_bytes}, estimated_output={estimated_output}"
        )
    print(
        f"选中{len(replacements)}个张量，预计输出{estimated_output / 1024**3:.3f}GiB",
        flush=True,
    )

    partial.parent.mkdir(parents=True, exist_ok=True)
    writer = GGUFWriter(partial, arch=str(base.fields["general.architecture"].contents()))
    name = f"ColorLM-v46-Qwen36-Mid-Cortex-L{args.first_layer}-L{args.last_layer}"
    copy_metadata(writer, base, {"general.name": name})
    writer.add_string("colorlm.qwen36_mid_cortex.format", "continuous-mid-cortex-v1")
    writer.add_string("colorlm.qwen36_mid_cortex.donor", args.donor.name)
    writer.add_uint32("colorlm.qwen36_mid_cortex.first_layer", args.first_layer)
    writer.add_uint32("colorlm.qwen36_mid_cortex.last_layer", args.last_layer)
    writer.add_uint32("colorlm.qwen36_mid_cortex.tensor_count", len(replacements))

    selected = []
    for tensor in base.tensors:
        source = replacements.get(tensor.name, tensor)
        selected.append(source)
        add_tensor_info(writer, tensor.name, source)
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
        "format": "colorlm-qwen36-continuous-mid-cortex-v1",
        "model": args.output.name,
        "base": args.base.name,
        "donor": args.donor.name,
        "first_layer": args.first_layer,
        "last_layer": args.last_layer,
        "preserved_base_prefix": [0, args.first_layer - 1],
        "preserved_base_suffix": [args.last_layer + 1, 39],
        "preserved_base_embedding_norm_output": True,
        "replacement_tensor_count": len(replacements),
        "base_replaced_payload_bytes": base_payload,
        "donor_payload_bytes": donor_payload,
        "output_bytes": args.output.stat().st_size,
        "tensors": tensor_records,
    }
    report_path = args.output.with_suffix(args.output.suffix + ".json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"v46中层皮层: {args.output}")
    print(f"构建报告: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
