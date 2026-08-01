"""构建 Qwen3.6 L36-L39 + final norm/output head 的终端连续段原型。"""

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


FIRST_LAYER = 36
LAST_LAYER = 39
TERMINAL_NAMES = {"output_norm.weight", "output.weight"}
MIN_FREE_BYTES = 25 * 1024**3


def parse_args() -> argparse.Namespace:
    models = ROOT / "fast16" / "models"
    parser = argparse.ArgumentParser(description="构建 Qwen3.6 终端连续段原型")
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
        default=models / "ColorLM-v32-Qwen36-Terminal-L36-L39.gguf",
    )
    return parser.parse_args()


def selected_name(name: str) -> bool:
    if name in TERMINAL_NAMES:
        return True
    return any(name.startswith(f"blk.{layer}.") for layer in range(FIRST_LAYER, LAST_LAYER + 1))


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
    if args.output.resolve() in {args.base.resolve(), args.donor.resolve()}:
        raise ValueError("输出不能覆盖基座或供体")
    if args.output.exists():
        raise FileExistsError(f"输出已存在: {args.output}")
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    if partial.exists():
        print(f"发现未完成临时文件，将原位截断重建: {partial}", flush=True)

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
    replacement_names = {name for name in base_map if selected_name(name)}
    if not replacement_names:
        raise RuntimeError("没有选中终端连续段张量")
    if replacement_names - set(donor_map):
        raise RuntimeError(f"供体缺少张量: {sorted(replacement_names - set(donor_map))[:4]}")

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
                "base_type": int(target.tensor_type),
                "donor_type": int(source.tensor_type),
                "base_bytes": int(target.data.nbytes),
                "donor_bytes": int(source.data.nbytes),
            }
        )

    estimated_output = args.base.stat().st_size - base_payload + donor_payload
    free_bytes = shutil.disk_usage(args.output.parent).free
    if free_bytes - estimated_output < MIN_FREE_BYTES:
        raise RuntimeError(
            "构建会突破25GiB磁盘安全线: "
            f"free={free_bytes}, estimated_output={estimated_output}"
        )
    print(
        f"选中 {len(replacements)} 个张量，预计输出 {estimated_output / 1024**3:.3f} GiB",
        flush=True,
    )

    base_arch = str(base.fields["general.architecture"].contents())
    writer = GGUFWriter(partial, arch=base_arch)
    copy_metadata(
        writer,
        base,
        {"general.name": "ColorLM-v32-Qwen36-Terminal-L36-L39"},
    )
    writer.add_string("colorlm.qwen36_terminal.format", "continuous-terminal-v1")
    writer.add_string("colorlm.qwen36_terminal.donor", args.donor.name)
    writer.add_uint32("colorlm.qwen36_terminal.first_layer", FIRST_LAYER)
    writer.add_uint32("colorlm.qwen36_terminal.last_layer", LAST_LAYER)
    writer.add_uint32("colorlm.qwen36_terminal.tensor_count", len(replacements))

    selected = []
    for tensor in base.tensors:
        source = replacements.get(tensor.name, tensor)
        selected.append((tensor.name, source))
        add_tensor_info(writer, tensor.name, source)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    try:
        for index, (_, source) in enumerate(selected, 1):
            writer.write_tensor_data(source.data)
            if index % 80 == 0 or index == len(selected):
                print(f"[{index}/{len(selected)}]", flush=True)
        writer.close()
        partial.replace(args.output)
    except BaseException:
        writer.close()
        raise

    report = {
        "format": "colormlm-qwen36-continuous-terminal-v1",
        "model": args.output.name,
        "base": args.base.name,
        "donor": args.donor.name,
        "first_layer": FIRST_LAYER,
        "last_layer": LAST_LAYER,
        "includes_final_norm": True,
        "includes_output_head": True,
        "replacement_tensor_count": len(replacements),
        "base_replaced_payload_bytes": base_payload,
        "donor_payload_bytes": donor_payload,
        "output_bytes": args.output.stat().st_size,
        "tensors": manifest,
    }
    report_path = args.output.with_suffix(args.output.suffix + ".json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"v32 原型: {args.output}")
    print(f"构建报告: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
