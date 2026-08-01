"""把 Qwen3.6 L39 的闭合 MoE 配对编译进 ColorLM v6 单 GGUF。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RESEARCH = ROOT / "fast16" / "research"
GGUF_PY = ROOT / "llama.cpp" / "gguf-py"
sys.path.insert(0, os.fspath(GGUF_PY))
sys.path.insert(0, os.fspath(RESEARCH))

from gguf import GGUFReader, GGUFWriter  # noqa: E402

from build_heterogeneous_slices import copy_metadata  # noqa: E402
from v31_qwen36_expert_pair.audit_qwen36_pair import (  # noqa: E402
    LAYER,
    PAIR_SUFFIXES,
    TOKEN_IDENTITY_FIELDS,
    tensor_digest,
    tokenizer_manifest,
)


def parse_args() -> argparse.Namespace:
    models = ROOT / "fast16" / "models"
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="构建 v31 Qwen3.6 L39 闭合专家配对")
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
        "--audit",
        type=Path,
        default=here / "pair_audit.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=models / "ColorLM-v31-Qwen36-L39-Expert-Pair.gguf",
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
    if args.output.resolve() in {args.base.resolve(), args.donor.resolve()}:
        raise ValueError("输出不能覆盖基座或供体")
    if args.output.exists():
        raise FileExistsError(f"输出已存在: {args.output}")
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    if partial.exists():
        print(f"发现未完成的构建临时文件，将原位截断重建: {partial}", flush=True)
    if not args.audit.is_file():
        raise FileNotFoundError(f"必须先完成配对审计: {args.audit}")

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if not audit.get("conclusion", {}).get("prototype_allowed", False):
        raise RuntimeError("配对审计未允许构建原型")

    base = GGUFReader(os.fspath(args.base), "r")
    donor = GGUFReader(os.fspath(args.donor), "r")
    base_tokenizer = tokenizer_manifest(base)
    donor_tokenizer = tokenizer_manifest(donor)
    identity_mismatches = [
        key
        for key in TOKEN_IDENTITY_FIELDS
        if base_tokenizer.get(key) != donor_tokenizer.get(key)
    ]
    if identity_mismatches:
        raise RuntimeError(f"基座与供体 token ID 身份不一致: {identity_mismatches}")

    base_map = {tensor.name: tensor for tensor in base.tensors}
    donor_map = {tensor.name: tensor for tensor in donor.tensors}
    replacement_names = {f"blk.{LAYER}.{suffix}" for suffix in PAIR_SUFFIXES}
    missing = sorted(replacement_names - set(base_map) | replacement_names - set(donor_map))
    if missing:
        raise RuntimeError(f"缺少替换张量: {missing}")

    replacements = {}
    manifest = []
    for name in sorted(replacement_names):
        source = donor_map[name]
        target = base_map[name]
        if tuple(source.shape) != tuple(target.shape):
            raise RuntimeError(f"张量形状不兼容: {name}")
        replacements[name] = source
        manifest.append(
            {
                "name": name,
                "shape": [int(value) for value in source.shape],
                "base_type": int(target.tensor_type),
                "donor_type": int(source.tensor_type),
                "base_bytes": int(target.data.nbytes),
                "donor_bytes": int(source.data.nbytes),
                "donor_sha256": tensor_digest(source),
            }
        )

    base_arch = str(base.fields["general.architecture"].contents())
    writer = GGUFWriter(partial, arch=base_arch)
    copy_metadata(
        writer,
        base,
        {"general.name": "ColorLM-v31-Qwen36-L39-Expert-Pair"},
    )
    writer.add_string("colorlm.qwen36_pair.format", "closed-moe-pair-v1")
    writer.add_string("colorlm.qwen36_pair.donor", args.donor.name)
    writer.add_uint32("colorlm.qwen36_pair.layer", LAYER)
    writer.add_uint32("colorlm.qwen36_pair.tensor_count", len(replacements))

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
        for index, (_, source) in enumerate(selected, start=1):
            writer.write_tensor_data(source.data)
            if index % 80 == 0 or index == len(selected):
                print(f"[{index}/{len(selected)}]", flush=True)
        writer.close()
        partial.replace(args.output)
    except BaseException:
        writer.close()
        raise

    manifest_bytes = json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    report = {
        "format": "colormlm-qwen36-closed-moe-pair-v1",
        "model": args.output.name,
        "base": {
            "name": args.base.name,
            "bytes": args.base.stat().st_size,
        },
        "donor": {
            "name": args.donor.name,
            "bytes": args.donor.stat().st_size,
        },
        "layer": LAYER,
        "replacement_tensor_count": len(replacements),
        "replacement_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "output_bytes": args.output.stat().st_size,
        "speed_contract": "不增加层数、每 token routed top-k 或第二前向",
        "tensors": manifest,
    }
    report_path = args.output.with_suffix(args.output.suffix + ".json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"v31 原型: {args.output}")
    print(f"构建报告: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
