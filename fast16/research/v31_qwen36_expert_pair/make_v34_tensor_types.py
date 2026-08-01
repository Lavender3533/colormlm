"""为 v33 生成逐张量精确量化表，仅压缩 Qwen3.6 MoE 权重。"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, os.fspath(ROOT / "llama.cpp" / "gguf-py"))

from gguf import GGUFReader  # noqa: E402


ROUTED_SUFFIXES = (
    "ffn_gate_exps.weight",
    "ffn_up_exps.weight",
    "ffn_down_exps.weight",
)
SHARED_SUFFIXES = (
    "ffn_gate_shexp.weight",
    "ffn_up_shexp.weight",
    "ffn_down_shexp.weight",
)


def parse_args() -> argparse.Namespace:
    models = ROOT / "fast16" / "models"
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="生成 v34 llama-quantize 张量类型表")
    parser.add_argument(
        "--model",
        type=Path,
        default=models / "ColorLM-v33-Qwen36-Global-MoE-Pair.gguf",
    )
    parser.add_argument("--output", type=Path, default=here / "v34_tensor_types.txt")
    parser.add_argument(
        "--max-compressed-layer",
        type=int,
        default=39,
        help="仅压缩0..N层的MoE；39表示全部40层。",
    )
    return parser.parse_args()


def target_type(name: str, current: str, max_compressed_layer: int) -> str:
    match = re.match(r"^blk\.(\d+)\.", name)
    if match is None or int(match.group(1)) > max_compressed_layer:
        return current
    if name.endswith(ROUTED_SUFFIXES):
        return "IQ3_S"
    if name.endswith(SHARED_SUFFIXES):
        return "F16"
    return current


def main() -> int:
    args = parse_args()
    if not 0 <= args.max_compressed_layer < 40:
        raise ValueError("--max-compressed-layer必须在0到39之间")
    reader = GGUFReader(os.fspath(args.model), "r")
    lines = []
    changed = []
    for tensor in reader.tensors:
        current = tensor.tensor_type.name
        wanted = target_type(tensor.name, current, args.max_compressed_layer)
        pattern = "^" + re.escape(tensor.name) + "$"
        lines.append(f"{pattern}={wanted}")
        if wanted != current:
            changed.append((tensor.name, current, wanted))

    if len(lines) != len(reader.tensors):
        raise RuntimeError("逐张量类型表不完整")
    expected_changed = (args.max_compressed_layer + 1) * (
        len(ROUTED_SUFFIXES) + len(SHARED_SUFFIXES)
    )
    if len(changed) != expected_changed:
        raise RuntimeError(f"预期改变{expected_changed}张量，实际{len(changed)}")
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"张量类型表: {args.output}")
    print(f"总张量: {len(lines)}；重数量化: {len(changed)}")
    for name, current, wanted in changed[:6]:
        print(f"{name}: {current} -> {wanted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
