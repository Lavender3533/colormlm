"""北极星专家页目录：只解析权重头部，不物化张量。

支持两类输入：

* Safetensors：每个专家拥有独立命名张量（Kimi K3、DeepSeek V4 等）；
* GGUF：``ffn_{gate,up,down}_exps.weight`` 打包专家银行。

目录记录本地文件中的精确字节区间。运行时据此直接读取 SSD 页，不需要
``from_pretrained``、``state_dict`` 或整张量反量化。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


FORMAT = "polaris-expert-page-catalog-v1"
DEFAULT_EXPERT_RE = re.compile(
    r"(?:^|\.)layers\.(?P<layer>\d+)\.(?:.*\.)?experts\."
    r"(?P<expert>\d+)\.(?P<component>.+)$"
)
GGUF_EXPERT_RE = re.compile(
    r"^blk\.(?P<layer>\d+)\.ffn_(?P<component>gate|up|down)_exps\.weight$"
)


@dataclass(frozen=True)
class Span:
    file: str
    offset: int
    length: int
    tensor: str
    component: str
    dtype: str
    shape: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "offset": self.offset,
            "length": self.length,
            "tensor": self.tensor,
            "component": self.component,
            "dtype": self.dtype,
            "shape": list(self.shape),
        }


def page_key(donor: str, layer: int, expert: int) -> str:
    return f"{donor}:L{layer:05d}:E{expert:05d}"


def _read_safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError(f"Safetensors 头不足 8 字节：{path}")
        header_len = int.from_bytes(raw, "little", signed=False)
        if header_len <= 1 or header_len > path.stat().st_size - 8:
            raise ValueError(f"Safetensors header_len 非法：{path} ({header_len})")
        header = json.loads(handle.read(header_len).decode("utf-8"))
    return 8 + header_len, header


def _coalesce_spans(spans: Sequence[Span]) -> tuple[list[Span], int]:
    """合并同文件、物理连续的张量，减少一次专家 miss 的系统调用数。"""

    ordered = sorted(spans, key=lambda item: (item.file, item.offset))
    merged: list[Span] = []
    for span in ordered:
        if merged and merged[-1].file == span.file and merged[-1].offset + merged[-1].length == span.offset:
            previous = merged.pop()
            merged.append(
                Span(
                    file=span.file,
                    offset=previous.offset,
                    length=previous.length + span.length,
                    tensor=f"{previous.tensor}|{span.tensor}",
                    component=f"{previous.component}|{span.component}",
                    dtype=f"{previous.dtype}|{span.dtype}",
                    shape=(),
                )
            )
        else:
            merged.append(span)
    return merged, len(ordered) - len(merged)


def _component_signatures(grouped: dict[tuple[int, int], list[Span]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for spans in grouped.values():
        signature = "|".join(sorted(span.component for span in spans))
        counts[signature] += 1
    return dict(sorted(counts.items()))


def _catalog(
    donor: str,
    source_format: str,
    grouped: dict[tuple[int, int], list[Span]],
    *,
    require_uniform_components: bool = True,
) -> dict[str, Any]:
    signatures = _component_signatures(grouped)
    if require_uniform_components and len(signatures) != 1:
        preview = ", ".join(f"{count}页[{signature}]" for signature, count in signatures.items())
        raise ValueError(
            "专家组件不完整或签名不一致；请同时传入包含其余组件的分片。"
            f"检测结果：{preview}"
        )
    entries: list[dict[str, Any]] = []
    coalesced_span_count = 0
    source_files: dict[str, int] = {}
    for (layer, expert), raw_spans in sorted(grouped.items()):
        spans, saved = _coalesce_spans(raw_spans)
        coalesced_span_count += saved
        for span in spans:
            if span.file not in source_files:
                source_files[span.file] = Path(span.file).stat().st_size
        entries.append(
            {
                "key": page_key(donor, layer, expert),
                "donor": donor,
                "layer": layer,
                "expert": expert,
                "nbytes": sum(span.length for span in spans),
                "spans": [span.to_dict() for span in spans],
            }
        )
    if not entries:
        raise ValueError("没有发现可分页的路由专家张量")
    return {
        "format": FORMAT,
        "donor": donor,
        "source_format": source_format,
        "page_granularity": "one routed expert across all weight components in one layer",
        "entries": entries,
        "summary": {
            "pages": len(entries),
            "payload_bytes": sum(entry["nbytes"] for entry in entries),
            "source_files": len(source_files),
            "physical_spans": sum(len(entry["spans"]) for entry in entries),
            "coalesced_spans_saved": coalesced_span_count,
            "component_signatures": signatures,
            "uniform_components": len(signatures) == 1,
        },
        "sources": [
            {"file": path, "size": size} for path, size in sorted(source_files.items())
        ],
    }


def build_safetensors_catalog(
    paths: Iterable[Path], donor: str, *, require_uniform_components: bool = True
) -> dict[str, Any]:
    grouped: dict[tuple[int, int], list[Span]] = defaultdict(list)
    for path in paths:
        path = path.resolve()
        data_start, header = _read_safetensors_header(path)
        file_size = path.stat().st_size
        for tensor, metadata in header.items():
            if tensor == "__metadata__":
                continue
            match = DEFAULT_EXPERT_RE.search(tensor)
            if match is None:
                continue
            start, end = (int(value) for value in metadata["data_offsets"])
            absolute_start = data_start + start
            absolute_end = data_start + end
            if not (0 <= absolute_start <= absolute_end <= file_size):
                raise ValueError(f"张量区间越界：{path}::{tensor}")
            grouped[(int(match["layer"]), int(match["expert"]))].append(
                Span(
                    file=str(path),
                    offset=absolute_start,
                    length=absolute_end - absolute_start,
                    tensor=tensor,
                    component=match["component"],
                    dtype=str(metadata["dtype"]),
                    shape=tuple(int(value) for value in metadata["shape"]),
                )
            )
    return _catalog(
        donor,
        "safetensors",
        grouped,
        require_uniform_components=require_uniform_components,
    )


def _load_gguf_reader() -> Any:
    root = Path(__file__).resolve().parents[4]
    gguf_py = root / "llama.cpp" / "gguf-py"
    sys.path.insert(0, str(gguf_py))
    from gguf import GGUFReader  # type: ignore[import-not-found]

    return GGUFReader


def build_gguf_catalog(paths: Iterable[Path], donor: str, n_experts: int | None) -> dict[str, Any]:
    GGUFReader = _load_gguf_reader()
    grouped: dict[tuple[int, int], list[Span]] = defaultdict(list)
    detected_n_experts = n_experts
    for path in paths:
        path = path.resolve()
        reader = GGUFReader(str(path), "r")
        candidates: list[tuple[Any, re.Match[str]]] = []
        for tensor in reader.tensors:
            match = GGUF_EXPERT_RE.match(tensor.name)
            if match is not None:
                candidates.append((tensor, match))
                if detected_n_experts is None:
                    detected_n_experts = int(tensor.shape[-1])
        if not detected_n_experts or detected_n_experts <= 0:
            raise ValueError("无法从 GGUF shape 推断专家数；请传 --experts")
        for tensor, match in candidates:
            if tensor.n_bytes % detected_n_experts != 0:
                raise ValueError(f"GGUF 专家银行不能等分：{tensor.name}")
            per_expert = tensor.n_bytes // detected_n_experts
            for expert in range(detected_n_experts):
                grouped[(int(match["layer"]), expert)].append(
                    Span(
                        file=str(path),
                        offset=int(tensor.data_offset) + expert * per_expert,
                        length=per_expert,
                        tensor=tensor.name,
                        component=match["component"],
                        dtype=tensor.tensor_type.name,
                        shape=tuple(int(value) for value in tensor.shape),
                    )
                )
        # 释放大文件的 numpy memmap 引用；本目录只保留数值 offset。
        candidates.clear()
        del reader
    return _catalog(donor, "gguf", grouped)


def validate_catalog(catalog: dict[str, Any], *, check_files: bool = True) -> None:
    if catalog.get("format") != FORMAT:
        raise ValueError("目录 format 不匹配")
    seen: set[str] = set()
    file_sizes: dict[str, int] = {}
    for entry in catalog.get("entries", []):
        key = str(entry["key"])
        if key in seen:
            raise ValueError(f"重复页 key：{key}")
        seen.add(key)
        total = 0
        for span in entry["spans"]:
            offset = int(span["offset"])
            length = int(span["length"])
            if offset < 0 or length <= 0:
                raise ValueError(f"非法页区间：{key}")
            if check_files:
                filename = str(span["file"])
                if filename not in file_sizes:
                    file_sizes[filename] = Path(filename).stat().st_size
                size = file_sizes[filename]
                if offset + length > size:
                    raise ValueError(f"页区间越界：{key}")
            total += length
        if total != int(entry["nbytes"]):
            raise ValueError(f"页大小校验失败：{key}")


def write_catalog(catalog: dict[str, Any], output: Path) -> None:
    validate_catalog(catalog)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _expand_inputs(inputs: Sequence[Path], suffix: str) -> list[Path]:
    expanded: list[Path] = []
    for item in inputs:
        if item.is_dir():
            expanded.extend(sorted(item.glob(f"*{suffix}")))
        else:
            expanded.append(item)
    if not expanded:
        raise ValueError("输入中没有发现权重分片")
    return expanded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="建立北极星专家页字节目录")
    parser.add_argument("--format", choices=("safetensors", "gguf"), required=True)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--donor", required=True)
    parser.add_argument("--experts", type=int, help="GGUF 每层专家数；默认从 shape 推断")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="仅做头部取证时允许专家组件签名不一致；运行时目录禁止使用",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    suffix = ".safetensors" if args.format == "safetensors" else ".gguf"
    paths = _expand_inputs(args.input, suffix)
    if args.format == "safetensors":
        catalog = build_safetensors_catalog(
            paths, args.donor, require_uniform_components=not args.allow_incomplete
        )
    else:
        catalog = build_gguf_catalog(paths, args.donor, args.experts)
    write_catalog(catalog, args.output)
    print(json.dumps(catalog["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
