"""Compile a donor GGUF into a self-identifying ColorLM v4 model file."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from .donor import build_qwen35_donor_plan
from .gguf_header import GGUFHeaderReader, validate_plan_sources


SOURCE_ARCH = b"qwen35moe"
TARGET_ARCH = b"colorlmv4"
SOURCE_NAME = b"Qwen3.5-35B-A3B"
TARGET_NAME = b"ColorLM-v4-SMoE"
EXPECTED_DONOR_BYTES = 13_583_383_168
EXPECTED_DONOR_SHA256 = "ca6e9542e9f9e34f28d59e1e335b3629825daf30a4c0b5c1ec9d2f898c061fb6"


@dataclass(frozen=True)
class TransplantResult:
    source: str
    output: str
    file_bytes: int
    metadata_bytes: int
    architecture_replacements: int
    name_replacements: int
    tensor_count: int
    tensor_bytes: int
    runtime_requires_donor: bool

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2) + "\n"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rewrite_metadata_prefix(metadata: bytes) -> tuple[bytes, int, int]:
    """Rewrite only equal-length GGUF strings, preserving every offset."""

    if len(SOURCE_ARCH) != len(TARGET_ARCH) or len(SOURCE_NAME) != len(TARGET_NAME):
        raise AssertionError("元数据替换必须保持字节长度")
    architecture_count = metadata.count(SOURCE_ARCH)
    name_count = metadata.count(SOURCE_NAME)
    if architecture_count < 2:
        raise ValueError("没有找到完整的供体架构元数据")
    if name_count < 1:
        raise ValueError("没有找到供体模型名")
    rewritten = metadata.replace(SOURCE_ARCH, TARGET_ARCH).replace(SOURCE_NAME, TARGET_NAME)
    if len(rewritten) != len(metadata):
        raise AssertionError("等长元数据改写改变了文件偏移")
    return rewritten, architecture_count, name_count


def transplant_qwen35_to_colorlm(
    source_path: str | Path,
    output_path: str | Path,
    *,
    verify_sha256: bool = True,
) -> TransplantResult:
    source_path = Path(source_path)
    output_path = Path(output_path)
    if source_path.resolve() == output_path.resolve():
        raise ValueError("输出文件不能覆盖供体文件")
    if source_path.stat().st_size != EXPECTED_DONOR_BYTES:
        raise ValueError(
            f"供体大小错误: {source_path.stat().st_size} != {EXPECTED_DONOR_BYTES}"
        )
    if verify_sha256:
        digest = _sha256(source_path)
        if digest != EXPECTED_DONOR_SHA256:
            raise ValueError(f"供体 SHA-256 错误: {digest}")

    reader = GGUFHeaderReader(source_path, "r")
    summary = reader.summary()
    if summary["architecture"] != SOURCE_ARCH.decode("ascii"):
        raise ValueError(f"供体架构错误: {summary['architecture']}")
    validation = validate_plan_sources(build_qwen35_donor_plan(), reader)
    if not validation.ok:
        raise ValueError(f"供体缺少 {len(validation.missing_sources)} 个必需张量")
    metadata_bytes = int(reader.data_offset)
    del reader

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        shutil.copyfile(source_path, temporary)
        with temporary.open("r+b") as model:
            metadata = model.read(metadata_bytes)
            rewritten, architecture_count, name_count = rewrite_metadata_prefix(metadata)
            model.seek(0)
            model.write(rewritten)
            model.flush()
            os.fsync(model.fileno())
        compiled = GGUFHeaderReader(temporary, "r")
        compiled_summary = compiled.summary()
        if compiled_summary["architecture"] != TARGET_ARCH.decode("ascii"):
            raise ValueError("编译后的模型没有 ColorLM v4 架构标识")
        if compiled_summary["name"] != TARGET_NAME.decode("ascii"):
            raise ValueError("编译后的模型名错误")
        del compiled
        os.replace(temporary, output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    return TransplantResult(
        source=str(source_path.resolve()),
        output=str(output_path.resolve()),
        file_bytes=output_path.stat().st_size,
        metadata_bytes=metadata_bytes,
        architecture_replacements=architecture_count,
        name_replacements=name_count,
        tensor_count=int(summary["tensor_count"]),
        tensor_bytes=int(summary["tensor_bytes"]),
        runtime_requires_donor=False,
    )
