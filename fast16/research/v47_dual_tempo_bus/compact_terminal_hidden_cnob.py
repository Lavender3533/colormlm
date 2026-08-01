"""从 CNOB 中提取 terminal hidden，生成可复现的小型云训练资产。"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path


HEADER = struct.Struct("<6I4qQ")
MAGIC = 0x424F4E43
VERSION = 1
F32 = 0
BASE_HIDDEN = 4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expect-records", type=int, required=True)
    parser.add_argument("--expect-width", type=int, default=2048)
    args = parser.parse_args()

    if args.output.exists() or args.manifest.exists():
        raise FileExistsError("拒绝覆盖已有 compact CNOB 或 manifest")
    if args.expect_records <= 0 or args.expect_width <= 0:
        raise ValueError("期望记录数和宽度必须为正数")

    records: dict[int, tuple[bytes, tuple[int, int, int, int]]] = {}
    total_records = 0
    with args.input.open("rb") as source:
        while True:
            encoded = source.read(HEADER.size)
            if not encoded:
                break
            if len(encoded) != HEADER.size:
                raise ValueError("输入 CNOB 头被截断")
            magic, version, kind, record, dtype, reserved, *tail = HEADER.unpack(encoded)
            ne = tuple(int(value) for value in tail[:4])
            payload_bytes = int(tail[4])
            if magic != MAGIC or version != VERSION or dtype != F32 or reserved != 0:
                raise ValueError("输入 CNOB 头/版本/dtype 不符合契约")
            payload = source.read(payload_bytes)
            if len(payload) != payload_bytes:
                raise ValueError("输入 CNOB payload 被截断")
            total_records += 1
            if kind != BASE_HIDDEN:
                continue
            if ne != (args.expect_width, 1, 1, 1) or payload_bytes != args.expect_width * 4:
                raise ValueError(f"record={record} hidden shape 错误: {ne}")
            if record in records:
                raise ValueError(f"record={record} 重复")
            records[int(record)] = (payload, ne)

    expected = list(range(args.expect_records))
    if sorted(records) != expected:
        raise ValueError("terminal hidden record 必须从 0 开始连续且数量精确")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as target:
        for record in expected:
            payload, ne = records[record]
            target.write(
                HEADER.pack(
                    MAGIC,
                    VERSION,
                    BASE_HIDDEN,
                    record,
                    F32,
                    0,
                    *ne,
                    len(payload),
                )
            )
            target.write(payload)

    manifest = {
        "format": "polaris-v47-terminal-hidden-compact-manifest-v1",
        "status": "research_training_asset",
        "source": str(args.input.resolve()),
        "source_sha256": sha256_file(args.input),
        "source_total_cnob_records": total_records,
        "output": args.output.name,
        "output_sha256": sha256_file(args.output),
        "output_bytes": args.output.stat().st_size,
        "hidden_records": len(records),
        "hidden_width": args.expect_width,
        "record_ids_contiguous": True,
        "claim_limit": "仅用于复现 Parallel Genome Head；不包含主模型权重，也不构成能力晋级。",
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
