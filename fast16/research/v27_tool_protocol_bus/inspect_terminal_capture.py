"""Inspect a default-off ColorLM CNOB terminal capture without loading tensors."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any


HEADER = struct.Struct("<6I4qQ")
MAGIC = 0x424F4E43
KIND_NAMES = {
    1: "base_logits",
    2: "donor_hidden",
    3: "donor_logits",
    4: "base_hidden",
}


def inspect(path: Path) -> dict[str, Any]:
    size = path.stat().st_size
    rows: list[dict[str, Any]] = []
    offset = 0
    with path.open("rb") as source:
        while offset < size:
            encoded = source.read(HEADER.size)
            if len(encoded) != HEADER.size:
                raise ValueError(f"截断的CNOB头: offset={offset}, bytes={len(encoded)}")
            magic, version, kind, record, dtype, reserved, *tail = HEADER.unpack(encoded)
            ne = tail[:4]
            payload_bytes = tail[4]
            if magic != MAGIC or version != 1 or reserved != 0:
                raise ValueError(
                    f"非法CNOB头: offset={offset}, magic={magic:#x}, "
                    f"version={version}, reserved={reserved}"
                )
            if kind not in KIND_NAMES:
                raise ValueError(f"未知CNOB kind: {kind}")
            payload = source.read(payload_bytes)
            if len(payload) != payload_bytes:
                raise ValueError(
                    f"截断的CNOB payload: offset={offset}, "
                    f"expected={payload_bytes}, actual={len(payload)}"
                )
            rows.append(
                {
                    "kind": kind,
                    "kind_name": KIND_NAMES[kind],
                    "record": record,
                    "dtype": dtype,
                    "shape": ne,
                    "payload_bytes": payload_bytes,
                }
            )
            offset += HEADER.size + payload_bytes
    if offset != size:
        raise ValueError(f"CNOB字节边界不一致: parsed={offset}, file={size}")
    return {
        "format": "colorlm-v27-terminal-capture-inspection-v1",
        "path": str(path.resolve()),
        "bytes": size,
        "records": rows,
        "kinds": sorted({row["kind_name"] for row in rows}),
        "complete": bool(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="检查ColorLM末层hidden/logits CNOB记录")
    parser.add_argument("capture", type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect(args.capture), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
