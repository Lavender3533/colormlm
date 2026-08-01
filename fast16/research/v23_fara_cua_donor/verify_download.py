"""校验v23 Fara教师模型的固定文件身份。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DONOR_DIR = ROOT / "fast16/models/donor/fara15_27b"
FILES = {
    "model": {
        "name": "Fara1.5-27B-Q5_K_M.gguf",
        "bytes": 20_513_802_048,
        "sha256": "91bb8785f51df1f175d8811f9b275228c566ca4b92249d8a851b57dfb42e86a8",
    },
    "mmproj": {
        "name": "mmproj-Fara1.5-27B-f16.gguf",
        "bytes": 927_607_456,
        "sha256": "6358165b66c42f375c69da8a990b4fa60ffd2a6b0175c90c48f7d15737ce059d",
    },
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(32 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="校验Fara1.5-27B v23供体文件")
    parser.add_argument("--full-hash", action="store_true")
    parser.add_argument(
        "--receipt",
        type=Path,
        default=ROOT / "fast16/research/v23_fara_cua_donor/download_receipt.json",
    )
    args = parser.parse_args()

    rows: dict[str, object] = {}
    valid = True
    for role, spec in FILES.items():
        path = DONOR_DIR / str(spec["name"])
        exists = path.is_file()
        size = path.stat().st_size if exists else None
        size_ok = size == spec["bytes"]
        actual_hash = digest(path) if args.full_hash and size_ok else None
        hash_ok = actual_hash == spec["sha256"] if args.full_hash else None
        row_ok = exists and size_ok and (hash_ok is True if args.full_hash else True)
        valid = valid and row_ok
        rows[role] = {
            "path": str(path),
            "exists": exists,
            "expected_bytes": spec["bytes"],
            "actual_bytes": size,
            "size_ok": size_ok,
            "expected_sha256": spec["sha256"],
            "actual_sha256": actual_hash,
            "hash_checked": args.full_hash,
            "hash_ok": hash_ok,
            "valid": row_ok,
        }

    report = {
        "format": "colorlm-v23-fara-download-receipt-v1",
        "revision": "dd7cba968d1a9c8feab0c2b85d93b117e6cc16fe",
        "full_hash": args.full_hash,
        "files": rows,
        "valid": valid,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
