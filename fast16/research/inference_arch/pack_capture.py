"""Pack CNOB binary tensor records and frozen teacher metadata into one NPZ."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Iterable

import numpy as np


HEADER = struct.Struct("<6I4qQ")
MAGIC = 0x424F4E43
KIND_NAMES = {1: "base_logits", 2: "donor_0_hidden", 3: "donor_0_logits"}
GGML_DTYPES = {0: np.dtype("<f4"), 1: np.dtype("<f2")}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_records(path: Path) -> dict[str, np.ndarray]:
    grouped: dict[int, list[tuple[int, np.ndarray]]] = {kind: [] for kind in KIND_NAMES}
    with path.open("rb") as source:
        while header_bytes := source.read(HEADER.size):
            if len(header_bytes) != HEADER.size:
                raise ValueError("capture header被截断")
            magic, version, kind, record, ggml_type, reserved, *tail = HEADER.unpack(
                header_bytes
            )
            ne = tuple(int(value) for value in tail[:4])
            payload_bytes = int(tail[4])
            if magic != MAGIC or version != 1 or reserved != 0:
                raise ValueError("capture header格式不受支持")
            if kind not in grouped or ggml_type not in GGML_DTYPES:
                raise ValueError(f"capture kind/type不受支持: {kind}/{ggml_type}")
            if ne[0] <= 0 or ne[1] <= 0 or ne[2:] != (1, 1):
                raise ValueError(f"capture tensor shape不受支持: {ne}")
            dtype = GGML_DTYPES[ggml_type]
            expected_bytes = int(np.prod(ne, dtype=np.int64)) * dtype.itemsize
            if payload_bytes != expected_bytes:
                raise ValueError("capture payload大小与shape/type不一致")
            payload = source.read(payload_bytes)
            if len(payload) != payload_bytes:
                raise ValueError("capture payload被截断")
            rows = np.frombuffer(payload, dtype=dtype).reshape(ne[1], ne[0]).copy()
            grouped[kind].append((record, rows))

    result: dict[str, np.ndarray] = {}
    for kind, name in KIND_NAMES.items():
        records = sorted(grouped[kind], key=lambda item: item[0])
        if not records or [item[0] for item in records] != list(range(len(records))):
            raise ValueError(f"{name}记录缺失、重复或序号不连续")
        widths = {item[1].shape[1] for item in records}
        if len(widths) != 1:
            raise ValueError(f"{name}宽度在记录间变化")
        result[name] = np.concatenate([item[1] for item in records], axis=0)

    row_counts = {value.shape[0] for value in result.values()}
    if len(row_counts) != 1:
        raise ValueError("三类capture tensor的输出位置数不一致")
    return result


def read_metadata(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    labels: list[int] = []
    task_ids: list[str] = []
    sample_ids: list[str] = []
    for index, row in enumerate(rows):
        label = row.get("target_token_id", row.get("label"))
        if not isinstance(label, int) or label < 0:
            raise ValueError(f"metadata第{index + 1}行缺少非负token label")
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"metadata第{index + 1}行缺少task_id")
        labels.append(label)
        task_ids.append(task_id)
        sample_ids.append(str(row.get("sample_id", f"row-{index:06d}")))
    return (
        np.asarray(labels, dtype=np.int64),
        np.asarray(task_ids, dtype=np.str_),
        np.asarray(sample_ids, dtype=np.str_),
    )


def read_base_ids(package: Path) -> tuple[np.ndarray, dict[str, object]]:
    manifest_path = package / "head.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = {item["name"]: item for item in manifest["tensors"]}
    record = records["mapping.base_ids"]
    if record["dtype"] != "I64" or len(record["ggml_shape"]) != 1:
        raise ValueError("mapping.base_ids契约不受支持")
    count = int(record["ggml_shape"][0])
    weights = package / manifest["weights"]["file"]
    with weights.open("rb") as source:
        source.seek(int(record["offset"]))
        payload = source.read(int(record["bytes"]))
    base_ids = np.frombuffer(payload, dtype="<i8").copy()
    if base_ids.size != count or np.unique(base_ids).size != count:
        raise ValueError("mapping.base_ids大小错误或存在重复")
    return base_ids, manifest


def pack_capture(
    records_path: Path,
    metadata_path: Path,
    head_package: Path,
    output_path: Path,
    manifest_path: Path,
) -> dict[str, object]:
    arrays = read_records(records_path)
    labels, task_ids, sample_ids = read_metadata(metadata_path)
    base_ids, head_manifest = read_base_ids(head_package)
    row_count = arrays["base_logits"].shape[0]
    if labels.size != row_count:
        raise ValueError(f"metadata行数{labels.size}与capture输出位置数{row_count}不一致")
    if arrays["donor_0_logits"].shape[1] != base_ids.size:
        raise ValueError("donor logits宽度与mapping.base_ids不一致")
    if labels.size and int(labels.max()) >= arrays["base_logits"].shape[1]:
        raise ValueError("metadata label超出base词表")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError("拒绝覆盖已有capture NPZ或manifest")
    temporary = output_path.with_suffix(output_path.suffix + ".part")
    with temporary.open("wb") as output:
        np.savez(
            output,
            base_logits=arrays["base_logits"],
            labels=labels,
            task_ids=task_ids,
            sample_ids=sample_ids,
            donor_0_logits=arrays["donor_0_logits"],
            donor_0_base_ids=base_ids,
            donor_0_hidden=arrays["donor_0_hidden"],
        )
    temporary.replace(output_path)

    report: dict[str, object] = {
        "format": "colorlm-neural-output-bus-capture-manifest-v1",
        "capture": str(output_path.resolve()),
        "capture_sha256": sha256_file(output_path),
        "records_sha256": sha256_file(records_path),
        "metadata_sha256": sha256_file(metadata_path),
        "head_manifest_sha256": hashlib.sha256(
            (head_package / "head.json").read_bytes()
        ).hexdigest(),
        "token_map_sha256": head_manifest["mapping"]["source_map_sha256"],
        "rows": row_count,
        "arrays": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in {
                **arrays,
                "labels": labels,
                "task_ids": task_ids,
                "sample_ids": sample_ids,
                "donor_0_base_ids": base_ids,
            }.items()
        },
        "invariants": {
            "alpha0_exact_base": True,
            "task_ids_are_metadata_only": True,
            "donor_base_ids_unique": True,
        },
    }
    manifest_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--head-package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = pack_capture(
        args.records, args.metadata, args.head_package, args.output, args.manifest
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
