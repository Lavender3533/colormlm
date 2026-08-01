"""生成 v47 训练器的可学习合成夹具；只验证管线，不作为模型能力证据。"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
HEADER = struct.Struct("<6I4qQ")
MAGIC = 0x424F4E43
VERSION = 1
BASE_HIDDEN = 4
F32 = 0


def sha256_json(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "selfcheck")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = args.output_dir / "sequence_dataset.jsonl"
    capture = args.output_dir / "initial_states.cnob"
    if not args.force and (dataset.exists() or capture.exists()):
        raise FileExistsError("合成夹具已存在；如需重建请使用 --force")

    rng = np.random.default_rng(47)
    rows = []
    hidden_rows = []
    record = 0
    split_sizes = {"train": 30, "validation": 12, "blind": 6}
    for split, count in split_sizes.items():
        for index in range(count):
            category = index % 3
            raw = rng.normal(0.0, 0.015, size=32).astype("<f4")
            raw[category] += 2.0
            raw[3 + category] += 0.8
            target = [100 + category, 200, 300 + category]
            task_id = f"synthetic-{split}-{index:02d}"
            prefix = [9000, category, index]
            rows.append(
                {
                    "format": "colorlm-v47-sequence-record-v1",
                    "task_id": task_id,
                    "group_id": f"group-{split}-{index:02d}",
                    "template_cluster_id": f"template-{split}-{index // 3:02d}",
                    "split": split,
                    "capability": "synthetic_tool_sequence",
                    "target_mode": "exact_token_sequence",
                    "target_text": f"synthetic category {category}",
                    "target_token_ids": target,
                    "capture_record": record,
                    "prefix_sha256": sha256_json(prefix),
                    "source_task_sha256": sha256_json({"id": task_id, "target": target}),
                    "metadata": {"synthetic": True, "category": category},
                }
            )
            hidden_rows.append((record, raw))
            record += 1

    dataset.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    with capture.open("wb") as output:
        for record, hidden in hidden_rows:
            payload = hidden.tobytes(order="C")
            output.write(
                HEADER.pack(
                    MAGIC,
                    VERSION,
                    BASE_HIDDEN,
                    record,
                    F32,
                    0,
                    len(hidden),
                    1,
                    1,
                    1,
                    len(payload),
                )
            )
            output.write(payload)
    result = {
        "format": "colorlm-v47-synthetic-fixture-v1",
        "warning": "pipeline_selfcheck_only_not_model_evidence",
        "dataset": str(dataset.resolve()),
        "capture": str(capture.resolve()),
        "record_count": len(rows),
        "split_counts": split_sizes,
        "hidden_width": 32,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

