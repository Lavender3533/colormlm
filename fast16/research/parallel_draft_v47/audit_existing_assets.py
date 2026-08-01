"""审计v43/v44现有资产能否真实支持v47完整轨迹草稿，不训练、不加载模型。"""

from __future__ import annotations

import argparse
import json
import struct
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from draft_core import read_jsonl, sha256_file


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
CNOB_HEADER = struct.Struct("<6I4qQ")
CNOB_MAGIC = 0x424F4E43


def scan_cnob(path: Path) -> dict[str, Any]:
    kinds: Counter[int] = Counter()
    shapes: defaultdict[str, Counter[str]] = defaultdict(Counter)
    records: set[int] = set()
    with path.open("rb") as source:
        while True:
            header = source.read(CNOB_HEADER.size)
            if not header:
                break
            if len(header) != CNOB_HEADER.size:
                raise ValueError("CNOB头截断")
            magic, version, kind, record, dtype, reserved, *tail = CNOB_HEADER.unpack(header)
            shape = tuple(int(value) for value in tail[:4])
            payload_bytes = int(tail[4])
            if magic != CNOB_MAGIC or version != 1 or reserved != 0 or payload_bytes < 0:
                raise ValueError("CNOB头无效")
            kinds[int(kind)] += 1
            shapes[str(kind)][str(shape)] += 1
            records.add(int(record))
            source.seek(payload_bytes, 1)
    return {
        "bytes": path.stat().st_size,
        "tensor_records": sum(kinds.values()),
        "logical_records": len(records),
        "kinds": {str(key): value for key, value in sorted(kinds.items())},
        "shapes": {key: dict(value) for key, value in shapes.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher",
        type=Path,
        default=WORKSPACE / "fast16/research/v44_critical_action_bus/critical-teacher-dev-v1.jsonl",
    )
    parser.add_argument(
        "--capture",
        type=Path,
        default=WORKSPACE / "fast16/research/v44_critical_action_bus/critical-states-dev-v1.cnob",
    )
    parser.add_argument(
        "--capture-manifest",
        type=Path,
        default=WORKSPACE / "fast16/research/v44_critical_action_bus/critical-capture-manifest-dev-v1.json",
    )
    parser.add_argument(
        "--span-audit",
        type=Path,
        default=WORKSPACE / "fast16/research/v44_critical_action_bus/critical-span-audit.json",
    )
    parser.add_argument(
        "--shortlist-report",
        type=Path,
        default=WORKSPACE / "fast16/research/v47_dual_tempo_bus/shortlist_v44_k32.json",
    )
    parser.add_argument("--output", type=Path, default=HERE / "evidence_gap_report.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("拒绝覆盖已有证据缺口报告")
    for path in (args.teacher, args.capture, args.capture_manifest, args.span_audit, args.shortlist_report):
        if not path.is_file():
            raise FileNotFoundError(path)

    teacher = read_jsonl(args.teacher)
    capture_manifest = json.loads(args.capture_manifest.read_text(encoding="utf-8"))
    span_audit = json.loads(args.span_audit.read_text(encoding="utf-8"))
    shortlist = json.loads(args.shortlist_report.read_text(encoding="utf-8"))
    positions: defaultdict[str, set[int]] = defaultdict(set)
    split_tasks: defaultdict[str, set[str]] = defaultdict(set)
    for row in teacher:
        task_id = str(row["task_id"])
        positions[task_id].add(int(row["token_index"]))
        split_tasks[str(row["split"])].add(task_id)
    consecutive_windows = 0
    tasks_with_window: set[str] = set()
    for task_id, values in positions.items():
        for start in values:
            if all(start + offset in values for offset in range(4)):
                consecutive_windows += 1
                tasks_with_window.add(task_id)

    source_model = str(capture_manifest.get("model", ""))
    capture_is_v38 = "v38" in source_model.lower() or capture_manifest.get("model_alias") == "ColorLM-v38-Qwen36-Shared-Sequence-Policy"
    teacher_hash = sha256_file(args.teacher)
    capture_hash_consistent = (
        capture_manifest.get("capture_sha256") == shortlist.get("capture_sha256")
        and capture_manifest.get("capture_sha256") == "e41b9da7f9c7042418f1fcca866524d2b079b18740f43d757c4e6fc09e7a53b9"
    )
    report = {
        "format": "colorlm-v47-parallel-draft-evidence-gap-v1",
        "decision": "insufficient_real_assets_stop_before_training",
        "existing_assets": {
            "teacher": {
                "path": str(args.teacher.resolve()),
                "sha256": teacher_hash,
                "manifest_sha256_matches": teacher_hash == capture_manifest.get("teacher_sha256"),
                "samples": len(teacher),
                "tasks": len(positions),
                "splits": {split: len(tasks) for split, tasks in sorted(split_tasks.items())},
                "selection": "每个语义字段首个判别token的稀疏teacher-forced位置",
                "has_validator_token_ids": all("validator_token_ids" in row for row in teacher),
                "has_oracle_full_token_ids": all("oracle_token_ids" in row for row in teacher),
                "four_consecutive_sparse_windows": consecutive_windows,
                "tasks_with_any_four_consecutive_sparse_window": len(tasks_with_window),
            },
            "capture": {
                "path": str(args.capture.resolve()),
                "declared_sha256": capture_manifest.get("capture_sha256"),
                "hash_cross_manifest_consistent": capture_hash_consistent,
                "source_model": source_model,
                "source_is_v38": capture_is_v38,
                "scan": scan_cnob(args.capture),
            },
            "span_audit": {
                "tasks": span_audit["summary"]["tasks"],
                "critical_token_occurrences": span_audit["summary"]["critical_token_occurrences"],
                "rows_store_complete_target_token_ids": all("target_token_ids" in row for row in span_audit["rows"]),
            },
            "shortlist_408": {
                "overall_covered": shortlist["overall"]["covered"],
                "overall_samples": shortlist["overall"]["sample_count"],
                "splits": sorted(shortlist["by_split"]),
                "gate_passed": shortlist["gate"]["passed"],
                "status": shortlist["status"],
                "warning": shortlist["warning"],
            },
        },
        "proof_boundaries": {
            "actually_proven": [
                "v36在408个稀疏teacher-forced关键位置上，冻结shortlist能包含该位置oracle target token",
                "408条记录各有一份2048维terminal hidden和一份248320维原生logits",
                "现有shortlist报告自身因无test而gate=false"
            ],
            "not_proven": [
                "v38原生首token复用",
                "同一anchor的一次性4-token完整轨迹shortlist覆盖",
                "v38自由贪心validator轨迹",
                "自由滚动候选命中与首拒绝位置",
                "平均接受长度",
                "解析或端到端加速"
            ]
        },
        "contract_gates": {
            "v38_anchor_capture": capture_is_v38,
            "train_validation_test_present": set(split_tasks) >= {"train", "validation", "test"},
            "full_oracle_token_sequences": all("oracle_token_ids" in row for row in teacher),
            "v38_free_roll_validator_sequences": all("validator_token_ids" in row for row in teacher),
            "complete_trajectory_shortlist_gate": False,
            "acceptance_prediction_allowed": False
        },
        "minimum_collection": [
            "只启动一次v38采集会话；每个anchor保存一次terminal hidden和原生top-32 token/logit，不保存未来完整词表投影",
            "每个anchor保存最近96个已提交token，并由v38温度0自由滚动最多4 token作为validator_token_ids",
            "同时保存完整oracle_token_ids与是否提前EOS；oracle只用于必要覆盖，不得喂入自由滚动head",
            "train/validation/test按group_id与template_cluster_id双重隔离，至少64个validation+test anchor并覆盖短/中/长上下文桶",
            "收集后先跑完整轨迹shortlist覆盖；未过0.995立即停止，禁止训练或改C++"
        ],
        "claim_limit": "本审计没有重新采集、训练或模拟；不得把408/408写成v47接受率或加速。"
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "decision": report["decision"], "gates": report["contract_gates"]}, ensure_ascii=False, indent=2))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
