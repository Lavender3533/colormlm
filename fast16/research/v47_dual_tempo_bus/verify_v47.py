"""验证 v47 静态合同、UTF-8、隔离拆分和可选训练自检产物。"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_dataset(rows: list[dict[str, Any]], schema: dict[str, Any]) -> None:
    required = set(schema["required"])
    allowed = set(schema["properties"])
    valid_splits = set(schema["properties"]["split"]["enum"])
    valid_modes = set(schema["properties"]["target_mode"]["enum"])
    sha_pattern = re.compile(r"^[0-9a-f]{64}$")
    for index, row in enumerate(rows, 1):
        missing = required - set(row)
        extra = set(row) - allowed
        if missing or extra:
            raise ValueError(f"dataset 第 {index} 行字段错误: missing={sorted(missing)}, extra={sorted(extra)}")
        if row.get("format") != "colorlm-v47-sequence-record-v1":
            raise ValueError(f"dataset 第 {index} 行 format 错误")
        for field in ("task_id", "group_id", "template_cluster_id", "capability", "target_text"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f"dataset 第 {index} 行 {field} 必须是非空字符串")
        if row.get("split") not in valid_splits or row.get("target_mode") not in valid_modes:
            raise ValueError(f"dataset 第 {index} 行 split/target_mode 不受支持")
        tokens = row.get("target_token_ids")
        if (
            not isinstance(tokens, list)
            or not 1 <= len(tokens) <= 512
            or not all(isinstance(token, int) and not isinstance(token, bool) and token >= 0 for token in tokens)
        ):
            raise ValueError(f"dataset 第 {index} 行 target_token_ids 无效")
        if not isinstance(row.get("capture_record"), int) or isinstance(row["capture_record"], bool) or row["capture_record"] < 0:
            raise ValueError(f"dataset 第 {index} 行 capture_record 无效")
        for field in ("prefix_sha256", "source_task_sha256"):
            if field in row and (not isinstance(row[field], str) or sha_pattern.fullmatch(row[field]) is None):
                raise ValueError(f"dataset 第 {index} 行 {field} 无效")
        if "metadata" in row and not isinstance(row["metadata"], dict):
            raise ValueError(f"dataset 第 {index} 行 metadata 必须是对象")
    task_ids = [str(row["task_id"]) for row in rows]
    records = [int(row["capture_record"]) for row in rows]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("dataset task_id 重复")
    if sorted(records) != list(range(len(rows))):
        raise ValueError("dataset capture_record 必须从 0 连续编号")
    for key in ("group_id", "template_cluster_id"):
        owner: dict[str, str] = {}
        for row in rows:
            value = str(row[key])
            split = str(row["split"])
            old = owner.setdefault(value, split)
            if old != split:
                raise ValueError(f"{key}={value!r} 跨 split 泄漏")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selfcheck-dir", type=Path, default=HERE / "selfcheck")
    args = parser.parse_args()

    checked_utf8 = 0
    bom_files = []
    for path in sorted(HERE.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".py", ".json", ".jsonl", ".md"}:
            continue
        raw = path.read_bytes()
        if raw.startswith(b"\xef\xbb\xbf"):
            bom_files.append(str(path))
        raw.decode("utf-8")
        checked_utf8 += 1
    if bom_files:
        raise ValueError("发现 UTF-8 BOM: " + ", ".join(bom_files))

    contract = read_json(HERE / "dual_tempo_contract.json")
    if contract.get("authoritative_base") != "ColorLM-v38-Qwen36-Shared-Sequence-Policy":
        raise ValueError("v47 合同错误修改了正式基座")
    if contract["lanes"]["slow"]["default_k"] != 0:
        raise ValueError("慢通道默认必须 K=0")
    read_json(HERE / "sequence_island.schema.json")
    frontend_schema = read_json(HERE / "frontend_design_ir.schema.json")
    if frontend_schema["properties"]["asset_policy"]["properties"]["allow_remote_random"].get("const") is not False:
        raise ValueError("前端 IR 必须禁止随机远程资产")

    result: dict[str, Any] = {
        "format": "colorlm-v47-static-selfcheck-v1",
        "utf8_files_checked": checked_utf8,
        "utf8_bom_count": 0,
        "static_contracts_passed": True,
        "synthetic_fixture_checked": False,
    }
    dataset_path = args.selfcheck_dir / "sequence_dataset.jsonl"
    report_path = args.selfcheck_dir / "fit_report.json"
    weights_path = args.selfcheck_dir / "sequence_island.npz"
    if dataset_path.exists():
        rows = read_jsonl(dataset_path)
        validate_dataset(rows, read_json(HERE / "sequence_island.schema.json"))
        result["synthetic_fixture_checked"] = True
        result["synthetic_records"] = len(rows)
    if report_path.exists():
        report = read_json(report_path)
        validation = report["metrics"]["validation"]
        numeric = [
            report["wall_seconds"],
            report["loss_first"],
            report["loss_last"],
            validation["exact_sequence_rate"],
            validation["aligned_token_accuracy"],
            validation["oov_token_rate"],
        ]
        if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in numeric):
            raise ValueError("fit report 含非有限指标")
        if report["loss_last"] >= report["loss_first"]:
            raise ValueError("合成训练 loss 未下降")
        if not weights_path.is_file() or report["prototype_gate"]["passed"] is not True:
            raise ValueError("合成训练器未通过自己的原型门")
        result["synthetic_training_checked"] = True
        result["synthetic_validation_exact_sequence_rate"] = validation["exact_sequence_rate"]
        result["synthetic_validation_token_accuracy"] = validation["aligned_token_accuracy"]

    output = args.selfcheck_dir / "verify_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
