#!/usr/bin/env python3
"""用现成 v38 在线选择 v47 Design Genome，并在一条冻结 validation 题上短门验收。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


MODEL = "ColorLM-v38-Qwen36-Shared-Sequence-Policy"


def _configure_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_task(path: Path, task_id: str) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = [row for row in rows if row.get("id") == task_id]
    if len(selected) != 1:
        raise ValueError(f"找不到唯一任务 {task_id}")
    return selected[0]


def _request_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(f"v38 请求失败: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError("v38 返回值不是 JSON 对象")
    return value


def run_gate(endpoint: str, task_id: str, output_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    root = Path(__file__).resolve().parents[3]
    design_dir = root / "fast16/research/parallel_design_ir_v47"
    frontend_dir = root / "fast16/research/parallel_frontend_v47"
    task_path = frontend_dir / "data/validation.jsonl"
    baseline_path = frontend_dir / "fixtures/ordinary_three_cards.html"
    task = _load_task(task_path, task_id)

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"输出目录非空，拒绝覆盖: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt = task["prompt"]
    lede = prompt.split("，", 1)[1] if "，" in prompt else prompt
    if len(lede) > 160:
        lede = lede[:160]
        if lede not in prompt:
            raise RuntimeError("自动 copy slot 不再是 prompt 连续片段")
    slots_payload = {
        "schema_version": "design-copy-slots-v1",
        "task_id": task_id,
        "prompt_sha256": _sha256(prompt.encode("utf-8")),
        "slots": [
            {"id": 0, "kind": "title", "text": task["title"]},
            {"id": 1, "kind": "lede", "text": lede},
        ],
    }

    sys.path.insert(0, str(design_dir))
    try:
        from compile_design_genome import compile_genome
        from decode_genome import decode_text
        from ir_core import validate_slots

        slot_errors = validate_slots(slots_payload, prompt)
        if slot_errors:
            raise ValueError("; ".join(slot_errors))
        slot_lines = [
            f"slot {item['id']} ({item['kind']}) = {item['text']}"
            for item in slots_payload["slots"]
        ]
        system = (
            (design_dir / "CATALOG_PROMPT.md").read_text(encoding="utf-8").strip()
            + "\n\n本请求 copy slots：\n"
            + "\n".join(slot_lines)
        )
        request_payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": 160,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "grammar": (design_dir / "design_genome.gbnf").read_text(encoding="utf-8"),
        }
        prompt_path = output_dir / "prompt.txt"
        slots_path = output_dir / "slots.json"
        request_path = output_dir / "request.json"
        prompt_path.write_text(prompt + "\n", encoding="utf-8")
        slots_path.write_text(
            json.dumps(slots_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        request_path.write_text(
            json.dumps(request_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        request_started = time.perf_counter()
        response = _request_json(
            endpoint.rstrip("/") + "/v1/chat/completions",
            request_payload,
            timeout=180.0,
        )
        request_seconds = time.perf_counter() - request_started
        choices = response.get("choices")
        message = choices[0].get("message") if isinstance(choices, list) and choices else None
        raw = message.get("content") if isinstance(message, dict) else None
        if not isinstance(raw, str) or not raw.strip():
            raise RuntimeError(f"v38 没有返回 Design Genome: {response}")
        raw_path = output_dir / "genome_raw.txt"
        response_path = output_dir / "response.json"
        raw_path.write_text(raw + "\n", encoding="utf-8")
        response_path.write_text(
            json.dumps(response, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        decoded = decode_text(raw)
        if decoded.genome is None or decoded.canonical is None:
            report = {
                "format": "polaris-v38-v47-live-validation-gate-v1",
                "status": "fail",
                "scope": "single_frozen_validation_task",
                "task_id": task_id,
                "model": MODEL,
                "endpoint": endpoint,
                "timing": {
                    "model_request_seconds": round(request_seconds, 4),
                    "total_seconds": round(time.perf_counter() - started, 4),
                },
                "failure_stage": "v47_legal_combination_projection",
                "model_output": {
                    "usage": response.get("usage", {}),
                    "raw_sha256": _sha256(raw.encode("utf-8")),
                    "decode": decoded.as_dict(),
                },
                "candidate": None,
                "gate": None,
                "retry_or_repair_applied": False,
                "claim_limit": "模型生成了完整 Genome，但组合非法；未编译 HTML、未重试、未修复。",
            }
            (output_dir / "live_report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            return report
        html, compile_report = compile_genome(
            decoded.genome,
            {item["id"]: item["text"] for item in slots_payload["slots"]},
        )
    finally:
        sys.path.pop(0)

    genome_path = output_dir / "genome.json"
    html_path = output_dir / "candidate.html"
    genome_path.write_text(decoded.canonical + "\n", encoding="utf-8")
    html_path.write_text(html, encoding="utf-8")

    sys.path.insert(0, str(frontend_dir))
    try:
        from validate_gates import evaluate_one

        baseline_gate = evaluate_one(task, baseline_path)
        candidate_gate = evaluate_one(task, html_path)
    finally:
        sys.path.pop(0)

    baseline_score = float(baseline_gate["audit"]["summary"]["final_score"])
    candidate_score = float(candidate_gate["audit"]["summary"]["final_score"])
    score_gain = round(candidate_score - baseline_score, 2)
    accepted = bool(candidate_gate["passed"] and score_gain >= 12.0)
    report = {
        "format": "polaris-v38-v47-live-validation-gate-v1",
        "status": "pass" if accepted else "fail",
        "scope": "single_frozen_validation_task",
        "task_id": task_id,
        "model": MODEL,
        "endpoint": endpoint,
        "timing": {
            "model_request_seconds": round(request_seconds, 4),
            "total_seconds": round(time.perf_counter() - started, 4),
        },
        "model_output": {
            "usage": response.get("usage", {}),
            "decode": decoded.as_dict(),
            "genome_sha256": _sha256(decoded.canonical.encode("utf-8")),
        },
        "compiler": compile_report,
        "candidate": {
            "path": str(html_path.resolve()),
            "sha256": _sha256(html.encode("utf-8")),
            "bytes": len(html.encode("utf-8")),
            "deterministic_repair_applied": False,
        },
        "gate": {
            "baseline_score": baseline_score,
            "candidate_score": candidate_score,
            "score_gain": score_gain,
            "candidate_passed": candidate_gate["passed"],
            "critical": candidate_gate["critical"],
            "score_checks": candidate_gate["score_checks"],
            "signals": candidate_gate["signals"],
        },
        "claim_limit": (
            "只是一条冻结 validation 题的在线 Genome 选择与静态门；"
            "编译器贡献不属于模型能力，未运行浏览器动作门。"
        ),
    }
    report_path = output_dir / "live_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    _configure_utf8()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8138")
    parser.add_argument("--task-id", default="pf47-validation-01")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("live_v38_v47_validation_01"),
    )
    args = parser.parse_args()
    try:
        report = run_gate(args.endpoint, args.task_id, args.output_dir)
    except Exception as error:  # noqa: BLE001 - 顶层输出精确失败
        print(f"live_gate_error: {error}", file=sys.stderr)
        return 1
    summary = {
        "status": report["status"],
        "task_id": report["task_id"],
        **report["timing"],
    }
    if report.get("gate"):
        summary.update(report["gate"])
    else:
        summary["failure_stage"] = report.get("failure_stage")
        summary["decode"] = report.get("model_output", {}).get("decode")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
