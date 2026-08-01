#!/usr/bin/env python3
"""构造 llama.cpp `/v1/chat/completions` 的 Design Genome 请求体。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ir_core import IRError, read_utf8_no_bom, validate_slots, write_utf8_no_bom


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("gbnf", "schema"), default="gbnf")
    parser.add_argument("--model", default="ColorLM-v38-Qwen36-Shared-Sequence-Policy")
    parser.add_argument("--prompt-file", required=True, type=Path)
    parser.add_argument("--slots", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prompt = read_utf8_no_bom(args.prompt_file).strip()
    slot_payload = json.loads(read_utf8_no_bom(args.slots))
    slot_errors = validate_slots(slot_payload, prompt)
    if slot_errors:
        raise IRError("\n".join(slot_errors))
    slot_lines = [f"slot {item['id']} ({item['kind']}) = {item['text']}" for item in slot_payload["slots"]]
    system = read_utf8_no_bom(ROOT / "CATALOG_PROMPT.md").strip() + "\n\n本请求 copy slots：\n" + "\n".join(slot_lines)
    request = {
        "model": args.model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 160 if args.mode == "gbnf" else 256,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if args.mode == "gbnf":
        request["grammar"] = read_utf8_no_bom(ROOT / "design_genome.gbnf")
    else:
        request["response_format"] = {
            "type": "json_schema",
            "schema": json.loads(read_utf8_no_bom(ROOT / "design_ir.llamacpp.schema.json")),
        }
    write_utf8_no_bom(args.output, json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({"mode": args.mode, "output": str(args.output.resolve()), "request_bytes": args.output.stat().st_size}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
