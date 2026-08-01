#!/usr/bin/env python3
"""Design Genome 输出解析、规范化、重复抑制与截断续写接口。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ir_core import HARD_MAX_BYTES, canonical_text, read_utf8_no_bom, validate_ir, write_utf8_no_bom


ROOT_MARKER = '{"v":"dg1"'


@dataclass
class DecodeResult:
    status: str
    genome: dict[str, Any] | None
    canonical: str | None
    source_bytes: int
    complete_objects: int
    repeated_roots: int
    recovered_closure: bool
    resume_prefix: str | None
    errors: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "design-genome-decode-report-v1",
            "status": self.status,
            "genome": self.genome,
            "canonical": self.canonical,
            "source_bytes": self.source_bytes,
            "complete_objects": self.complete_objects,
            "repeated_roots": self.repeated_roots,
            "recovered_closure": self.recovered_closure,
            "resume_prefix": self.resume_prefix,
            "errors": self.errors,
        }


def strip_transport(text: str) -> str:
    text = text.lstrip("\ufeff \t\r\n")
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline >= 0:
            text = text[first_newline + 1:]
        closing = text.rfind("```")
        if closing >= 0:
            text = text[:closing]
    return text.strip()


def balanced_objects(text: str) -> list[str]:
    """字符串感知地提取所有闭合根对象；重复对象不会污染第一个结果。"""
    result: list[str] = []
    for start, char in enumerate(text):
        if char != "{":
            continue
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            current = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    in_string = False
                continue
            if current == '"':
                in_string = True
            elif current in "[{":
                depth += 1
            elif current in "]}":
                depth -= 1
                if depth == 0:
                    result.append(text[start:index + 1])
                    break
                if depth < 0:
                    break
    return result


def close_truncated_object(text: str) -> str | None:
    """只修复完整 JSON token 后缺失的引号/括号；不猜测缺失语义槽。"""
    start = text.rfind(ROOT_MARKER)
    if start < 0:
        return None
    candidate = text[start:].strip()
    stack: list[str] = []
    in_string = False
    escaped = False
    for current in candidate:
        if in_string:
            if escaped:
                escaped = False
            elif current == "\\":
                escaped = True
            elif current == '"':
                in_string = False
            continue
        if current == '"':
            in_string = True
        elif current == "{":
            stack.append("}")
        elif current == "[":
            stack.append("]")
        elif current in "}]":
            if not stack or stack[-1] != current:
                return None
            stack.pop()
    if in_string:
        candidate += '"'
    candidate = candidate.rstrip()
    if candidate.endswith(":"):
        return None
    if candidate.endswith(","):
        candidate = candidate[:-1]
    return candidate + "".join(reversed(stack))


def safe_resume_prefix(text: str) -> str | None:
    """返回最后一个结构分隔点；调用方应保留前缀并从 GBNF 状态续写。"""
    start = text.rfind(ROOT_MARKER)
    if start < 0:
        return None
    candidate = text[start:]
    in_string = False
    escaped = False
    last_safe = -1
    for index, current in enumerate(candidate):
        if in_string:
            if escaped:
                escaped = False
            elif current == "\\":
                escaped = True
            elif current == '"':
                in_string = False
            continue
        if current == '"':
            in_string = True
        elif current in ",[]{}":
            last_safe = index
    return candidate[:last_safe + 1] if last_safe >= 0 else None


def decode_text(raw: str) -> DecodeResult:
    source_bytes = len(raw.encode("utf-8"))
    errors: list[str] = []
    if source_bytes > HARD_MAX_BYTES * 2:
        errors.append(f"输出 {source_bytes} 字节，超过恢复窗口 {HARD_MAX_BYTES * 2}")
    text = strip_transport(raw[: HARD_MAX_BYTES * 2])
    roots = text.count(ROOT_MARKER)
    objects = balanced_objects(text)
    semantic_errors: list[str] = []
    for candidate in objects:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as error:
            semantic_errors.append(str(error))
            continue
        current = validate_ir(payload, enforce_target_size=True)
        if not current:
            canonical = canonical_text(payload)
            return DecodeResult("ok" if candidate == text else "recovered_duplicate_or_wrapper", payload, canonical, source_bytes, len(objects), roots, False, None, errors)
        semantic_errors.extend(current)
    repaired = close_truncated_object(text)
    if repaired:
        try:
            payload = json.loads(repaired)
            current = validate_ir(payload, enforce_target_size=True)
            if not current:
                canonical = canonical_text(payload)
                return DecodeResult("recovered_closure", payload, canonical, source_bytes, len(objects), roots, True, None, errors)
            semantic_errors.extend(current)
        except json.JSONDecodeError as error:
            semantic_errors.append(str(error))
    errors.extend(dict.fromkeys(semantic_errors))
    prefix = safe_resume_prefix(text)
    status = "needs_resume" if roots and not objects else "rejected"
    if status == "needs_resume":
        errors.append("截断发生在完整槽位之前；禁止猜测缺失基因，应携带 resume_prefix 继续受约束生成")
    elif not roots:
        errors.append("未找到紧凑 Genome 根标记")
    return DecodeResult(status, None, None, source_bytes, len(objects), roots, False, prefix, errors)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", "-o", type=Path, help="写出规范单行 JSON")
    parser.add_argument("--report", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = decode_text(read_utf8_no_bom(args.input))
    if args.output and result.canonical is not None:
        write_utf8_no_bom(args.output, result.canonical + "\n")
    if args.report:
        write_utf8_no_bom(args.report, json.dumps(result.as_dict(), ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    return 0 if result.genome is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
