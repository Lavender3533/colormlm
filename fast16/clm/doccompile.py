"""Compile local UTF-8 documents into training-free CLM memory records."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator


TEXT_EXTENSIONS = {
    ".md",
    ".txt",
    ".py",
    ".rs",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".toml",
    ".json",
    ".ps1",
    ".bat",
}
SKIP_DIRECTORIES = {
    ".git",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "models",
    "node_modules",
    "target",
}
SYMBOL = re.compile(
    r"^(?:pub\s+)?(?:async\s+)?(?:def|class|fn|struct|enum|trait)\s+([^\s(:<{]+)"
)
BOX_DRAWING = re.compile(r"[─━│┃┌┐└┘├┤┬┴┼╭╮╰╯]+")


def _compact_line(line: str) -> str:
    if BOX_DRAWING.search(line):
        line = BOX_DRAWING.sub(" ", line)
        line = re.sub(r" {2,}", " ", line).strip()
    return line


def _iter_files(inputs: Iterable[str | Path], max_files: int | None) -> Iterator[Path]:
    found: set[Path] = set()
    for raw_path in inputs:
        path = Path(raw_path).resolve()
        candidates = [path] if path.is_file() else path.rglob("*") if path.is_dir() else []
        for candidate in candidates:
            if not candidate.is_file() or candidate.suffix.lower() not in TEXT_EXTENSIONS:
                continue
            if any(part in SKIP_DIRECTORIES for part in candidate.parts):
                continue
            found.add(candidate)
    ordered = sorted(found, key=lambda item: str(item).lower())
    if max_files is not None:
        ordered = ordered[:max_files]
    yield from ordered


def _record_key(path: Path, heading: str, lines: list[str]) -> str:
    symbols = []
    for line in lines:
        match = SYMBOL.match(line.strip())
        if match:
            symbols.append(match.group(1))
            if len(symbols) == 3:
                break
    first_line = next((line.strip() for line in lines if line.strip()), "")
    parts = [heading] if heading else [path.stem]
    parts.extend(symbols)
    if not heading and len(parts) == 1 and first_line:
        parts.append(first_line[:96])
    return " | ".join(dict.fromkeys(part for part in parts if part))


def _aliases(key: str) -> list[str]:
    aliases: list[str] = []
    terms = re.findall(r"[A-Za-z][A-Za-z0-9.+_-]{1,15}", key)
    for term in terms:
        if term.lower() in {"the", "and", "src", "data"}:
            continue
        aliases.extend((term, f"什么是{term}"))
    chinese = re.findall(r"[\u4e00-\u9fff]{2,12}", key)
    aliases.extend(chinese[:2])
    return list(dict.fromkeys(alias for alias in aliases if alias and alias != key))[:8]


def _alias_quality(alias: str, value: str) -> tuple[int, int]:
    subject = alias.removeprefix("什么是").strip()
    lowered_subject = subject.lower()
    lowered_value = value.lower()
    occurrences = lowered_value.count(lowered_subject) if lowered_subject else 0
    definition = 0
    for marker in (f"{lowered_subject} (", f"{lowered_subject} 是", f"{lowered_subject}："):
        if marker in lowered_value:
            definition += 1
    return definition * 100 + occurrences * 10, -len(value)


def _truncate_utf8(text: str, max_bytes: int) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def _alias_window(
    candidates: list[dict[str, str]],
    best_index: int,
    max_bytes: int,
    alias: str,
) -> str:
    value = candidates[best_index]["value"]
    subject = alias.removeprefix("什么是").strip()
    position = value.lower().find(subject.lower())
    if position > 0:
        line_start = value.rfind("\n", 0, position) + 1
        header = value.split("\n", 1)[0] if value.startswith("[") else ""
        value = (header + "\n" if header else "") + value[line_start:]
    for candidate in candidates[best_index + 1 :]:
        continuation = candidate["value"]
        if continuation.startswith("[") and "\n" in continuation:
            continuation = continuation.split("\n", 1)[1]
        combined = value.rstrip() + "\n" + continuation.lstrip()
        value = _truncate_utf8(combined, max_bytes)
        if len(value.encode("utf-8")) >= max_bytes - 3:
            break
    return value


def _split_document(path: Path, text: str, max_value_bytes: int) -> Iterator[dict[str, str]]:
    lines = [
        _compact_line(line)
        for line in text.replace("\r\n", "\n").replace("\r", "\n").replace("\0", "").splitlines()
    ]
    heading = ""
    chunk: list[str] = []

    def flush() -> dict[str, str] | None:
        nonlocal chunk
        content = "\n".join(chunk).strip()
        chunk = []
        if not content:
            return None
        value = f"[{path.name}]\n{content}"
        return {"key": _record_key(path, heading, content.splitlines()), "value": value}

    for line in lines:
        stripped = line.strip()
        is_heading = stripped.startswith("#")
        is_symbol = SYMBOL.match(stripped) is not None
        candidate = "\n".join(chunk + [line]).encode("utf-8")
        if chunk and (len(candidate) > max_value_bytes or is_heading or is_symbol):
            record = flush()
            if record:
                yield record
        if is_heading:
            heading = stripped.lstrip("#").strip()
        encoded = line.encode("utf-8")
        if len(encoded) <= max_value_bytes:
            chunk.append(line)
            continue
        start = 0
        while start < len(encoded):
            piece = encoded[start : start + max_value_bytes].decode("utf-8", errors="ignore")
            if piece:
                chunk.append(piece)
                record = flush()
                if record:
                    yield record
            start += max_value_bytes
    record = flush()
    if record:
        yield record


def compile_documents(
    inputs: Iterable[str | Path],
    output_path: str | Path,
    *,
    max_value_bytes: int = 384,
    max_files: int | None = None,
) -> dict[str, int | str]:
    if max_value_bytes < 64:
        raise ValueError("max_value_bytes must be at least 64")
    files = list(_iter_files(inputs, max_files))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = 0
    source_bytes = 0
    value_bytes = 0
    base_records: list[dict[str, str]] = []
    grouped: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        source_bytes += len(text.encode("utf-8"))
        for record in _split_document(path, text, max_value_bytes):
            base_records.append(record)
            grouped[record["key"]].append(record)

    variants = list(base_records)
    for key, candidates in grouped.items():
        for alias in _aliases(key):
            best_index = max(
                range(len(candidates)),
                key=lambda index: _alias_quality(alias, candidates[index]["value"]),
            )
            variants.append(
                {
                    "key": alias,
                    "value": _alias_window(candidates, best_index, max_value_bytes, alias),
                }
            )

    seen: set[tuple[str, str]] = set()
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for variant in variants:
            identity = (variant["key"], variant["value"])
            if identity in seen:
                continue
            seen.add(identity)
            output.write(json.dumps(variant, ensure_ascii=False, separators=(",", ":")) + "\n")
            records += 1
            value_bytes += len(variant["value"].encode("utf-8"))

    return {
        "output": str(output_path.resolve()),
        "files": len(files),
        "records": records,
        "source_bytes": source_bytes,
        "value_bytes": value_bytes,
        "max_value_bytes": max_value_bytes,
    }
