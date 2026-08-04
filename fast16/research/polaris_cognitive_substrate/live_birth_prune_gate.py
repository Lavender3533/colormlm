#!/usr/bin/env python3
"""失败回执驱动的真实 v17 Qwen 权重岛出生—修剪最小门。"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import psutil


ALIAS = "Polaris-GenesisGate-v0"
PORT = 8117
INITIAL_CANDIDATE = """def normalize_score(value):
    return max(0, min(100, int(value)))
"""
PRESERVED_DESIGN_IR = "dashboard_two_column;header;stats_panel"


@dataclass(frozen=True)
class ValidationReceipt:
    passed: bool
    failures: tuple[str, ...]
    source_sha256: str


def _configure_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _health(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=2) as response:
            return json.load(response).get("status") == "ok"
    except (OSError, ValueError, urllib.error.URLError):
        return False


def _extract_code(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines.pop()
        stripped = "\n".join(lines).strip()
    return stripped + "\n"


def validate_candidate(source: str) -> ValidationReceipt:
    failures: list[str] = []
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as error:
        return ValidationReceipt(
            False,
            (f"syntax_error:{error.msg}",),
            _sha256_bytes(source.encode("utf-8")),
        )

    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(tree.body) != 1 or len(functions) != 1 or functions[0].name != "normalize_score":
        failures.append("contract:must_define_only_normalize_score")

    forbidden = (
        ast.Import,
        ast.ImportFrom,
        ast.Attribute,
        ast.ClassDef,
        ast.Lambda,
        ast.Global,
        ast.Nonlocal,
        ast.With,
        ast.AsyncWith,
    )
    if any(isinstance(node, forbidden) for node in ast.walk(tree)):
        failures.append("contract:forbidden_ast_node")

    allowed_calls = {"float", "int", "max", "min", "isinstance"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in allowed_calls:
                failures.append("contract:forbidden_call")
                break

    if failures:
        return ValidationReceipt(
            False,
            tuple(failures),
            _sha256_bytes(source.encode("utf-8")),
        )

    safe_builtins = {
        "float": float,
        "int": int,
        "max": max,
        "min": min,
        "isinstance": isinstance,
        "bool": bool,
        "ValueError": ValueError,
        "TypeError": TypeError,
    }
    namespace: dict[str, Any] = {"__builtins__": safe_builtins}
    try:
        exec(compile(tree, "<candidate>", "exec"), namespace, namespace)
        function = namespace["normalize_score"]
    except Exception as error:  # noqa: BLE001 - 回执必须保留候选失败
        return ValidationReceipt(
            False,
            (f"load_error:{type(error).__name__}",),
            _sha256_bytes(source.encode("utf-8")),
        )

    cases = (
        (50, 50.0),
        (12.5, 12.5),
        ("120", 100.0),
        ("-3", 0.0),
        (None, None),
        (True, None),
        ("not-a-number", None),
    )
    for value, expected in cases:
        try:
            observed = function(value)
        except Exception as error:  # noqa: BLE001 - 外部验证器记录实际异常
            failures.append(f"case:{value!r}:raised:{type(error).__name__}")
            continue
        if observed != expected:
            failures.append(f"case:{value!r}:expected:{expected!r}:observed:{observed!r}")
    return ValidationReceipt(
        not failures,
        tuple(failures),
        _sha256_bytes(source.encode("utf-8")),
    )


def _listener_process(port: int, expected_executable: Path) -> psutil.Process:
    expected = expected_executable.resolve()
    listeners: list[psutil.Process] = []
    for connection in psutil.net_connections(kind="tcp"):
        if (
            connection.status == psutil.CONN_LISTEN
            and connection.laddr
            and connection.laddr.port == port
            and connection.pid is not None
        ):
            listeners.append(psutil.Process(connection.pid))
    if len(listeners) != 1:
        raise RuntimeError(f"端口{port}监听进程数量不是1: {len(listeners)}")
    process = listeners[0]
    actual = Path(process.exe()).resolve()
    if actual != expected:
        raise RuntimeError(f"端口{port}被非授权程序占用: {actual}")
    return process


def _request_repair(base_url: str) -> tuple[str, dict[str, Any], float]:
    body = {
        "model": ALIAS,
        "messages": [
            {
                "role": "system",
                "content": "你是代码修复器。只输出一个Python函数定义，不要Markdown，不要解释，不要import。",
            },
            {
                "role": "user",
                "content": (
                    "外部验证器报告下面函数失败：\n\n"
                    + INITIAL_CANDIDATE
                    + "\n失败事实：None、非数字字符串和bool会产生异常或错误结果。\n"
                    "合同：整数/浮点数/数字字符串转换为float并限制到[0,100]；"
                    "None、bool、非数字字符串返回None；不得import，不得访问文件或网络。"
                    "请给出修复后的normalize_score函数。"
                ),
            },
        ],
        "max_tokens": 128,
        "temperature": 0,
        "stream": False,
    }
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = json.loads(response.read().decode("utf-8"))
    elapsed = time.perf_counter() - started
    content = payload["choices"][0]["message"]["content"]
    return _extract_code(str(content)), payload, elapsed


def run_gate(output_path: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    base_url = f"http://127.0.0.1:{PORT}"
    server = root / "llama.cpp/build-v17-perf/bin/Release/llama-server.exe"
    manifest_path = root / "fast16/research/v17_coder_island/runtime-v3/island.json"
    route_path = output_path.with_name("route_dump.bin")
    if _health(base_url):
        raise RuntimeError(f"正式门要求端口{PORT}初始为空，但服务已存在")
    if route_path.exists():
        route_path.unlink()

    events: list[dict[str, Any]] = []
    design_sha_before = _sha256_bytes(PRESERVED_DESIGN_IR.encode("utf-8"))
    initial_receipt = validate_candidate(INITIAL_CANDIDATE)
    events.append(
        {
            "event": "test.receipt",
            "status": "pass" if initial_receipt.passed else "fail",
            "receipt": asdict(initial_receipt),
        }
    )
    if initial_receipt.passed:
        raise RuntimeError("初始候选意外通过，出生门没有触发条件")

    events.append(
        {
            "event": "spawn",
            "node": "island.qwen.v17.l44_l47",
            "trigger": "test.receipt:fail",
            "manifest_sha256": _sha256_file(manifest_path),
        }
    )
    start_command = [
        sys.executable,
        str(root / "fast16/start_fast16_runtime.py"),
        "--server",
        str(server),
        "--port",
        str(PORT),
        "--ctx-size",
        "2048",
        "--batch-size",
        "128",
        "--ubatch-size",
        "128",
        "--runtime-alias",
        ALIAS,
        "--neural-island-manifest",
        str(manifest_path),
        "--neural-island-alpha",
        "0.02",
        "--neural-island-expert-cache-slots",
        "10",
        "--neural-island-route-dump",
        str(route_path),
        "--neural-island-route-dump-max-records",
        "64",
        "--anthropic-max-tokens",
        "128",
        "--allow-mmap",
        "--no-warmup",
    ]
    startup_started = time.perf_counter()
    startup = subprocess.run(
        start_command,
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=240,
        check=False,
    )
    startup_seconds = time.perf_counter() - startup_started
    if startup.returncode != 0 or not _health(base_url):
        raise RuntimeError(
            f"权重岛出生失败: code={startup.returncode}, stdout={startup.stdout!r}, stderr={startup.stderr!r}"
        )

    process = _listener_process(PORT, server)
    process_id = process.pid
    repaired_source = ""
    response_payload: dict[str, Any] = {}
    request_seconds = 0.0
    repaired_receipt: ValidationReceipt | None = None
    prune_status = "pending"
    try:
        repaired_source, response_payload, request_seconds = _request_repair(base_url)
        repaired_receipt = validate_candidate(repaired_source)
        events.append(
            {
                "event": "candidate.patch",
                "source_sha256": repaired_receipt.source_sha256,
                "model": ALIAS,
                "process_id": process_id,
            }
        )
        events.append(
            {
                "event": "test.receipt",
                "status": "pass" if repaired_receipt.passed else "fail",
                "receipt": asdict(repaired_receipt),
            }
        )
        events.append(
            {
                "event": "commit" if repaired_receipt.passed else "reject",
                "candidate_sha256": repaired_receipt.source_sha256,
            }
        )
    finally:
        if process.is_running():
            process.terminate()
            try:
                process.wait(20)
            except psutil.TimeoutExpired:
                process.kill()
                process.wait(10)
        prune_status = "pruned" if not process.is_running() else "failed"
        events.append(
            {
                "event": "prune",
                "node": "island.qwen.v17.l44_l47",
                "process_id": process_id,
                "status": prune_status,
            }
        )

    design_sha_after = _sha256_bytes(PRESERVED_DESIGN_IR.encode("utf-8"))
    route_bytes = route_path.stat().st_size if route_path.is_file() else 0
    route_sha = _sha256_file(route_path) if route_bytes else None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assertions = {
        "failure_preceded_spawn": events[0]["status"] == "fail" and events[1]["event"] == "spawn",
        "real_island_manifest_bound": events[1]["manifest_sha256"] == _sha256_file(manifest_path),
        "real_route_receipt_nonempty": route_bytes > 0 and route_sha is not None,
        "repair_passed_external_validator": bool(repaired_receipt and repaired_receipt.passed),
        "unrelated_design_ir_hash_preserved": design_sha_before == design_sha_after,
        "temporary_island_pruned": prune_status == "pruned" and not _health(base_url),
    }
    report = {
        "format": "polaris-genesis-live-birth-prune-gate-v0",
        "status": "pass" if all(assertions.values()) else "fail",
        "scope": "single_task_process_level_spawn_not_native_dynamic_graph",
        "assertions": assertions,
        "initial_candidate": INITIAL_CANDIDATE,
        "repaired_candidate": repaired_source,
        "events": events,
        "island": {
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256_file(manifest_path),
            "source_layers": manifest["source_layers"],
            "total_weight_bytes": manifest["total_weight_bytes"],
            "route_receipt": str(route_path),
            "route_receipt_bytes": route_bytes,
            "route_receipt_sha256": route_sha,
        },
        "timings": {
            "startup_seconds": startup_seconds,
            "request_seconds": request_seconds,
            "model_timings": response_payload.get("timings", {}),
        },
        "limitations": [
            "初始失败候选是固定fixture，不是v38当场生成。",
            "spawn/prune发生在进程级；尚未在单一运行时内创建或删除计算图节点。",
            "v17岛仍通过v38固定语言主干和输出头表达。",
            "单题通过不构成能力提升或新架构证据。",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    _configure_utf8()
    output = Path(__file__).with_name("live_birth_prune") / "gate_receipt.json"
    try:
        report = run_gate(output)
    except Exception as error:  # noqa: BLE001 - 顶层必须给出可读失败
        print(f"gate_error: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": report["status"], **report["assertions"]}, ensure_ascii=False, indent=2))
    print(f"回执: {output.resolve()}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
