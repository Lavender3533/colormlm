"""启动单个模型、采集指定深层激活，然后可靠关闭服务。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
BRIDGE_TOOL = Path(__file__).with_name("activation_bridge.py")
DEFAULT_SERVER = ROOT / "llama.cpp" / "build-v16-vulkan" / "bin" / "Release" / "llama-server.exe"


class CaptureError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="一次只加载一个模型，完成v18深层激活采集后自动释放内存"
    )
    parser.add_argument("--server", type=Path, default=DEFAULT_SERVER)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--alias", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--ctx-size", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--cpu-moe-layers", type=int, required=True)
    parser.add_argument(
        "--no-mmap",
        action="store_true",
        help="主干模型可用；完整29.7GB供体应保留mmap。",
    )
    parser.add_argument("--tensor", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=64)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=18)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--startup-timeout", type=float, default=420.0)
    parser.add_argument("--log-prefix", type=Path)
    return parser.parse_args()


def resolve_existing(path: Path, label: str) -> Path:
    result = path if path.is_absolute() else ROOT / path
    result = result.resolve()
    if not result.is_file():
        raise CaptureError(f"找不到{label}: {result}")
    return result


def resolve_output(path: Path) -> Path:
    return (path if path.is_absolute() else ROOT / path).resolve()


def native_process_path(path: Path) -> str:
    """Avoid narrow-argv corruption when the absolute workspace path is non-ASCII."""

    try:
        return os.fspath(path.relative_to(ROOT))
    except ValueError:
        return os.fspath(path)


def read_json_url(url: str, timeout: float = 2.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise CaptureError(f"HTTP响应不是JSON对象: {url}")
    return value


def port_is_live(port: int) -> bool:
    try:
        read_json_url(f"http://127.0.0.1:{port}/health")
        return True
    except Exception:
        return False


def wait_ready(process: subprocess.Popen[bytes], port: int, alias: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error = "服务尚未响应"
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise CaptureError(f"llama-server提前退出，退出码={return_code}")
        try:
            health = read_json_url(f"http://127.0.0.1:{port}/health")
            models = read_json_url(f"http://127.0.0.1:{port}/v1/models")
            ids = {
                str(item.get("id"))
                for item in models.get("data", [])
                if isinstance(item, dict)
            }
            if health.get("status") == "ok" and alias in ids:
                return
            last_error = f"health={health.get('status')}, models={sorted(ids)}"
        except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError) as error:
            last_error = str(error)
        time.sleep(0.5)
    raise CaptureError(f"等待模型加载超时: {last_error}")


def stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=20)


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if not 1 <= args.port <= 65535:
        raise CaptureError("--port超出范围")
    if (
        args.ctx_size <= 0
        or args.threads <= 0
        or args.gpu_layers < 0
        or args.cpu_moe_layers < 0
    ):
        raise CaptureError("ctx-size、threads、gpu-layers或cpu-moe-layers无效")
    if args.max_records <= 0 or args.startup_timeout <= 0 or args.request_timeout <= 0:
        raise CaptureError("超时和max-records必须为正数")

    server = resolve_existing(args.server, "llama-server")
    model = resolve_existing(args.model, "模型")
    prompts = resolve_existing(args.prompts, "校准prompt")
    dump = resolve_output(args.dump)
    receipt = resolve_output(args.receipt)
    if dump.exists() and dump.stat().st_size:
        raise CaptureError(f"为避免记录上限和旧数据混淆，dump必须为空或不存在: {dump}")
    if port_is_live(args.port):
        raise CaptureError(f"端口{args.port}已有服务，拒绝误用或覆盖")

    dump.parent.mkdir(parents=True, exist_ok=True)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    log_prefix = resolve_output(
        args.log_prefix
        or receipt.parent / f"{receipt.stem}.server"
    )
    log_prefix.parent.mkdir(parents=True, exist_ok=True)
    stdout_path = Path(str(log_prefix) + ".stdout.log")
    stderr_path = Path(str(log_prefix) + ".stderr.log")
    report_path = Path(str(log_prefix) + ".run.json")

    command = [
        str(server),
        "--model", native_process_path(model),
        "--alias", args.alias,
        "--n-gpu-layers", str(args.gpu_layers),
        "--n-cpu-moe", str(args.cpu_moe_layers),
        "--threads", str(args.threads),
        "--ctx-size", str(args.ctx_size),
        "--parallel", "1",
        "--batch-size", "512",
        "--ubatch-size", "512",
        "--cache-ram", "0",
        "--ctx-checkpoints", "0",
        "--fit", "off",
        "--no-warmup",
        "--flash-attn", "on",
        "--cache-type-k", "q8_0",
        "--cache-type-v", "q8_0",
        "--jinja",
        "--reasoning", "off",
        "--host", "127.0.0.1",
        "--port", str(args.port),
    ]
    if args.no_mmap:
        command.append("--no-mmap")

    environment = os.environ.copy()
    for name in list(environment):
        if name.startswith("COLORLM_"):
            environment.pop(name, None)
    environment.update(
        {
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "GGML_SCHED_MERGE_CPU_SYNC": "1",
            "GGML_SCHED_SKIP_CPU_FINAL_SYNC": "1",
            "GGML_SCHED_BATCH_CPU_READ": "1",
            "COLORLM_ACTIVATION_DUMP": native_process_path(dump),
            "COLORLM_ACTIVATION_TENSOR": args.tensor,
            "COLORLM_ACTIVATION_DUMP_MAX_RECORDS": str(args.max_records),
        }
    )
    environment.pop("GGML_VK_SPIN_FENCE", None)

    started = time.monotonic()
    process: subprocess.Popen[bytes] | None = None
    collect_return_code: int | None = None
    failure: str | None = None
    with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
        try:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
            )
            print(f"正在加载 {args.alias}，PID={process.pid}...", flush=True)
            wait_ready(process, args.port, args.alias, args.startup_timeout)
            print(f"模型已就绪，开始采集 {args.tensor}", flush=True)
            collect_command = [
                sys.executable,
                str(BRIDGE_TOOL),
                "collect",
                "--endpoint", f"http://127.0.0.1:{args.port}",
                "--expect-model", args.alias,
                "--dump", str(dump),
                "--layer", str(args.layer),
                "--stage", args.stage,
                "--prompts", str(prompts),
                "--output", str(receipt),
                "--seed", str(args.seed),
                "--timeout", str(args.request_timeout),
            ]
            completed = subprocess.run(
                collect_command,
                cwd=ROOT,
                env=environment,
                check=False,
            )
            collect_return_code = completed.returncode
            if collect_return_code != 0:
                raise CaptureError(f"activation_bridge collect失败，退出码={collect_return_code}")
        except Exception as error:
            failure = str(error)
        finally:
            if process is not None:
                stop_process(process)

    report = {
        "format": "colorlm-v18-single-model-capture-run-v1",
        "result": "ok" if failure is None else "error",
        "error": failure,
        "model": str(model),
        "model_bytes": model.stat().st_size,
        "alias": args.alias,
        "server": str(server),
        "port": args.port,
        "gpu_layers": args.gpu_layers,
        "cpu_moe_layers": args.cpu_moe_layers,
        "pid": process.pid if process is not None else None,
        "tensor": args.tensor,
        "layer": args.layer,
        "stage": args.stage,
        "dump": str(dump),
        "receipt": str(receipt),
        "collect_return_code": collect_return_code,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }
    write_report(report_path, report)
    if failure is not None:
        print(f"采集失败: {failure}", file=sys.stderr)
        print(f"服务日志: {stderr_path}", file=sys.stderr)
        return 1
    print(f"采集完成并已释放模型: {receipt}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CaptureError, OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
