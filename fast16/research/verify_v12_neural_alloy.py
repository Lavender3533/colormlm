"""Static and live contract verifier for ColorLM v12 Neural Alloy."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path


FAST16 = Path(__file__).resolve().parent.parent
ROOT = FAST16.parent
if str(FAST16) not in sys.path:
    sys.path.insert(0, str(FAST16))

from start_fast16_runtime import validate_alloy_plan  # noqa: E402


DEFAULT_PLAN = Path("fast16/models/ColorLM-v12-Neural-Alloy.alloyplan.json")
DEFAULT_PACKAGE = Path("fast16/models/ColorLM-v12-Neural-Alloy.clmpkg.json")
DEFAULT_REPORT = Path("fast16/research/v12_neural_alloy_smoke_report.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证ColorLM v12 Neural Alloy交付契约")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--full-core-sha256", action="store_true")
    parser.add_argument("--require-running", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json_url(url: str) -> dict[str, object]:
    with urllib.request.urlopen(url, timeout=3) as response:
        return json.load(response)


def main() -> int:
    args = parse_args()
    plan_path = resolve(args.plan)
    package_path = resolve(args.package)
    report_path = resolve(args.report)

    try:
        alloy = validate_alloy_plan(ROOT, plan_path, args.full_core_sha256)
        package = load_json(package_path)
        report = load_json(report_path)
        plan_sha = sha256_file(plan_path)

        require(package.get("format") == "colorlm-model-package-v4", "模型包格式错误")
        require(package.get("name") == alloy["alias"], "模型包名称与合金计划不一致")
        require(package["alloy_plan"]["sha256"] == plan_sha, "模型包未锁定当前合金计划")
        require(int(package["effective_bytes"]) == int(load_json(plan_path)["effective_bytes"]),
                "模型包有效字节数不一致")

        dll = ROOT / "build" / "bin" / "Release" / "llama.dll"
        server = ROOT / "build" / "bin" / "Release" / "llama-server.exe"
        dll_sha = sha256_file(dll)
        server_sha = sha256_file(server)
        require(load_json(plan_path)["runtime"]["llama_dll_sha256"] == dll_sha,
                "合金计划未锁定当前llama.dll")
        require(package["runtime"]["llama_dll_sha256"] == dll_sha,
                "模型包未锁定当前llama.dll")
        launcher = resolve(Path(str(package["runtime"]["entry"])))
        require(package["runtime"]["entry_sha256"] == sha256_file(launcher),
                "模型包未锁定当前正式启动器")
        require(package["evidence"]["smoke_report_sha256"] == sha256_file(report_path),
                "模型包未锁定当前烟测报告")
        require(package["evidence"]["verifier_sha256"] == sha256_file(Path(__file__)),
                "模型包未锁定当前验收脚本")
        require(report["alloy_plan_sha256"] == plan_sha, "烟测报告未锁定当前合金计划")
        require(report["runtime"]["llama_dll_sha256"] == dll_sha,
                "烟测报告未锁定当前llama.dll")
        require(report["runtime"]["llama_server_sha256"] == server_sha,
                "烟测报告未锁定当前llama-server.exe")

        if args.require_running:
            base_url = f"http://127.0.0.1:{alloy['port']}"
            health = read_json_url(f"{base_url}/health")
            models = read_json_url(f"{base_url}/v1/models")
            props = read_json_url(f"{base_url}/props")
            require(health.get("status") == "ok", "运行服务健康检查失败")
            require(any(item.get("id") == alloy["alias"] for item in models.get("data", [])),
                    "运行服务alias不匹配")
            require(int(props["default_generation_settings"]["n_ctx"]) == int(alloy["context"]),
                    "运行服务上下文不匹配")
            actual_model = str(props.get("model_path", "")).replace("\\", "/").lower()
            require(actual_model == str(alloy["model_relative"]).lower(), "运行服务核心GGUF不匹配")

        summary = {
            "model": alloy["alias"],
            "plan_sha256": plan_sha,
            "dll_sha256": dll_sha,
            "effective_bytes": package["effective_bytes"],
            "core_sha256": "verified" if args.full_core_sha256 else "not-requested",
            "running": args.require_running,
            "result": "passed",
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, RuntimeError) as error:
        print(f"v12验收失败: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
