"""Verify the static and live ColorLM v13 causal-sparse package contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any


FAST16 = Path(__file__).resolve().parent.parent
ROOT = FAST16.parent
if str(FAST16) not in sys.path:
    sys.path.insert(0, str(FAST16))

from start_fast16_runtime import validate_k3_plan  # noqa: E402


DEFAULT_PACKAGE = Path("fast16/models/ColorLM-v13-Causal-Sparse-L12.clmpkg.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证ColorLM v13因果稀疏模型包")
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--require-running", action="store_true")
    parser.add_argument("--full-core-sha256", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON根节点必须是对象: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json_url(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=5) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise RuntimeError(f"服务返回值不是JSON对象: {url}")
    return value


def main() -> int:
    args = parse_args()
    try:
        package_path = resolve(args.package)
        package = load_json(package_path)
        require(package.get("format") == "colorlm-model-package-v5", "模型包格式错误")
        require(package.get("name") == "ColorLM-v13-Causal-Sparse-L12", "模型名错误")

        model_dir = package_path.parent
        core = model_dir / str(package["core"]["file"])
        require(core.stat().st_size == int(package["core"]["bytes"]), "核心GGUF尺寸不匹配")
        if args.full_core_sha256:
            require(sha256_file(core) == package["core"]["sha256"], "核心GGUF SHA-256不匹配")

        plan_path = model_dir / str(package["k3_plan"]["file"])
        require(sha256_file(plan_path) == package["k3_plan"]["sha256"], "K3计划SHA-256不匹配")
        _, sites = validate_k3_plan(ROOT, plan_path)
        require([int(site["site"]) for site in sites] == [12], "v13必须只包含L12 K3站点")

        weight_bytes = 0
        candidate_count = 0
        for site in sites:
            for candidate in site["candidates"]:
                manifest = load_json(Path(str(candidate["capsule"])) / "capsule.json")
                weight_bytes += int(manifest["runtime_total_bytes"])
                weight_bytes += int(manifest["runtime_router"]["bytes"])
                candidate_count += 1
        require(weight_bytes == int(package["k3_plan"]["weight_bytes"]), "K3权重字节数不匹配")
        require(candidate_count == int(package["k3_plan"]["candidate_count"]), "K3候选数不匹配")
        require(int(package["effective_bytes"]) == core.stat().st_size + weight_bytes,
                "有效权重字节数不匹配")

        runtime = package["runtime"]
        launcher = resolve(Path(str(runtime["entry"])))
        require(sha256_file(launcher) == runtime["entry_sha256"], "启动器SHA-256不匹配")
        dll = ROOT / "build" / "bin" / "Release" / "llama.dll"
        require(sha256_file(dll) == runtime["llama_dll_sha256"], "llama.dll SHA-256不匹配")
        report = (model_dir / str(package["evidence"]["report"])).resolve()
        require(sha256_file(report) == package["evidence"]["report_sha256"], "研究报告SHA-256不匹配")

        if args.require_running:
            base_url = "http://127.0.0.1:8102"
            health = read_json_url(f"{base_url}/health")
            models = read_json_url(f"{base_url}/v1/models")
            props = read_json_url(f"{base_url}/props")
            require(health.get("status") == "ok", "服务健康检查失败")
            require(any(item.get("id") == package["name"] for item in models.get("data", [])),
                    "运行服务alias不匹配")
            settings = props.get("default_generation_settings", {})
            require(int(settings.get("n_ctx", 0)) == int(runtime["context"]), "上下文长度不匹配")
            actual_model = str(props.get("model_path", "")).replace("\\", "/").lower()
            require(actual_model == f"fast16/models/{package['core']['file']}".lower(),
                    "运行服务核心GGUF不匹配")

        print(json.dumps({
            "model": package["name"],
            "effective_bytes": package["effective_bytes"],
            "k3_sites": [12],
            "k3_candidates": candidate_count,
            "core_sha256": "verified" if args.full_core_sha256 else "previously-verified",
            "running": args.require_running,
            "result": "passed",
        }, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, RuntimeError) as error:
        print(f"v13验收失败: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
