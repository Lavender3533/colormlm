"""Start the standalone Fast16 Runtime v1 server on Vulkan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


DEFAULT_PORT = 8096
DEFAULT_CTX_SIZE = 4096
ALIAS = "ColorLM-Fast16-v1"
NEURAL_BUS_ALIAS = "ColorLM-Neural-Bus-v1"
NEURAL_BUS_V2_ALIAS = "ColorLM-Neural-Bus-v2"
K3_NEURAL_BUS_ALIAS = "ColorLM-v9-K3-rc1"
CPU_MOE_LAYERS = 29
SLOT_POOL_LAYERS = 8
SLOT_POOL_K = 28
NEURAL_BUS_V1_CAPSULE = Path(
    "fast16/research/neural_bus_capsules/coder_next_l47_e471_q4_0"
)
NEURAL_BUS_V2_PRIMARY = Path(
    "fast16/research/neural_bus_capsules/coder_next_l47_e471_v2_q4_0"
)
NEURAL_BUS_V2_SECONDARY = Path(
    "fast16/research/neural_bus_capsules/coder_next_l47_e0_v2_q4_0"
)
NEURAL_BUS_V2_REPORT = Path("fast16/research/neural_bus_v2_capsule_build_report.json")
K3_NEURAL_BUS_PLAN = Path("fast16/models/ColorLM-v9-K3-rc1.k3plan.json")
COLORLM_LAYER_COUNT = 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="启动独立Fast16 Runtime v1")
    parser.add_argument(
        "--server",
        type=Path,
        help="覆盖llama-server路径；用于隔离研究构建。",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="覆盖核心GGUF路径；不能与--alloy-plan同时使用。",
    )
    parser.add_argument("--port", type=int)
    parser.add_argument("--runtime-alias", help="覆盖API暴露的模型版本名。")
    parser.add_argument("--ctx-size", type=int)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="逻辑prompt batch；默认512。大核心叠加策略头在8GiB显存上可降到256。",
    )
    parser.add_argument(
        "--ubatch-size",
        type=int,
        default=512,
        help="物理micro-batch；不得大于batch-size，默认512。",
    )
    parser.add_argument(
        "--alloy-plan",
        type=Path,
        help="由统一Neural Alloy计划选择基座、Coder胶囊、K3计划和默认运行参数。",
    )
    parser.add_argument(
        "--verify-alloy-core-sha256",
        action="store_true",
        help="启动前完整校验Neural Alloy核心GGUF；正式验收使用，日常启动默认只校验尺寸。",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="只解析并校验完整运行契约，打印最终配置后退出，不启动模型。",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="CPU专家和生成使用的线程数；当前RX 5700 XT + i5-12400F实测8最优。",
    )
    parser.add_argument(
        "--no-merge-cpu-sync",
        action="store_true",
        help="关闭CPU split同步合并，仅用于回归对照。",
    )
    parser.add_argument(
        "--no-skip-cpu-final-sync",
        action="store_true",
        help="恢复CPU backend尾部同步，仅用于回归对照。",
    )
    parser.add_argument(
        "--no-batch-cpu-read",
        action="store_true",
        help="关闭Vulkan到pinned host的批量读取，仅用于回归对照。",
    )
    slot_pool = parser.add_mutually_exclusive_group()
    slot_pool.add_argument(
        "--slot-pool",
        action="store_true",
        help="启用实验性精确专家槽池；相邻A/B尚未证明有稳定收益。",
    )
    slot_pool.add_argument(
        "--no-slot-pool",
        action="store_true",
        help="显式关闭实验性精确专家槽池（当前默认）。",
    )
    fence = parser.add_mutually_exclusive_group()
    fence.add_argument(
        "--spin-fence",
        action="store_true",
        help="显式启用Vulkan自旋等待；ColorLM v6默认不启用。",
    )
    fence.add_argument(
        "--no-spin-fence",
        action="store_true",
        help="显式关闭Vulkan自旋等待。",
    )
    neural_bus = parser.add_mutually_exclusive_group()
    neural_bus.add_argument(
        "--neural-bus",
        action="store_true",
        help="启用Neural Bus v1 Coder残差胶囊。",
    )
    neural_bus.add_argument(
        "--neural-bus-v2",
        action="store_true",
        help="启用Neural Bus v2双胶囊隐藏状态top-1路由。",
    )
    parser.add_argument(
        "--neural-bus-alpha",
        type=float,
        help="覆盖Coder残差alpha；设为0时不加载胶囊、不构建图节点。",
    )
    parser.add_argument("--neural-bus-sites", default="12,28")
    parser.add_argument("--neural-bus-target-ratio", type=float, default=0.08)
    parser.add_argument("--neural-bus-sharpness", type=float, default=4.0)
    parser.add_argument("--neural-bus-router-bias", type=float)
    parser.add_argument("--neural-bus-router-floor", type=float)
    parser.add_argument(
        "--neural-bus-trace",
        action="store_true",
        help="只用于短验证：记录前8次胶囊残差的RMS。",
    )
    parser.add_argument(
        "--neural-block-package",
        type=Path,
        help="启用同图完整Coder神经块，目录内必须包含block.json与weights.bin。",
    )
    parser.add_argument("--neural-block-alpha", type=float, default=0.04)
    parser.add_argument("--neural-block-site", type=int, default=35)
    parser.add_argument("--neural-block-target-ratio", type=float, default=0.04)
    parser.add_argument("--neural-block-sharpness", type=float, default=4.0)
    parser.add_argument(
        "--anthropic-max-tokens",
        type=int,
        default=1024,
        help="Anthropic兼容接口的单次输出硬上限；防止Claude客户端请求32K后失控长生成。",
    )
    parser.add_argument(
        "--verify-neural-block-weights",
        action="store_true",
        help="启动前校验完整0.876GiB运行包SHA-256。",
    )
    parser.add_argument(
        "--neural-island-manifest",
        type=Path,
        help="启用连续Coder神经岛；指向colorlm-neural-island-runtime-v1清单。",
    )
    parser.add_argument("--neural-island-alpha", type=float, default=0.02)
    parser.add_argument("--neural-island-site", type=int, default=35)
    parser.add_argument("--neural-island-target-ratio", type=float, default=0.04)
    parser.add_argument("--neural-island-sharpness", type=float, default=4.0)
    parser.add_argument(
        "--neural-island-expert-cache-slots",
        type=int,
        default=0,
        help="每个神经岛层的精确GPU热专家槽数；0关闭，建议32。",
    )
    parser.add_argument(
        "--neural-island-expert-cache-policy",
        choices=("lru", "lfu97"),
        default="lru",
        help="精确专家缓存淘汰策略；lfu97只改变驻留次序，不改变模型数学输出。",
    )
    parser.add_argument(
        "--verify-neural-island-weights",
        action="store_true",
        help="启动前校验神经岛四层全部运行权重SHA-256。",
    )
    parser.add_argument(
        "--neural-island-route-dump",
        type=Path,
        help="仅用于短校准：导出四层单token decode真实top-k专家ID。",
    )
    parser.add_argument(
        "--neural-island-route-dump-max-records",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--neural-island-teacher-dump",
        type=Path,
        help="仅用于蒸馏校准：导出岛输入hidden与原始delta成对张量。",
    )
    parser.add_argument(
        "--neural-island-teacher-dump-max-records",
        type=int,
        default=256,
    )
    parser.add_argument("--neural-island-teacher-dump-site", type=int, default=35)
    parser.add_argument(
        "--neural-island-stage-dump",
        type=Path,
        help="仅用于蒸馏校准：分别导出四个供体层的输入与原始残差。",
    )
    parser.add_argument(
        "--neural-island-stage-dump-max-records",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--neural-island-l47-moe-dump",
        type=Path,
        help="仅用于蒸馏校准：导出L47 post-attention norm与MoE+shared残差。",
    )
    parser.add_argument(
        "--neural-island-l47-moe-dump-max-records",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--neural-island-micro-l47-package",
        type=Path,
        help="研究候选：prefill保留完整L47，仅在单token decode使用低秩微层。",
    )
    parser.add_argument(
        "--verify-neural-island-micro-l47-weights",
        action="store_true",
    )
    parser.add_argument(
        "--neural-island-micro-l47-moe-package",
        type=Path,
        help="研究候选：保留L47 Attention/KV，仅在decode用低秩支路替换MoE+shared。",
    )
    parser.add_argument(
        "--verify-neural-island-micro-l47-moe-weights",
        action="store_true",
    )
    parser.add_argument(
        "--neural-output-head-package",
        type=Path,
        help="启用v19供体末端双输出头；目录内必须包含head.json与weights.bin。",
    )
    parser.add_argument("--neural-output-head-alpha", type=float, default=0.0)
    parser.add_argument(
        "--verify-neural-output-head-weights",
        action="store_true",
        help="启动前校验约245MiB供体末端输出头SHA-256。",
    )
    parser.add_argument(
        "--sequence-policy-package",
        type=Path,
        help="启用v29隐藏状态条件化的稀疏序列策略头。",
    )
    parser.add_argument(
        "--verify-sequence-policy-weights",
        action="store_true",
        help="启动前校验v29序列策略头权重SHA-256。",
    )
    parser.add_argument(
        "--neural-output-capture",
        type=Path,
        help=(
            "研究模式：未启用输出头时采集base末层hidden与原始logits；"
            "启用v19输出头时保持原三张量采集契约。"
        ),
    )
    parser.add_argument(
        "--neural-output-capture-max-records",
        type=int,
        default=256,
        help="每类输出张量最多写入的记录数。",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="关闭llama-server启动warmup；用于要求第0条严格对齐的张量采集。",
    )
    parser.add_argument(
        "--allow-mmap",
        action="store_true",
        help=(
            "允许核心GGUF使用文件映射；仅改变物理装载，"
            "用于Vulkan pinned-host预算不足的短研究门。"
        ),
    )
    parser.add_argument(
        "--spec-type",
        choices=(
            "none",
            "ngram-cache",
            "ngram-simple",
            "ngram-map-k",
            "ngram-map-k4v",
            "ngram-mod",
        ),
        default="none",
        help="无草稿模型的推测解码类型；连续神经岛默认none，ngram仅用于隔离研究。",
    )
    parser.add_argument("--spec-ngram-match", type=int, default=16)
    parser.add_argument("--spec-ngram-min", type=int, default=4)
    parser.add_argument("--spec-ngram-max", type=int, default=16)
    parser.add_argument(
        "--k3-plan",
        type=Path,
        nargs="?",
        const=K3_NEURAL_BUS_PLAN,
        help="启用按站点K3宏胶囊；不带路径时使用v9-K3-rc1正式计划。",
    )
    parser.add_argument(
        "--k3-alpha",
        type=float,
        help="覆盖计划内所有K3站点alpha；设为0可精确退回无K3路径。",
    )
    parser.add_argument(
        "--k3-trace",
        action="store_true",
        help="只用于短验证：记录前8次K3门控后残差RMS。",
    )
    parser.add_argument(
        "--intelligence-plan",
        type=Path,
        help="启用由反事实next-token NLL校准的no-op/Coder/K3隐藏态路由。",
    )
    parser.add_argument(
        "--intelligence-trace",
        action="store_true",
        help="只用于短验证：记录三路能力路由份额。",
    )
    force_path = parser.add_mutually_exclusive_group()
    force_path.add_argument(
        "--force-path",
        choices=("auto", "no_op", "coder", "k3"),
        default="auto",
        help="仅用于反事实校准，强制全部路由站走同一路径。",
    )
    force_path.add_argument(
        "--force-site-paths",
        help=(
            "仅用于逐站反事实校准，格式如12=coder,28=no_op；"
            "未列出的层回退到全局auto。"
        ),
    )
    parser.add_argument(
        "--hidden-dump",
        type=Path,
        help="仅用于短校准：导出指定站点的attn_post_norm隐藏态。",
    )
    parser.add_argument("--hidden-dump-sites", default="12,28")
    parser.add_argument("--hidden-dump-max-records", type=int, default=256)
    parser.add_argument(
        "--k3-feature-dump",
        type=Path,
        help="仅用于短校准：导出K3残差的6维连续特征。",
    )
    parser.add_argument("--k3-feature-dump-sites", default="28")
    parser.add_argument("--k3-feature-dump-max-records", type=int, default=256)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_force_site_paths(value: str | None) -> tuple[str | None, dict[int, str]]:
    if value is None:
        return None, {}
    if not value or value.startswith(",") or value.endswith(","):
        raise ValueError("--force-site-paths列表边界无效")

    result: dict[int, str] = {}
    for item in value.split(","):
        if item.count("=") != 1:
            raise ValueError(f"--force-site-paths项目无效: {item}")
        layer_text, path = item.split("=", 1)
        if not layer_text or not layer_text.isascii() or not layer_text.isdigit():
            raise ValueError(f"--force-site-paths层号无效: {layer_text}")
        layer = int(layer_text)
        if not 0 <= layer < COLORLM_LAYER_COUNT:
            raise ValueError(
                f"--force-site-paths层号超出模型范围[0,{COLORLM_LAYER_COUNT - 1}]: {layer}"
            )
        if layer in result:
            raise ValueError(f"--force-site-paths层号重复: {layer}")
        if path not in {"no_op", "coder", "k3"}:
            raise ValueError(f"--force-site-paths路径无效: {path}")
        result[layer] = path

    normalized = ",".join(f"{layer}={path}" for layer, path in result.items())
    return normalized, result


def validate_k3_plan(root: Path, requested: Path) -> tuple[Path, list[dict[str, object]]]:
    plan_path = requested if requested.is_absolute() else root / requested
    plan_path = plan_path.resolve()
    if not plan_path.is_file():
        raise RuntimeError(f"找不到K3站点计划: {plan_path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan_format = plan.get("format")
    if plan_format not in {"colorlm-k3-site-plan-v1", "colorlm-k3-site-plan-v2"}:
        raise RuntimeError(f"不支持的K3站点计划: {plan_path}")
    sites = plan.get("sites")
    if not isinstance(sites, list) or not sites:
        raise RuntimeError("K3站点计划至少需要一个站点")

    checked: list[dict[str, object]] = []
    seen: set[int] = set()
    for record in sites:
        site = int(record["site"])
        if site in seen:
            raise RuntimeError(f"K3站点重复: {site}")
        seen.add(site)
        sources = record.get("candidates") if plan_format.endswith("v2") else [record]
        if not isinstance(sources, list) or (
            plan_format.endswith("v2") and not 2 <= len(sources) <= 8
        ):
            raise RuntimeError(f"K3 v2站点{site}必须包含2到8颗候选胶囊")
        candidates: list[dict[str, object]] = []
        for source in sources:
            capsule = Path(str(source["capsule"]))
            if not capsule.is_absolute():
                capsule = plan_path.parent / capsule
            capsule = capsule.resolve()
            manifest_path = capsule / "capsule.json"
            if not manifest_path.is_file():
                raise RuntimeError(f"K3胶囊缺少capsule.json: {capsule}")
            expected = str(source["manifest_sha256"]).lower()
            actual = sha256_file(manifest_path)
            if len(expected) != 64 or actual != expected:
                raise RuntimeError(f"K3胶囊清单SHA-256不匹配: {capsule}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            capsule_format = manifest.get("format")
            supported_capsules = {
                "colorlm-kimi-k3-latent-macro-capsule-v1",
                "colorlm-kimi-k3-latent-macro-capsule-v3",
                "colorlm-kimi-k3-latent-macro-capsule-v4",
            }
            if plan_format.endswith("v2"):
                supported_capsules = {
                    "colorlm-kimi-k3-latent-macro-capsule-v2",
                    "colorlm-kimi-k3-latent-macro-capsule-v3",
                    "colorlm-kimi-k3-latent-macro-capsule-v4",
                }
            else:
                # 单候选计划可以安全复用带router的v2胶囊；C++加载器会验证
                # router，但单候选执行只消费专家本体，不把router当能力标签。
                supported_capsules.add("colorlm-kimi-k3-latent-macro-capsule-v2")
            if capsule_format not in supported_capsules:
                raise RuntimeError(f"K3胶囊格式不受支持: {capsule}")
            if plan_format.endswith("v2") and not (capsule / "router.f32").is_file():
                raise RuntimeError(f"K3 v2胶囊缺少router.f32: {capsule}")
            candidates.append(
                {
                    "k3_layer": int(manifest["layer"]),
                    "expert": int(manifest["expert"]),
                    "capsule": str(capsule),
                }
            )
        checked.append(
            {
                "site": site,
                "candidates": candidates,
            }
        )
    return plan_path, checked


def require_sha256(value: object, label: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise RuntimeError(f"{label}不是有效SHA-256")
    return digest


def resolve_plan_path(plan_path: Path, value: object, label: str) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = plan_path.parent / path
    path = path.resolve()
    if not path.exists():
        raise RuntimeError(f"找不到{label}: {path}")
    return path


def validate_intelligence_plan(
    root: Path,
    requested: Path,
) -> tuple[Path, list[dict[str, object]], int]:
    plan_path = requested if requested.is_absolute() else root / requested
    plan_path = plan_path.resolve()
    if not plan_path.is_file():
        raise RuntimeError(f"找不到Intelligence Router站点计划: {plan_path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (
        plan.get("format") != "colorlm-intelligence-site-plan-v1"
        or plan.get("status") != "candidate"
        or int(plan.get("input_width", 0)) != 2048
        or plan.get("routes") != ["no_op", "coder", "k3"]
    ):
        raise RuntimeError(f"不支持的Intelligence Router站点计划: {plan_path}")

    sites = plan.get("sites")
    if not isinstance(sites, list) or not sites:
        raise RuntimeError("Intelligence Router站点计划至少需要一个站点")
    checked: list[dict[str, object]] = []
    total_bytes = 0
    seen: set[int] = set()
    for record in sites:
        site = int(record["site"])
        temperature = float(record.get("temperature", 1.0))
        if site in seen or site < 0 or not 0.01 <= temperature <= 4.0:
            raise RuntimeError(f"Intelligence Router站点无效或重复: {site}")
        seen.add(site)
        router = resolve_plan_path(plan_path, record["router"], "Intelligence Router目录")
        if not router.is_dir():
            raise RuntimeError(f"Intelligence Router不是目录: {router}")
        manifest_path = router / "router.json"
        expected_manifest = require_sha256(
            record["manifest_sha256"], "Intelligence Router清单SHA-256"
        )
        if not manifest_path.is_file() or sha256_file(manifest_path) != expected_manifest:
            raise RuntimeError(f"Intelligence Router清单SHA-256不匹配: {router}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("format") != "colorlm-intelligence-router-v1"
            or manifest.get("runtime_layout")
            != "two-headerless-f32-le-row-major-v1"
            or int(manifest.get("input_width", 0)) != 2048
            or int(manifest.get("route_count", 0)) != 3
            or manifest.get("routes") != ["no_op", "coder", "k3"]
        ):
            raise RuntimeError(f"Intelligence Router清单结构不匹配: {router}")
        tensors = manifest.get("tensors")
        expected_tensors = {
            "weight": ("weight.f32", [2048, 3], 2048 * 3 * 4),
            "bias": ("bias.f32", [3], 3 * 4),
        }
        if not isinstance(tensors, dict) or set(tensors) != set(expected_tensors):
            raise RuntimeError(f"Intelligence Router必须包含weight/bias: {router}")
        for name, (filename, shape, size) in expected_tensors.items():
            tensor = tensors[name]
            tensor_path = router / str(tensor["file"])
            expected_sha = require_sha256(
                tensor["sha256"], f"Intelligence Router {name} SHA-256"
            )
            if (
                tensor.get("file") != filename
                or tensor.get("dtype") != "float32-le"
                or tensor.get("ggml_shape") != shape
                or int(tensor.get("bytes", -1)) != size
                or not tensor_path.is_file()
                or tensor_path.stat().st_size != size
                or sha256_file(tensor_path) != expected_sha
            ):
                raise RuntimeError(f"Intelligence Router {name}校验失败: {router}")
            total_bytes += size
        checked.append(
            {
                "site": site,
                "temperature": temperature,
                "router": str(router),
                "manifest_sha256": expected_manifest,
            }
        )
    return plan_path, checked, total_bytes


def validate_coder_capsule(
    capsule: Path,
    expected_manifest_sha256: object,
    expected_weight_bytes: object,
) -> int:
    if not capsule.is_dir():
        raise RuntimeError(f"找不到Coder胶囊: {capsule}")
    manifest_path = capsule / "capsule.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Coder胶囊缺少capsule.json: {capsule}")
    expected_manifest = require_sha256(
        expected_manifest_sha256, "Coder胶囊清单SHA-256"
    )
    if sha256_file(manifest_path) != expected_manifest:
        raise RuntimeError(f"Coder胶囊清单SHA-256不匹配: {capsule}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "colorlm-neural-bus-capsule-v1":
        raise RuntimeError(f"Coder胶囊格式不受支持: {capsule}")
    if (
        int(manifest.get("input_width", 0)) != 2048
        or int(manifest.get("intermediate_width", 0)) != 512
        or int(manifest.get("output_width", 0)) != 2048
        or manifest.get("dtype") != "Q4_0"
        or manifest.get("activation") != "SwiGLU"
    ):
        raise RuntimeError(f"Coder胶囊结构不匹配: {capsule}")

    tensors = manifest.get("tensors")
    expected_shapes = {
        "gate": [2048, 512],
        "up": [2048, 512],
        "down": [512, 2048],
    }
    if not isinstance(tensors, dict) or set(tensors) != set(expected_shapes):
        raise RuntimeError(f"Coder胶囊必须包含gate/up/down三张量: {capsule}")
    total_bytes = 0
    for name, shape in expected_shapes.items():
        record = tensors[name]
        tensor_path = capsule / str(record["file"])
        expected_bytes = int(record["bytes"])
        expected_sha = require_sha256(record["sha256"], f"Coder {name} SHA-256")
        if list(record.get("ggml_shape", [])) != shape:
            raise RuntimeError(f"Coder {name}形状不匹配: {capsule}")
        if not tensor_path.is_file() or tensor_path.stat().st_size != expected_bytes:
            raise RuntimeError(f"Coder {name}尺寸不匹配: {tensor_path}")
        if sha256_file(tensor_path) != expected_sha:
            raise RuntimeError(f"Coder {name} SHA-256不匹配: {tensor_path}")
        total_bytes += expected_bytes
    if total_bytes != int(expected_weight_bytes):
        raise RuntimeError("Coder胶囊权重字节数与Neural Alloy计划不一致")
    return total_bytes


def validate_alloy_plan(
    root: Path,
    requested: Path,
    verify_core_sha256: bool,
) -> dict[str, object]:
    plan_path = requested if requested.is_absolute() else root / requested
    plan_path = plan_path.resolve()
    if not plan_path.is_file():
        raise RuntimeError(f"找不到Neural Alloy计划: {plan_path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan_format = plan.get("format")
    if plan_format not in {
        "colorlm-neural-slice-abi-v1",
        "colorlm-neural-slice-abi-v2",
    }:
        raise RuntimeError(f"不支持的Neural Alloy计划: {plan_path}")

    core = plan.get("core")
    if not isinstance(core, dict):
        raise RuntimeError("Neural Alloy计划缺少core")
    model = resolve_plan_path(plan_path, core["file"], "Neural Alloy核心GGUF")
    core_bytes = int(core["bytes"])
    core_sha256 = require_sha256(core["sha256"], "Neural Alloy核心SHA-256")
    if not model.is_file() or model.stat().st_size != core_bytes:
        raise RuntimeError(f"Neural Alloy核心GGUF尺寸不匹配: {model}")
    if verify_core_sha256 and sha256_file(model) != core_sha256:
        raise RuntimeError(f"Neural Alloy核心GGUF SHA-256不匹配: {model}")
    try:
        model_relative = model.relative_to(root).as_posix()
    except ValueError as error:
        raise RuntimeError("Neural Alloy核心GGUF必须位于项目目录内") from error

    lineage = core.get("lineage")
    if not isinstance(lineage, list) or not lineage:
        raise RuntimeError("Neural Alloy核心缺少供体血统")
    for stage in lineage:
        manifest = resolve_plan_path(plan_path, stage["manifest"], "供体血统清单")
        expected = require_sha256(stage["manifest_sha256"], "供体血统清单SHA-256")
        if not manifest.is_file() or sha256_file(manifest) != expected:
            raise RuntimeError(f"供体血统清单SHA-256不匹配: {manifest}")

    families = plan.get("residual_families")
    if not isinstance(families, list):
        raise RuntimeError("Neural Alloy计划缺少residual_families")
    family_by_abi = {str(family.get("abi")): family for family in families}
    known_abis = {
        "colorlm-neural-bus-q4-swiglu-v1",
        "colorlm-k3-site-plan-v2",
    }
    if plan_format == "colorlm-neural-slice-abi-v2":
        known_abis.add("colorlm-intelligence-router-linear-v1")
    if set(family_by_abi) != known_abis or len(family_by_abi) != len(families):
        raise RuntimeError("Neural Alloy残差ABI集合与计划版本不匹配")

    coder = family_by_abi["colorlm-neural-bus-q4-swiglu-v1"]
    coder_capsule = resolve_plan_path(plan_path, coder["capsule"], "Coder胶囊")
    coder_bytes = validate_coder_capsule(
        coder_capsule,
        coder["capsule_manifest_sha256"],
        coder["weight_bytes"],
    )
    coder_sites = [int(site) for site in coder["sites"]]
    if not coder_sites or len(coder_sites) != len(set(coder_sites)):
        raise RuntimeError("Coder站点不能为空或重复")
    coder_alpha = float(coder["alpha"])
    coder_target_ratio = float(coder["target_ratio"])
    coder_sharpness = float(coder.get("sharpness", 4.0))
    if not 0.0 < coder_alpha <= 1.0 or coder_target_ratio <= 0.0 or coder_sharpness <= 0.0:
        raise RuntimeError("Coder残差参数超出范围")

    k3 = family_by_abi["colorlm-k3-site-plan-v2"]
    k3_plan = resolve_plan_path(plan_path, k3["site_plan"], "K3站点计划")
    expected_k3_plan = require_sha256(k3["site_plan_sha256"], "K3站点计划SHA-256")
    if not k3_plan.is_file() or sha256_file(k3_plan) != expected_k3_plan:
        raise RuntimeError(f"K3站点计划SHA-256不匹配: {k3_plan}")
    k3_plan, k3_sites = validate_k3_plan(root, k3_plan)
    candidate_count = sum(len(site["candidates"]) for site in k3_sites)
    if candidate_count != int(k3["candidate_count"]):
        raise RuntimeError("K3候选数量与Neural Alloy计划不一致")
    k3_bytes = 0
    for site in k3_sites:
        for candidate in site["candidates"]:
            manifest = json.loads(
                (Path(str(candidate["capsule"])) / "capsule.json").read_text(encoding="utf-8")
            )
            k3_bytes += int(manifest["runtime_total_bytes"])
            k3_bytes += int(manifest["runtime_router"]["bytes"])
    if k3_bytes != int(k3["weight_bytes"]):
        raise RuntimeError("K3胶囊权重字节数与Neural Alloy计划不一致")

    intelligence_plan: Path | None = None
    intelligence_sites: list[dict[str, object]] = []
    intelligence_bytes = 0
    if plan_format == "colorlm-neural-slice-abi-v2":
        intelligence = family_by_abi["colorlm-intelligence-router-linear-v1"]
        intelligence_plan = resolve_plan_path(
            plan_path, intelligence["site_plan"], "Intelligence Router站点计划"
        )
        expected_plan = require_sha256(
            intelligence["site_plan_sha256"], "Intelligence Router站点计划SHA-256"
        )
        if sha256_file(intelligence_plan) != expected_plan:
            raise RuntimeError("Intelligence Router站点计划SHA-256不匹配")
        intelligence_plan, intelligence_sites, intelligence_bytes = (
            validate_intelligence_plan(root, intelligence_plan)
        )
        if intelligence_bytes != int(intelligence["weight_bytes"]):
            raise RuntimeError("Intelligence Router权重字节数与Neural Alloy计划不一致")
        declared_sites = [int(site) for site in intelligence["sites"]]
        actual_sites = [int(site["site"]) for site in intelligence_sites]
        k3_site_ids = {int(site["site"]) for site in k3_sites}
        if (
            declared_sites != actual_sites
            or not declared_sites
            or any(
                site not in coder_sites or site not in k3_site_ids
                for site in declared_sites
            )
        ):
            raise RuntimeError("Intelligence Router站点必须是Coder/K3公共站点的非空子集")
        if intelligence.get("routes") != ["no_op", "coder", "k3"]:
            raise RuntimeError("Intelligence Router路由通道顺序不匹配")
        calibration_report = resolve_plan_path(
            plan_path, intelligence["calibration_report"], "路由校准报告"
        )
        expected_report = require_sha256(
            intelligence["calibration_report_sha256"], "路由校准报告SHA-256"
        )
        if not calibration_report.is_file() or sha256_file(calibration_report) != expected_report:
            raise RuntimeError("路由校准报告SHA-256不匹配")

    effective_bytes = int(plan["effective_bytes"])
    if effective_bytes != core_bytes + coder_bytes + k3_bytes + intelligence_bytes:
        raise RuntimeError("Neural Alloy有效字节数计算不一致")

    runtime = plan.get("runtime")
    if not isinstance(runtime, dict) or runtime.get("device") != "Vulkan":
        raise RuntimeError("Neural Alloy v1运行时必须为Vulkan")
    alias = str(runtime["alias"])
    port = int(runtime["port"])
    context = int(runtime["context"])
    if not alias or not 1 <= port <= 65535 or context <= 0:
        raise RuntimeError("Neural Alloy运行时参数无效")

    return {
        "path": plan_path,
        "sha256": sha256_file(plan_path),
        "model": model,
        "model_relative": model_relative,
        "alias": alias,
        "port": port,
        "context": context,
        "coder_capsule": coder_capsule,
        "coder_sites": ",".join(str(site) for site in coder_sites),
        "coder_alpha": coder_alpha,
        "coder_target_ratio": coder_target_ratio,
        "coder_sharpness": coder_sharpness,
        "k3_plan": k3_plan,
        "k3_sites": k3_sites,
        "intelligence_plan": intelligence_plan,
        "intelligence_sites": intelligence_sites,
        "intelligence_bytes": intelligence_bytes,
        "core_sha256_verified": verify_core_sha256,
    }


def server_ready(
    base_url: str,
    alias: str = ALIAS,
    expected_context: int | None = None,
    expected_model: str | None = None,
) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=2) as response:
            if json.load(response).get("status") != "ok":
                return False
        with urllib.request.urlopen(f"{base_url}/v1/models", timeout=2) as response:
            models = json.load(response).get("data", [])
        if not any(model.get("id") == alias for model in models):
            return False
        if expected_context is not None or expected_model is not None:
            with urllib.request.urlopen(f"{base_url}/props", timeout=2) as response:
                props = json.load(response)
            if expected_context is not None and int(
                props.get("default_generation_settings", {}).get("n_ctx", -1)
            ) != expected_context:
                return False
            if expected_model is not None:
                actual_model = str(props.get("model_path", "")).replace("\\", "/").lower()
                if actual_model != expected_model.replace("\\", "/").lower():
                    return False
        return True
    except Exception:
        return False


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    runtime = root / "fast16" / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    server = args.server or root / "build" / "bin" / "Release" / "llama-server.exe"
    if not server.is_absolute():
        server = root / server

    alloy: dict[str, object] | None = None
    if args.alloy_plan is not None and args.model is not None:
        print("--model不能与--alloy-plan同时使用", file=sys.stderr)
        return 1
    if args.alloy_plan is not None:
        if args.neural_bus_v2:
            print("Neural Alloy v1不允许覆盖为Neural Bus v2", file=sys.stderr)
            return 1
        try:
            alloy = validate_alloy_plan(root, args.alloy_plan, args.verify_alloy_core_sha256)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, RuntimeError) as error:
            print(str(error), file=sys.stderr)
            return 1
        args.neural_bus = True
        args.k3_plan = Path(str(alloy["k3_plan"]))
        alloy_intelligence = alloy.get("intelligence_plan")
        if alloy_intelligence is not None:
            alloy_intelligence_path = Path(str(alloy_intelligence))
            if (
                args.intelligence_plan is not None
                and args.intelligence_plan.resolve() != alloy_intelligence_path.resolve()
            ):
                print("Neural Alloy计划不允许覆盖Intelligence Router", file=sys.stderr)
                return 1
            args.intelligence_plan = alloy_intelligence_path
        if args.neural_bus_alpha is None:
            args.neural_bus_alpha = float(alloy["coder_alpha"])
        args.neural_bus_sites = str(alloy["coder_sites"])
        args.neural_bus_target_ratio = float(alloy["coder_target_ratio"])
        args.neural_bus_sharpness = float(alloy["coder_sharpness"])
        if args.port is None:
            args.port = int(alloy["port"])
        if args.ctx_size is None:
            args.ctx_size = int(alloy["context"])
        if args.runtime_alias is None:
            args.runtime_alias = str(alloy["alias"])

    if args.port is None:
        args.port = DEFAULT_PORT
    if args.ctx_size is None:
        args.ctx_size = DEFAULT_CTX_SIZE
    if args.neural_bus_alpha is None:
        args.neural_bus_alpha = 0.08
    if not 1 <= args.port <= 65535 or args.ctx_size <= 0:
        print("--port或--ctx-size超出范围", file=sys.stderr)
        return 1
    if not 0.0 <= args.neural_bus_alpha <= 1.0:
        print("--neural-bus-alpha必须在[0, 1]内", file=sys.stderr)
        return 1
    if not 0.0 <= args.neural_block_alpha <= 1.0:
        print("--neural-block-alpha必须在[0, 1]内", file=sys.stderr)
        return 1
    if not 0 <= args.neural_block_site < COLORLM_LAYER_COUNT:
        print("--neural-block-site超出ColorLM层范围", file=sys.stderr)
        return 1
    if not 0.0001 <= args.neural_block_target_ratio <= 1.0:
        print("--neural-block-target-ratio必须在[0.0001, 1]内", file=sys.stderr)
        return 1
    if not 0.25 <= args.neural_block_sharpness <= 16.0:
        print("--neural-block-sharpness必须在[0.25, 16]内", file=sys.stderr)
        return 1
    if not 0.0 <= args.neural_island_alpha <= 1.0:
        print("--neural-island-alpha必须在[0, 1]内", file=sys.stderr)
        return 1
    if not 0 <= args.neural_island_site < COLORLM_LAYER_COUNT:
        print("--neural-island-site超出ColorLM层范围", file=sys.stderr)
        return 1
    if not 0.0001 <= args.neural_island_target_ratio <= 1.0:
        print("--neural-island-target-ratio必须在[0.0001, 1]内", file=sys.stderr)
        return 1
    if not 0.25 <= args.neural_island_sharpness <= 16.0:
        print("--neural-island-sharpness必须在[0.25, 16]内", file=sys.stderr)
        return 1
    if args.neural_island_expert_cache_slots != 0 and not (
        10 <= args.neural_island_expert_cache_slots <= 512
        ):
        print(
            "--neural-island-expert-cache-slots必须为0或[10, 512]内的整数",
            file=sys.stderr,
        )
        return 1
    if args.neural_island_expert_cache_slots == 0 and args.neural_island_expert_cache_policy != "lru":
        print("关闭专家缓存时策略必须为lru", file=sys.stderr)
        return 1
    if not 1 <= args.neural_island_route_dump_max_records <= 100000:
        print("--neural-island-route-dump-max-records必须在[1, 100000]内", file=sys.stderr)
        return 1
    if not 1 <= args.neural_island_teacher_dump_max_records <= 100000:
        print("--neural-island-teacher-dump-max-records必须在[1, 100000]内", file=sys.stderr)
        return 1
    if not 1 <= args.neural_island_stage_dump_max_records <= 100000:
        print("--neural-island-stage-dump-max-records必须在[1, 100000]内", file=sys.stderr)
        return 1
    if not 1 <= args.neural_island_l47_moe_dump_max_records <= 100000:
        print("--neural-island-l47-moe-dump-max-records必须在[1, 100000]内", file=sys.stderr)
        return 1
    if not 0.0 <= args.neural_output_head_alpha <= 1.0:
        print("--neural-output-head-alpha必须在[0, 1]内", file=sys.stderr)
        return 1
    if not 1 <= args.neural_output_capture_max_records <= 100000:
        print("--neural-output-capture-max-records必须在[1, 100000]内", file=sys.stderr)
        return 1
    if not 1 <= args.batch_size <= 2048:
        print("--batch-size必须在[1, 2048]内", file=sys.stderr)
        return 1
    if not 1 <= args.ubatch_size <= args.batch_size:
        print("--ubatch-size必须在[1, batch-size]内", file=sys.stderr)
        return 1
    if not 1 <= args.anthropic_max_tokens <= 65536:
        print("--anthropic-max-tokens必须在[1, 65536]内", file=sys.stderr)
        return 1
    if not (
        2 <= args.spec_ngram_match <= 256
        and 1 <= args.spec_ngram_min <= args.spec_ngram_max <= 256
    ):
        print("ngram推测参数必须满足match>=2且1<=min<=max<=256", file=sys.stderr)
        return 1
    if args.k3_alpha is not None and not 0.0 <= args.k3_alpha <= 1.0:
        print("--k3-alpha必须在[0, 1]内", file=sys.stderr)
        return 1
    if args.hidden_dump_max_records <= 0:
        print("--hidden-dump-max-records必须为正数", file=sys.stderr)
        return 1
    if args.k3_feature_dump_max_records <= 0:
        print("--k3-feature-dump-max-records必须为正数", file=sys.stderr)
        return 1
    try:
        force_site_paths_encoded, force_site_paths = parse_force_site_paths(
            args.force_site_paths
        )
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1

    neural_bus_requested = args.neural_bus or args.neural_bus_v2
    neural_bus_enabled = neural_bus_requested and args.neural_bus_alpha != 0.0
    neural_block_enabled = (
        args.neural_block_package is not None and args.neural_block_alpha != 0.0
    )
    neural_island_enabled = (
        args.neural_island_manifest is not None and args.neural_island_alpha != 0.0
    )
    if args.neural_island_route_dump is not None and not neural_island_enabled:
        print("专家路由dump要求同时启用非零alpha的Neural Island", file=sys.stderr)
        return 1
    if args.neural_island_teacher_dump is not None and not neural_island_enabled:
        print("教师dump要求同时启用非零alpha的Neural Island", file=sys.stderr)
        return 1
    if args.neural_island_stage_dump is not None and not neural_island_enabled:
        print("分层教师dump要求同时启用非零alpha的Neural Island", file=sys.stderr)
        return 1
    if args.neural_island_l47_moe_dump is not None and not neural_island_enabled:
        print("L47 MoE教师dump要求同时启用非零alpha的Neural Island", file=sys.stderr)
        return 1
    if args.neural_island_micro_l47_package is not None and not neural_island_enabled:
        print("L47微层要求同时启用非零alpha的Neural Island", file=sys.stderr)
        return 1
    if args.neural_island_micro_l47_moe_package is not None and not neural_island_enabled:
        print("L47微MoE要求同时启用非零alpha的Neural Island", file=sys.stderr)
        return 1
    if (
        args.neural_island_micro_l47_package is not None
        and args.neural_island_stage_dump is not None
    ):
        print("L47微层运行候选不能与全层教师dump同时启用", file=sys.stderr)
        return 1
    if (
        args.neural_island_micro_l47_package is not None
        and args.neural_island_micro_l47_moe_package is not None
    ):
        print("完整L47微层与L47微MoE候选互斥", file=sys.stderr)
        return 1
    if (
        args.neural_island_l47_moe_dump is not None
        and (
            args.neural_island_stage_dump is not None
            or args.neural_island_teacher_dump is not None
            or args.neural_island_micro_l47_package is not None
            or args.neural_island_micro_l47_moe_package is not None
        )
    ):
        print("L47 MoE教师dump必须独占微岛研究采集路径", file=sys.stderr)
        return 1
    if not 0 <= args.neural_island_teacher_dump_site < COLORLM_LAYER_COUNT:
        print("--neural-island-teacher-dump-site超出ColorLM层范围", file=sys.stderr)
        return 1
    neural_output_head_enabled = (
        args.neural_output_head_package is not None
        and args.neural_output_head_alpha != 0.0
    )
    sequence_policy_enabled = args.sequence_policy_package is not None
    if neural_output_head_enabled and not neural_island_enabled:
        print("末端双输出头要求同时启用Neural Island", file=sys.stderr)
        return 1
    k3_enabled = args.k3_plan is not None and args.k3_alpha != 0.0
    if alloy is not None:
        model = Path(str(alloy["model"]))
        model_relative = str(alloy["model_relative"])
    elif args.model is not None:
        model = args.model if args.model.is_absolute() else root / args.model
        model = model.resolve()
        try:
            model_relative = os.fspath(model.relative_to(root))
        except ValueError:
            model_relative = os.fspath(model)
    else:
        model = root / "fast16" / "models" / "ColorLM-v6-Q3Router-Fused-A1.gguf"
        model_relative = "fast16/models/ColorLM-v6-Q3Router-Fused-A1.gguf"
    for path, label in ((server, "llama-server"), (model, "核心GGUF")):
        if not path.is_file():
            print(f"找不到{label}: {path}", file=sys.stderr)
            return 1

    neural_block_package: Path | None = None
    if neural_block_enabled:
        neural_block_package = args.neural_block_package
        if not neural_block_package.is_absolute():
            neural_block_package = root / neural_block_package
        manifest = neural_block_package / "block.json"
        weights = neural_block_package / "weights.bin"
        if not manifest.is_file() or not weights.is_file():
            print(f"Neural Block运行包不完整: {neural_block_package}", file=sys.stderr)
            return 1

    neural_island_manifest: Path | None = None
    neural_island_contract: dict[str, object] | None = None
    if neural_island_enabled:
        neural_island_manifest = args.neural_island_manifest
        if not neural_island_manifest.is_absolute():
            neural_island_manifest = root / neural_island_manifest
        neural_island_manifest = neural_island_manifest.resolve()
        try:
            neural_island_contract = json.loads(
                neural_island_manifest.read_text(encoding="utf-8")
            )
            if (
                neural_island_contract.get("format")
                != "colorlm-neural-island-runtime-v1"
                or neural_island_contract.get("formal") is not True
                or neural_island_contract.get("source_layers") != [44, 45, 46, 47]
                or int(neural_island_contract.get("target_site", -1))
                != args.neural_island_site
            ):
                raise RuntimeError("神经岛运行契约不匹配")
        except (OSError, ValueError, TypeError, json.JSONDecodeError, RuntimeError) as error:
            print(f"Neural Island清单无效: {error}", file=sys.stderr)
            return 1

    neural_output_head_package: Path | None = None
    neural_output_head_manifest_sha256: str | None = None
    if neural_output_head_enabled:
        neural_output_head_package = args.neural_output_head_package
        if not neural_output_head_package.is_absolute():
            neural_output_head_package = root / neural_output_head_package
        neural_output_head_package = neural_output_head_package.resolve()
        head_manifest = neural_output_head_package / "head.json"
        head_weights = neural_output_head_package / "weights.bin"
        try:
            head_contract = json.loads(head_manifest.read_text(encoding="utf-8"))
            if (
                head_contract.get("format")
                != "colorlm-neural-output-head-runtime-v2"
                or head_contract.get("formal") is not True
                or int(head_contract.get("source", {}).get("vocab_size", -1))
                != 151936
                or int(head_contract.get("target", {}).get("vocab_size", -1))
                != 248320
                or head_contract.get("mapping", {}).get("method")
                != "exact-tokenizer.ggml.tokens-raw-bytes"
                or int(head_contract.get("mapping", {}).get("target_collisions", -1))
                != 0
                or head_contract.get("mapping", {}).get("projection_layout")
                != "mapped-only-q6-k-raw-rows"
                or int(head_contract.get("mapping", {}).get("source_row_bytes", -1))
                != 1680
                or not head_weights.is_file()
                or head_weights.stat().st_size
                != int(head_contract.get("weights", {}).get("bytes", -1))
            ):
                raise RuntimeError("末端输出头运行契约不匹配")
            neural_output_head_manifest_sha256 = hashlib.sha256(
                head_manifest.read_bytes()
            ).hexdigest()
        except (OSError, ValueError, TypeError, json.JSONDecodeError, RuntimeError) as error:
            print(f"Neural Output Head运行包无效: {error}", file=sys.stderr)
            return 1

    sequence_policy_package: Path | None = None
    sequence_policy_manifest_sha256: str | None = None
    if sequence_policy_enabled:
        sequence_policy_package = args.sequence_policy_package
        if not sequence_policy_package.is_absolute():
            sequence_policy_package = root / sequence_policy_package
        sequence_policy_package = sequence_policy_package.resolve()
        policy_manifest = sequence_policy_package / "policy.json"
        policy_weights = sequence_policy_package / "weights.bin"
        try:
            policy_contract = json.loads(policy_manifest.read_text(encoding="utf-8"))
            policy_format = policy_contract.get("format")
            static_v1 = policy_format == "colorlm-sequence-policy-runtime-v1"
            dynamic_v2 = policy_format == "colorlm-sequence-policy-runtime-v2"
            pca_noop_v3 = policy_format == "colorlm-sequence-policy-runtime-v3"
            common_valid = (
                policy_contract.get("formal") is True
                and policy_contract.get("architecture") == "ColorLMV4"
                and int(policy_contract.get("hidden_size", -1)) == 2048
                and int(policy_contract.get("base_vocab_size", -1)) == 248320
                and policy_weights.is_file()
                and policy_weights.stat().st_size
                == int(policy_contract.get("weights", {}).get("bytes", -1))
            )
            static_valid = (
                static_v1
                and 1 <= int(policy_contract.get("candidate_token_count", -1)) <= 256
                and policy_contract.get("normalization") == "per-sample-l2"
            )
            dynamic_valid = (
                dynamic_v2
                and policy_contract.get("mode") == "dynamic-low-rank-bilinear"
                and int(policy_contract.get("embedding_size", -1)) == 2048
                and 1 <= int(policy_contract.get("rank", -1)) <= 256
                and 1 <= int(policy_contract.get("static_token_count", -1))
                < int(policy_contract.get("candidate_capacity", -1)) <= 256
                and policy_contract.get("hidden_normalization") == "per-sample-l2"
                and policy_contract.get("embedding_normalization") == "per-token-l2"
            )
            candidate_count = int(policy_contract.get("candidate_token_count", -1))
            pca_noop_valid = (
                pca_noop_v3
                and policy_contract.get("mode")
                == "pca-rank8-multiclass-noop-sparse"
                and 4 <= candidate_count <= 16
                and int(policy_contract.get("class_count", -1))
                == candidate_count + 1
                and int(policy_contract.get("pca_rank", -1)) == 8
                and int(policy_contract.get("no_op_class", -1)) == 0
                and policy_contract.get("no_op_rule")
                == "exact-no-op-iff-class-0-is-argmax"
                and policy_contract.get("hidden_normalization")
                == "per-sample-l2"
                and int(policy_contract.get("tensor_count", -1)) == 6
            )
            if not common_valid or not (static_valid or dynamic_valid or pca_noop_valid):
                raise RuntimeError("序列策略头运行契约不匹配")
            if pca_noop_v3:
                expected_weights_sha = str(
                    policy_contract.get("weights", {}).get("sha256", "")
                )
                if sha256_file(policy_weights) != expected_weights_sha:
                    raise RuntimeError("v43策略头weights.bin SHA-256不匹配")
            sequence_policy_manifest_sha256 = hashlib.sha256(
                policy_manifest.read_bytes()
            ).hexdigest()
        except (OSError, ValueError, TypeError, json.JSONDecodeError, RuntimeError) as error:
            print(f"Sequence Policy运行包无效: {error}", file=sys.stderr)
            return 1

    k3_plan_path: Path | None = None
    k3_sites: list[dict[str, object]] = []
    if alloy is not None:
        k3_plan_path = Path(str(alloy["k3_plan"]))
        k3_sites = list(alloy["k3_sites"])
    elif args.k3_plan is not None:
        try:
            k3_plan_path, k3_sites = validate_k3_plan(root, args.k3_plan)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, RuntimeError) as error:
            print(str(error), file=sys.stderr)
            return 1

    intelligence_plan_path: Path | None = None
    intelligence_sites: list[dict[str, object]] = []
    intelligence_bytes = 0
    if args.intelligence_plan is not None:
        try:
            intelligence_plan_path, intelligence_sites, intelligence_bytes = (
                validate_intelligence_plan(root, args.intelligence_plan)
            )
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, RuntimeError) as error:
            print(str(error), file=sys.stderr)
            return 1
    intelligence_enabled = (
        intelligence_plan_path is not None and neural_bus_enabled and k3_enabled
    )
    if intelligence_plan_path is not None and not intelligence_enabled:
        print(
            "Intelligence Router需要同时启用Coder与K3；关闭任一alpha时路由器自动旁路。",
            file=sys.stderr,
        )

    capsule_paths = (
        (Path(str(alloy["coder_capsule"])),)
        if alloy is not None
        else (
            (root / NEURAL_BUS_V2_PRIMARY, root / NEURAL_BUS_V2_SECONDARY)
            if args.neural_bus_v2
            else (root / NEURAL_BUS_V1_CAPSULE,)
        )
    )
    if neural_bus_enabled and any(not path.is_dir() for path in capsule_paths):
        print(
            "找不到Neural Bus胶囊: "
            + ", ".join(str(path) for path in capsule_paths)
            + "\n请先运行对应的Neural Bus胶囊构建器",
            file=sys.stderr,
        )
        return 1

    default_alias = (
        NEURAL_BUS_V2_ALIAS
        if args.neural_bus_v2 and neural_bus_enabled
        else NEURAL_BUS_ALIAS
        if args.neural_bus and neural_bus_enabled
        else K3_NEURAL_BUS_ALIAS
        if k3_enabled
        else ALIAS
    )
    alias = args.runtime_alias or default_alias
    base_url = f"http://127.0.0.1:{args.port}"
    if (
        not args.validate_only
        and args.force_path == "auto"
        and not force_site_paths
        and args.hidden_dump is None
        and args.neural_output_capture is None
        and server_ready(base_url, alias, args.ctx_size, model_relative)
    ):
        print(f"Fast16 Runtime v1 已就绪: {base_url}")
        return 0

    router_bias = args.neural_bus_router_bias
    router_floor = args.neural_bus_router_floor
    if args.neural_bus_v2 and (router_bias is None or router_floor is None):
        report_path = root / NEURAL_BUS_V2_REPORT
        if not report_path.is_file():
            print(f"找不到Neural Bus v2校准报告: {report_path}", file=sys.stderr)
            return 1
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if router_bias is None:
            router_bias = float(report["router"]["recommended_bias"])
        if router_floor is None:
            router_floor = float(report["router"]["no_op"]["threshold"])

    command = [
        str(server),
        "--model",
        model_relative,
        "--alias",
        alias,
        "--n-gpu-layers",
        "99",
        "--n-cpu-moe",
        str(CPU_MOE_LAYERS),
        "--threads",
        str(args.threads),
        "--ctx-size",
        str(args.ctx_size),
        "--parallel",
        "1",
        "--batch-size",
        str(args.batch_size),
        "--ubatch-size",
        str(args.ubatch_size),
        "--cache-ram",
        "0",
        "--ctx-checkpoints",
        "0",
        "--spec-type",
        args.spec_type,
        "--flash-attn",
        "on",
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "q8_0",
        "--jinja",
        "--reasoning",
        "off",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
    ]
    if not args.allow_mmap:
        command.append("--no-mmap")
    if args.spec_type == "ngram-mod":
        command.extend(
            [
                "--spec-ngram-mod-n-match",
                str(args.spec_ngram_match),
                "--spec-ngram-mod-n-min",
                str(args.spec_ngram_min),
                "--spec-ngram-mod-n-max",
                str(args.spec_ngram_max),
            ]
        )
    if args.no_warmup:
        command.append("--no-warmup")
    if k3_plan_path is not None or neural_block_enabled or neural_island_enabled:
        # Side-loaded capsules, blocks, islands, and their expert caches are
        # allocated after the base-model placement decision.
        # llama.cpp's automatic fit probe builds transient graphs and cannot
        # account for these buffers reliably, so preserve the known v6 split.
        command.extend(["--fit", "off"])

    environment = os.environ.copy()
    if args.spin_fence:
        environment["GGML_VK_SPIN_FENCE"] = "1"
    else:
        environment.pop("GGML_VK_SPIN_FENCE", None)
    if args.no_merge_cpu_sync:
        environment.pop("GGML_SCHED_MERGE_CPU_SYNC", None)
    else:
        environment["GGML_SCHED_MERGE_CPU_SYNC"] = "1"
    if args.no_skip_cpu_final_sync:
        environment.pop("GGML_SCHED_SKIP_CPU_FINAL_SYNC", None)
    else:
        environment["GGML_SCHED_SKIP_CPU_FINAL_SYNC"] = "1"
    if args.no_batch_cpu_read:
        environment.pop("GGML_SCHED_BATCH_CPU_READ", None)
    else:
        environment["GGML_SCHED_BATCH_CPU_READ"] = "1"
    if args.slot_pool:
        environment["COLORLM_MOE_SLOT_POOL"] = "1"
        environment["COLORLM_MOE_SLOT_LAYERS"] = str(SLOT_POOL_LAYERS)
        environment["COLORLM_MOE_SLOT_K"] = str(SLOT_POOL_K)
    else:
        environment.pop("COLORLM_MOE_SLOT_POOL", None)
        environment.pop("COLORLM_MOE_SLOT_LAYERS", None)
        environment.pop("COLORLM_MOE_SLOT_K", None)
    environment.pop("COLORLM_ALLOY_Q3_PATH", None)
    environment.pop("COLORLM_ALLOY_ALPHA", None)
    for name in (
        "COLORLM_NEURAL_BUS_CAPSULE",
        "COLORLM_NEURAL_BUS_CAPSULE_2",
        "COLORLM_NEURAL_BUS_ALPHA",
        "COLORLM_NEURAL_BUS_SITES",
        "COLORLM_NEURAL_BUS_TARGET_RATIO",
        "COLORLM_NEURAL_BUS_SHARPNESS",
        "COLORLM_NEURAL_BUS_ROUTER_BIAS",
        "COLORLM_NEURAL_BUS_ROUTER_FLOOR",
        "COLORLM_NEURAL_BUS_TRACE",
        "COLORLM_NEURAL_BUS_K3_PLAN",
        "COLORLM_NEURAL_BUS_K3_CAPSULE",
        "COLORLM_NEURAL_BUS_K3_ALPHA",
        "COLORLM_NEURAL_BUS_K3_SITES",
        "COLORLM_NEURAL_BUS_K3_TRACE",
        "COLORLM_NEURAL_BUS_INTELLIGENCE_PLAN",
        "COLORLM_NEURAL_BUS_INTELLIGENCE_TRACE",
        "COLORLM_NEURAL_BUS_FORCE_PATH",
        "COLORLM_NEURAL_BUS_FORCE_PATH_BY_SITE",
        "COLORLM_HIDDEN_DUMP",
        "COLORLM_HIDDEN_DUMP_SITES",
        "COLORLM_HIDDEN_DUMP_MAX_RECORDS",
        "COLORLM_NEURAL_BLOCK_ALPHA",
        "COLORLM_NEURAL_BLOCK_SITE",
        "COLORLM_NEURAL_BLOCK_PACKAGE",
        "COLORLM_NEURAL_BLOCK_TARGET_RATIO",
        "COLORLM_NEURAL_BLOCK_SHARPNESS",
        "COLORLM_NEURAL_BLOCK_VERIFY_WEIGHTS",
        "COLORLM_NEURAL_ISLAND_ALPHA",
        "COLORLM_NEURAL_ISLAND_SITE",
        "COLORLM_NEURAL_ISLAND_MANIFEST",
        "COLORLM_NEURAL_ISLAND_TARGET_RATIO",
        "COLORLM_NEURAL_ISLAND_SHARPNESS",
        "COLORLM_NEURAL_ISLAND_EXPERT_CACHE_SLOTS",
        "COLORLM_NEURAL_ISLAND_EXPERT_CACHE_POLICY",
        "COLORLM_NEURAL_ISLAND_VERIFY_WEIGHTS",
        "COLORLM_NEURAL_ISLAND_ROUTE_DUMP",
        "COLORLM_NEURAL_ISLAND_ROUTE_DUMP_MAX_RECORDS",
        "COLORLM_NEURAL_ISLAND_TEACHER_DUMP",
        "COLORLM_NEURAL_ISLAND_TEACHER_DUMP_MAX_RECORDS",
        "COLORLM_NEURAL_ISLAND_TEACHER_DUMP_SITE",
        "COLORLM_NEURAL_ISLAND_STAGE_DUMP",
        "COLORLM_NEURAL_ISLAND_STAGE_DUMP_MAX_RECORDS",
        "COLORLM_NEURAL_ISLAND_L47_MOE_DUMP",
        "COLORLM_NEURAL_ISLAND_L47_MOE_DUMP_MAX_RECORDS",
        "COLORLM_NEURAL_ISLAND_MICRO_L47_PACKAGE",
        "COLORLM_NEURAL_ISLAND_MICRO_L47_VERIFY_WEIGHTS",
        "COLORLM_NEURAL_ISLAND_MICRO_L47_MOE_PACKAGE",
        "COLORLM_NEURAL_ISLAND_MICRO_L47_MOE_VERIFY_WEIGHTS",
        "COLORLM_NEURAL_OUTPUT_HEAD_ALPHA",
        "COLORLM_NEURAL_OUTPUT_HEAD_PACKAGE",
        "COLORLM_NEURAL_OUTPUT_HEAD_MANIFEST_SHA256",
        "COLORLM_NEURAL_OUTPUT_HEAD_VERIFY_WEIGHTS",
        "COLORLM_SEQUENCE_POLICY_PACKAGE",
        "COLORLM_SEQUENCE_POLICY_MANIFEST_SHA256",
        "COLORLM_SEQUENCE_POLICY_VERIFY_WEIGHTS",
        "COLORLM_NEURAL_OUTPUT_CAPTURE",
        "COLORLM_NEURAL_OUTPUT_CAPTURE_ARM",
        "COLORLM_NEURAL_OUTPUT_CAPTURE_MAX_RECORDS",
        "COLORLM_ANTHROPIC_MAX_TOKENS",
    ):
        environment.pop(name, None)
    environment["COLORLM_ANTHROPIC_MAX_TOKENS"] = str(args.anthropic_max_tokens)
    if neural_bus_enabled:
        environment.update(
            {
                "COLORLM_NEURAL_BUS_CAPSULE": str(capsule_paths[0]),
                "COLORLM_NEURAL_BUS_ALPHA": str(args.neural_bus_alpha),
                "COLORLM_NEURAL_BUS_SITES": args.neural_bus_sites,
                "COLORLM_NEURAL_BUS_TARGET_RATIO": str(args.neural_bus_target_ratio),
                "COLORLM_NEURAL_BUS_SHARPNESS": str(args.neural_bus_sharpness),
            }
        )
        if args.neural_bus_v2:
            environment["COLORLM_NEURAL_BUS_CAPSULE_2"] = str(capsule_paths[1])
            environment["COLORLM_NEURAL_BUS_ROUTER_BIAS"] = str(router_bias)
            environment["COLORLM_NEURAL_BUS_ROUTER_FLOOR"] = str(router_floor)
        if args.neural_bus_trace:
            environment["COLORLM_NEURAL_BUS_TRACE"] = "1"
    if neural_block_enabled:
        environment.update(
            {
                "COLORLM_NEURAL_BLOCK_ALPHA": str(args.neural_block_alpha),
                "COLORLM_NEURAL_BLOCK_SITE": str(args.neural_block_site),
                "COLORLM_NEURAL_BLOCK_PACKAGE": str(neural_block_package),
                "COLORLM_NEURAL_BLOCK_TARGET_RATIO": str(
                    args.neural_block_target_ratio
                ),
                "COLORLM_NEURAL_BLOCK_SHARPNESS": str(args.neural_block_sharpness),
            }
        )
        if args.verify_neural_block_weights:
            environment["COLORLM_NEURAL_BLOCK_VERIFY_WEIGHTS"] = "1"
    if neural_island_enabled:
        environment.update(
            {
                "COLORLM_NEURAL_ISLAND_ALPHA": str(args.neural_island_alpha),
                "COLORLM_NEURAL_ISLAND_SITE": str(args.neural_island_site),
                "COLORLM_NEURAL_ISLAND_MANIFEST": str(neural_island_manifest),
                "COLORLM_NEURAL_ISLAND_TARGET_RATIO": str(
                    args.neural_island_target_ratio
                ),
                "COLORLM_NEURAL_ISLAND_SHARPNESS": str(
                    args.neural_island_sharpness
                ),
                "COLORLM_NEURAL_ISLAND_EXPERT_CACHE_SLOTS": str(
                    args.neural_island_expert_cache_slots
                ),
                "COLORLM_NEURAL_ISLAND_EXPERT_CACHE_POLICY": (
                    args.neural_island_expert_cache_policy
                ),
            }
        )
        if args.verify_neural_island_weights:
            environment["COLORLM_NEURAL_ISLAND_VERIFY_WEIGHTS"] = "1"
        if args.neural_island_route_dump is not None:
            route_dump = args.neural_island_route_dump
            if not route_dump.is_absolute():
                route_dump = root / route_dump
            route_dump = route_dump.resolve()
            route_dump.parent.mkdir(parents=True, exist_ok=True)
            if route_dump.exists():
                print(f"神经岛路由dump已存在，拒绝覆盖: {route_dump}", file=sys.stderr)
                return 1
            environment["COLORLM_NEURAL_ISLAND_ROUTE_DUMP"] = str(route_dump)
            environment["COLORLM_NEURAL_ISLAND_ROUTE_DUMP_MAX_RECORDS"] = str(
                args.neural_island_route_dump_max_records
            )
        if args.neural_island_teacher_dump is not None:
            teacher_dump = args.neural_island_teacher_dump
            if not teacher_dump.is_absolute():
                teacher_dump = root / teacher_dump
            teacher_dump = teacher_dump.resolve()
            teacher_dump.parent.mkdir(parents=True, exist_ok=True)
            if teacher_dump.exists():
                print(f"神经岛教师dump已存在，拒绝覆盖: {teacher_dump}", file=sys.stderr)
                return 1
            environment["COLORLM_NEURAL_ISLAND_TEACHER_DUMP"] = str(teacher_dump)
            environment["COLORLM_NEURAL_ISLAND_TEACHER_DUMP_MAX_RECORDS"] = str(
                args.neural_island_teacher_dump_max_records
            )
            environment["COLORLM_NEURAL_ISLAND_TEACHER_DUMP_SITE"] = str(
                args.neural_island_teacher_dump_site
            )
        if args.neural_island_stage_dump is not None:
            stage_dump = args.neural_island_stage_dump
            if not stage_dump.is_absolute():
                stage_dump = root / stage_dump
            stage_dump = stage_dump.resolve()
            stage_dump.parent.mkdir(parents=True, exist_ok=True)
            if stage_dump.exists():
                print(f"神经岛分层dump已存在，拒绝覆盖: {stage_dump}", file=sys.stderr)
                return 1
            environment["COLORLM_NEURAL_ISLAND_STAGE_DUMP"] = str(stage_dump)
            environment["COLORLM_NEURAL_ISLAND_STAGE_DUMP_MAX_RECORDS"] = str(
                args.neural_island_stage_dump_max_records
            )
        if args.neural_island_l47_moe_dump is not None:
            moe_dump = args.neural_island_l47_moe_dump
            if not moe_dump.is_absolute():
                moe_dump = root / moe_dump
            moe_dump = moe_dump.resolve()
            moe_dump.parent.mkdir(parents=True, exist_ok=True)
            if moe_dump.exists():
                print(f"L47 MoE教师dump已存在，拒绝覆盖: {moe_dump}", file=sys.stderr)
                return 1
            environment["COLORLM_NEURAL_ISLAND_L47_MOE_DUMP"] = str(moe_dump)
            environment["COLORLM_NEURAL_ISLAND_L47_MOE_DUMP_MAX_RECORDS"] = str(
                args.neural_island_l47_moe_dump_max_records
            )
        if args.neural_island_micro_l47_package is not None:
            micro_package = args.neural_island_micro_l47_package
            if not micro_package.is_absolute():
                micro_package = root / micro_package
            micro_package = micro_package.resolve()
            if not (micro_package / "micro_stage.json").is_file() or not (
                micro_package / "weights.bin"
            ).is_file():
                print(f"L47微层运行包不完整: {micro_package}", file=sys.stderr)
                return 1
            environment["COLORLM_NEURAL_ISLAND_MICRO_L47_PACKAGE"] = str(
                micro_package
            )
            if args.verify_neural_island_micro_l47_weights:
                environment["COLORLM_NEURAL_ISLAND_MICRO_L47_VERIFY_WEIGHTS"] = "1"
        if args.neural_island_micro_l47_moe_package is not None:
            micro_moe_package = args.neural_island_micro_l47_moe_package
            if not micro_moe_package.is_absolute():
                micro_moe_package = root / micro_moe_package
            micro_moe_package = micro_moe_package.resolve()
            if not (micro_moe_package / "micro_stage.json").is_file() or not (
                micro_moe_package / "weights.bin"
            ).is_file():
                print(f"L47微MoE运行包不完整: {micro_moe_package}", file=sys.stderr)
                return 1
            environment["COLORLM_NEURAL_ISLAND_MICRO_L47_MOE_PACKAGE"] = str(
                micro_moe_package
            )
            if args.verify_neural_island_micro_l47_moe_weights:
                environment[
                    "COLORLM_NEURAL_ISLAND_MICRO_L47_MOE_VERIFY_WEIGHTS"
                ] = "1"
    if neural_output_head_enabled:
        environment.update(
            {
                "COLORLM_NEURAL_OUTPUT_HEAD_ALPHA": str(
                    args.neural_output_head_alpha
                ),
                "COLORLM_NEURAL_OUTPUT_HEAD_PACKAGE": str(
                    neural_output_head_package
                ),
                "COLORLM_NEURAL_OUTPUT_HEAD_MANIFEST_SHA256": str(
                    neural_output_head_manifest_sha256
                ),
            }
        )
        if args.verify_neural_output_head_weights:
            environment["COLORLM_NEURAL_OUTPUT_HEAD_VERIFY_WEIGHTS"] = "1"
    if sequence_policy_enabled:
        environment.update(
            {
                "COLORLM_SEQUENCE_POLICY_PACKAGE": str(sequence_policy_package),
                "COLORLM_SEQUENCE_POLICY_MANIFEST_SHA256": str(
                    sequence_policy_manifest_sha256
                ),
            }
        )
        if args.verify_sequence_policy_weights:
            environment["COLORLM_SEQUENCE_POLICY_VERIFY_WEIGHTS"] = "1"
    if args.neural_output_capture is not None:
        capture_path = args.neural_output_capture
        if not capture_path.is_absolute():
            capture_path = root / capture_path
        capture_path = capture_path.resolve()
        capture_path.parent.mkdir(parents=True, exist_ok=True)
        capture_arm = capture_path.with_suffix(capture_path.suffix + ".arm")
        if capture_arm.exists():
            print(f"输出采集arm文件已存在，拒绝启动: {capture_arm}", file=sys.stderr)
            return 1
        environment["COLORLM_NEURAL_OUTPUT_CAPTURE"] = str(capture_path)
        environment["COLORLM_NEURAL_OUTPUT_CAPTURE_ARM"] = str(capture_arm)
        environment["COLORLM_NEURAL_OUTPUT_CAPTURE_MAX_RECORDS"] = str(
            args.neural_output_capture_max_records
        )
    if k3_plan_path is not None:
        environment["COLORLM_NEURAL_BUS_K3_PLAN"] = str(k3_plan_path)
        if args.k3_alpha is not None:
            environment["COLORLM_NEURAL_BUS_K3_ALPHA"] = str(args.k3_alpha)
        if args.k3_trace:
            environment["COLORLM_NEURAL_BUS_K3_TRACE"] = "1"
    if intelligence_enabled:
        environment["COLORLM_NEURAL_BUS_INTELLIGENCE_PLAN"] = str(
            intelligence_plan_path
        )
        if args.intelligence_trace:
            environment["COLORLM_NEURAL_BUS_INTELLIGENCE_TRACE"] = "1"
    if args.force_path != "auto":
        environment["COLORLM_NEURAL_BUS_FORCE_PATH"] = args.force_path
    if force_site_paths_encoded is not None:
        environment["COLORLM_NEURAL_BUS_FORCE_PATH_BY_SITE"] = (
            force_site_paths_encoded
        )
    if args.hidden_dump is not None:
        hidden_dump = args.hidden_dump if args.hidden_dump.is_absolute() else root / args.hidden_dump
        environment["COLORLM_HIDDEN_DUMP"] = str(hidden_dump.resolve())
        environment["COLORLM_HIDDEN_DUMP_SITES"] = args.hidden_dump_sites
        environment["COLORLM_HIDDEN_DUMP_MAX_RECORDS"] = str(
            args.hidden_dump_max_records
        )
    if args.k3_feature_dump is not None:
        k3_feature_dump = (
            args.k3_feature_dump
            if args.k3_feature_dump.is_absolute()
            else root / args.k3_feature_dump
        )
        environment["COLORLM_K3_FEATURE_DUMP"] = str(k3_feature_dump.resolve())
        environment["COLORLM_K3_FEATURE_DUMP_SITES"] = args.k3_feature_dump_sites
        environment["COLORLM_K3_FEATURE_DUMP_MAX_RECORDS"] = str(
            args.k3_feature_dump_max_records
        )
    environment.update(
        {
            "COLORLM_V4_KERNEL_LAYERS": "0",
            "COLORLM_V4_RECURRENCE_ALPHA": "0",
            "COLORLM_V4_COGNITIVE_ROUNDS": "0",
            "COLORLM_V4_SEMANTIC_ALPHA": "0",
            "COLORLM_V4_PLASTIC_RANK": "0",
        }
    )

    if args.validate_only:
        contract = {
            "alias": alias,
            "url": base_url,
            "model": model_relative,
            "context": args.ctx_size,
            "neural_bus": {
                "enabled": neural_bus_enabled,
                "alpha": args.neural_bus_alpha,
                "sites": args.neural_bus_sites if neural_bus_enabled else None,
            },
            "neural_block": {
                "enabled": neural_block_enabled,
                "package": str(neural_block_package) if neural_block_enabled else None,
                "alpha": args.neural_block_alpha if neural_block_enabled else 0.0,
                "site": args.neural_block_site if neural_block_enabled else None,
            },
            "neural_island": {
                "enabled": neural_island_enabled,
                "manifest": str(neural_island_manifest) if neural_island_enabled else None,
                "alpha": args.neural_island_alpha if neural_island_enabled else 0.0,
                "site": args.neural_island_site if neural_island_enabled else None,
                "expert_cache_slots": (
                    args.neural_island_expert_cache_slots
                    if neural_island_enabled
                    else 0
                ),
                "expert_cache_policy": (
                    args.neural_island_expert_cache_policy
                    if neural_island_enabled
                    else "lru"
                ),
                "source_layers": (
                    neural_island_contract.get("source_layers")
                    if neural_island_contract is not None
                    else []
                ),
                "route_dump": (
                    str(args.neural_island_route_dump)
                    if args.neural_island_route_dump is not None
                    else None
                ),
                "route_dump_max_records": args.neural_island_route_dump_max_records,
                "teacher_dump": (
                    str(args.neural_island_teacher_dump)
                    if args.neural_island_teacher_dump is not None
                    else None
                ),
                "teacher_dump_max_records": args.neural_island_teacher_dump_max_records,
                "teacher_dump_site": args.neural_island_teacher_dump_site,
                "stage_dump": (
                    str(args.neural_island_stage_dump)
                    if args.neural_island_stage_dump is not None
                    else None
                ),
                "stage_dump_max_records": args.neural_island_stage_dump_max_records,
                "l47_moe_dump": (
                    str(args.neural_island_l47_moe_dump)
                    if args.neural_island_l47_moe_dump is not None
                    else None
                ),
                "l47_moe_dump_max_records": args.neural_island_l47_moe_dump_max_records,
                "micro_l47_package": (
                    str(args.neural_island_micro_l47_package)
                    if args.neural_island_micro_l47_package is not None
                    else None
                ),
                "micro_l47_moe_package": (
                    str(args.neural_island_micro_l47_moe_package)
                    if args.neural_island_micro_l47_moe_package is not None
                    else None
                ),
            },
            "neural_output_head": {
                "enabled": neural_output_head_enabled,
                "package": (
                    str(neural_output_head_package)
                    if neural_output_head_enabled
                    else None
                ),
                "alpha": (
                    args.neural_output_head_alpha
                    if neural_output_head_enabled
                    else 0.0
                ),
                "manifest_sha256": (
                    neural_output_head_manifest_sha256
                    if neural_output_head_enabled
                    else None
                ),
            },
            "sequence_policy": {
                "enabled": sequence_policy_enabled,
                "package": (
                    str(sequence_policy_package)
                    if sequence_policy_enabled
                    else None
                ),
                "manifest_sha256": (
                    sequence_policy_manifest_sha256
                    if sequence_policy_enabled
                    else None
                ),
                "format": (
                    policy_contract.get("format")
                    if sequence_policy_enabled
                    else None
                ),
                "mode": (
                    policy_contract.get("mode", "static-linear")
                    if sequence_policy_enabled
                    else None
                ),
                "internal_exact_no_op": (
                    policy_contract.get("no_op_rule")
                    == "exact-no-op-iff-class-0-is-argmax"
                    if sequence_policy_enabled
                    else False
                ),
                "activation": "explicit-tools-only",
                "no_tools_physical_graph_bypass": True,
                "parallel_slots": 1,
            },
            "anthropic_max_tokens": args.anthropic_max_tokens,
            "speculative_decoding": {
                "type": args.spec_type,
                "ngram_match": args.spec_ngram_match,
                "ngram_min": args.spec_ngram_min,
                "ngram_max": args.spec_ngram_max,
            },
            "k3": {
                "enabled": k3_enabled,
                "alpha_override": args.k3_alpha,
                "plan": str(k3_plan_path) if k3_enabled else None,
            },
            "intelligence_router": {
                "enabled": intelligence_enabled,
                "plan": str(intelligence_plan_path) if intelligence_enabled else None,
                "sites": intelligence_sites if intelligence_enabled else [],
                "weight_bytes": intelligence_bytes if intelligence_enabled else 0,
                "routes": ["no_op", "coder", "k3"] if intelligence_enabled else [],
                "force_path": args.force_path,
                "force_path_by_site": {
                    str(layer): path for layer, path in force_site_paths.items()
                },
                "force_path_by_site_env": force_site_paths_encoded,
            },
            "hidden_dump": {
                "enabled": args.hidden_dump is not None,
                "path": str(args.hidden_dump) if args.hidden_dump is not None else None,
                "sites": args.hidden_dump_sites if args.hidden_dump is not None else None,
                "max_records": args.hidden_dump_max_records,
            },
            "k3_feature_dump": {
                "enabled": args.k3_feature_dump is not None,
                "path": str(args.k3_feature_dump) if args.k3_feature_dump is not None else None,
                "sites": args.k3_feature_dump_sites if args.k3_feature_dump is not None else None,
                "max_records": args.k3_feature_dump_max_records,
                "features": [
                    "hidden_rms", "native_rms", "k3_delta_rms",
                    "hidden_delta_cos", "native_delta_cos", "energy_gate",
                ],
            },
            "alloy_plan": str(alloy["path"]) if alloy is not None else None,
            "alloy_plan_sha256": str(alloy["sha256"]) if alloy is not None else None,
            "core_sha256_verified": (
                bool(alloy["core_sha256_verified"]) if alloy is not None else False
            ),
            "result": "validated",
        }
        print(json.dumps(contract, ensure_ascii=False, indent=2))
        return 0

    suffix = "spin" if args.spin_fence else "baseline" if args.no_merge_cpu_sync else "optimized"
    suffix += f"-spec-{args.spec_type}"
    if args.slot_pool:
        suffix += f"-slot-{SLOT_POOL_LAYERS}x{SLOT_POOL_K}"
    if neural_bus_enabled:
        suffix += "-neural-bus-v2" if args.neural_bus_v2 else "-neural-bus"
    if neural_block_enabled:
        suffix += f"-neural-block-l{args.neural_block_site}"
    if neural_island_enabled:
        suffix += f"-neural-island-l{args.neural_island_site}"
        if args.neural_island_expert_cache_slots:
            suffix += (
                f"-cache{args.neural_island_expert_cache_slots}"
                f"-{args.neural_island_expert_cache_policy}"
            )
        if args.neural_island_route_dump is not None:
            suffix += "-route-dump"
        if args.neural_island_teacher_dump is not None:
            suffix += "-teacher-dump"
        if args.neural_island_stage_dump is not None:
            suffix += "-stage-dump"
        if args.neural_island_l47_moe_dump is not None:
            suffix += "-l47-moe-dump"
        if args.neural_island_micro_l47_package is not None:
            suffix += "-micro-l47"
        if args.neural_island_micro_l47_moe_package is not None:
            suffix += "-micro-l47-moe"
    if neural_output_head_enabled:
        head_alpha_suffix = format(args.neural_output_head_alpha, ".9g").replace("-", "m").replace(".", "p")
        suffix += f"-output-head-a{head_alpha_suffix}"
    if sequence_policy_enabled:
        suffix += "-sequence-policy"
    if k3_enabled:
        suffix += "-k3"
    if intelligence_enabled:
        suffix += "-intelligence"
    if args.force_path != "auto":
        suffix += f"-force-{args.force_path}"
    if force_site_paths:
        force_sites_suffix = "_".join(
            f"{layer}-{path}" for layer, path in force_site_paths.items()
        )
        suffix += f"-force-sites-{force_sites_suffix}"
    if args.hidden_dump is not None:
        suffix += "-hidden-dump"
    if args.k3_feature_dump is not None:
        suffix += "-k3-feature-dump"
    stdout = (runtime / f"fast16-v1-{suffix}.stdout.log").open("ab")
    stderr = (runtime / f"fast16-v1-{suffix}.stderr.log").open("ab")
    creation_flags = 0
    if sys.platform == "win32":
        creation_flags = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS
    process = subprocess.Popen(
        command,
        cwd=root,
        stdin=subprocess.DEVNULL,
        stdout=stdout,
        stderr=stderr,
        env=environment,
        creationflags=creation_flags,
        close_fds=True,
    )
    stdout.close()
    stderr.close()

    for _ in range(360):
        if server_ready(base_url, alias, args.ctx_size, model_relative):
            mode = "SPIN_FENCE" if args.spin_fence else "baseline" if args.no_merge_cpu_sync else "optimized"
            if not args.no_batch_cpu_read:
                mode += "+batch-read"
            mode += f"+spec-{args.spec_type}"
            if args.slot_pool:
                mode += f"+slot-{SLOT_POOL_LAYERS}x{SLOT_POOL_K}"
            if neural_bus_enabled:
                bus_version = "v2" if args.neural_bus_v2 else "v1"
                mode += f"+neural-bus-{bus_version}(alpha={args.neural_bus_alpha})"
            if neural_block_enabled:
                mode += (
                    f"+neural-block-L{args.neural_block_site}"
                    f"(alpha={args.neural_block_alpha})"
                )
            if neural_island_enabled:
                mode += (
                    f"+neural-island-L44-L47@L{args.neural_island_site}"
                    f"(alpha={args.neural_island_alpha},"
                    f"cache={args.neural_island_expert_cache_slots},"
                    f"policy={args.neural_island_expert_cache_policy})"
                )
            if neural_output_head_enabled:
                mode += f"+neural-output-head(alpha={args.neural_output_head_alpha})"
            if sequence_policy_enabled:
                if policy_contract.get("format") == "colorlm-sequence-policy-runtime-v3":
                    mode += "+sequence-policy-v3(explicit-tools-only,internal-exact-noop)"
                else:
                    mode += "+sequence-policy(explicit-tools-only)"
            if k3_enabled:
                description = ",".join(
                    f"L{item['site']}<-"
                    + "/".join(
                        f"K3L{candidate['k3_layer']}E{candidate['expert']}"
                        for candidate in item["candidates"]
                    )
                    for item in k3_sites
                )
                mode += f"+k3[{description}]"
            if intelligence_enabled:
                sites = ",".join(str(item["site"]) for item in intelligence_sites)
                force_description = (
                    force_site_paths_encoded
                    if force_site_paths_encoded is not None
                    else args.force_path
                )
                mode += f"+intelligence-router[L{sites};force={force_description}]"
            if alloy is not None:
                verification = "full-sha" if alloy["core_sha256_verified"] else "size+manifest-sha"
                mode += f"+alloy-plan({str(alloy['sha256'])[:12]},{verification})"
            print(f"Fast16 Runtime v1 已启动: {base_url} [{mode}]")
            return 0
        return_code = process.poll()
        if return_code is not None:
            print(
                f"Fast16子进程提前退出，退出码{ return_code }；"
                f"请查看 fast16/runtime/fast16-v1-{suffix}.stderr.log",
                file=sys.stderr,
            )
            return return_code if return_code != 0 else 1
        time.sleep(0.5)
    print(f"Fast16启动失败，请查看 fast16/runtime/fast16-v1-{suffix}.stderr.log", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
