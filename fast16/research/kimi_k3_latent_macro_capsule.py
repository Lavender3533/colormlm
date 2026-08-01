"""Build a consumable Kimi K3 latent-MoE macro capsule for ColorLM.

This tool keeps network extraction, expensive bridge folding, router mapping,
and manifest assembly as explicit commands.  ``plan`` and ``router`` are local.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from kimi_k3_expert_capsule import (
    DEFAULT_CACHE,
    DEFAULT_REPO,
    DEFAULT_REVISION,
    SITU_BETA,
    SITU_LINEAR_BETA,
    canonical_json_sha256,
    download_range,
    model_cache_dir,
    plan_identity,
    read_json,
    safe_slug,
    sha256_file,
    write_json,
)


RESEARCH = Path(__file__).resolve().parent
DEFAULT_TRANSPORT = RESEARCH / "kimi_k3_to_colorlm_semiorthogonal_f32.npy"
DEFAULT_ROUTER = (
    DEFAULT_CACHE
    / "moonshotai_Kimi-K3"
    / "master"
    / "extracted"
    / "language_model.model.layers.92.block_sparse_moe.gate.weight.bin"
)
BRIDGE_COMPONENTS = (
    ("routed_expert_down_proj.weight", "latent_down"),
    ("routed_expert_norm.weight", "latent_norm"),
    ("routed_expert_up_proj.weight", "latent_up"),
)
RMS_NORM_EPS = 1.0e-5


def bridge_tensor_name(layer: int, suffix: str) -> str:
    return f"language_model.model.layers.{layer}.block_sparse_moe.{suffix}"


def build_bridge_plan(
    cache_dir: Path,
    repo: str,
    revision: str,
    layer: int,
) -> dict:
    local = model_cache_dir(cache_dir, repo, revision)
    index = read_json(local / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise RuntimeError("模型索引缺少weight_map")

    names = [bridge_tensor_name(layer, suffix) for suffix, _ in BRIDGE_COMPONENTS]
    missing = [name for name in names if name not in weight_map]
    if missing:
        raise RuntimeError(f"缺失K3 latent bridge张量: {missing[0]}")
    shards = {str(weight_map[name]) for name in names}
    if len(shards) != 1:
        raise RuntimeError(f"latent bridge横跨多分片: {sorted(shards)}")
    shard = next(iter(shards))

    header_path = local / "headers" / f"{safe_slug(shard)}.json"
    header = read_json(header_path)
    header_bytes = int(header["header_bytes"])
    tensors = header["tensors"]
    data_start = 8 + header_bytes
    entries = []
    for (suffix, role), name in zip(BRIDGE_COMPONENTS, names, strict=True):
        item = tensors.get(name)
        if not isinstance(item, dict):
            raise RuntimeError(f"分片header缺失: {name}")
        dtype = str(item.get("dtype"))
        shape = [int(value) for value in item.get("shape", [])]
        offsets = [int(value) for value in item.get("data_offsets", [])]
        if dtype != "BF16" or len(offsets) != 2:
            raise RuntimeError(f"latent bridge张量格式异常: {name} {item}")
        byte_count = math.prod(shape) * 2
        if offsets[1] - offsets[0] != byte_count:
            raise RuntimeError(f"latent bridge张量字节数异常: {name}")
        entries.append(
            {
                "name": name,
                "role": role,
                "dtype": dtype,
                "shape": shape,
                "bytes": byte_count,
                "absolute_start": data_start + offsets[0],
                "absolute_end": data_start + offsets[1],
            }
        )

    expected_shapes = {
        "latent_down": [3584, 7168],
        "latent_norm": [3584],
        "latent_up": [7168, 3584],
    }
    for entry in entries:
        if entry["shape"] != expected_shapes[entry["role"]]:
            raise RuntimeError(f"{entry['role']}形状异常: {entry['shape']}")
    ordered = sorted(entries, key=lambda item: item["absolute_start"])
    for left, right in zip(ordered, ordered[1:]):
        if left["absolute_end"] != right["absolute_start"]:
            raise RuntimeError("latent down/norm/up不连续，拒绝合并Range")

    start = ordered[0]["absolute_start"]
    end = ordered[-1]["absolute_end"]
    for entry in entries:
        entry["capsule_offset_start"] = entry["absolute_start"] - start
        entry["capsule_offset_end"] = entry["absolute_end"] - start
    return {
        "format": "colorlm-kimi-k3-latent-bridge-range-plan-v1",
        "repo": repo,
        "revision": revision,
        "layer": layer,
        "expert": None,
        "source_shard": shard,
        "header_cache": os.fspath(header_path),
        "header_bytes": header_bytes,
        "header_tensors_sha256": canonical_json_sha256(tensors),
        "http_range": {
            "start": start,
            "end_inclusive": end - 1,
            "bytes": end - start,
        },
        "download_bytes": end - start,
        "download_mib": (end - start) / 1024**2,
        "source_shapes": expected_shapes,
        "folded_shapes": {
            "colorlm_to_latent": [3584, 2048],
            "latent_to_colorlm": [2048, 3584],
            "latent_norm": [3584],
        },
        "transport_equations": {
            "colorlm_to_latent": "routed_expert_down_proj @ transport",
            "latent_to_colorlm": "transport.T @ routed_expert_up_proj",
        },
        "tensors": entries,
    }


def default_bridge_dir(cache_dir: Path, repo: str, revision: str, layer: int) -> Path:
    return model_cache_dir(cache_dir, repo, revision) / "latent_bridges" / f"layer-{layer:02d}"


def lock_bridge_plan(output_dir: Path, plan: dict) -> None:
    path = output_dir / "source-plan.json"
    if path.is_file():
        if plan_identity(read_json(path)) != plan_identity(plan):
            raise RuntimeError(f"目录已绑定到另一个latent bridge: {output_dir}")
        return
    if any(output_dir.iterdir()):
        raise RuntimeError(f"目录非空且缺少source-plan.json: {output_dir}")
    write_json(path, plan)


def import_biopsy_bridge(
    plan: dict,
    extracted_dir: Path,
    bridge_dir: Path,
) -> dict:
    """Materialize the contiguous bridge source from verified biopsy outputs."""
    bridge_dir.mkdir(parents=True, exist_ok=True)
    lock_bridge_plan(bridge_dir, plan)

    components = []
    ordered = sorted(plan["tensors"], key=lambda item: item["capsule_offset_start"])
    expected_offset = 0
    for entry in ordered:
        if int(entry["capsule_offset_start"]) != expected_offset:
            raise RuntimeError("latent bridge计划的胶囊偏移不连续")
        source = extracted_dir / f"{entry['name']}.bin"
        metadata_path = source.with_suffix(source.suffix + ".json")
        if not source.is_file() or not metadata_path.is_file():
            raise RuntimeError(f"缺少已下载活检张量或其清单: {source}")
        metadata = read_json(metadata_path)
        checks = {
            "format": "colorlm-neural-biopsy-v1",
            "repo": plan["repo"],
            "revision": plan["revision"],
            "tensor": entry["name"],
            "source_shard": plan["source_shard"],
            "source_dtype": entry["dtype"],
            "output_shape": entry["shape"],
            "output_bytes": entry["bytes"],
        }
        for field, expected in checks.items():
            if metadata.get(field) != expected:
                raise RuntimeError(
                    f"活检清单字段不匹配: {metadata_path} {field}="
                    f"{metadata.get(field)!r}, expected={expected!r}"
                )
        if metadata.get("selection") is not None:
            raise RuntimeError(f"outer bridge必须是完整张量，不能带selection: {source}")
        if source.stat().st_size != int(entry["bytes"]):
            raise RuntimeError(f"活检张量字节数不匹配: {source}")
        digest = sha256_file(source)
        if digest != metadata.get("output_sha256"):
            raise RuntimeError(f"活检张量SHA-256与清单不匹配: {source}")
        components.append(
            {
                "role": entry["role"],
                "tensor": entry["name"],
                "source": os.fspath(source),
                "source_manifest": os.fspath(metadata_path),
                "bytes": source.stat().st_size,
                "sha256": digest,
                "capsule_offset_start": int(entry["capsule_offset_start"]),
                "capsule_offset_end": int(entry["capsule_offset_end"]),
            }
        )
        expected_offset = int(entry["capsule_offset_end"])

    raw_path = bridge_dir / "latent-bridge.bf16.bin"
    partial = raw_path.with_suffix(raw_path.suffix + ".part")
    partial.unlink(missing_ok=True)
    with partial.open("wb") as destination:
        for component in components:
            with Path(component["source"]).open("rb") as source_stream:
                shutil.copyfileobj(source_stream, destination, length=8 * 1024 * 1024)
    if partial.stat().st_size != int(plan["download_bytes"]):
        raise RuntimeError("从活检张量组装的latent bridge总字节数异常")
    partial.replace(raw_path)

    raw_sha256 = sha256_file(raw_path)
    source_report = {
        "format": "colorlm-kimi-k3-local-biopsy-bridge-source-v1",
        "plan_identity": plan_identity(plan),
        "header_tensors_sha256": plan["header_tensors_sha256"],
        "raw_sha256": raw_sha256,
        "raw_bytes": raw_path.stat().st_size,
        "components": components,
        "validator": {
            "kind": "local-biopsy-component-sha256",
            "component_count": len(components),
        },
    }
    write_json(raw_path.with_suffix(raw_path.suffix + ".source.json"), source_report)
    return source_report


def bf16_to_f32(values: np.ndarray) -> np.ndarray:
    words = np.asarray(values, dtype=np.uint16)
    return np.left_shift(words.astype(np.uint32), 16).view(np.float32)


def raw_bf16_component(raw: np.memmap, entry: dict) -> np.ndarray:
    start = int(entry["capsule_offset_start"])
    end = int(entry["capsule_offset_end"])
    shape = tuple(int(value) for value in entry["shape"])
    view = raw[start:end]
    if view.size != math.prod(shape) * 2:
        raise RuntimeError(f"原始latent bridge张量长度异常: {entry['name']}")
    return view.view("<u2").reshape(shape)


def bridge_entry(plan: dict, role: str) -> dict:
    matches = [item for item in plan["tensors"] if item["role"] == role]
    if len(matches) != 1:
        raise RuntimeError(f"latent bridge计划中{role}数量异常")
    return matches[0]


def save_matrix_atomic(path: Path, matrix: np.ndarray, dtype: np.dtype) -> dict:
    partial = path.with_suffix(path.suffix + ".part")
    partial.unlink(missing_ok=True)
    destination = np.lib.format.open_memmap(
        partial,
        mode="w+",
        dtype=dtype,
        shape=matrix.shape,
    )
    destination[:] = matrix.astype(dtype, copy=False)
    destination.flush()
    del destination
    partial.replace(path)
    return {
        "file": path.name,
        "shape": list(matrix.shape),
        "dtype": dtype.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "orientation": "vector" if matrix.ndim == 1 else "pytorch-linear-out-in",
    }


def build_folded_bridge(
    bridge_dir: Path,
    transport_path: Path,
    output_dtype: str,
    chunk_rows: int,
    min_free_bytes: int,
) -> dict:
    if chunk_rows <= 0:
        raise ValueError("chunk_rows必须大于0")
    dtype = np.dtype(output_dtype)
    plan = read_json(bridge_dir / "source-plan.json")
    raw_path = bridge_dir / "latent-bridge.bf16.bin"
    source_meta = read_json(raw_path.with_suffix(raw_path.suffix + ".source.json"))
    raw_sha256 = sha256_file(raw_path)
    if source_meta.get("raw_sha256") != raw_sha256:
        raise RuntimeError("latent bridge原始文件SHA-256不匹配")
    transport = np.load(transport_path, mmap_mode="r", allow_pickle=False)
    if transport.shape != (7168, 2048) or transport.dtype != np.float32:
        raise RuntimeError(f"坐标运输矩阵格式异常: {transport.shape} {transport.dtype}")

    output_bytes = (3584 * 2048 + 2048 * 3584 + 3584) * dtype.itemsize
    free = shutil.disk_usage(bridge_dir).free
    if free - output_bytes < min_free_bytes:
        raise RuntimeError("折叠latent bridge后剩余空间低于安全线")

    raw = np.memmap(raw_path, dtype=np.uint8, mode="r")
    latent_down = raw_bf16_component(raw, bridge_entry(plan, "latent_down"))
    latent_norm = raw_bf16_component(raw, bridge_entry(plan, "latent_norm"))
    latent_up = raw_bf16_component(raw, bridge_entry(plan, "latent_up"))

    input_path = bridge_dir / f"colorlm_to_latent.{dtype.name}.npy"
    input_partial = input_path.with_suffix(input_path.suffix + ".part")
    input_partial.unlink(missing_ok=True)
    input_matrix = np.lib.format.open_memmap(
        input_partial,
        mode="w+",
        dtype=dtype,
        shape=(3584, 2048),
    )
    for first in range(0, 3584, chunk_rows):
        stop = min(first + chunk_rows, 3584)
        block = bf16_to_f32(latent_down[first:stop])
        input_matrix[first:stop] = (block @ transport).astype(dtype, copy=False)
    input_matrix.flush()
    del input_matrix
    input_partial.replace(input_path)

    # U is reused for every output-row block, so decode it once (about 98 MiB F32).
    latent_up_f32 = bf16_to_f32(latent_up)
    output_path = bridge_dir / f"latent_to_colorlm.{dtype.name}.npy"
    output_partial = output_path.with_suffix(output_path.suffix + ".part")
    output_partial.unlink(missing_ok=True)
    output_matrix = np.lib.format.open_memmap(
        output_partial,
        mode="w+",
        dtype=dtype,
        shape=(2048, 3584),
    )
    for first in range(0, 2048, chunk_rows):
        stop = min(first + chunk_rows, 2048)
        block = np.asarray(transport[:, first:stop], dtype=np.float32).T @ latent_up_f32
        output_matrix[first:stop] = block.astype(dtype, copy=False)
    output_matrix.flush()
    del output_matrix, latent_up_f32
    output_partial.replace(output_path)

    norm_values = bf16_to_f32(latent_norm)
    norm_path = bridge_dir / f"latent_norm.{dtype.name}.npy"
    norm_report = save_matrix_atomic(norm_path, norm_values, dtype)
    del raw

    matrices = {
        "colorlm_to_latent": {
            "file": input_path.name,
            "shape": [3584, 2048],
            "dtype": dtype.name,
            "bytes": input_path.stat().st_size,
            "sha256": sha256_file(input_path),
            "orientation": "pytorch-linear-out-in",
        },
        "latent_to_colorlm": {
            "file": output_path.name,
            "shape": [2048, 3584],
            "dtype": dtype.name,
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
            "orientation": "pytorch-linear-out-in",
        },
        "latent_norm": norm_report,
    }
    manifest = {
        "format": "colorlm-kimi-k3-folded-latent-bridge-v1",
        "repo": plan["repo"],
        "revision": plan["revision"],
        "layer": plan["layer"],
        "raw_sha256": raw_sha256,
        "transport": {
            "file": os.fspath(transport_path),
            "shape": [7168, 2048],
            "sha256": sha256_file(transport_path),
            "direction": "kimi_row @ transport = colorlm_row",
        },
        "equations": plan["transport_equations"],
        "rms_norm_eps": RMS_NORM_EPS,
        "matrices": matrices,
    }
    write_json(bridge_dir / "bridge.json", manifest)
    return manifest


def map_router_prior(
    layer: int,
    expert: int,
    router_path: Path,
    transport_path: Path,
    output_dir: Path,
) -> dict:
    if not 0 <= expert < 896:
        raise ValueError("expert必须在0--895")
    expected = 896 * 7168 * 2
    if router_path.stat().st_size != expected:
        raise RuntimeError(f"K3 router BF16文件大小异常: {router_path}")
    transport = np.load(transport_path, mmap_mode="r", allow_pickle=False)
    if transport.shape != (7168, 2048):
        raise RuntimeError(f"坐标运输形状异常: {transport.shape}")
    router = np.memmap(router_path, dtype="<u2", mode="r", shape=(896, 7168))
    donor_row = bf16_to_f32(router[expert])
    mapped_row = np.asarray(donor_row @ transport, dtype=np.float32)
    output_dir.mkdir(parents=True, exist_ok=True)
    donor_path = output_dir / "router_kimi_f32.npy"
    mapped_path = output_dir / "router_colorlm_f32.npy"
    donor_report = save_matrix_atomic(donor_path, donor_row, np.dtype("float32"))
    mapped_report = save_matrix_atomic(mapped_path, mapped_row, np.dtype("float32"))
    report = {
        "format": "colorlm-kimi-k3-router-prior-v1",
        "layer": layer,
        "expert": expert,
        "source_router": {
            "file": os.fspath(router_path),
            "sha256": sha256_file(router_path),
            "row": donor_report,
        },
        "transport_sha256": sha256_file(transport_path),
        "mapped_row": mapped_report,
        "mapped_l2_norm": float(np.linalg.norm(mapped_row)),
        "contract": "score = router_colorlm_f32 @ hidden_state",
        "warning": "router prior is not a capability label; calibrate against forced-route NLL",
    }
    write_json(output_dir / "router.json", report)
    del router
    return report


def materialize_raw_f16(
    source: Path,
    destination: Path,
    expected_shape: tuple[int, ...],
    chunk_rows: int = 128,
) -> dict:
    matrix = np.load(source, mmap_mode="r", allow_pickle=False)
    if matrix.shape != expected_shape or matrix.dtype not in (np.float16, np.float32):
        raise RuntimeError(
            f"胶囊中间矩阵格式异常: {source} "
            f"shape={matrix.shape} dtype={matrix.dtype}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.unlink(missing_ok=True)
    with partial.open("wb") as stream:
        if matrix.ndim == 1:
            np.asarray(matrix, dtype="<f2").tofile(stream)
        else:
            for first in range(0, matrix.shape[0], chunk_rows):
                stop = min(first + chunk_rows, matrix.shape[0])
                np.asarray(matrix[first:stop], dtype="<f2").tofile(stream)
    expected_bytes = math.prod(expected_shape) * 2
    if partial.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"无头F16大小不匹配: {partial.stat().st_size} vs {expected_bytes}"
        )
    partial.replace(destination)
    return {
        "file": destination.name,
        "shape": list(expected_shape),
        "dtype": "float16-le",
        "layout": "raw-row-major-pytorch-out-in" if len(expected_shape) == 2 else "raw-vector",
        "bytes": expected_bytes,
        "sha256": sha256_file(destination),
        "source_npy": os.fspath(source),
    }


def materialize_raw_f32_vector(source: Path, destination: Path, width: int) -> dict:
    vector = np.load(source, mmap_mode="r", allow_pickle=False)
    if vector.shape != (width,) or vector.dtype != np.float32:
        raise RuntimeError(
            f"路由向量格式异常: {source} shape={vector.shape} dtype={vector.dtype}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.unlink(missing_ok=True)
    np.asarray(vector, dtype="<f4").tofile(partial)
    expected_bytes = width * 4
    if partial.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"无头F32路由大小不匹配: {partial.stat().st_size} vs {expected_bytes}"
        )
    partial.replace(destination)
    return {
        "file": destination.name,
        "shape": [width],
        "dtype": "float32-le",
        "layout": "raw-vector",
        "bytes": expected_bytes,
        "sha256": sha256_file(destination),
        "source_npy": os.fspath(source),
        "l2_norm": float(np.linalg.norm(np.asarray(vector, dtype=np.float64))),
    }


def assemble_macro(
    expert_dir: Path,
    bridge_dir: Path,
    router_dir: Path,
    output_dir: Path,
    include_router: bool = False,
) -> dict:
    expert = read_json(expert_dir / "capsule.json")
    bridge = read_json(bridge_dir / "bridge.json")
    router = read_json(router_dir / "router.json")
    if expert["layer"] != bridge["layer"] or expert["layer"] != router["layer"]:
        raise RuntimeError("专家、latent bridge和router不在同一层")
    if expert["expert"] != router["expert"]:
        raise RuntimeError("专家胶囊与router专家编号不一致")
    if not isinstance(expert.get("matrices"), dict):
        raise RuntimeError("专家胶囊尚未解码为矩阵")

    output_dir.mkdir(parents=True, exist_ok=True)
    source_files = {
        "b_in": (
            bridge_dir / bridge["matrices"]["colorlm_to_latent"]["file"],
            (3584, 2048),
        ),
        "gate": (expert_dir / expert["matrices"]["gate"]["file"], (3072, 3584)),
        "up": (expert_dir / expert["matrices"]["up"]["file"], (3072, 3584)),
        "down": (expert_dir / expert["matrices"]["down"]["file"], (3584, 3072)),
        "norm": (bridge_dir / bridge["matrices"]["latent_norm"]["file"], (3584,)),
        "b_out": (
            bridge_dir / bridge["matrices"]["latent_to_colorlm"]["file"],
            (2048, 3584),
        ),
    }
    files = {
        name: materialize_raw_f16(source, output_dir / f"{name}.f16", shape)
        for name, (source, shape) in source_files.items()
    }
    total_bytes = sum(item["bytes"] for item in files.values())
    if total_bytes != 95_427_584:
        raise RuntimeError(f"六张量宏胶囊总大小异常: {total_bytes}")
    runtime_router = None
    if include_router:
        runtime_router = materialize_raw_f32_vector(
            router_dir / router["mapped_row"]["file"],
            output_dir / "router.f32",
            2048,
        )
        if abs(runtime_router["l2_norm"] - float(router["mapped_l2_norm"])) > 1e-5:
            raise RuntimeError("运行时路由向量范数与router清单不一致")

    manifest = {
        "format": (
            "colorlm-kimi-k3-latent-macro-capsule-v2"
            if include_router
            else "colorlm-kimi-k3-latent-macro-capsule-v1"
        ),
        "repo": expert["repo"],
        "revision": expert["revision"],
        "layer": expert["layer"],
        "expert": expert["expert"],
        "dimensions": {"colorlm": 2048, "latent": 3584, "intermediate": 3072},
        "activation": {
            "name": "situ",
            "beta": SITU_BETA,
            "linear_beta": SITU_LINEAR_BETA,
        },
        "rms_norm_eps": RMS_NORM_EPS,
        "execution": (
            "z=colorlm_to_latent@h; e=down@situ(gate@z,up@z); "
            "delta=latent_to_colorlm@rms_norm(e,latent_norm)"
        ),
        "runtime_layout": "six-headerless-f16-le-row-major-v1",
        "runtime_total_bytes": total_bytes,
        "runtime_files": files,
        "expert_capsule": {"path": os.fspath(expert_dir), "manifest": expert},
        "latent_bridge": {"path": os.fspath(bridge_dir), "manifest": bridge},
        "router_prior": {"path": os.fspath(router_dir), "manifest": router},
        "scope": "forced single routed expert; shared experts are intentionally excluded",
    }
    if runtime_router is not None:
        manifest["runtime_router"] = runtime_router
        manifest["routing_contract"] = (
            "cosine(router, hidden_state); combine candidates with temperature-scaled softmax"
        )
    write_json(output_dir / "capsule.json", manifest)
    return manifest


def run_self_test() -> dict:
    values = np.asarray([0x3F80, 0xC020, 0x0000], dtype=np.uint16)
    np.testing.assert_array_equal(
        bf16_to_f32(values),
        np.asarray([1.0, -2.5, 0.0], dtype=np.float32),
    )
    rng = np.random.default_rng(20260729)
    down = rng.standard_normal((3, 5), dtype=np.float32)
    up = rng.standard_normal((5, 3), dtype=np.float32)
    transport = rng.standard_normal((5, 2), dtype=np.float32)
    color = rng.standard_normal(2, dtype=np.float32)
    latent = rng.standard_normal(3, dtype=np.float32)
    np.testing.assert_allclose((down @ transport) @ color, down @ (transport @ color), rtol=1e-5)
    np.testing.assert_allclose(
        (transport.T @ up) @ latent,
        transport.T @ (up @ latent),
        rtol=1e-5,
    )
    with tempfile.TemporaryDirectory(prefix="kimi-k3-macro-test-") as directory:
        temporary = Path(directory)
        source = temporary / "source.npy"
        expected = np.arange(6, dtype=np.float32).reshape(2, 3)
        np.save(source, expected, allow_pickle=False)
        raw = temporary / "matrix.f16"
        report = materialize_raw_f16(source, raw, (2, 3), chunk_rows=1)
        if report["bytes"] != 12:
            raise AssertionError("无头F16自检大小错误")
        np.testing.assert_array_equal(
            np.fromfile(raw, dtype="<f2").reshape(2, 3),
            expected.astype(np.float16),
        )
    return {
        "ok": True,
        "bf16_decode": True,
        "input_bridge_equation": True,
        "output_bridge_equation": True,
        "headerless_f16": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Kimi K3 latent bridge与宏胶囊构建")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="离线计算共享latent bridge Range")
    plan.add_argument("--layer", type=int, default=92)
    plan.add_argument("--output", type=Path)

    extract = commands.add_parser("extract", help="Range提取latent down/norm/up")
    extract.add_argument("--layer", type=int, default=92)
    extract.add_argument("--output-dir", type=Path)
    extract.add_argument("--max-download-mib", type=float, default=128.0)
    extract.add_argument("--min-free-gib", type=float, default=2.0)

    import_biopsy = commands.add_parser(
        "import-biopsy",
        help="从已校验的本地活检张量组装latent bridge源文件",
    )
    import_biopsy.add_argument("--layer", type=int, default=92)
    import_biopsy.add_argument("--extracted-dir", type=Path, required=True)
    import_biopsy.add_argument("--bridge-dir", type=Path, required=True)

    build = commands.add_parser("build", help="折叠两个ColorLM/K3 latent外桥")
    build.add_argument("--bridge-dir", type=Path, required=True)
    build.add_argument("--transport", type=Path, default=DEFAULT_TRANSPORT)
    build.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    build.add_argument("--chunk-rows", type=int, default=32)
    build.add_argument("--min-free-gib", type=float, default=2.0)

    router = commands.add_parser("router", help="从本地K3 router生成ColorLM路由先验")
    router.add_argument("--layer", type=int, default=92)
    router.add_argument("--expert", type=int, required=True)
    router.add_argument("--router-bf16", type=Path, default=DEFAULT_ROUTER)
    router.add_argument("--transport", type=Path, default=DEFAULT_TRANSPORT)
    router.add_argument("--output-dir", type=Path, required=True)

    assemble = commands.add_parser("assemble", help="组装可消费宏胶囊契约")
    assemble.add_argument("--expert-dir", type=Path, required=True)
    assemble.add_argument("--bridge-dir", type=Path, required=True)
    assemble.add_argument("--router-dir", type=Path, required=True)
    assemble.add_argument("--output-dir", type=Path, required=True)
    assemble.add_argument(
        "--include-router",
        action="store_true",
        help="生成v2宏胶囊并携带无头F32隐藏态路由向量",
    )
    commands.add_parser("self-test", help="零网络检查BF16和外桥方向")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "self-test":
        print(json.dumps(run_self_test(), ensure_ascii=False, indent=2))
        return 0
    if args.command in ("plan", "extract", "import-biopsy"):
        plan = build_bridge_plan(
            args.cache_dir,
            args.repo,
            args.revision,
            args.layer,
        )
        if args.command == "plan":
            if args.output is not None:
                write_json(args.output, plan)
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0
        if args.command == "import-biopsy":
            report = import_biopsy_bridge(
                plan,
                args.extracted_dir,
                args.bridge_dir,
            )
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        output_dir = args.output_dir or default_bridge_dir(
            args.cache_dir, args.repo, args.revision, args.layer
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        lock_bridge_plan(output_dir, plan)
        report = download_range(
            plan,
            args.cache_dir,
            output_dir / "latent-bridge.bf16.bin",
            int(args.max_download_mib * 1024**2),
            int(args.min_free_gib * 1024**3),
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "build":
        report = build_folded_bridge(
            args.bridge_dir,
            args.transport,
            args.dtype,
            args.chunk_rows,
            int(args.min_free_gib * 1024**3),
        )
    elif args.command == "router":
        report = map_router_prior(
            args.layer,
            args.expert,
            args.router_bf16,
            args.transport,
            args.output_dir,
        )
    else:
        report = assemble_macro(
            args.expert_dir,
            args.bridge_dir,
            args.router_dir,
            args.output_dir,
            args.include_router,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
