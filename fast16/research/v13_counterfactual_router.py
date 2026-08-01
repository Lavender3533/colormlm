"""ColorLM v13 反事实 next-token NLL 采集与小型智能路由器校准。

该工具不负责启动或停止模型。三个原子路径必须由操作者依次启动：
no_op、coder、k3。采集用 logit_bias 强制同一个 teacher token，再读取
llama-server 在采样器之前计算的原始 softmax logprob。目标 token 无需进入
top-k，因此每条保留记录都是精确 next-token NLL。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import struct
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


FORMAT_TEACHER = "colorlm-v13-counterfactual-teacher-v1"
FORMAT_SHARD = "colorlm-v13-counterfactual-route-shard-v1"
FORMAT_ROUTER = "colorlm-v13-intelligence-router-v1"
FORMAT_TEACHER_MANIFEST = "colorlm-v13-counterfactual-teacher-manifest-v1"
FORMAT_SHARD_MANIFEST = "colorlm-v13-counterfactual-route-shard-manifest-v1"
HIDDEN_MAGIC = 0x394D4C43
K3_FEATURE_MAGIC = 0x46334C43
HIDDEN_HEADER = struct.Struct("<IIiI4qQ")
DEFAULT_SITES = (12, 28)


class V13Error(RuntimeError):
    """可预期且应直接展示给操作者的契约错误。"""


@dataclass(frozen=True)
class HiddenRecord:
    layer: int
    tensor_type: int
    shape: tuple[int, int, int, int]
    values: np.ndarray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_prefix(path: Path, byte_count: int) -> str:
    digest = hashlib.sha256()
    remaining = byte_count
    with path.open("rb") as stream:
        while remaining:
            chunk = stream.read(min(1024 * 1024, remaining))
            if not chunk:
                raise V13Error(
                    f"计算前缀SHA-256时文件提前结束: {path}, remaining={remaining}"
                )
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise V13Error(f"{path}:{line_number} 不是有效JSON: {error}") from error
            if not isinstance(value, dict):
                raise V13Error(f"{path}:{line_number} 必须是JSON对象")
            rows.append(value)
    if not rows:
        raise V13Error(f"{path} 没有数据")
    return rows


def read_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise V13Error(f"{description}不是有效JSON对象: {path}: {error}") from error
    if not isinstance(value, dict):
        raise V13Error(f"{description}必须是JSON对象: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")


def manifest_path(path: Path) -> Path:
    return Path(str(path) + ".manifest.json")


def relative_to_manifest(path: Path, manifest: Path) -> str:
    return os.path.relpath(path.resolve(), manifest.parent.resolve()).replace("\\", "/")


def resolve_manifest_path(raw: str, manifest: Path) -> Path:
    candidate = Path(raw)
    return candidate if candidate.is_absolute() else (manifest.parent / candidate).resolve()


def normalize_endpoint(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/v1"):
        endpoint = endpoint[:-3]
    if not endpoint.startswith(("http://", "https://")):
        endpoint = "http://" + endpoint
    return endpoint


def http_json(
    endpoint: str,
    path: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    url = normalize_endpoint(endpoint) + path
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"content-type": "application/json", "x-api-key": "local-v13-research"},
        method="GET" if payload is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise V13Error(f"请求失败 {url}: {error}") from error
    if not isinstance(value, dict):
        raise V13Error(f"{url} 返回值不是JSON对象")
    return value


def endpoint_snapshot(endpoint: str, expect_model: str | None) -> dict[str, Any]:
    health = http_json(endpoint, "/health", timeout=5)
    models = http_json(endpoint, "/v1/models", timeout=5)
    props = http_json(endpoint, "/props", timeout=5)
    if health.get("status") != "ok":
        raise V13Error(f"服务健康检查失败: {health}")
    model_ids = [str(item.get("id")) for item in models.get("data", []) if isinstance(item, dict)]
    if expect_model and expect_model not in model_ids:
        raise V13Error(f"服务alias不匹配，期望 {expect_model}，实际 {model_ids}")
    settings = props.get("default_generation_settings", {})
    return {
        "endpoint": normalize_endpoint(endpoint),
        "models": model_ids,
        "model_path": props.get("model_path"),
        "context": settings.get("n_ctx") if isinstance(settings, dict) else None,
    }


def tokenize(endpoint: str, content: str, add_special: bool) -> list[int]:
    response = http_json(
        endpoint,
        "/tokenize",
        {"content": content, "add_special": add_special, "parse_special": True},
    )
    tokens = response.get("tokens")
    if not isinstance(tokens, list) or not all(isinstance(token, int) for token in tokens):
        raise V13Error("/tokenize 没有返回整数token数组")
    return [int(token) for token in tokens]


def render_task_prompt(endpoint: str, task: dict[str, Any]) -> str:
    prompt = task.get("prompt")
    if isinstance(prompt, str):
        return prompt
    messages = task.get("messages")
    if not isinstance(messages, list) or not messages:
        raise V13Error(f"任务 {task.get('id')} 必须提供 prompt 或 messages")
    payload: dict[str, Any] = {
        "messages": messages,
        "add_generation_prompt": True,
    }
    if "tools" in task:
        payload["tools"] = task["tools"]
    response = http_json(endpoint, "/apply-template", payload)
    rendered = response.get("prompt")
    if not isinstance(rendered, str):
        raise V13Error("/apply-template 没有返回prompt")
    return rendered


def prepare_teacher(args: argparse.Namespace) -> int:
    snapshot = endpoint_snapshot(args.endpoint, args.expect_model)
    tasks = read_jsonl(args.tasks)
    rows: list[dict[str, Any]] = []
    boundary_fallbacks = 0
    for task_index, task in enumerate(tasks):
        task_id = str(task.get("id") or f"task-{task_index:03d}")
        target = task.get("target")
        if not isinstance(target, str) or not target:
            raise V13Error(f"任务 {task_id} 缺少非空target")
        prompt = render_task_prompt(args.endpoint, task)
        prompt_tokens = tokenize(args.endpoint, prompt, add_special=True)
        combined_tokens = tokenize(args.endpoint, prompt + target, add_special=True)
        if combined_tokens[: len(prompt_tokens)] == prompt_tokens:
            target_tokens = combined_tokens[len(prompt_tokens) :]
            boundary_mode = "combined"
        else:
            target_tokens = tokenize(args.endpoint, target, add_special=False)
            boundary_mode = "separate_target"
            boundary_fallbacks += 1
        if not target_tokens:
            raise V13Error(f"任务 {task_id} 的target没有产生token")
        limit = min(len(target_tokens), args.max_target_tokens)
        for token_index in range(limit):
            prefix = prompt_tokens + target_tokens[:token_index]
            if len(prefix) > args.max_prefix_tokens:
                break
            rows.append(
                {
                    "format": FORMAT_TEACHER,
                    "sample_id": f"{task_id}:{token_index:04d}",
                    "task_id": task_id,
                    "token_index": token_index,
                    "prefix_tokens": prefix,
                    "target_token_id": target_tokens[token_index],
                    "boundary_mode": boundary_mode,
                }
            )
            if len(rows) >= args.max_samples:
                break
        if len(rows) >= args.max_samples:
            break
    if not rows:
        raise V13Error("没有生成任何teacher token")
    write_jsonl(args.output, rows)
    manifest = {
        "format": FORMAT_TEACHER_MANIFEST,
        "source_tasks": str(args.tasks.resolve()),
        "source_tasks_sha256": sha256_file(args.tasks),
        "teacher": args.output.name,
        "teacher_sha256": sha256_file(args.output),
        "sample_count": len(rows),
        "task_count": len({row["task_id"] for row in rows}),
        "boundary_fallbacks": boundary_fallbacks,
        "max_target_tokens": args.max_target_tokens,
        "max_prefix_tokens": args.max_prefix_tokens,
        "tokenizer_endpoint": snapshot,
    }
    write_json(manifest_path(args.output), manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def read_hidden_dump(
    path: Path,
    start_offset: int = 0,
    end_offset: int | None = None,
    *,
    allow_empty: bool = False,
    expected_magic: int = HIDDEN_MAGIC,
) -> list[HiddenRecord]:
    file_size = path.stat().st_size
    end = file_size if end_offset is None else end_offset
    if start_offset < 0 or end < start_offset or end > file_size:
        raise V13Error(
            f"隐藏状态字节范围无效: start={start_offset}, end={end}, size={file_size}"
        )
    records: list[HiddenRecord] = []
    with path.open("rb") as stream:
        stream.seek(start_offset)
        while stream.tell() < end:
            record_offset = stream.tell()
            if end - record_offset < HIDDEN_HEADER.size:
                raise V13Error(
                    f"隐藏状态范围尾部header不完整: offset={record_offset}, end={end}"
                )
            raw_header = stream.read(HIDDEN_HEADER.size)
            if len(raw_header) != HIDDEN_HEADER.size:
                raise V13Error(f"隐藏状态文件尾部header不完整: {path}")
            magic, version, layer, tensor_type, ne0, ne1, ne2, ne3, payload_bytes = (
                HIDDEN_HEADER.unpack(raw_header)
            )
            if magic != expected_magic or version != 1:
                raise V13Error(
                    f"隐藏状态header不受支持: offset={record_offset}, "
                    f"magic={magic:#x}, version={version}"
                )
            if payload_bytes > end - stream.tell():
                raise V13Error(
                    f"隐藏状态payload越过采集边界: layer={layer}, "
                    f"offset={record_offset}, end={end}"
                )
            payload = stream.read(payload_bytes)
            if len(payload) != payload_bytes:
                raise V13Error(f"隐藏状态payload不完整: layer={layer}")
            if tensor_type != 0 or payload_bytes % 4 != 0:
                raise V13Error(
                    f"隐藏状态必须是GGML_TYPE_F32，实际type={tensor_type}, bytes={payload_bytes}"
                )
            if any(dimension <= 0 for dimension in (ne0, ne1, ne2, ne3)):
                raise V13Error(
                    f"隐藏状态shape必须全为正数: shape={(ne0, ne1, ne2, ne3)}"
                )
            expected = int(ne0 * ne1 * ne2 * ne3 * 4)
            if expected != payload_bytes:
                raise V13Error(
                    f"隐藏状态尺寸不一致: shape={(ne0, ne1, ne2, ne3)}, bytes={payload_bytes}"
                )
            values = np.frombuffer(payload, dtype="<f4").astype(np.float32, copy=True)
            if not np.isfinite(values).all():
                raise V13Error(f"隐藏状态包含NaN或Inf: layer={layer}, offset={record_offset}")
            records.append(
                HiddenRecord(
                    layer=int(layer),
                    tensor_type=int(tensor_type),
                    shape=(int(ne0), int(ne1), int(ne2), int(ne3)),
                    values=values,
                )
            )
        if stream.tell() != end:
            raise V13Error(f"隐藏状态范围没有在记录边界结束: actual={stream.tell()}, end={end}")
    if not records and not allow_empty:
        raise V13Error(f"隐藏状态文件为空: {path}")
    return records


def align_hidden_records(
    records: list[HiddenRecord],
    rows: list[dict[str, Any]],
    sites: tuple[int, ...],
) -> dict[int, np.ndarray]:
    """严格对齐本次请求新增的记录；不扫描、不跳过，也不接受尾随记录。"""
    if not sites:
        raise V13Error("sites不能为空")
    expected_count = len(rows) * len(sites)
    if len(records) != expected_count:
        raise V13Error(
            f"隐藏状态新增记录数不精确: expected={expected_count}, actual={len(records)}"
        )
    n_embd: int | None = None
    matrices: dict[int, list[np.ndarray]] = {site: [] for site in sites}
    for row_index, row in enumerate(rows):
        prefix_len = len(row["prefix_tokens"])
        begin = row_index * len(sites)
        group = records[begin : begin + len(sites)]
        for record, site in zip(group, sites):
            if (
                record.layer != site
                or record.shape[1] != prefix_len
                or record.shape[2:] != (1, 1)
            ):
                actual = [{"layer": item.layer, "shape": item.shape} for item in group]
                raise V13Error(
                    f"隐藏状态组与 {row['sample_id']} 不一致: "
                    f"expected_sites={list(sites)}, prefix_tokens={prefix_len}, actual={actual}"
                )
        for record, site in zip(group, sites):
            if n_embd is None:
                n_embd = record.shape[0]
            if record.shape[0] != n_embd:
                raise V13Error("隐藏状态宽度在记录间发生变化")
            if record.values.size != math.prod(record.shape):
                raise V13Error(f"隐藏状态payload元素数与shape不一致: layer={site}")
            token_rows = record.values.reshape(-1, record.shape[0])
            if not np.isfinite(token_rows[-1]).all():
                raise V13Error(f"隐藏状态末token包含NaN或Inf: layer={site}")
            matrices[site].append(token_rows[-1].copy())
    if n_embd is None:
        raise V13Error("没有可对齐的隐藏状态")
    return {
        site: np.stack(vectors).astype(np.float32, copy=False)
        for site, vectors in matrices.items()
    }


def collect_route(args: argparse.Namespace) -> int:
    snapshot = endpoint_snapshot(args.endpoint, args.expect_model)
    teacher = read_jsonl(args.teacher)
    if any(row.get("format") != FORMAT_TEACHER for row in teacher):
        raise V13Error("teacher格式不受支持")
    if args.max_samples:
        teacher = teacher[: args.max_samples]
    hidden_start_offset: int | None = None
    hidden_records_before: int | None = None
    hidden_prefix_sha256: str | None = None
    hidden_file_identity: tuple[int, int] | None = None
    if args.hidden_dump:
        if not args.hidden_dump.is_file():
            raise V13Error(f"隐藏状态dump尚不存在: {args.hidden_dump}")
        hidden_stat = args.hidden_dump.stat()
        hidden_start_offset = hidden_stat.st_size
        hidden_file_identity = (hidden_stat.st_dev, hidden_stat.st_ino)
        hidden_prefix_sha256 = sha256_prefix(args.hidden_dump, hidden_start_offset)
        prefix_records = read_hidden_dump(
            args.hidden_dump,
            0,
            hidden_start_offset,
            allow_empty=True,
        )
        hidden_records_before = len(prefix_records)
    feature_start_offset: int | None = None
    feature_records_before: int | None = None
    feature_prefix_sha256: str | None = None
    feature_file_identity: tuple[int, int] | None = None
    if args.k3_feature_dump:
        if not args.k3_feature_dump.is_file():
            raise V13Error(f"K3特征dump尚不存在: {args.k3_feature_dump}")
        feature_stat = args.k3_feature_dump.stat()
        feature_start_offset = feature_stat.st_size
        feature_file_identity = (feature_stat.st_dev, feature_stat.st_ino)
        feature_prefix_sha256 = sha256_prefix(args.k3_feature_dump, feature_start_offset)
        prefix_records = read_hidden_dump(
            args.k3_feature_dump,
            0,
            feature_start_offset,
            allow_empty=True,
            expected_magic=K3_FEATURE_MAGIC,
        )
        feature_records_before = len(prefix_records)
    started = time.perf_counter()
    output_rows: list[dict[str, Any]] = []
    exact_count = 0
    for index, row in enumerate(teacher):
        prefix = row.get("prefix_tokens")
        target_id = row.get("target_token_id")
        if not isinstance(prefix, list) or not all(isinstance(token, int) for token in prefix):
            raise V13Error(f"teacher样本 {row.get('sample_id')} 的prefix_tokens无效")
        if not isinstance(target_id, int):
            raise V13Error(f"teacher样本 {row.get('sample_id')} 的target_token_id无效")
        request_started = time.perf_counter()
        response = http_json(
            args.endpoint,
            "/completion",
            {
                "prompt": prefix,
                "n_predict": 1,
                "n_probs": args.n_probs,
                "logit_bias": [[target_id, 100.0]],
                "temperature": -1.0,
                "samplers": ["temperature"],
                "seed": 1,
                "stream": False,
                "cache_prompt": False,
                "return_tokens": True,
                "post_sampling_probs": False,
            },
            timeout=args.timeout,
        )
        probabilities = response.get("completion_probabilities")
        if not isinstance(probabilities, list) or len(probabilities) != 1:
            raise V13Error(
                f"样本 {row['sample_id']} 没有返回单token completion_probabilities"
            )
        token_info = probabilities[0]
        generated_id = token_info.get("id") if isinstance(token_info, dict) else None
        if generated_id != target_id:
            raise V13Error(
                f"样本 {row['sample_id']} 强制token失败：期望{target_id}，实际{generated_id}"
            )
        raw_logprob = token_info.get("logprob") if isinstance(token_info, dict) else None
        if (
            not isinstance(raw_logprob, (int, float))
            or not math.isfinite(raw_logprob)
            or raw_logprob > 1e-6
            or raw_logprob <= -1e30
        ):
            raise V13Error(f"样本 {row['sample_id']} 没有精确原始logprob")
        target_logprob = float(raw_logprob)
        exact_count += 1
        output_rows.append(
            {
                "format": FORMAT_SHARD,
                "route": args.route,
                "sample_id": row["sample_id"],
                "task_id": row["task_id"],
                "token_index": row["token_index"],
                "prefix_token_count": len(prefix),
                "target_token_id": target_id,
                "target_logprob": target_logprob,
                "target_nll": -target_logprob,
                "exact": True,
                "censoring": None,
                "generated_token_id": generated_id,
                "n_probs_requested": args.n_probs,
                "request_seconds": round(time.perf_counter() - request_started, 6),
                "hidden_index": index if args.hidden_dump else None,
                "k3_feature_index": index if args.k3_feature_dump else None,
            }
        )

    hidden_record: dict[str, Any] | None = None
    hidden_output: Path | None = None
    if args.hidden_dump:
        assert (
            hidden_start_offset is not None
            and hidden_records_before is not None
            and hidden_prefix_sha256 is not None
            and hidden_file_identity is not None
        )
        hidden_stat = args.hidden_dump.stat()
        hidden_end_offset = hidden_stat.st_size
        if (hidden_stat.st_dev, hidden_stat.st_ino) != hidden_file_identity:
            raise V13Error("隐藏状态dump在采集期间被替换")
        if hidden_end_offset < hidden_start_offset:
            raise V13Error(
                "隐藏状态dump在采集期间被截断或重写: "
                f"start={hidden_start_offset}, end={hidden_end_offset}"
            )
        if sha256_prefix(args.hidden_dump, hidden_start_offset) != hidden_prefix_sha256:
            raise V13Error("隐藏状态dump在采集期间重写了既有字节")
        records = read_hidden_dump(
            args.hidden_dump,
            hidden_start_offset,
            hidden_end_offset,
        )
        sites = tuple(args.sites)
        matrices = align_hidden_records(records, teacher, sites)
        hidden_output = args.hidden_output or Path(str(args.output) + ".hidden.npz")
        hidden_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            hidden_output,
            sample_ids=np.asarray([str(row["sample_id"]) for row in teacher]),
            **{f"layer_{site}": matrix for site, matrix in matrices.items()},
        )
        hidden_record = {
            "file": None,
            "sha256": sha256_file(hidden_output),
            "sites": list(sites),
            "n_embd": int(next(iter(matrices.values())).shape[1]),
            "byte_start_offset": hidden_start_offset,
            "byte_end_offset": hidden_end_offset,
            "records_before_start": hidden_records_before,
            "records_appended": len(records),
            "prefix_sha256_at_start": hidden_prefix_sha256,
            "alignment": "strict-appended-byte-range-v1",
        }

    feature_record: dict[str, Any] | None = None
    feature_output: Path | None = None
    if args.k3_feature_dump:
        assert (
            feature_start_offset is not None
            and feature_records_before is not None
            and feature_prefix_sha256 is not None
            and feature_file_identity is not None
        )
        feature_stat = args.k3_feature_dump.stat()
        feature_end_offset = feature_stat.st_size
        if (feature_stat.st_dev, feature_stat.st_ino) != feature_file_identity:
            raise V13Error("K3特征dump在采集期间被替换")
        if feature_end_offset < feature_start_offset:
            raise V13Error("K3特征dump在采集期间被截断或重写")
        if sha256_prefix(args.k3_feature_dump, feature_start_offset) != feature_prefix_sha256:
            raise V13Error("K3特征dump在采集期间重写了既有字节")
        feature_records = read_hidden_dump(
            args.k3_feature_dump,
            feature_start_offset,
            feature_end_offset,
            expected_magic=K3_FEATURE_MAGIC,
        )
        feature_sites = tuple(args.k3_feature_sites)
        feature_matrices = align_hidden_records(feature_records, teacher, feature_sites)
        if any(matrix.shape[1] != 6 for matrix in feature_matrices.values()):
            raise V13Error("K3残差特征宽度必须为6")
        feature_output = args.k3_feature_output or Path(str(args.output) + ".k3_features.npz")
        feature_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            feature_output,
            sample_ids=np.asarray([str(row["sample_id"]) for row in teacher]),
            feature_names=np.asarray([
                "hidden_rms", "native_rms", "k3_delta_rms",
                "hidden_delta_cos", "native_delta_cos", "energy_gate",
            ]),
            **{
                f"layer_{site}": matrix
                for site, matrix in feature_matrices.items()
            },
        )
        feature_record = {
            "file": None,
            "sha256": sha256_file(feature_output),
            "sites": list(feature_sites),
            "feature_width": 6,
            "feature_names": [
                "hidden_rms", "native_rms", "k3_delta_rms",
                "hidden_delta_cos", "native_delta_cos", "energy_gate",
            ],
            "byte_start_offset": feature_start_offset,
            "byte_end_offset": feature_end_offset,
            "records_before_start": feature_records_before,
            "records_appended": len(feature_records),
            "prefix_sha256_at_start": feature_prefix_sha256,
            "alignment": "strict-appended-byte-range-v1",
        }

    write_jsonl(args.output, output_rows)
    shard_manifest = manifest_path(args.output)
    if hidden_record is not None and hidden_output is not None:
        hidden_record["file"] = relative_to_manifest(hidden_output, shard_manifest)
    if feature_record is not None and feature_output is not None:
        feature_record["file"] = relative_to_manifest(feature_output, shard_manifest)
    manifest = {
        "format": FORMAT_SHARD_MANIFEST,
        "route": args.route,
        "route_attestation": "process-level --force-path plus expected API alias",
        "teacher": str(args.teacher.resolve()),
        "teacher_sha256": sha256_file(args.teacher),
        "shard": args.output.name,
        "shard_sha256": sha256_file(args.output),
        "sample_count": len(output_rows),
        "exact_count": exact_count,
        "exact_coverage": exact_count / len(output_rows),
        "n_probs": args.n_probs,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "server": snapshot,
        "expected_model": args.expect_model,
        "hidden": hidden_record,
        "k3_features": feature_record,
    }
    write_json(shard_manifest, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def parse_shards(values: list[str]) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for value in values:
        if "=" not in value:
            raise V13Error("--shard必须写成 route=path")
        route, raw_path = value.split("=", 1)
        route = route.strip()
        path = Path(raw_path).resolve()
        if not route or route in seen:
            raise V13Error(f"重复或空route: {route}")
        if not path.is_file():
            raise V13Error(f"找不到route shard: {path}")
        seen.add(route)
        result.append((route, path))
    if len(result) < 2:
        raise V13Error("至少需要两个反事实route shard")
    return result


def unit_l2(features: np.ndarray) -> np.ndarray:
    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
        raise V13Error(f"隐藏特征必须是非空二维矩阵，实际shape={features.shape}")
    if not np.isfinite(features).all():
        raise V13Error("隐藏特征包含NaN或Inf")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= np.float32(1e-8)):
        raise V13Error("隐藏特征包含非有限或近零L2范数")
    normalized = features / norms
    if not np.isfinite(normalized).all():
        raise V13Error("L2单位化结果包含NaN或Inf")
    return normalized.astype(np.float32, copy=False)


def softmax(logits: np.ndarray) -> np.ndarray:
    if logits.ndim != 2 or not np.isfinite(logits).all():
        raise V13Error("softmax logits必须是有限二维矩阵")
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    denominator = np.sum(exp, axis=1, keepdims=True)
    probs = exp / denominator
    if not np.isfinite(probs).all() or np.any(denominator <= 0):
        raise V13Error("softmax产生非有限概率")
    return probs


def fit_softmax(
    features: np.ndarray,
    labels: np.ndarray,
    class_count: int,
    epochs: int,
    learning_rate: float,
    l2: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
        raise V13Error("softmax拟合的features/labels形状不一致")
    sample_count, width = features.shape
    if sample_count == 0 or width == 0 or class_count < 2:
        raise V13Error("softmax拟合需要非空样本、特征和至少两个类别")
    if not np.isfinite(features).all():
        raise V13Error("softmax拟合输入包含NaN或Inf")
    if np.any(labels < 0) or np.any(labels >= class_count):
        raise V13Error("softmax拟合标签超出类别范围")
    if not all(math.isfinite(value) for value in (learning_rate, l2)):
        raise V13Error("softmax拟合超参数包含NaN或Inf")
    weights = np.zeros((class_count, width), dtype=np.float32)
    bias = np.zeros(class_count, dtype=np.float32)
    mw = np.zeros_like(weights)
    vw = np.zeros_like(weights)
    mb = np.zeros_like(bias)
    vb = np.zeros_like(bias)
    one_hot = np.eye(class_count, dtype=np.float32)[labels]
    beta1, beta2 = 0.9, 0.999
    loss = math.inf
    for epoch in range(1, epochs + 1):
        logits = features @ weights.T + bias
        probs = softmax(logits)
        loss = float(
            -np.mean(np.log(np.maximum(probs[np.arange(sample_count), labels], 1e-12)))
            + 0.5 * l2 * np.sum(weights * weights)
        )
        if not math.isfinite(loss):
            raise V13Error(f"softmax拟合loss非有限: epoch={epoch}")
        error = (probs - one_hot) / sample_count
        grad_w = error.T @ features + l2 * weights
        grad_b = np.sum(error, axis=0)
        if not np.isfinite(grad_w).all() or not np.isfinite(grad_b).all():
            raise V13Error(f"softmax拟合梯度非有限: epoch={epoch}")
        mw = beta1 * mw + (1.0 - beta1) * grad_w
        vw = beta2 * vw + (1.0 - beta2) * (grad_w * grad_w)
        mb = beta1 * mb + (1.0 - beta1) * grad_b
        vb = beta2 * vb + (1.0 - beta2) * (grad_b * grad_b)
        mw_hat = mw / (1.0 - beta1**epoch)
        vw_hat = vw / (1.0 - beta2**epoch)
        mb_hat = mb / (1.0 - beta1**epoch)
        vb_hat = vb / (1.0 - beta2**epoch)
        weights -= learning_rate * mw_hat / (np.sqrt(vw_hat) + 1e-8)
        bias -= learning_rate * mb_hat / (np.sqrt(vb_hat) + 1e-8)
        if not np.isfinite(weights).all() or not np.isfinite(bias).all():
            raise V13Error(f"softmax拟合权重非有限: epoch={epoch}")
    return weights.astype(np.float32), bias.astype(np.float32), loss


def accuracy(features: np.ndarray, labels: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> float:
    if len(features) == 0 or len(features) != len(labels):
        raise V13Error("accuracy需要非空且等长的features/labels")
    if not all(np.isfinite(value).all() for value in (features, weights, bias)):
        raise V13Error("accuracy输入包含NaN或Inf")
    logits = features @ weights.T + bias
    if not np.isfinite(logits).all():
        raise V13Error("accuracy logits包含NaN或Inf")
    predictions = np.argmax(logits, axis=1)
    return float(np.mean(predictions == labels))


def split_indices(
    task_ids: list[str],
    heldout_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    groups = sorted(set(task_ids))
    rng = random.Random(seed)
    rng.shuffle(groups)
    if len(groups) < 2:
        raise V13Error(
            "至少需要两个有效task才能按任务留出；禁止把同一答案的相邻token随机拆到训练/留出集"
        )
    heldout_groups = set(
        groups[: max(1, min(len(groups) - 1, round(len(groups) * heldout_fraction)))]
    )
    test = np.asarray([index for index, task in enumerate(task_ids) if task in heldout_groups])
    train = np.asarray([index for index, task in enumerate(task_ids) if task not in heldout_groups])
    return train, test, "task_group"


def is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def validate_teacher_artifact(
    teacher_path: Path,
    expected_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    teacher_path = teacher_path.resolve()
    if not teacher_path.is_file():
        raise V13Error(f"teacher文件不存在: {teacher_path}")
    actual_sha256 = sha256_file(teacher_path)
    if actual_sha256 != expected_sha256:
        raise V13Error(f"teacher SHA-256不匹配: {teacher_path}")
    teacher_manifest_path = manifest_path(teacher_path)
    if not teacher_manifest_path.is_file():
        raise V13Error(f"缺少teacher manifest: {teacher_manifest_path}")
    teacher_manifest = read_json_object(teacher_manifest_path, "teacher manifest")
    if teacher_manifest.get("format") != FORMAT_TEACHER_MANIFEST:
        raise V13Error(f"teacher manifest格式不受支持: {teacher_manifest_path}")
    manifest_teacher = teacher_manifest.get("teacher")
    if not isinstance(manifest_teacher, str):
        raise V13Error("teacher manifest缺少teacher路径")
    if resolve_manifest_path(manifest_teacher, teacher_manifest_path) != teacher_path:
        raise V13Error("teacher manifest记录的文件与route shard引用不一致")
    if teacher_manifest.get("teacher_sha256") != actual_sha256:
        raise V13Error("teacher manifest内SHA-256与真实文件不一致")
    rows = read_jsonl(teacher_path)
    if teacher_manifest.get("sample_count") != len(rows):
        raise V13Error("teacher manifest sample_count与真实行数不一致")
    seen: set[str] = set()
    for row_number, row in enumerate(rows, 1):
        sample_id = row.get("sample_id")
        prefix = row.get("prefix_tokens")
        if row.get("format") != FORMAT_TEACHER:
            raise V13Error(f"teacher第{row_number}行format不受支持")
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
            raise V13Error(f"teacher第{row_number}行sample_id为空或重复")
        if not isinstance(row.get("task_id"), str) or not row["task_id"]:
            raise V13Error(f"teacher样本 {sample_id} 缺少task_id")
        if not isinstance(row.get("token_index"), int):
            raise V13Error(f"teacher样本 {sample_id} 的token_index无效")
        if not isinstance(prefix, list) or not prefix or not all(isinstance(token, int) for token in prefix):
            raise V13Error(f"teacher样本 {sample_id} 的prefix_tokens无效")
        if not isinstance(row.get("target_token_id"), int):
            raise V13Error(f"teacher样本 {sample_id} 的target_token_id无效")
        seen.add(sample_id)
    return rows, teacher_manifest


def validate_route_shard(
    route: str,
    path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], Path, list[dict[str, Any]]]:
    path = path.resolve()
    rows = read_jsonl(path)
    shard_manifest_path = manifest_path(path)
    if not shard_manifest_path.is_file():
        raise V13Error(f"缺少route shard manifest: {shard_manifest_path}")
    manifest = read_json_object(shard_manifest_path, "route shard manifest")
    if manifest.get("format") != FORMAT_SHARD_MANIFEST:
        raise V13Error(f"route shard manifest格式不受支持: {shard_manifest_path}")
    if manifest.get("route") != route:
        raise V13Error(f"route shard manifest路线不一致: expected={route}")
    if manifest.get("shard") != path.name:
        raise V13Error(f"route shard manifest文件名不匹配: {path}")
    actual_shard_sha256 = sha256_file(path)
    if manifest.get("shard_sha256") != actual_shard_sha256:
        raise V13Error(f"route shard SHA-256不匹配: {path}")
    if manifest.get("sample_count") != len(rows):
        raise V13Error(f"route shard sample_count不匹配: {path}")

    seen: set[str] = set()
    exact_count = 0
    for row_number, row in enumerate(rows, 1):
        sample_id = row.get("sample_id")
        if row.get("format") != FORMAT_SHARD:
            raise V13Error(f"{path} 第{row_number}行format不受支持")
        if row.get("route") != route:
            raise V13Error(f"{path} 第{row_number}行route不一致")
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
            raise V13Error(f"{path} 第{row_number}行sample_id为空或重复")
        logprob = row.get("target_logprob")
        nll = row.get("target_nll")
        if not is_finite_number(logprob) or not is_finite_number(nll):
            raise V13Error(f"route样本 {sample_id} 的logprob/NLL非有限")
        if float(logprob) > 1e-6 or float(logprob) <= -1e30:
            raise V13Error(f"route样本 {sample_id} 的原始logprob超出有效范围")
        if not math.isclose(float(nll), -float(logprob), rel_tol=1e-7, abs_tol=1e-7):
            raise V13Error(f"route样本 {sample_id} 的NLL不等于-logprob")
        target_id = row.get("target_token_id")
        if not isinstance(target_id, int) or row.get("generated_token_id") != target_id:
            raise V13Error(f"route样本 {sample_id} 的强制生成token与target不一致")
        if not isinstance(row.get("task_id"), str) or not isinstance(row.get("token_index"), int):
            raise V13Error(f"route样本 {sample_id} 的task/token_index无效")
        if row.get("exact") is True:
            exact_count += 1
        elif row.get("exact") is not False:
            raise V13Error(f"route样本 {sample_id} 的exact必须是布尔值")
        seen.add(sample_id)

    expected_coverage = exact_count / len(rows)
    if manifest.get("exact_count") != exact_count:
        raise V13Error(f"route shard exact_count不匹配: {path}")
    coverage = manifest.get("exact_coverage")
    if not is_finite_number(coverage) or not math.isclose(
        float(coverage), expected_coverage, rel_tol=1e-9, abs_tol=1e-12
    ):
        raise V13Error(f"route shard exact_coverage不匹配: {path}")

    teacher_raw = manifest.get("teacher")
    teacher_sha256 = manifest.get("teacher_sha256")
    if not isinstance(teacher_raw, str) or not isinstance(teacher_sha256, str):
        raise V13Error(f"route shard缺少teacher路径或SHA-256: {path}")
    teacher_path = resolve_manifest_path(teacher_raw, shard_manifest_path)
    teacher_rows, _ = validate_teacher_artifact(teacher_path, teacher_sha256)
    teacher_by_id = {str(row["sample_id"]): row for row in teacher_rows}
    if not seen.issubset(teacher_by_id):
        raise V13Error(f"route shard包含teacher中不存在的sample_id: {path}")
    for row in rows:
        sample_id = str(row["sample_id"])
        teacher = teacher_by_id[sample_id]
        if (
            row.get("task_id") != teacher.get("task_id")
            or row.get("token_index") != teacher.get("token_index")
            or row.get("target_token_id") != teacher.get("target_token_id")
            or row.get("prefix_token_count") != len(teacher["prefix_tokens"])
        ):
            raise V13Error(f"route样本与teacher字段不一致: {sample_id}")

    expected_model = manifest.get("expected_model")
    server = manifest.get("server")
    aliases = server.get("models") if isinstance(server, dict) else None
    if (
        not isinstance(expected_model, str)
        or not expected_model
        or not isinstance(aliases, list)
        or not all(isinstance(alias, str) for alias in aliases)
        or expected_model not in aliases
    ):
        raise V13Error(f"route shard expected_model不在server model aliases中: {path}")
    return rows, manifest, shard_manifest_path, teacher_rows


def calibrate_router(args: argparse.Namespace) -> int:
    shards = parse_shards(args.shard)
    classes = ["no_op", "coder", "k3"]
    if len(shards) != 3 or {route for route, _ in shards} != set(classes):
        raise V13Error("正式v13校准必须且只能提供no_op、coder、k3三个route shard")
    path_by_route = dict(shards)
    shards = [(route, path_by_route[route]) for route in classes]
    if args.feature_route not in classes:
        raise V13Error(f"feature-route {args.feature_route} 不在shard中")

    rows_by_route: dict[str, dict[str, dict[str, Any]]] = {}
    manifests: dict[str, tuple[Path, dict[str, Any]]] = {}
    teacher_orders: dict[str, list[str]] = {}
    for route, path in shards:
        rows, manifest, mpath, teacher_rows = validate_route_shard(route, path)
        rows_by_route[route] = {str(row["sample_id"]): row for row in rows}
        manifests[route] = (mpath, manifest)
        teacher_orders[route] = [
            str(row["sample_id"])
            for row in teacher_rows
            if str(row["sample_id"]) in rows_by_route[route]
        ]
    key_sets = [set(mapping) for mapping in rows_by_route.values()]
    if any(keys != key_sets[0] for keys in key_sets[1:]):
        raise V13Error("各route shard的sample_id集合不完全一致")
    if any(order != teacher_orders[classes[0]] for order in teacher_orders.values()):
        raise V13Error("各route shard的teacher样本顺序不完全一致")
    teacher_hashes = {value[1]["teacher_sha256"] for value in manifests.values()}
    if len(teacher_hashes) != 1:
        raise V13Error("各route shard并非来自同一teacher")
    first_manifest_path, first_manifest = manifests[classes[0]]
    source_teacher_path = resolve_manifest_path(first_manifest["teacher"], first_manifest_path)
    source_teacher_manifest_path = manifest_path(source_teacher_path)

    feature_rows = rows_by_route[args.feature_route]
    ordered_ids = teacher_orders[classes[0]]
    selected_ids: list[str] = []
    labels: list[int] = []
    margins: list[float] = []
    task_ids: list[str] = []
    hidden_indices: list[int] = []
    censored = 0
    low_margin = 0
    for sample_id in ordered_ids:
        route_rows = [rows_by_route[route][sample_id] for route in classes]
        if any(
            row.get("target_token_id") != route_rows[0].get("target_token_id")
            or row.get("task_id") != route_rows[0].get("task_id")
            or row.get("token_index") != route_rows[0].get("token_index")
            for row in route_rows[1:]
        ):
            raise V13Error(f"反事实样本不一致: {sample_id}")
        if any(row.get("exact") is not True for row in route_rows):
            censored += 1
            continue
        nlls = np.asarray([float(row["target_nll"]) for row in route_rows], dtype=np.float64)
        if not np.isfinite(nlls).all():
            raise V13Error(f"反事实样本NLL非有限: {sample_id}")
        order = np.argsort(nlls)
        margin = float(nlls[order[1]] - nlls[order[0]])
        if not math.isfinite(margin):
            raise V13Error(f"反事实样本margin非有限: {sample_id}")
        if margin < args.margin:
            low_margin += 1
            continue
        hidden_index = feature_rows[sample_id].get("hidden_index")
        if not isinstance(hidden_index, int):
            raise V13Error(f"feature route缺少hidden_index: {sample_id}")
        selected_ids.append(sample_id)
        labels.append(int(order[0]))
        margins.append(margin)
        task_ids.append(str(route_rows[0]["task_id"]))
        hidden_indices.append(hidden_index)
    if len(selected_ids) < args.min_samples:
        raise V13Error(
            f"可训练token只有{len(selected_ids)}个，少于--min-samples={args.min_samples}；"
            f"censored={censored}, low_margin={low_margin}"
        )

    feature_manifest_path, feature_manifest = manifests[args.feature_route]
    hidden = feature_manifest.get("hidden")
    if not isinstance(hidden, dict) or not isinstance(hidden.get("file"), str):
        raise V13Error("feature route manifest缺少隐藏状态sidecar")
    hidden_path = resolve_manifest_path(hidden["file"], feature_manifest_path)
    if not hidden_path.is_file():
        raise V13Error(f"隐藏状态sidecar不存在: {hidden_path}")
    if sha256_file(hidden_path) != hidden.get("sha256"):
        raise V13Error("隐藏状态sidecar SHA-256不匹配")
    hidden_sites_raw = hidden.get("sites")
    n_embd = hidden.get("n_embd")
    if (
        not isinstance(hidden_sites_raw, list)
        or not hidden_sites_raw
        or not all(isinstance(site, int) and not isinstance(site, bool) and site >= 0 for site in hidden_sites_raw)
        or len(set(hidden_sites_raw)) != len(hidden_sites_raw)
    ):
        raise V13Error("隐藏状态sidecar sites无效")
    if not isinstance(n_embd, int) or isinstance(n_embd, bool) or n_embd <= 0:
        raise V13Error("隐藏状态sidecar n_embd无效")
    hidden_sites = tuple(int(site) for site in hidden_sites_raw)
    sites = hidden_sites if args.sites is None else tuple(args.sites)
    missing_sites = [site for site in sites if site not in hidden_sites]
    if missing_sites:
        raise V13Error(f"--sites不在隐藏状态sidecar中: {missing_sites}")

    hidden_matrices: dict[int, np.ndarray] = {}
    try:
        with np.load(hidden_path, allow_pickle=False) as archive:
            if "sample_ids" not in archive:
                raise V13Error("隐藏状态sidecar缺少sample_ids")
            archived_ids = [str(value) for value in archive["sample_ids"].tolist()]
            if archived_ids != ordered_ids:
                raise V13Error("隐藏状态sidecar的sample_ids或顺序与teacher不一致")
            for site in hidden_sites:
                key = f"layer_{site}"
                if key not in archive:
                    raise V13Error(f"隐藏状态sidecar缺少 {key}")
                matrix = np.asarray(archive[key], dtype=np.float32)
                if matrix.shape != (len(ordered_ids), n_embd):
                    raise V13Error(
                        f"隐藏状态sidecar {key} shape不一致: "
                        f"expected={(len(ordered_ids), n_embd)}, actual={matrix.shape}"
                    )
                if not np.isfinite(matrix).all():
                    raise V13Error(f"隐藏状态sidecar {key} 包含NaN或Inf")
                hidden_matrices[site] = matrix.copy()
    except (OSError, ValueError) as error:
        raise V13Error(f"无法读取隐藏状态sidecar: {hidden_path}: {error}") from error

    for sample_id, index in zip(selected_ids, hidden_indices):
        if index < 0 or index >= len(archived_ids) or archived_ids[index] != sample_id:
            raise V13Error(f"隐藏状态索引与sample_id不一致: {sample_id}")
    for expected_index, sample_id in enumerate(ordered_ids):
        hidden_index = feature_rows[sample_id].get("hidden_index")
        if hidden_index != expected_index:
            raise V13Error(
                f"feature route的hidden_index不是完整连续sidecar索引: {sample_id}"
            )

    y = np.asarray(labels, dtype=np.int64)
    train_index, test_index, split_kind = split_indices(task_ids, args.heldout_fraction, args.seed)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    class_counts = {classes[index]: int(np.sum(y == index)) for index in range(len(classes))}
    site_reports: list[dict[str, Any]] = []
    fitted_sites: list[dict[str, Any]] = []
    for site in sites:
        raw_features = hidden_matrices[site][hidden_indices]
        features = unit_l2(raw_features)
        weights, bias, train_loss = fit_softmax(
            features[train_index],
            y[train_index],
            len(classes),
            args.epochs,
            args.learning_rate,
            args.l2,
        )
        heldout_accuracy = accuracy(features[test_index], y[test_index], weights, bias)
        train_accuracy = accuracy(features[train_index], y[train_index], weights, bias)
        majority_class = int(np.bincount(y[train_index], minlength=len(classes)).argmax())
        majority_accuracy = float(np.mean(y[test_index] == majority_class))
        no_op_accuracy = (
            float(np.mean(y[test_index] == classes.index("no_op"))) if "no_op" in classes else None
        )
        promotable = heldout_accuracy >= majority_accuracy + args.min_lift
        site_report = {
            "site": site,
            "train_accuracy": train_accuracy,
            "heldout_accuracy": heldout_accuracy,
            "majority_route": classes[majority_class],
            "majority_heldout_accuracy": majority_accuracy,
            "always_no_op_heldout_accuracy": no_op_accuracy,
            "required_lift": args.min_lift,
            "promotable": promotable,
            "train_loss": train_loss,
            "artifact_fit_scope": "train_tasks_only",
            "heldout_used_in_fit": False,
            "independent_final_test_required_before_full_data_refit": True,
        }
        site_reports.append(site_report)
        fitted_sites.append(
            {
                "site": site,
                "input_width": int(raw_features.shape[1]),
                "weights": weights,
                "bias": bias,
                "report": site_report,
            }
        )

    status = "candidate" if all(item["promotable"] for item in site_reports) else "rejected_control"
    site_plan_path = output_dir / "ColorLM-v13-Intelligence-Router.intplan.json"
    if status != "candidate":
        site_plan_path.unlink(missing_ok=True)

    artifact_records: list[dict[str, Any]] = []
    for fitted in fitted_sites:
        site = int(fitted["site"])
        weights = np.asarray(fitted["weights"], dtype=np.float32)
        bias = np.asarray(fitted["bias"], dtype=np.float32)
        if not np.isfinite(weights).all() or not np.isfinite(bias).all():
            raise V13Error(f"拒绝写入非有限路由权重: layer={site}")
        site_dir = output_dir / f"layer_{site}"
        site_dir.mkdir(parents=True, exist_ok=True)
        weight_path = site_dir / "weight.f32"
        bias_path = site_dir / "bias.f32"
        weights.astype("<f4", copy=False).tofile(weight_path)
        bias.astype("<f4", copy=False).tofile(bias_path)
        weight_record = {
            "file": weight_path.name,
            "dtype": "float32-le",
            "ggml_shape": [int(fitted["input_width"]), len(classes)],
            "bytes": weight_path.stat().st_size,
            "sha256": sha256_file(weight_path),
        }
        bias_record = {
            "file": bias_path.name,
            "dtype": "float32-le",
            "ggml_shape": [len(classes)],
            "bytes": bias_path.stat().st_size,
            "sha256": sha256_file(bias_path),
        }
        site_manifest = {
            "format": "colorlm-intelligence-router-v1",
            "status": status,
            "deployable": status == "candidate",
            "runtime_layout": "two-headerless-f32-le-row-major-v1",
            "input_width": int(fitted["input_width"]),
            "input_transform": "per-token-l2-unit",
            "route_count": len(classes),
            "routes": classes,
            "tensors": {"weight": weight_record, "bias": bias_record},
            "artifact_fit_scope": "train_tasks_only",
            "heldout_used_in_fit": False,
            "independent_final_test_required_before_full_data_refit": True,
            "calibration": fitted["report"],
        }
        site_manifest_path = site_dir / "router.json"
        write_json(site_manifest_path, site_manifest)
        artifact_records.append(
            {
                "site": site,
                "directory": site_dir.name,
                "manifest": site_manifest_path.name,
                "manifest_sha256": sha256_file(site_manifest_path),
                "weight": weight_record,
                "bias": bias_record,
            }
        )

    router_manifest = {
        "format": FORMAT_ROUTER,
        "status": status,
        "classes": classes,
        "input": {
            "tensor": "attn_post_norm",
            "transform": "per-token-l2-unit",
            "n_embd": n_embd,
            "sites": list(sites),
            "hidden_sidecar_sites": list(hidden_sites),
        },
        "supervision": {
            "applies_to_sites": list(sites),
            "scope": "single_site" if len(sites) == 1 else "multi_site",
            "routes": classes,
        },
        "label": "argmin exact counterfactual next-token NLL",
        "margin": args.margin,
        "sample_accounting": {
            "teacher_total": len(ordered_ids),
            "selected": len(selected_ids),
            "censored": censored,
            "low_margin": low_margin,
            "class_counts": class_counts,
            "mean_margin": float(np.mean(margins)),
        },
        "split": {
            "kind": split_kind,
            "train": len(train_index),
            "heldout": len(test_index),
            "seed": args.seed,
        },
        "optimizer": {
            "name": "full-batch Adam softmax regression",
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "l2": args.l2,
        },
        "artifact_fit_scope": "train_tasks_only",
        "heldout_used_in_fit": False,
        "independent_final_test_required_before_full_data_refit": True,
        "sites": site_reports,
        "artifacts": artifact_records,
        "source_teacher_sha256": next(iter(teacher_hashes)),
        "source_teacher_manifest": str(source_teacher_manifest_path),
        "source_teacher_manifest_sha256": sha256_file(source_teacher_manifest_path),
        "source_shards": [
            {
                "route": route,
                "file": str(path),
                "sha256": sha256_file(path),
                "manifest": str(manifests[route][0]),
                "manifest_sha256": sha256_file(manifests[route][0]),
                "expected_model": manifests[route][1]["expected_model"],
            }
            for route, path in shards
        ],
        "integrity_verification": {
            "status": "passed",
            "route_shards": "format, route, filename, SHA-256, counts, exact coverage and rows",
            "teacher": "path, manifest format, SHA-256, sample count and sample identity",
            "server_aliases": "expected model present for every route",
            "hidden_sidecar": "SHA-256, sites, n_embd, sample order, shapes and finite values",
        },
        "runtime_contract": {
            "logits": "W @ l2_unit(attn_post_norm) + bias",
            "probabilities": "softmax(logits)",
            "no_op": "must create no residual nodes when selected by a future lazy executor",
            "current_runtime_support": True,
            "execution": "dense-residual-v1",
        },
        "runtime_plan": {
            "generated": status == "candidate",
            "reason": (
                "all selected sites exceeded the heldout control gate"
                if status == "candidate"
                else "at least one selected site did not exceed the heldout control gate"
            ),
            "file": site_plan_path.name if status == "candidate" else None,
        },
    }
    calibration_report = output_dir / "calibration_report.json"
    write_json(calibration_report, router_manifest)
    site_plan: dict[str, Any] | None = None
    if status == "candidate":
        site_plan = {
            "format": "colorlm-intelligence-site-plan-v1",
            "status": "candidate",
            "name": "ColorLM-v13-Intelligence-Router",
            "input_width": n_embd,
            "input_transform": "per-token-l2-unit",
            "routes": classes,
            "routing": "counterfactual-next-token-NLL calibrated linear softmax",
            "execution": "dense-residual-v1",
            "supervision_sites": list(sites),
            "sites": [
                {
                    "site": int(artifact["site"]),
                    "temperature": 1.0,
                    "router": artifact["directory"],
                    "manifest_sha256": artifact["manifest_sha256"],
                }
                for artifact in artifact_records
            ],
            "calibration_report": calibration_report.name,
            "calibration_report_sha256": sha256_file(calibration_report),
        }
        write_json(site_plan_path, site_plan)
    print(
        json.dumps(
            {
                "calibration": router_manifest,
                "site_plan": str(site_plan_path) if site_plan is not None else None,
                "site_plan_sha256": sha256_file(site_plan_path) if site_plan is not None else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def inspect_dump(args: argparse.Namespace) -> int:
    records = read_hidden_dump(args.path)
    summary = {
        "path": str(args.path.resolve()),
        "sha256": sha256_file(args.path),
        "record_count": len(records),
        "records": [
            {"index": index, "layer": record.layer, "type": record.tensor_type, "shape": record.shape}
            for index, record in enumerate(records)
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def self_test(_: argparse.Namespace) -> int:
    rng = np.random.default_rng(7)
    labels = np.repeat(np.arange(3), 16).astype(np.int64)
    features = rng.normal(0.0, 0.03, size=(len(labels), 8)).astype(np.float32)
    features[np.arange(len(labels)), labels] += 1.0
    features = unit_l2(features)
    weights, bias, loss = fit_softmax(features, labels, 3, 180, 0.04, 1e-4)
    score = accuracy(features, labels, weights, bias)
    if score < 0.98 or not math.isfinite(loss):
        raise V13Error(f"softmax自测失败: accuracy={score}, loss={loss}")

    with tempfile.TemporaryDirectory(prefix="colorlm-v13-") as directory:
        dump = Path(directory) / "hidden.bin"
        rows = [
            {"sample_id": "a", "prefix_tokens": [1, 2, 3, 4]},
            {"sample_id": "b", "prefix_tokens": [1, 2, 3, 4, 5]},
        ]

        def append_record(stream: Any, layer: int, tokens: int, values: np.ndarray) -> None:
            payload = values.astype("<f4", copy=False)
            stream.write(
                HIDDEN_HEADER.pack(
                    HIDDEN_MAGIC,
                    1,
                    layer,
                    0,
                    8,
                    tokens,
                    1,
                    1,
                    payload.nbytes,
                )
            )
            stream.write(payload.tobytes())

        with dump.open("wb") as stream:
            # warmup故意与首个teacher拥有相同长度，字节边界必须把它隔离在外。
            for layer, tokens in ((12, 4), (28, 4)):
                values = np.arange(tokens * 8, dtype="<f4") + layer
                append_record(stream, layer, tokens, values)
            start_offset = stream.tell()
            for layer, tokens in ((12, 4), (28, 4), (12, 5), (28, 5)):
                values = np.arange(tokens * 8, dtype="<f4") + layer
                append_record(stream, layer, tokens, values)
        prefix_records = read_hidden_dump(dump, 0, start_offset, allow_empty=True)
        records = read_hidden_dump(dump, start_offset, dump.stat().st_size)
        matrices = align_hidden_records(records, rows, (12, 28))
        if len(prefix_records) != 2 or matrices[12].shape != (2, 8):
            raise V13Error("隐藏状态解析/对齐自测失败")
        try:
            align_hidden_records(read_hidden_dump(dump), rows, (12, 28))
        except V13Error:
            pass
        else:
            raise V13Error("warmup与teacher同长度时必须依赖字节边界而不是扫描匹配")

        nan_dump = Path(directory) / "hidden_nan.bin"
        with nan_dump.open("wb") as stream:
            values = np.zeros(32, dtype="<f4")
            values[3] = np.nan
            append_record(stream, 12, 4, values)
        try:
            read_hidden_dump(nan_dump)
        except V13Error:
            pass
        else:
            raise V13Error("隐藏状态NaN负例没有被拒绝")

        try:
            unit_l2(np.asarray([[0.0, 0.0]], dtype=np.float32))
        except V13Error:
            pass
        else:
            raise V13Error("近零范数特征负例没有被拒绝")
    print(json.dumps({"result": "passed", "softmax_accuracy": score, "loss": loss}, indent=2))
    return 0


def parse_sites(raw: str) -> tuple[int, ...]:
    try:
        sites = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("sites必须是逗号分隔整数") from error
    if not sites or len(set(sites)) != len(sites) or any(site < 0 for site in sites):
        raise argparse.ArgumentTypeError("sites不能为空、重复或为负")
    return sites


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ColorLM v13反事实NLL采集与智能路由校准")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="将UTF-8任务JSONL变成teacher token前缀")
    prepare.add_argument("--endpoint", default="http://127.0.0.1:8110")
    prepare.add_argument("--expect-model")
    prepare.add_argument("--tasks", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--max-target-tokens", type=int, default=16)
    prepare.add_argument("--max-prefix-tokens", type=int, default=384)
    prepare.add_argument("--max-samples", type=int, default=64)
    prepare.set_defaults(handler=prepare_teacher)

    collect = subparsers.add_parser("collect", help="采集一个进程级强制路径的next-token NLL")
    collect.add_argument("--endpoint", default="http://127.0.0.1:8110")
    collect.add_argument("--expect-model", required=True)
    collect.add_argument("--route", required=True)
    collect.add_argument("--teacher", type=Path, required=True)
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--n-probs", type=int, default=1)
    collect.add_argument("--timeout", type=float, default=90.0)
    collect.add_argument("--max-samples", type=int)
    collect.add_argument("--hidden-dump", type=Path)
    collect.add_argument("--hidden-output", type=Path)
    collect.add_argument("--sites", type=parse_sites, default=DEFAULT_SITES)
    collect.add_argument("--k3-feature-dump", type=Path)
    collect.add_argument("--k3-feature-output", type=Path)
    collect.add_argument("--k3-feature-sites", type=parse_sites, default=(28,))
    collect.set_defaults(handler=collect_route)

    calibrate = subparsers.add_parser("calibrate", help="合并反事实分片并拟合每站点softmax路由")
    calibrate.add_argument("--shard", action="append", required=True, help="route=path，可重复")
    calibrate.add_argument("--feature-route", default="no_op")
    calibrate.add_argument(
        "--sites",
        type=parse_sites,
        help="只拟合指定站点，例如12或28；默认使用隐藏sidecar中的全部站点",
    )
    calibrate.add_argument("--output-dir", type=Path, required=True)
    calibrate.add_argument("--margin", type=float, default=0.05)
    calibrate.add_argument("--min-samples", type=int, default=12)
    calibrate.add_argument("--heldout-fraction", type=float, default=0.25)
    calibrate.add_argument("--min-lift", type=float, default=0.10)
    calibrate.add_argument("--epochs", type=int, default=200)
    calibrate.add_argument("--learning-rate", type=float, default=0.03)
    calibrate.add_argument("--l2", type=float, default=1e-3)
    calibrate.add_argument("--seed", type=int, default=13)
    calibrate.set_defaults(handler=calibrate_router)

    inspect = subparsers.add_parser("inspect-dump", help="只读检查CLM9隐藏状态dump")
    inspect.add_argument("path", type=Path)
    inspect.set_defaults(handler=inspect_dump)

    test = subparsers.add_parser("self-test", help="不启动模型的解析与优化器自测")
    test.set_defaults(handler=self_test)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.command == "prepare":
        if args.max_target_tokens <= 0 or args.max_prefix_tokens <= 0 or args.max_samples <= 0:
            raise V13Error("prepare的token/sample限制必须为正数")
    elif args.command == "collect":
        if (
            args.n_probs <= 0
            or not math.isfinite(args.timeout)
            or args.timeout <= 0
            or (args.max_samples is not None and args.max_samples <= 0)
        ):
            raise V13Error("collect的n-probs/timeout/max-samples必须为正数")
        if args.hidden_output and not args.hidden_dump:
            raise V13Error("--hidden-output必须和--hidden-dump一起使用")
        if args.k3_feature_output and not args.k3_feature_dump:
            raise V13Error("--k3-feature-output必须和--k3-feature-dump一起使用")
    elif args.command == "calibrate":
        float_values = (
            args.margin,
            args.heldout_fraction,
            args.min_lift,
            args.learning_rate,
            args.l2,
        )
        if not all(math.isfinite(value) for value in float_values):
            raise V13Error("calibrate浮点参数不能是NaN或Inf")
        if args.margin < 0 or not 0.0 < args.heldout_fraction < 1.0:
            raise V13Error("margin必须非负，heldout-fraction必须在(0,1)内")
        if not 0.0 <= args.min_lift <= 1.0:
            raise V13Error("min-lift必须在[0,1]内")
        if args.min_samples < 4 or args.epochs <= 0 or args.learning_rate <= 0 or args.l2 < 0:
            raise V13Error("calibrate参数超出范围")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_args(args)
        return int(args.handler(args))
    except (OSError, KeyError, TypeError, ValueError, V13Error) as error:
        print(f"v13失败: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
