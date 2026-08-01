"""Collect, inspect, and fit a deep-activation coordinate bridge for ColorLM v18.

The fitter consumes paired CLM9 traces.  It never starts a model itself during
``fit`` and keeps train/held-out separation at whole-prompt granularity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from scipy.linalg import svd as scipy_svd
except ImportError:  # pragma: no cover - NumPy remains a valid fallback.
    scipy_svd = None


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BASELINE = (
    ROOT / "fast16" / "research" / "coder_next_to_colorlm_orthogonal_f32.npy"
)
CLM9_MAGIC = 0x394D4C43
CLM9_HEADER = struct.Struct("<IIiI4qQ")
RECEIPT_FORMAT = "colorlm-activation-capture-v2"
REPORT_FORMAT = "colorlm-deep-activation-bridge-v1"


class BridgeError(RuntimeError):
    """A contract violation that should be shown directly to the operator."""


@dataclass(frozen=True)
class HiddenRecord:
    offset: int
    end_offset: int
    layer: int
    tensor_type: int
    shape: tuple[int, int, int, int]
    values: np.ndarray

    @property
    def width(self) -> int:
        return self.shape[0]

    @property
    def token_count(self) -> int:
        return self.shape[1] * self.shape[2] * self.shape[3]

    def rows(self) -> np.ndarray:
        # GGML stores ne[0] contiguously, so each row is one token position.
        return self.values.reshape(self.token_count, self.width)


@dataclass(frozen=True)
class PairedPrompt:
    prompt_id: str
    base: np.ndarray
    donor: np.ndarray


@dataclass(frozen=True)
class TokenBoundaryMap:
    by_end: dict[int, int]
    token_count: int
    mapped_count: int
    excluded_count: int
    prompt_bytes: int
    consumed_bytes: int


def sha256_file(path: Path, byte_count: int | None = None) -> str:
    digest = hashlib.sha256()
    remaining = byte_count
    with path.open("rb") as stream:
        while remaining is None or remaining > 0:
            size = 1024 * 1024 if remaining is None else min(1024 * 1024, remaining)
            chunk = stream.read(size)
            if not chunk:
                break
            digest.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    if remaining not in (None, 0):
        raise BridgeError(f"文件短于请求的SHA-256前缀: {path}, remaining={remaining}")
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BridgeError(f"无法读取JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise BridgeError(f"JSON根必须是对象: {path}")
    return value


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
                raise BridgeError(f"{path}:{line_number} JSON无效: {error}") from error
            if not isinstance(value, dict):
                raise BridgeError(f"{path}:{line_number} 必须是JSON对象")
            rows.append(value)
    if not rows:
        raise BridgeError(f"JSONL没有记录: {path}")
    return rows


def read_clm9(
    path: Path,
    start_offset: int = 0,
    end_offset: int | None = None,
) -> list[HiddenRecord]:
    if not path.is_file():
        raise BridgeError(f"隐藏态dump不存在: {path}")
    size = path.stat().st_size
    end = size if end_offset is None else end_offset
    if start_offset < 0 or end < start_offset or end > size:
        raise BridgeError(
            f"CLM9字节范围无效: start={start_offset}, end={end}, size={size}"
        )
    records: list[HiddenRecord] = []
    final_position = start_offset
    with path.open("rb") as stream:
        stream.seek(start_offset)
        while stream.tell() < end:
            offset = stream.tell()
            raw = stream.read(CLM9_HEADER.size)
            if len(raw) != CLM9_HEADER.size:
                raise BridgeError(f"CLM9 header不完整: {path}, offset={offset}")
            magic, version, layer, tensor_type, *tail = CLM9_HEADER.unpack(raw)
            ne0, ne1, ne2, ne3, payload_bytes = tail
            shape = (int(ne0), int(ne1), int(ne2), int(ne3))
            if magic != CLM9_MAGIC or version != 1:
                raise BridgeError(
                    f"CLM9 header不受支持: offset={offset}, magic={magic:#x}, "
                    f"version={version}"
                )
            if tensor_type != 0:
                raise BridgeError(f"只接受GGML_TYPE_F32，实际type={tensor_type}")
            if any(dimension <= 0 for dimension in shape):
                raise BridgeError(f"CLM9 shape非法: {shape}")
            expected = math.prod(shape) * 4
            if payload_bytes != expected or payload_bytes > end - stream.tell():
                raise BridgeError(
                    f"CLM9 payload尺寸错误: shape={shape}, bytes={payload_bytes}, "
                    f"expected={expected}"
                )
            payload = stream.read(payload_bytes)
            values = np.frombuffer(payload, dtype="<f4").astype(np.float32, copy=True)
            if not np.isfinite(values).all():
                raise BridgeError(f"CLM9包含NaN或Inf: layer={layer}, offset={offset}")
            records.append(
                HiddenRecord(
                    offset=offset,
                    end_offset=stream.tell(),
                    layer=int(layer),
                    tensor_type=int(tensor_type),
                    shape=shape,
                    values=values,
                )
            )
        final_position = stream.tell()
    if final_position != end:
        raise BridgeError(
            f"CLM9范围未在记录边界结束: actual={final_position}, end={end}"
        )
    return records


def normalize_endpoint(endpoint: str) -> str:
    result = endpoint.rstrip("/")
    if result.endswith("/v1"):
        result = result[:-3]
    if not result.startswith(("http://", "https://")):
        result = "http://" + result
    return result


def http_json(
    endpoint: str,
    path: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 60.0,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        normalize_endpoint(endpoint) + path,
        data=data,
        headers={"content-type": "application/json"},
        method="GET" if data is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise BridgeError(f"HTTP请求失败: {request.full_url}: {error}") from error
    if not isinstance(value, dict):
        raise BridgeError(f"HTTP响应不是JSON对象: {request.full_url}")
    return value


def canonical_piece(piece: Any) -> str | list[int]:
    if isinstance(piece, str):
        return piece
    if isinstance(piece, list) and all(isinstance(item, int) for item in piece):
        return [int(item) for item in piece]
    raise BridgeError(f"token piece格式不受支持: {piece!r}")


def piece_bytes(piece: Any) -> bytes:
    canonical = canonical_piece(piece)
    if isinstance(canonical, str):
        return canonical.encode("utf-8")
    try:
        return bytes(canonical)
    except ValueError as error:
        raise BridgeError(f"token piece包含无效字节: {canonical!r}") from error


def map_token_end_boundaries(
    prompt: str, tokens: list[dict[str, Any]]
) -> TokenBoundaryMap:
    """Map token rows to cumulative UTF-8 byte ends in the original prompt.

    Pieces that do not match the current prompt cursor are treated as special or
    otherwise unmappable.  They do not advance the cursor, so a following normal
    piece can still resume alignment after a BOS/control token.
    """

    raw_prompt = prompt.encode("utf-8")
    cursor = 0
    by_end: dict[int, int] = {}
    excluded = 0
    for token_index, token in enumerate(tokens):
        if not isinstance(token, dict) or "piece" not in token:
            raise BridgeError(f"token记录缺少piece: index={token_index}")
        raw_piece = piece_bytes(token["piece"])
        if not raw_piece or raw_prompt[cursor : cursor + len(raw_piece)] != raw_piece:
            excluded += 1
            continue
        cursor += len(raw_piece)
        if cursor in by_end:
            raise BridgeError(f"token累计字节结束边界重复: byte={cursor}")
        by_end[cursor] = token_index
    return TokenBoundaryMap(
        by_end=by_end,
        token_count=len(tokens),
        mapped_count=len(by_end),
        excluded_count=excluded,
        prompt_bytes=len(raw_prompt),
        consumed_bytes=cursor,
    )


def tokenize_with_pieces(endpoint: str, text: str) -> list[dict[str, Any]]:
    response = http_json(
        endpoint,
        "/tokenize",
        {
            "content": text,
            "add_special": True,
            "parse_special": True,
            "with_pieces": True,
        },
        timeout=15.0,
    )
    tokens = response.get("tokens")
    if not isinstance(tokens, list) or not tokens:
        raise BridgeError("/tokenize没有返回token")
    result: list[dict[str, Any]] = []
    for token in tokens:
        if not isinstance(token, dict) or not isinstance(token.get("id"), int):
            raise BridgeError(f"/tokenize记录无效: {token!r}")
        result.append({"id": int(token["id"]), "piece": canonical_piece(token.get("piece"))})
    return result


def wait_for_dump(path: Path, previous_size: int, timeout: float) -> int:
    deadline = time.monotonic() + timeout
    stable_since: float | None = None
    last_size = previous_size
    while time.monotonic() < deadline:
        current = path.stat().st_size if path.is_file() else 0
        if current > previous_size and current == last_size:
            if stable_since is None:
                stable_since = time.monotonic()
            elif time.monotonic() - stable_since >= 0.15:
                return current
        else:
            stable_since = None
            last_size = current
        time.sleep(0.05)
    raise BridgeError(f"等待隐藏态dump追加超时: {path}")


def collect(args: argparse.Namespace) -> int:
    prompts = read_jsonl(args.prompts)
    ids: set[str] = set()
    normalized: list[tuple[str, str]] = []
    for index, row in enumerate(prompts):
        prompt_id = str(row.get("id", row.get("prompt_id", index)))
        text = row.get("text", row.get("prompt"))
        if prompt_id in ids or not isinstance(text, str) or not text:
            raise BridgeError(f"prompt记录无效或ID重复: {prompt_id}")
        ids.add(prompt_id)
        normalized.append((prompt_id, text))

    models = http_json(args.endpoint, "/v1/models", timeout=10.0)
    model_ids = [
        str(item.get("id"))
        for item in models.get("data", [])
        if isinstance(item, dict)
    ]
    if args.expect_model and args.expect_model not in model_ids:
        raise BridgeError(f"模型alias不匹配: expected={args.expect_model}, actual={model_ids}")
    model = args.expect_model or (model_ids[0] if model_ids else None)
    if model is None:
        raise BridgeError("服务没有模型alias")

    captures: list[dict[str, Any]] = []
    for prompt_id, text in normalized:
        tokens = tokenize_with_pieces(args.endpoint, text)
        before = args.dump.stat().st_size if args.dump.is_file() else 0
        http_json(
            args.endpoint,
            "/completion",
            {
                "prompt": text,
                "n_predict": 1,
                "temperature": 0.0,
                "seed": args.seed,
                "cache_prompt": False,
            },
            timeout=args.timeout,
        )
        after = wait_for_dump(args.dump, before, min(args.timeout, 10.0))
        records = read_clm9(args.dump, before, after)
        candidates = [
            record
            for record in records
            if record.layer == args.layer and record.token_count == len(tokens)
        ]
        if len(candidates) != 1:
            observed = [
                {"layer": record.layer, "shape": record.shape, "offset": record.offset}
                for record in records
            ]
            raise BridgeError(
                f"{prompt_id} 无法唯一定位prefill隐藏态: tokens={len(tokens)}, "
                f"candidates={len(candidates)}, observed={observed}"
            )
        record = candidates[0]
        captures.append(
            {
                "prompt_id": prompt_id,
                "prompt": text,
                "prompt_utf8_bytes": len(text.encode("utf-8")),
                "prompt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "token_count": len(tokens),
                "tokens": tokens,
                "request_dump_start": before,
                "request_dump_end": after,
                "record_offset": record.offset,
                "record_end_offset": record.end_offset,
                "shape": list(record.shape),
            }
        )
        print(f"{prompt_id}: {len(tokens)} tokens")

    final_size = args.dump.stat().st_size
    receipt = {
        "format": RECEIPT_FORMAT,
        "endpoint": normalize_endpoint(args.endpoint),
        "model": model,
        "dump": os.fspath(args.dump.resolve()),
        "dump_prefix_bytes": final_size,
        "dump_prefix_sha256": sha256_file(args.dump, final_size),
        "layer": args.layer,
        "stage": args.stage,
        "capture_count": len(captures),
        "captures": captures,
    }
    write_json(args.output, receipt)
    print(json.dumps({"receipt": str(args.output), "captures": len(captures)}, ensure_ascii=False))
    return 0


def load_receipt(path: Path) -> tuple[dict[str, Any], dict[str, HiddenRecord]]:
    receipt = read_json(path)
    if receipt.get("format") != RECEIPT_FORMAT:
        raise BridgeError(f"采集收据格式错误: {path}")
    dump = Path(str(receipt.get("dump", "")))
    prefix_bytes = int(receipt.get("dump_prefix_bytes", -1))
    if not dump.is_file() or dump.stat().st_size < prefix_bytes:
        raise BridgeError(f"采集dump缺失或被截断: {dump}")
    if sha256_file(dump, prefix_bytes) != receipt.get("dump_prefix_sha256"):
        raise BridgeError(f"采集dump前缀SHA-256不匹配: {dump}")

    captures = receipt.get("captures")
    if not isinstance(captures, list) or len(captures) != int(
        receipt.get("capture_count", -1)
    ):
        raise BridgeError(f"采集收据记录数量错误: {path}")
    records: dict[str, HiddenRecord] = {}
    for item in captures:
        if not isinstance(item, dict):
            raise BridgeError(f"采集收据记录不是对象: {path}")
        prompt_id = str(item["prompt_id"])
        if prompt_id in records:
            raise BridgeError(f"采集收据prompt ID重复: {path}, prompt={prompt_id}")
        prompt = item.get("prompt")
        tokens = item.get("tokens")
        if not isinstance(prompt, str) or not isinstance(tokens, list):
            raise BridgeError(
                f"v2采集收据必须保存原始prompt与token pieces: {path}, prompt={prompt_id}"
            )
        prompt_bytes = prompt.encode("utf-8")
        if (
            len(prompt_bytes) != int(item.get("prompt_utf8_bytes", -1))
            or hashlib.sha256(prompt_bytes).hexdigest() != item.get("prompt_sha256")
        ):
            raise BridgeError(f"采集收据prompt内容或SHA-256漂移: {path}, prompt={prompt_id}")
        start = int(item["record_offset"])
        end = int(item["record_end_offset"])
        parsed = read_clm9(dump, start, end)
        if len(parsed) != 1:
            raise BridgeError(f"收据记录范围不唯一: {path}, prompt={prompt_id}")
        record = parsed[0]
        if list(record.shape) != item.get("shape"):
            raise BridgeError(f"收据shape漂移: {path}, prompt={prompt_id}")
        if record.token_count != len(tokens) or record.token_count != int(
            item.get("token_count", -1)
        ):
            raise BridgeError(
                f"收据token数与隐藏态行数不一致: {path}, prompt={prompt_id}, "
                f"hidden={record.token_count}, tokens={len(tokens)}"
            )
        records[prompt_id] = record
    return receipt, records


def pair_receipts(
    base_path: Path, donor_path: Path
) -> tuple[list[PairedPrompt], dict[str, Any]]:
    base_receipt, base_records = load_receipt(base_path)
    donor_receipt, donor_records = load_receipt(donor_path)
    base_items = {str(item["prompt_id"]): item for item in base_receipt["captures"]}
    donor_items = {str(item["prompt_id"]): item for item in donor_receipt["captures"]}
    common = sorted(set(base_items) & set(donor_items))
    paired: list[PairedPrompt] = []
    skipped: list[dict[str, Any]] = []
    prompt_alignment: list[dict[str, Any]] = []
    total_matched_tokens = 0
    for prompt_id in common:
        base_item = base_items[prompt_id]
        donor_item = donor_items[prompt_id]
        base_prompt = base_item["prompt"]
        donor_prompt = donor_item["prompt"]
        base_tokens = base_item["tokens"]
        donor_tokens = donor_item["tokens"]
        if base_prompt != donor_prompt:
            mismatch = {
                "prompt_id": prompt_id,
                "paired": False,
                "reason": "original_prompt_mismatch",
                "base_tokens": len(base_tokens),
                "donor_tokens": len(donor_tokens),
                "matched_tokens": 0,
                "match_coverage": 0.0,
            }
            skipped.append(mismatch)
            prompt_alignment.append(mismatch)
            continue
        base_boundaries = map_token_end_boundaries(base_prompt, base_tokens)
        donor_boundaries = map_token_end_boundaries(donor_prompt, donor_tokens)
        common_ends = sorted(
            set(base_boundaries.by_end) & set(donor_boundaries.by_end)
        )
        base = base_records[prompt_id].rows()
        donor = donor_records[prompt_id].rows()
        denominator = max(
            base_boundaries.mapped_count, donor_boundaries.mapped_count, 1
        )
        alignment = {
            "prompt_id": prompt_id,
            "prompt_utf8_bytes": len(base_prompt.encode("utf-8")),
            "base_tokens": base_boundaries.token_count,
            "donor_tokens": donor_boundaries.token_count,
            "base_mappable_tokens": base_boundaries.mapped_count,
            "donor_mappable_tokens": donor_boundaries.mapped_count,
            "base_excluded_tokens": base_boundaries.excluded_count,
            "donor_excluded_tokens": donor_boundaries.excluded_count,
            "base_prompt_byte_coverage": base_boundaries.consumed_bytes
            / max(base_boundaries.prompt_bytes, 1),
            "donor_prompt_byte_coverage": donor_boundaries.consumed_bytes
            / max(donor_boundaries.prompt_bytes, 1),
            "matched_tokens": len(common_ends),
            # Conservative endpoint coverage: one shared boundary counts once,
            # while the denominator is the more finely split tokenizer.
            "match_coverage": len(common_ends) / denominator,
            "base_match_coverage": len(common_ends)
            / max(base_boundaries.mapped_count, 1),
            "donor_match_coverage": len(common_ends)
            / max(donor_boundaries.mapped_count, 1),
            "last_matched_end_byte": common_ends[-1] if common_ends else 0,
        }
        if base.shape[1] != donor.shape[1]:
            alignment.update(
                {"paired": False, "reason": "activation_width_mismatch"}
            )
            skipped.append(alignment)
            prompt_alignment.append(alignment)
            continue
        if not common_ends:
            alignment.update(
                {"paired": False, "reason": "no_common_utf8_end_boundary"}
            )
            skipped.append(alignment)
            prompt_alignment.append(alignment)
            continue
        base_indices = np.asarray(
            [base_boundaries.by_end[end] for end in common_ends], dtype=np.int64
        )
        donor_indices = np.asarray(
            [donor_boundaries.by_end[end] for end in common_ends], dtype=np.int64
        )
        paired.append(PairedPrompt(prompt_id, base[base_indices], donor[donor_indices]))
        total_matched_tokens += len(common_ends)
        alignment["paired"] = True
        prompt_alignment.append(alignment)
    accepted_alignment = [item for item in prompt_alignment if item.get("paired")]
    metadata = {
        "base": {
            "receipt": str(base_path.resolve()),
            "model": base_receipt.get("model"),
            "layer": base_receipt.get("layer"),
            "stage": base_receipt.get("stage"),
        },
        "donor": {
            "receipt": str(donor_path.resolve()),
            "model": donor_receipt.get("model"),
            "layer": donor_receipt.get("layer"),
            "stage": donor_receipt.get("stage"),
        },
        "common_prompts": len(common),
        "paired_prompts": len(paired),
        "total_base_tokens": sum(item["base_tokens"] for item in accepted_alignment),
        "total_donor_tokens": sum(
            item["donor_tokens"] for item in accepted_alignment
        ),
        "total_matched_tokens": total_matched_tokens,
        "mean_match_coverage": float(
            np.mean([item["match_coverage"] for item in accepted_alignment])
        )
        if accepted_alignment
        else 0.0,
        "prompt_alignment": prompt_alignment,
        "skipped": skipped,
    }
    return paired, metadata


def unit_rows(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, np.float32(1e-12)), norms[:, 0]


def value_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def row_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_unit, _ = unit_rows(left)
    right_unit, _ = unit_rows(right)
    return np.sum(left_unit * right_unit, axis=1)


def relative_rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    numerator = float(np.mean(np.square(prediction - target), dtype=np.float64))
    denominator = float(np.mean(np.square(target), dtype=np.float64))
    return math.sqrt(numerator / max(denominator, 1e-30))


def optimal_positive_scale(prediction: np.ndarray, target: np.ndarray) -> float:
    numerator = float(np.sum(prediction * target, dtype=np.float64))
    denominator = float(np.sum(prediction * prediction, dtype=np.float64))
    return max(numerator / max(denominator, 1e-30), 1e-6)


def concatenate(groups: Iterable[PairedPrompt]) -> tuple[np.ndarray, np.ndarray]:
    items = list(groups)
    return (
        np.concatenate([item.base for item in items], axis=0),
        np.concatenate([item.donor for item in items], axis=0),
    )


def subsample_pair(
    base: np.ndarray,
    donor: np.ndarray,
    limit: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if len(base) <= limit:
        return base, donor
    indices = np.sort(rng.choice(len(base), size=limit, replace=False))
    return base[indices], donor[indices]


def thin_svd(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if scipy_svd is not None:
        try:
            return scipy_svd(
                matrix,
                full_matrices=False,
                overwrite_a=True,
                check_finite=False,
                lapack_driver="gesdd",
            )
        except np.linalg.LinAlgError:
            # Rank-deficient nullspace completions can make divide-and-conquer
            # SVD fail on some OpenBLAS builds; the QR driver is slower but
            # considerably more robust for this offline calibration path.
            return scipy_svd(
                np.ascontiguousarray(matrix),
                full_matrices=False,
                overwrite_a=True,
                check_finite=False,
                lapack_driver="gesvd",
            )
    return np.linalg.svd(matrix, full_matrices=False)


def nullspace_anchored_procrustes(
    cross: np.ndarray,
    baseline_base_to_donor: np.ndarray,
    rank_relative_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Resolve rank-deficient Procrustes freedom toward the old bridge.

    The signal singular vectors remain the exact orthogonal Procrustes optimum.
    Only the zero-singular-value completion is selected by minimum Frobenius
    distance to the existing embedding bridge.
    """
    left, singular_values, right_t = thin_svd(cross)
    threshold = max(
        float(np.max(singular_values)) * rank_relative_tolerance,
        1e-12,
    )
    rank = int(np.count_nonzero(singular_values > threshold))
    width = cross.shape[0]
    if rank >= width:
        return (
            np.ascontiguousarray(left @ right_t, dtype=np.float32),
            singular_values,
            rank,
        )
    if rank == 0:
        return (
            np.ascontiguousarray(baseline_base_to_donor, dtype=np.float32),
            singular_values,
            rank,
        )

    signal_left = left[:, :rank]
    signal_right = right_t[:rank, :].T
    null_left = left[:, rank:]
    null_right = right_t[rank:, :].T
    null_alignment = np.asarray(
        null_left.T @ baseline_base_to_donor @ null_right,
        dtype=np.float32,
    )
    anchor_left, _anchor_values, anchor_right_t = thin_svd(null_alignment)
    completion = anchor_left @ anchor_right_t
    fitted = (
        signal_left @ signal_right.T
        + null_left @ completion @ null_right.T
    )
    return np.ascontiguousarray(fitted, dtype=np.float32), singular_values, rank


def fit_bridge(
    train_base: np.ndarray,
    train_donor: np.ndarray,
    baseline_donor_to_base: np.ndarray,
    prior_samples: float,
    min_scale: float,
    max_scale: float,
    bridge_strategy: str = "full_procrustes",
    rank_relative_tolerance: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, np.ndarray]:
    base_unit, base_norm = unit_rows(train_base)
    donor_unit, donor_norm = unit_rows(train_donor)
    baseline_base_to_donor = baseline_donor_to_base.T
    cross = np.asarray(base_unit.T @ donor_unit, dtype=np.float32)
    if prior_samples > 0:
        cross += np.float32(prior_samples) * baseline_base_to_donor
    if bridge_strategy == "full_procrustes":
        left, singular_values, right_t = thin_svd(cross)
        base_to_donor = np.ascontiguousarray(left @ right_t, dtype=np.float32)
    elif bridge_strategy == "nullspace_anchored":
        if prior_samples > 0:
            raise BridgeError(
                "nullspace_anchored与全空间等效样本先验不能同时启用"
            )
        base_to_donor, singular_values, _rank = nullspace_anchored_procrustes(
            cross,
            baseline_base_to_donor,
            rank_relative_tolerance,
        )
    else:
        raise BridgeError(f"未知桥策略: {bridge_strategy}")
    ratios = donor_norm / np.maximum(base_norm, np.float32(1e-12))
    raw_scale = float(np.median(ratios))
    scale = float(np.clip(raw_scale, min_scale, max_scale))

    # Runtime matrices multiply column hidden states: y=W_in*x, x=W_out*y.
    input_weight = np.ascontiguousarray(scale * base_to_donor.T, dtype=np.float32)
    output_weight = np.ascontiguousarray((1.0 / scale) * base_to_donor, dtype=np.float32)
    donor_to_base_orthogonal = np.ascontiguousarray(base_to_donor.T, dtype=np.float32)
    return (
        input_weight,
        output_weight,
        donor_to_base_orthogonal,
        scale,
        raw_scale,
        singular_values,
    )


def evaluate_map(
    base: np.ndarray,
    donor: np.ndarray,
    base_to_donor: np.ndarray,
    scale: float | None = None,
) -> dict[str, Any]:
    prediction = base @ base_to_donor
    fitted_scale = optimal_positive_scale(prediction, donor) if scale is None else scale
    scaled = fitted_scale * prediction
    return {
        "cosine": value_summary(row_cosine(prediction, donor)),
        "relative_rmse": relative_rmse(scaled, donor),
        "scale": float(fitted_scale),
    }


def heldout_record_lifts(
    groups: list[PairedPrompt],
    baseline_base_to_donor: np.ndarray,
    fitted_base_to_donor: np.ndarray,
) -> tuple[list[dict[str, Any]], float]:
    rows: list[dict[str, Any]] = []
    for group in groups:
        baseline = float(np.median(row_cosine(group.base @ baseline_base_to_donor, group.donor)))
        fitted = float(np.median(row_cosine(group.base @ fitted_base_to_donor, group.donor)))
        rows.append(
            {
                "prompt_id": group.prompt_id,
                "tokens": len(group.base),
                "baseline_median_cosine": baseline,
                "activation_median_cosine": fitted,
                "lift": fitted - baseline,
            }
        )
    positive = sum(item["lift"] > 0 for item in rows) / max(len(rows), 1)
    return rows, positive


def run_fit(args: argparse.Namespace) -> int:
    groups, pairing = pair_receipts(args.base_receipt, args.donor_receipt)
    if len(groups) < 2:
        raise BridgeError(
            "至少需要两个能产生共同UTF-8结束边界的prompt，且正式门要求更多；"
            f"当前paired={len(groups)}"
        )
    width = groups[0].base.shape[1]
    if any(item.base.shape[1] != width for item in groups):
        raise BridgeError("隐藏宽度不一致")
    baseline = np.asarray(np.load(args.baseline, allow_pickle=False), dtype=np.float32)
    if baseline.shape != (width, width) or not np.isfinite(baseline).all():
        raise BridgeError(f"嵌入桥形状/数值无效: {baseline.shape}, expected={(width, width)}")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(groups))
    heldout_count = max(1, int(round(len(groups) * args.heldout_fraction)))
    heldout_count = min(heldout_count, len(groups) - 1)
    heldout_ids = set(int(index) for index in order[:heldout_count])
    heldout_groups = [item for index, item in enumerate(groups) if index in heldout_ids]
    train_groups = [item for index, item in enumerate(groups) if index not in heldout_ids]
    train_base, train_donor = concatenate(train_groups)
    test_base, test_donor = concatenate(heldout_groups)
    train_base, train_donor = subsample_pair(
        train_base, train_donor, args.max_train_tokens, rng
    )
    test_base, test_donor = subsample_pair(
        test_base, test_donor, args.max_heldout_tokens, rng
    )

    started = time.perf_counter()
    (
        input_weight,
        output_weight,
        donor_to_base,
        scale,
        raw_scale,
        singular_values,
    ) = fit_bridge(
        train_base,
        train_donor,
        baseline,
        args.prior_samples,
        args.min_scale,
        args.max_scale,
        args.bridge_strategy,
        args.rank_relative_tolerance,
    )
    fit_seconds = time.perf_counter() - started
    base_to_donor = donor_to_base.T
    baseline_base_to_donor = baseline.T

    identity_metrics = evaluate_map(test_base, test_donor, np.eye(width, dtype=np.float32))
    # Compare both bridges at the same train-derived donor amplitude.  Using
    # a separate held-out optimal scale for the baseline would reward shrinking
    # its prediction toward zero and would not describe the runtime contract.
    baseline_metrics = evaluate_map(
        test_base, test_donor, baseline_base_to_donor, scale
    )
    baseline_optimal_metrics = evaluate_map(
        test_base, test_donor, baseline_base_to_donor
    )
    activation_metrics = evaluate_map(test_base, test_donor, base_to_donor, scale)
    activation_optimal_metrics = evaluate_map(
        test_base, test_donor, base_to_donor
    )
    per_prompt, positive_rate = heldout_record_lifts(
        heldout_groups, baseline_base_to_donor, base_to_donor
    )
    cosine_lift = (
        activation_metrics["cosine"]["median"]
        - baseline_metrics["cosine"]["median"]
    )
    nrmse_ratio = activation_metrics["relative_rmse"] / max(
        baseline_metrics["relative_rmse"], 1e-12
    )
    identity = np.eye(width, dtype=np.float32)
    orthogonality_rmse = float(
        np.linalg.norm(base_to_donor.T @ base_to_donor - identity)
        / math.sqrt(identity.size)
    )
    cycle_rmse = float(
        np.linalg.norm(output_weight @ input_weight - identity)
        / math.sqrt(identity.size)
    )
    paired_tokens = sum(len(item.base) for item in groups)
    gates = {
        "enough_prompts": len(groups) >= args.min_prompts,
        "enough_tokens": paired_tokens >= args.min_tokens,
        "heldout_cosine_lift": cosine_lift >= args.min_cosine_lift,
        "heldout_nrmse_ratio": nrmse_ratio <= args.max_nrmse_ratio,
        "positive_prompt_rate": positive_rate >= args.min_positive_prompt_rate,
        "reversible": cycle_rmse <= 5e-5 and orthogonality_rmse <= 5e-5,
        "scale_in_contract": args.min_scale <= raw_scale <= args.max_scale,
    }
    promoted = all(gates.values())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_path = args.output_dir / "coder_activation_input_weight_f32.npy"
    output_path = args.output_dir / "coder_activation_output_weight_f32.npy"
    compatibility_path = args.output_dir / "coder_to_colorlm_activation_orthogonal_f32.npy"
    np.save(input_path, input_weight, allow_pickle=False)
    np.save(output_path, output_weight, allow_pickle=False)
    np.save(compatibility_path, donor_to_base, allow_pickle=False)
    report = {
        "format": REPORT_FORMAT,
        "pairing": pairing,
        "method": {
            "name": args.bridge_strategy,
            "fit_direction": "ColorLM-L35-attn-residual rows -> Coder-L44-input rows",
            "prior": str(args.baseline.resolve()),
            "prior_sha256": sha256_file(args.baseline),
            "prior_equivalent_samples": args.prior_samples,
            "rank_relative_tolerance": args.rank_relative_tolerance,
            "train_split_unit": "whole prompt",
            "seed": args.seed,
        },
        "samples": {
            "paired_prompts": len(groups),
            "paired_tokens": paired_tokens,
            "train_prompts": len(train_groups),
            "heldout_prompts": len(heldout_groups),
            "train_tokens_used": len(train_base),
            "heldout_tokens_used": len(test_base),
        },
        "fit": {
            "seconds": fit_seconds,
            "scale": scale,
            "raw_median_scale": raw_scale,
            "singular_value_min": float(np.min(singular_values)),
            "singular_value_median": float(np.median(singular_values)),
            "singular_value_max": float(np.max(singular_values)),
            "effective_rank": int(
                np.count_nonzero(
                    singular_values
                    > max(
                        float(np.max(singular_values))
                        * args.rank_relative_tolerance,
                        1e-12,
                    )
                )
            ),
            "orthogonality_rmse": orthogonality_rmse,
            "cycle_rmse": cycle_rmse,
        },
        "heldout": {
            "identity": identity_metrics,
            "embedding_bridge": baseline_metrics,
            "embedding_bridge_optimal_scale_diagnostic": baseline_optimal_metrics,
            "activation_bridge": activation_metrics,
            "activation_bridge_optimal_scale_diagnostic": activation_optimal_metrics,
            "median_cosine_lift_vs_embedding": cosine_lift,
            "nrmse_ratio_vs_embedding": nrmse_ratio,
            "positive_prompt_rate": positive_rate,
            "per_prompt": per_prompt,
        },
        "runtime": {
            "input_weight": {
                "file": input_path.name,
                "sha256": sha256_file(input_path),
                "contract": "donor_column = W_in @ colorlm_column",
            },
            "output_weight": {
                "file": output_path.name,
                "sha256": sha256_file(output_path),
                "contract": "colorlm_delta_column = W_out @ donor_delta_column",
            },
            "orthogonal_compatibility": {
                "file": compatibility_path.name,
                "sha256": sha256_file(compatibility_path),
                "contract": "donor_row @ T = colorlm_row; legacy compiler uses T/T.T",
            },
        },
        "promotion": {
            "gates": gates,
            "decision": "candidate" if promoted else "reject",
            "thresholds": {
                "min_prompts": args.min_prompts,
                "min_tokens": args.min_tokens,
                "min_cosine_lift": args.min_cosine_lift,
                "max_nrmse_ratio": args.max_nrmse_ratio,
                "min_positive_prompt_rate": args.min_positive_prompt_rate,
            },
        },
    }
    report_path = args.output_dir / "activation_bridge_report.json"
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if promoted or args.allow_rejected else 2


def inspect_dump(args: argparse.Namespace) -> int:
    records = read_clm9(args.path)
    result = {
        "path": str(args.path.resolve()),
        "sha256": sha256_file(args.path),
        "record_count": len(records),
        "records": [
            {
                "offset": record.offset,
                "end_offset": record.end_offset,
                "layer": record.layer,
                "shape": list(record.shape),
                "tokens": record.token_count,
            }
            for record in records
        ],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def append_fixture_record(stream: Any, layer: int, values: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(values, dtype="<f4")
    tokens, width = contiguous.shape
    payload = contiguous.tobytes(order="C")
    stream.write(
        CLM9_HEADER.pack(
            CLM9_MAGIC, 1, layer, 0, width, tokens, 1, 1, len(payload)
        )
    )
    stream.write(payload)


def self_test(_: argparse.Namespace) -> int:
    rng = np.random.default_rng(18)
    bilingual_prompt = "hello世界!"
    base_tokenization = [
        {"id": 1, "piece": ""},
        {"id": 2, "piece": "hello"},
        {"id": 3, "piece": "世"},
        {"id": 4, "piece": "界"},
        {"id": 5, "piece": "!"},
    ]
    donor_tokenization = [
        {"id": 10, "piece": "<BOS>"},
        {"id": 11, "piece": "he"},
        {"id": 12, "piece": "llo"},
        {"id": 13, "piece": [255]},
        {"id": 14, "piece": list("世界".encode("utf-8"))},
        {"id": 15, "piece": "!"},
    ]
    base_boundaries = map_token_end_boundaries(bilingual_prompt, base_tokenization)
    donor_boundaries = map_token_end_boundaries(bilingual_prompt, donor_tokenization)
    shared_ends = sorted(set(base_boundaries.by_end) & set(donor_boundaries.by_end))
    if shared_ends != [5, 11, 12]:
        raise BridgeError(f"跨词表UTF-8边界自测失败: {shared_ends}")
    if base_boundaries.excluded_count != 1 or donor_boundaries.excluded_count != 2:
        raise BridgeError(
            "特殊token/不可映射piece排除自测失败: "
            f"base={base_boundaries.excluded_count}, "
            f"donor={donor_boundaries.excluded_count}"
        )
    if (
        base_boundaries.consumed_bytes != len(bilingual_prompt.encode("utf-8"))
        or donor_boundaries.consumed_bytes != len(bilingual_prompt.encode("utf-8"))
    ):
        raise BridgeError("中英文prompt累计UTF-8字节自测失败")

    width = 32
    raw = rng.normal(size=(width, width)).astype(np.float32)
    left, _, right_t = np.linalg.svd(raw, full_matrices=False)
    truth = np.asarray(left @ right_t, dtype=np.float32)
    groups: list[PairedPrompt] = []
    with tempfile.TemporaryDirectory(prefix="colorlm-v18-bridge-") as directory:
        base_dump = Path(directory) / "base.bin"
        donor_dump = Path(directory) / "donor.bin"
        with base_dump.open("wb") as base_stream, donor_dump.open("wb") as donor_stream:
            for index in range(8):
                base = rng.normal(size=(24, width)).astype(np.float32)
                donor = 1.25 * (base @ truth)
                donor += rng.normal(scale=0.005, size=donor.shape).astype(np.float32)
                append_fixture_record(base_stream, 35, base)
                append_fixture_record(donor_stream, 43, donor)
                groups.append(PairedPrompt(str(index), base, donor))
        parsed_base = read_clm9(base_dump)
        parsed_donor = read_clm9(donor_dump)
        if len(parsed_base) != 8 or len(parsed_donor) != 8:
            raise BridgeError("CLM9自测解析数量错误")

        def fixture_receipt(
            dump: Path, records: list[HiddenRecord], layer: int, stage: str
        ) -> dict[str, Any]:
            size = dump.stat().st_size
            captures = []
            for index, record in enumerate(records):
                tokens = [
                    {"id": token, "piece": f"p{token}"}
                    for token in range(record.token_count)
                ]
                prompt = "".join(str(token["piece"]) for token in tokens)
                prompt_bytes = prompt.encode("utf-8")
                captures.append(
                    {
                        "prompt_id": str(index),
                        "prompt": prompt,
                        "prompt_utf8_bytes": len(prompt_bytes),
                        "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
                        "token_count": record.token_count,
                        "tokens": tokens,
                        "request_dump_start": record.offset,
                        "request_dump_end": record.end_offset,
                        "record_offset": record.offset,
                        "record_end_offset": record.end_offset,
                        "shape": list(record.shape),
                    }
                )
            return {
                "format": RECEIPT_FORMAT,
                "endpoint": "fixture",
                "model": stage,
                "dump": str(dump.resolve()),
                "dump_prefix_bytes": size,
                "dump_prefix_sha256": sha256_file(dump, size),
                "layer": layer,
                "stage": stage,
                "capture_count": len(captures),
                "captures": captures,
            }

        base_receipt = Path(directory) / "base.receipt.json"
        donor_receipt = Path(directory) / "donor.receipt.json"
        write_json(base_receipt, fixture_receipt(base_dump, parsed_base, 35, "attn_residual"))
        write_json(donor_receipt, fixture_receipt(donor_dump, parsed_donor, 43, "l_out"))
        paired, pairing_metadata = pair_receipts(base_receipt, donor_receipt)
        if (
            len(paired) != 8
            or sum(len(item.base) for item in paired) != 192
            or pairing_metadata["total_matched_tokens"] != 192
            or any(
                item["match_coverage"] != 1.0
                for item in pairing_metadata["prompt_alignment"]
            )
        ):
            raise BridgeError("采集收据配对自测失败")

        # Exercise the full receipt path with genuinely different tokenizers.
        # The shared rows must follow equal UTF-8 prefix ends, not token IDs or
        # equal sequence lengths.
        cross_base_dump = Path(directory) / "cross-base.bin"
        cross_donor_dump = Path(directory) / "cross-donor.bin"
        cross_base_values = np.arange(5 * width, dtype=np.float32).reshape(5, width)
        cross_donor_values = np.arange(6 * width, dtype=np.float32).reshape(6, width)
        with (
            cross_base_dump.open("wb") as base_stream,
            cross_donor_dump.open("wb") as donor_stream,
        ):
            append_fixture_record(base_stream, 35, cross_base_values)
            append_fixture_record(donor_stream, 43, cross_donor_values)
        cross_base_record = read_clm9(cross_base_dump)[0]
        cross_donor_record = read_clm9(cross_donor_dump)[0]

        def cross_receipt(
            dump: Path,
            record: HiddenRecord,
            layer: int,
            stage: str,
            tokens: list[dict[str, Any]],
        ) -> dict[str, Any]:
            prompt_bytes = bilingual_prompt.encode("utf-8")
            return {
                "format": RECEIPT_FORMAT,
                "endpoint": "fixture",
                "model": stage,
                "dump": str(dump.resolve()),
                "dump_prefix_bytes": dump.stat().st_size,
                "dump_prefix_sha256": sha256_file(dump),
                "layer": layer,
                "stage": stage,
                "capture_count": 1,
                "captures": [
                    {
                        "prompt_id": "cross-vocab",
                        "prompt": bilingual_prompt,
                        "prompt_utf8_bytes": len(prompt_bytes),
                        "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
                        "token_count": len(tokens),
                        "tokens": tokens,
                        "request_dump_start": record.offset,
                        "request_dump_end": record.end_offset,
                        "record_offset": record.offset,
                        "record_end_offset": record.end_offset,
                        "shape": list(record.shape),
                    }
                ],
            }

        cross_base_receipt = Path(directory) / "cross-base.receipt.json"
        cross_donor_receipt = Path(directory) / "cross-donor.receipt.json"
        write_json(
            cross_base_receipt,
            cross_receipt(
                cross_base_dump,
                cross_base_record,
                35,
                "attn_residual",
                base_tokenization,
            ),
        )
        write_json(
            cross_donor_receipt,
            cross_receipt(
                cross_donor_dump,
                cross_donor_record,
                43,
                "l_out",
                donor_tokenization,
            ),
        )
        cross_pairs, cross_metadata = pair_receipts(
            cross_base_receipt, cross_donor_receipt
        )
        if (
            len(cross_pairs) != 1
            or not np.array_equal(
                cross_pairs[0].base, cross_base_values[[1, 3, 4]]
            )
            or not np.array_equal(
                cross_pairs[0].donor, cross_donor_values[[2, 4, 5]]
            )
            or cross_metadata["total_matched_tokens"] != 3
            or cross_metadata["prompt_alignment"][0]["match_coverage"] != 0.75
        ):
            raise BridgeError("跨词表采集收据端到端配对自测失败")

    train_base, train_donor = concatenate(groups[:6])
    input_weight, output_weight, donor_to_base, scale, raw_scale, _ = fit_bridge(
        train_base,
        train_donor,
        np.eye(width, dtype=np.float32),
        prior_samples=2.0,
        min_scale=0.25,
        max_scale=4.0,
    )
    test_base, test_donor = concatenate(groups[6:])
    baseline_cos = float(np.median(row_cosine(test_base, test_donor)))
    fitted_cos = float(np.median(row_cosine(test_base @ donor_to_base.T, test_donor)))
    cycle = float(np.max(np.abs(output_weight @ input_weight - np.eye(width))))
    if fitted_cos < 0.90 or fitted_cos - baseline_cos < 0.70:
        raise BridgeError(
            f"Procrustes自测精度不足: baseline={baseline_cos}, fitted={fitted_cos}"
        )
    if abs(scale - 1.25) > 0.02 or abs(raw_scale - 1.25) > 0.02 or cycle > 1e-4:
        raise BridgeError(
            f"缩放/可逆性自测失败: scale={scale}, raw={raw_scale}, cycle={cycle}"
        )

    anchor_width = 16
    anchor_rank = 5
    anchor_raw = rng.normal(size=(anchor_width, anchor_width)).astype(np.float32)
    anchor_left, _, anchor_right_t = np.linalg.svd(
        anchor_raw, full_matrices=False
    )
    anchor_baseline = np.asarray(anchor_left @ anchor_right_t, dtype=np.float32)
    signal_left, _, _ = np.linalg.svd(
        rng.normal(size=(anchor_width, anchor_width)).astype(np.float32),
        full_matrices=False,
    )
    signal_right, _, _ = np.linalg.svd(
        rng.normal(size=(anchor_width, anchor_width)).astype(np.float32),
        full_matrices=False,
    )
    signal_values = np.linspace(3.0, 1.0, anchor_rank, dtype=np.float32)
    rank_deficient_cross = (
        signal_left[:, :anchor_rank]
        @ np.diag(signal_values)
        @ signal_right[:, :anchor_rank].T
    )
    anchored, _, observed_rank = nullspace_anchored_procrustes(
        np.asarray(rank_deficient_cross, dtype=np.float32),
        anchor_baseline,
        rank_relative_tolerance=1e-5,
    )
    standard_left, _, standard_right_t = thin_svd(
        np.asarray(rank_deficient_cross, dtype=np.float32)
    )
    standard = standard_left @ standard_right_t
    anchored_orthogonality = float(
        np.max(np.abs(anchored.T @ anchored - np.eye(anchor_width)))
    )
    anchored_objective = float(np.sum(rank_deficient_cross * anchored))
    standard_objective = float(np.sum(rank_deficient_cross * standard))
    anchored_distance = float(np.linalg.norm(anchored - anchor_baseline))
    standard_distance = float(np.linalg.norm(standard - anchor_baseline))
    if (
        observed_rank != anchor_rank
        or anchored_orthogonality > 1e-4
        or abs(anchored_objective - standard_objective) > 1e-3
        or anchored_distance > standard_distance + 1e-4
    ):
        raise BridgeError(
            "nullspace anchored自测失败: "
            f"rank={observed_rank}, orth={anchored_orthogonality}, "
            f"objective_delta={anchored_objective - standard_objective}, "
            f"distance={anchored_distance}/{standard_distance}"
        )
    print("v18 activation bridge self-test: ok")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ColorLM v18深层激活坐标桥工具")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser("inspect", help="只读检查CLM9隐藏态dump")
    inspect.add_argument("path", type=Path)
    inspect.set_defaults(handler=inspect_dump)

    capture = subparsers.add_parser("collect", help="向已启用dump的服务发送短校准prompt")
    capture.add_argument("--endpoint", required=True)
    capture.add_argument("--expect-model")
    capture.add_argument("--dump", type=Path, required=True)
    capture.add_argument("--layer", type=int, required=True)
    capture.add_argument("--stage", required=True)
    capture.add_argument("--prompts", type=Path, required=True)
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--seed", type=int, default=18)
    capture.add_argument("--timeout", type=float, default=90.0)
    capture.set_defaults(handler=collect)

    fit = subparsers.add_parser("fit", help="拟合并留出验证激活坐标桥")
    fit.add_argument("--base-receipt", type=Path, required=True)
    fit.add_argument("--donor-receipt", type=Path, required=True)
    fit.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    fit.add_argument("--output-dir", type=Path, required=True)
    fit.add_argument("--prior-samples", type=float, default=512.0)
    fit.add_argument(
        "--bridge-strategy",
        choices=("full_procrustes", "nullspace_anchored"),
        default="full_procrustes",
    )
    fit.add_argument("--rank-relative-tolerance", type=float, default=1e-6)
    fit.add_argument("--heldout-fraction", type=float, default=0.25)
    fit.add_argument("--max-train-tokens", type=int, default=4096)
    fit.add_argument("--max-heldout-tokens", type=int, default=2048)
    fit.add_argument("--min-prompts", type=int, default=6)
    fit.add_argument("--min-tokens", type=int, default=512)
    fit.add_argument("--min-cosine-lift", type=float, default=0.03)
    fit.add_argument("--max-nrmse-ratio", type=float, default=0.95)
    fit.add_argument("--min-positive-prompt-rate", type=float, default=0.67)
    fit.add_argument("--min-scale", type=float, default=0.25)
    fit.add_argument("--max-scale", type=float, default=4.0)
    fit.add_argument("--seed", type=int, default=18)
    fit.add_argument(
        "--allow-rejected",
        action="store_true",
        help="候选未过门时仍返回0；报告始终明确标为reject。",
    )
    fit.set_defaults(handler=run_fit)

    test = subparsers.add_parser("self-test", help="不启动模型的CLM9与拟合器自测")
    test.set_defaults(handler=self_test)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.command == "collect" and (args.layer < 0 or args.timeout <= 0):
        raise BridgeError("collect的layer/timeout无效")
    if args.command == "fit":
        if not 0 < args.heldout_fraction < 0.5:
            raise BridgeError("heldout-fraction必须在(0, 0.5)内")
        if args.prior_samples < 0 or args.min_scale <= 0 or args.max_scale <= args.min_scale:
            raise BridgeError("先验强度或scale边界无效")
        if args.rank_relative_tolerance <= 0:
            raise BridgeError("rank-relative-tolerance必须为正数")
        if args.bridge_strategy == "nullspace_anchored" and args.prior_samples > 0:
            raise BridgeError("nullspace_anchored要求--prior-samples 0")
        if min(
            args.max_train_tokens,
            args.max_heldout_tokens,
            args.min_prompts,
            args.min_tokens,
        ) <= 0:
            raise BridgeError("token/prompt限制必须为正数")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_args(args)
        return int(args.handler(args))
    except BridgeError as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
