"""Download the two byte ranges backing the v16 donor block, with resume support."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

import requests


ROOT = Path(__file__).resolve().parents[2]
RESEARCH = Path(__file__).resolve().parent
DEFAULT_PLAN = RESEARCH / "v16_coder_neural_block_plan.json"
DEFAULT_OUTPUT = RESEARCH / "neural_blocks" / "qwen3_coder_next_l47" / "source"
MODELSCOPE = "https://modelscope.cn"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按精确Range提取Qwen3-Coder-Next L47，不下载完整模型"
    )
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--download",
        action="store_true",
        help="实际下载；默认仅做本地契约和空间预检",
    )
    parser.add_argument("--min-free-gib", type=float, default=30.0)
    parser.add_argument(
        "--request-mib",
        type=float,
        default=16.0,
        help="每次HTTP子Range大小；小块可降低镜像长连接中断损失",
    )
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="并行获取独立子Range的连接数；落盘仍按字节顺序提交。",
    )
    parser.add_argument(
        "--import-full-shard",
        type=Path,
        action="append",
        default=[],
        help=(
            "从已经下载的完整safetensors导入计划Range；可重复指定。"
            "已有.part前缀会先逐字节校验，再从断点继续。"
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def source_url(repo: str, revision: str, shard: str) -> str:
    return (
        f"{MODELSCOPE}/models/{quote(repo, safe='/')}/resolve/"
        f"{quote(revision, safe='')}/{quote(shard, safe='/')}"
    )


def inspect_local(output: Path, ranges: list[dict]) -> tuple[int, list[dict]]:
    remaining = 0
    status: list[dict] = []
    for segment in ranges:
        final = output / segment["file"]
        partial = final.with_suffix(final.suffix + ".part")
        expected = int(segment["bytes"])
        if final.is_file():
            current = final.stat().st_size
            state = "complete" if current == expected else "invalid-final"
        elif partial.is_file():
            current = partial.stat().st_size
            state = "partial" if current <= expected else "invalid-partial"
        else:
            current = 0
            state = "missing"
        if state.startswith("invalid"):
            raise RuntimeError(f"已有Range文件尺寸异常: {final}")
        remaining += expected - current
        status.append(
            {
                "file": segment["file"],
                "state": state,
                "bytes": current,
                "expected_bytes": expected,
            }
        )
    return remaining, status


def import_full_shard(full_shard: Path, output: Path, segment: dict) -> dict:
    """Import one byte range without duplicating an already verified prefix."""
    final = output / segment["file"]
    partial = final.with_suffix(final.suffix + ".part")
    expected = int(segment["bytes"])
    start = int(segment["start"])
    end = int(segment["end_exclusive"])
    if full_shard.name != segment["source_shard"]:
        raise RuntimeError(
            f"完整shard文件名与Range来源不匹配: {full_shard.name} != "
            f"{segment['source_shard']}"
        )
    if not full_shard.is_file() or full_shard.stat().st_size < end:
        raise RuntimeError(f"完整shard缺失或长度不足: {full_shard}")
    if final.is_file():
        if final.stat().st_size != expected:
            raise RuntimeError(f"完成文件尺寸异常: {final}")
        return {
            "file": final.name,
            "bytes": expected,
            "sha256": sha256_file(final),
            "resumed": False,
            "imported_from": os.fspath(full_shard.resolve()),
        }

    completed = partial.stat().st_size if partial.is_file() else 0
    if completed > expected:
        raise RuntimeError(f"临时文件超过计划范围: {partial}")
    with full_shard.open("rb") as source:
        if completed:
            source.seek(start)
            with partial.open("rb") as existing:
                checked = 0
                while checked < completed:
                    count = min(8 * 1024 * 1024, completed - checked)
                    left = existing.read(count)
                    right = source.read(count)
                    if left != right:
                        raise RuntimeError(
                            f"已有Range断点与完整shard不一致，偏移约{checked}字节: {partial}"
                        )
                    checked += len(left)
        source.seek(start + completed)
        with partial.open("ab") as target:
            remaining = expected - completed
            while remaining:
                chunk = source.read(min(8 * 1024 * 1024, remaining))
                if not chunk:
                    raise RuntimeError(f"完整shard在目标Range内意外结束: {full_shard}")
                target.write(chunk)
                remaining -= len(chunk)
    if partial.stat().st_size != expected:
        raise RuntimeError(f"导入Range长度不匹配: {partial}")
    partial.replace(final)
    return {
        "file": final.name,
        "bytes": expected,
        "sha256": sha256_file(final),
        "resumed": completed > 0,
        "imported_from": os.fspath(full_shard.resolve()),
    }


def write_receipt(plan_path: Path, output: Path, plan: dict, receipts: list[dict]) -> Path:
    receipt = {
        "format": "colorlm-neural-block-source-v1",
        "plan": os.fspath(plan_path.resolve()),
        "plan_sha256": sha256_file(plan_path),
        "source": plan["source"],
        "total_bytes": sum(int(item["bytes"]) for item in receipts),
        "ranges": receipts,
    }
    receipt_path = output / "source.json"
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return receipt_path


def download_segment(
    session: requests.Session,
    repo: str,
    revision: str,
    output: Path,
    segment: dict,
    request_bytes: int,
    retries: int,
    workers: int,
) -> dict:
    final = output / segment["file"]
    partial = final.with_suffix(final.suffix + ".part")
    expected = int(segment["bytes"])
    if final.is_file():
        if final.stat().st_size != expected:
            raise RuntimeError(f"完成文件尺寸异常: {final}")
        return {
            "file": final.name,
            "bytes": expected,
            "sha256": sha256_file(final),
            "resumed": False,
        }

    completed = partial.stat().st_size if partial.is_file() else 0
    if completed > expected:
        raise RuntimeError(f"临时文件超过计划范围: {partial}")
    resumed = completed > 0
    if workers > 1 and completed < expected:
        remote_base = int(segment["start"])
        remote_limit = int(segment["end_exclusive"])

        def fetch_chunk(remote_start: int, remote_end: int) -> bytes:
            chunk_expected = remote_end - remote_start + 1
            for attempt in range(1, retries + 1):
                try:
                    response = requests.get(
                        source_url(repo, revision, segment["source_shard"]),
                        headers={
                            "Range": f"bytes={remote_start}-{remote_end}",
                            "User-Agent": "ColorLM-Neural-Block-Extractor/1.0",
                        },
                        timeout=(15, 60),
                    )
                    if response.status_code != 206:
                        response.close()
                        raise RuntimeError(
                            f"服务端未执行Range请求: HTTP {response.status_code}"
                        )
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(
                        f"bytes {remote_start}-{remote_end}/"
                    ):
                        response.close()
                        raise RuntimeError(f"Content-Range不匹配: {content_range}")
                    data = response.content
                    if len(data) != chunk_expected:
                        raise RuntimeError(
                            f"子Range长度不匹配: {len(data)} vs {chunk_expected}"
                        )
                    return data
                except (requests.RequestException, RuntimeError):
                    if attempt == retries:
                        raise
                    time.sleep(min(attempt, 3))
            raise AssertionError("unreachable")

        next_start = remote_base + completed
        pending: deque[tuple[int, int, Future[bytes]]] = deque()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            def submit_one() -> bool:
                nonlocal next_start
                if next_start >= remote_limit:
                    return False
                remote_start = next_start
                remote_end = min(remote_limit - 1, remote_start + request_bytes - 1)
                pending.append(
                    (remote_start, remote_end, executor.submit(fetch_chunk, remote_start, remote_end))
                )
                next_start = remote_end + 1
                return True

            for _ in range(workers):
                submit_one()
            with partial.open("ab") as target:
                while pending:
                    remote_start, remote_end, future = pending.popleft()
                    expected_offset = remote_start - remote_base
                    if target.tell() != expected_offset:
                        raise RuntimeError(
                            f"并行Range提交偏移漂移: {target.tell()} vs {expected_offset}"
                        )
                    target.write(future.result())
                    target.flush()
                    completed = target.tell()
                    submit_one()
                    print(
                        f"  {completed / 1024**2:.1f}/{expected / 1024**2:.1f} MiB",
                        flush=True,
                    )

    while completed < expected:
        remote_start = int(segment["start"]) + completed
        remote_end = min(
            int(segment["end_exclusive"]) - 1,
            remote_start + request_bytes - 1,
        )
        chunk_expected = remote_end - remote_start + 1
        for attempt in range(1, retries + 1):
            before = partial.stat().st_size if partial.is_file() else 0
            try:
                response = session.get(
                    source_url(repo, revision, segment["source_shard"]),
                    headers={"Range": f"bytes={remote_start}-{remote_end}"},
                    stream=True,
                    timeout=(15, 120),
                )
                if response.status_code != 206:
                    response.close()
                    raise RuntimeError(
                        f"服务端未执行Range请求: HTTP {response.status_code}"
                    )
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {remote_start}-{remote_end}/"):
                    response.close()
                    raise RuntimeError(f"Content-Range不匹配: {content_range}")
                with partial.open("ab") as stream:
                    for chunk in response.iter_content(4 * 1024 * 1024):
                        if chunk:
                            stream.write(chunk)
                written = partial.stat().st_size - before
                if written != chunk_expected:
                    raise RuntimeError(
                        f"子Range长度不匹配: {written} vs {chunk_expected}"
                    )
                break
            except (requests.RequestException, RuntimeError):
                # A failed response may have appended a prefix. Keep it and let
                # the outer loop resume at the exact next byte.
                after = partial.stat().st_size if partial.is_file() else 0
                if after > before:
                    completed = after
                    break
                if attempt == retries:
                    raise
                time.sleep(min(attempt, 3))
        completed = partial.stat().st_size
        if completed > expected:
            raise RuntimeError(f"Range临时文件超过目标长度: {partial}")
        print(
            f"  {completed / 1024**2:.1f}/{expected / 1024**2:.1f} MiB",
            flush=True,
        )
    if partial.stat().st_size != expected:
        raise RuntimeError(
            f"Range下载长度不匹配: {partial.stat().st_size} vs {expected}"
        )
    partial.replace(final)
    return {
        "file": final.name,
        "bytes": expected,
        "sha256": sha256_file(final),
        "resumed": resumed,
    }


def main() -> int:
    args = parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan.get("format") != "colorlm-neural-block-abi-v1":
        raise RuntimeError("不支持的Neural Block计划格式")
    ranges = plan["source_ranges"]
    if not ranges or sum(int(item["bytes"]) for item in ranges) != int(
        plan["budget"]["bf16_total_bytes"]
    ):
        raise RuntimeError("Range总字节数与ABI预算不一致")
    tensor_bytes = sum(int(item["bytes"]) for item in plan["tensors"])
    if tensor_bytes != int(plan["budget"]["bf16_total_bytes"]):
        raise RuntimeError("张量总字节数与ABI预算不一致")

    args.output.mkdir(parents=True, exist_ok=True)
    imported: dict[str, dict] = {}
    for full_shard in args.import_full_shard:
        matches = [
            segment for segment in ranges
            if segment["source_shard"] == full_shard.name
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"完整shard必须精确匹配一个计划Range: {full_shard}，匹配数={len(matches)}"
            )
        result = import_full_shard(full_shard, args.output, matches[0])
        imported[result["file"]] = result
        print(f"已从完整shard导入: {result['file']}")

    remaining, status = inspect_local(args.output, ranges)
    free = shutil.disk_usage(args.output).free
    reserve = int(args.min_free_gib * 1024**3)
    print(f"Range段: {len(ranges)}")
    print(f"待下载: {remaining / 1024**3:.3f} GiB")
    print(f"当前空闲: {free / 1024**3:.3f} GiB")
    for item in status:
        print(
            f"{item['state']:>8}  {item['bytes'] / 1024**2:8.1f}/"
            f"{item['expected_bytes'] / 1024**2:8.1f} MiB  {item['file']}"
        )
    if free - remaining < reserve:
        raise RuntimeError(
            f"下载后空闲空间将低于{args.min_free_gib:.1f} GiB安全线"
        )
    if not args.download:
        if remaining:
            print("预检通过；加 --download 或 --import-full-shard 补齐剩余Range。")
            return 0
        receipts = []
        for segment in ranges:
            path = args.output / segment["file"]
            receipts.append(imported.get(path.name, {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "resumed": False,
            }))
        receipt_path = write_receipt(args.plan, args.output, plan, receipts)
        print(f"提取完成: {receipt_path}")
        return 0

    source = plan["source"]
    request_bytes = int(args.request_mib * 1024**2)
    if request_bytes <= 0 or args.retries <= 0 or args.workers <= 0:
        raise ValueError("request-mib、retries和workers必须为正数")
    session = requests.Session()
    session.headers["User-Agent"] = "ColorLM-Neural-Block-Extractor/1.0"
    receipts = []
    for ordinal, segment in enumerate(ranges, start=1):
        print(f"下载Range {ordinal}/{len(ranges)}: {segment['file']}", flush=True)
        receipts.append(
            download_segment(
                session,
                source["repo"],
                source["revision"],
                args.output,
                segment,
                request_bytes,
                args.retries,
                args.workers,
            )
        )
    receipt_path = write_receipt(args.plan, args.output, plan, receipts)
    print(f"提取完成: {receipt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
