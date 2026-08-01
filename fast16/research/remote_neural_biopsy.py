"""Inspect and extract tensor byte ranges from ModelScope Safetensors shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import struct
import sys
from pathlib import Path
from urllib.parse import quote

import requests


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = ROOT / "fast16" / "research" / "biopsy_cache"
MODELSCOPE = "https://modelscope.cn"
JSON_LIMIT = 64 * 1024 * 1024

DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


class RemoteSafetensors:
    def __init__(self, repo: str, revision: str, cache_dir: Path) -> None:
        if repo.count("/") != 1:
            raise ValueError("repo必须是namespace/model格式")
        self.repo = repo
        self.revision = revision
        self.cache_dir = cache_dir / safe_slug(repo) / safe_slug(revision)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "ColorLM-Neural-Biopsy/1.0"
        self.index = self._load_index()
        self.weight_map: dict[str, str] = self.index["weight_map"]

    def file_url(self, path: str) -> str:
        return (
            f"{MODELSCOPE}/models/{quote(self.repo, safe='/')}/resolve/"
            f"{quote(self.revision, safe='')}/{quote(path, safe='/')}"
        )

    def _download_json(self, path: str, destination: Path) -> dict:
        if destination.is_file():
            return json.loads(destination.read_text(encoding="utf-8"))
        response = self.session.get(self.file_url(path), stream=True, timeout=(15, 60))
        response.raise_for_status()
        content_length = int(response.headers.get("Content-Length", "0") or 0)
        if content_length > JSON_LIMIT:
            response.close()
            raise RuntimeError(f"JSON文件超过{JSON_LIMIT}字节限制: {path}")
        data = bytearray()
        for chunk in response.iter_content(1024 * 1024):
            data.extend(chunk)
            if len(data) > JSON_LIMIT:
                response.close()
                raise RuntimeError(f"JSON响应超过{JSON_LIMIT}字节限制: {path}")
        parsed = json.loads(data.decode("utf-8"))
        destination.write_text(
            json.dumps(parsed, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        return parsed

    def _load_index(self) -> dict:
        path = self.cache_dir / "model.safetensors.index.json"
        index = self._download_json("model.safetensors.index.json", path)
        if not isinstance(index.get("weight_map"), dict):
            raise RuntimeError("模型索引缺少weight_map")
        return index

    def _range_bytes(self, path: str, start: int, end: int) -> bytes:
        response = self.session.get(
            self.file_url(path),
            headers={"Range": f"bytes={start}-{end}"},
            stream=True,
            timeout=(15, 60),
        )
        if response.status_code != 206:
            response.close()
            raise RuntimeError(
                f"服务端未执行Range请求，已中止以避免整分片下载: HTTP {response.status_code}"
            )
        expected = end - start + 1
        content_range = response.headers.get("Content-Range", "")
        if not content_range.startswith(f"bytes {start}-{end}/"):
            response.close()
            raise RuntimeError(f"Content-Range不匹配: {content_range}")
        data = response.content
        if len(data) != expected:
            raise RuntimeError(f"Range长度不匹配: {len(data)} vs {expected}")
        return data

    def shard_header(self, shard: str) -> tuple[int, dict]:
        cache = self.cache_dir / "headers" / f"{safe_slug(shard)}.json"
        if cache.is_file():
            saved = json.loads(cache.read_text(encoding="utf-8"))
            return int(saved["header_bytes"]), saved["tensors"]

        prefix = self._range_bytes(shard, 0, 7)
        header_bytes = struct.unpack("<Q", prefix)[0]
        if header_bytes <= 2 or header_bytes > JSON_LIMIT:
            raise RuntimeError(f"Safetensors头长度异常: {header_bytes}")
        raw = self._range_bytes(shard, 8, 8 + header_bytes - 1)
        header = json.loads(raw.decode("utf-8"))
        tensors = {name: value for name, value in header.items() if name != "__metadata__"}
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(
            json.dumps(
                {"header_bytes": header_bytes, "tensors": tensors},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        return header_bytes, tensors

    def tensor_info(self, name: str) -> dict:
        shard = self.weight_map.get(name)
        if shard is None:
            raise KeyError(f"张量不存在: {name}")
        header_bytes, tensors = self.shard_header(shard)
        item = tensors.get(name)
        if item is None:
            raise RuntimeError(f"分片头中找不到张量: {name}")
        dtype = item["dtype"]
        shape = [int(value) for value in item["shape"]]
        offsets = [int(value) for value in item["data_offsets"]]
        if dtype not in DTYPE_BYTES:
            raise RuntimeError(f"不支持的Safetensors dtype: {dtype}")
        logical_bytes = math.prod(shape) * DTYPE_BYTES[dtype]
        if offsets[1] - offsets[0] != logical_bytes:
            raise RuntimeError(f"张量字节数不匹配: {name}")
        data_start = 8 + header_bytes
        return {
            "name": name,
            "shard": shard,
            "dtype": dtype,
            "shape": shape,
            "bytes": logical_bytes,
            "absolute_start": data_start + offsets[0],
            "absolute_end": data_start + offsets[1],
        }

    def matching(self, pattern: str) -> list[str]:
        matcher = re.compile(pattern)
        return sorted(name for name in self.weight_map if matcher.search(name))

    def download_file(
        self,
        path: str,
        output: Path,
        max_download_bytes: int,
        min_free_bytes: int,
    ) -> dict:
        if output.is_file():
            size = output.stat().st_size
        else:
            response = self.session.get(self.file_url(path), stream=True, timeout=(15, 120))
            response.raise_for_status()
            content_length = int(response.headers.get("Content-Length", "0") or 0)
            if content_length > max_download_bytes:
                response.close()
                raise RuntimeError(f"文件超过下载配额: {content_length}字节")
            output.parent.mkdir(parents=True, exist_ok=True)
            free = shutil.disk_usage(output.parent).free
            required = content_length or max_download_bytes
            if free - required < min_free_bytes:
                response.close()
                raise RuntimeError("下载后剩余空间将低于安全线")
            partial = output.with_suffix(output.suffix + ".part")
            written = 0
            with partial.open("wb") as stream:
                for chunk in response.iter_content(1024 * 1024):
                    if not chunk:
                        continue
                    written += len(chunk)
                    if written > max_download_bytes:
                        response.close()
                        stream.close()
                        partial.unlink(missing_ok=True)
                        raise RuntimeError("流式响应超过下载配额")
                    stream.write(chunk)
            if content_length and written != content_length:
                partial.unlink(missing_ok=True)
                raise RuntimeError(f"文件长度不匹配: {written} vs {content_length}")
            partial.replace(output)
            size = written
        digest = hashlib.sha256(output.read_bytes()).hexdigest()
        return {
            "repo": self.repo,
            "revision": self.revision,
            "path": path,
            "output": os.fspath(output),
            "bytes": size,
            "sha256": digest,
        }

    def extract(
        self,
        name: str,
        output: Path,
        axis0: tuple[int, int] | None,
        max_download_bytes: int,
        min_free_bytes: int,
    ) -> dict:
        info = self.tensor_info(name)
        start = info["absolute_start"]
        end = info["absolute_end"]
        output_shape = list(info["shape"])
        selection = None

        if axis0 is not None:
            if not output_shape:
                raise ValueError("标量张量不支持axis0切片")
            first, stop = axis0
            if not 0 <= first < stop <= output_shape[0]:
                raise ValueError(f"axis0范围越界: {first}:{stop} vs {output_shape[0]}")
            row_bytes = math.prod(output_shape[1:]) * DTYPE_BYTES[info["dtype"]]
            start += first * row_bytes
            end = start + (stop - first) * row_bytes
            output_shape[0] = stop - first
            selection = {"axis": 0, "start": first, "stop": stop}

        expected = end - start
        if expected > max_download_bytes:
            raise RuntimeError(
                f"计划下载{expected / 1024**3:.3f}GiB，超过单次配额"
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(output.parent).free
        if free - expected < min_free_bytes:
            raise RuntimeError(
                f"下载后剩余空间将低于{min_free_bytes / 1024**3:.1f}GiB安全线"
            )

        partial = output.with_suffix(output.suffix + ".part")
        if output.is_file():
            if output.stat().st_size != expected:
                raise RuntimeError(f"已有输出大小不一致: {output}")
        else:
            completed = partial.stat().st_size if partial.is_file() else 0
            if completed > expected:
                raise RuntimeError(f"临时文件大于计划范围: {partial}")
            if completed < expected:
                remote_start = start + completed
                remote_end = end - 1
                response = self.session.get(
                    self.file_url(info["shard"]),
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
                if partial.stat().st_size != expected:
                    raise RuntimeError(
                        f"下载字节数不匹配: {partial.stat().st_size} vs {expected}"
                    )
            partial.replace(output)

        digest = hashlib.sha256()
        with output.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
        report = {
            "format": "colorlm-neural-biopsy-v1",
            "repo": self.repo,
            "revision": self.revision,
            "tensor": name,
            "source_shard": info["shard"],
            "source_dtype": info["dtype"],
            "source_shape": info["shape"],
            "selection": selection,
            "output_shape": output_shape,
            "output_bytes": expected,
            "output_sha256": digest.hexdigest(),
            "output": output.name,
        }
        output.with_suffix(output.suffix + ".json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return report


def parse_axis0(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    match = re.fullmatch(r"(\d+):(\d+)", value)
    if not match:
        raise argparse.ArgumentTypeError("axis0必须是start:stop")
    return int(match.group(1)), int(match.group(2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ModelScope远程Safetensors神经活检")
    parser.add_argument("--repo", required=True, help="ModelScope namespace/model")
    parser.add_argument("--revision", default="master")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    commands = parser.add_subparsers(dest="command", required=True)

    inspect = commands.add_parser("inspect", help="只读取索引和分片头")
    inspect.add_argument("--match", default=".*")
    inspect.add_argument("--limit", type=int, default=40)
    inspect.add_argument("--shapes", action="store_true")

    metadata = commands.add_parser("metadata", help="下载受大小限制的JSON元数据")
    metadata.add_argument("--path", required=True, help="仓库内JSON路径")
    metadata.add_argument("--output", type=Path)

    file_command = commands.add_parser("file", help="下载受大小和磁盘安全线限制的小文件")
    file_command.add_argument("--path", required=True, help="仓库内文件路径")
    file_command.add_argument("--output", type=Path)
    file_command.add_argument("--max-download-mib", type=float, default=64.0)
    file_command.add_argument("--min-free-gib", type=float, default=30.0)

    extract = commands.add_parser("extract", help="下载单个张量或axis-0切片")
    extract.add_argument("--tensor", required=True)
    extract.add_argument("--axis0", type=parse_axis0)
    extract.add_argument("--output", type=Path)
    extract.add_argument("--max-download-gib", type=float, default=4.0)
    extract.add_argument("--min-free-gib", type=float, default=30.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    remote = RemoteSafetensors(args.repo, args.revision, args.cache_dir)
    if args.command == "inspect":
        names = remote.matching(args.match)
        print(f"repo={args.repo} matching={len(names)}")
        for name in names[: args.limit]:
            if args.shapes:
                info = remote.tensor_info(name)
                print(
                    f"{name}\t{info['dtype']}\t{info['shape']}\t"
                    f"{info['bytes'] / 1024**2:.3f}MiB\t{info['shard']}"
                )
            else:
                print(f"{name}\t{remote.weight_map[name]}")
        return 0

    if args.command == "metadata":
        output = args.output
        if output is None:
            output = remote.cache_dir / "metadata" / safe_slug(args.path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = remote._download_json(args.path, output)
        print(
            json.dumps(
                {
                    "repo": args.repo,
                    "revision": args.revision,
                    "path": args.path,
                    "output": os.fspath(output),
                    "bytes": output.stat().st_size,
                    "top_level_type": type(payload).__name__,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "file":
        output = args.output
        if output is None:
            output = remote.cache_dir / "metadata" / safe_slug(args.path)
        report = remote.download_file(
            args.path,
            output,
            int(args.max_download_mib * 1024**2),
            int(args.min_free_gib * 1024**3),
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    output = args.output
    if output is None:
        suffix = f".axis0-{args.axis0[0]}-{args.axis0[1]}" if args.axis0 else ""
        output = remote.cache_dir / "extracted" / f"{safe_slug(args.tensor)}{suffix}.bin"
    report = remote.extract(
        args.tensor,
        output,
        args.axis0,
        int(args.max_download_gib * 1024**3),
        int(args.min_free_gib * 1024**3),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
