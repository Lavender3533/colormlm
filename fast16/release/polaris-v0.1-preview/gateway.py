#!/usr/bin/env python3
"""Polaris v0.1 Preview：为现有 v38 服务提供明确标注的草稿代理。"""

from __future__ import annotations

import argparse
import http.client
import json
import socket
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable
from urllib.parse import SplitResult, urlsplit


VERSION = "v0.1-preview"
TARGET_MODEL = "ColorLM-v38-Qwen36-Shared-Sequence-Policy"
VERIFICATION_HEADER = "draft-only"
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
REQUEST_EXCLUDED_HEADERS = HOP_BY_HOP_HEADERS | {"content-length", "host"}
RESPONSE_EXCLUDED_HEADERS = HOP_BY_HOP_HEADERS | {
    "access-control-allow-origin",
    "content-length",
    "date",
    "server",
    "x-polaris-verification",
}
INDEX_HTML = Path(__file__).with_name("index.html").read_bytes()


@dataclass(frozen=True)
class GatewayConfig:
    """网关运行参数。"""

    upstream_url: str = "http://127.0.0.1:8138"
    target_model: str = TARGET_MODEL
    upstream_timeout_seconds: float = 600.0

    def parsed_upstream(self) -> SplitResult:
        parsed = urlsplit(self.upstream_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("--upstream 必须是有效的 http(s) URL")
        return parsed


class PolarisHTTPServer(ThreadingHTTPServer):
    """携带不可变代理配置的多线程 HTTP 服务。"""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address: tuple[str, int], config: GatewayConfig):
        self.config = config
        super().__init__(server_address, PolarisRequestHandler)


class PolarisRequestHandler(BaseHTTPRequestHandler):
    """状态页、体验页和透明 API 代理。"""

    protocol_version = "HTTP/1.1"
    server_version = "PolarisPreview/0.1"
    sys_version = ""

    @property
    def config(self) -> GatewayConfig:
        return self.server.config  # type: ignore[attr-defined, no-any-return]

    def log_message(self, message_format: str, *args: object) -> None:
        print(
            f"[{self.log_date_time_string()}] {self.client_address[0]} "
            f"{message_format % args}",
            flush=True,
        )

    def do_OPTIONS(self) -> None:  # noqa: N802 - 标准库处理器命名约定
        self.send_response(204)
        self._send_common_headers()
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, PATCH, DELETE, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Authorization, Content-Type, X-API-Key, Anthropic-Version, Anthropic-Beta",
        )
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path == "/":
            self._send_local(200, INDEX_HTML, "text/html; charset=utf-8")
            return
        if path == "/polaris/status":
            self._send_json(200, self._status_payload())
            return
        if path.startswith("/polaris/"):
            self._send_json(404, {"error": "未知的 Polaris Preview 路径"})
            return
        self._proxy()

    def do_HEAD(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path == "/":
            self._send_local(200, INDEX_HTML, "text/html; charset=utf-8")
            return
        if path == "/polaris/status":
            body = self._json_bytes(self._status_payload())
            self.send_response(200)
            self._send_common_headers()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            return
        self._proxy()

    def do_POST(self) -> None:  # noqa: N802
        if urlsplit(self.path).path.startswith("/polaris/"):
            self._send_json(405, {"error": "该状态接口仅支持 GET"})
            return
        self._proxy()

    def do_PUT(self) -> None:  # noqa: N802
        self._proxy()

    def do_PATCH(self) -> None:  # noqa: N802
        self._proxy()

    def do_DELETE(self) -> None:  # noqa: N802
        self._proxy()

    def _status_payload(self) -> dict[str, object]:
        return {
            "service": "Polaris v0.1 Preview",
            "version": VERSION,
            "gateway_status": "ready",
            "upstream": self.config.upstream_url,
            "upstream_tcp_reachable": self._upstream_tcp_reachable(),
            "upstream_model": self.config.target_model,
            "exact_verifier": "not_ready",
            "output_mode": "draft_only",
            "capabilities": {
                "openai_proxy": True,
                "anthropic_proxy": True,
                "json": True,
                "sse": True,
                "full_depth_verification": False,
                "k3": False,
            },
            "notice": "当前输出仅为 v38 草稿，不代表 FullDepth 验证结果，也不具备 K3 能力。",
        }

    def _upstream_tcp_reachable(self) -> bool:
        parsed = self.config.parsed_upstream()
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        try:
            with socket.create_connection((parsed.hostname, port), timeout=0.25):
                return True
        except OSError:
            return False

    def _proxy(self) -> None:
        upstream = self.config.parsed_upstream()
        request_body = self._read_request_body()
        request_body, stream_requested = self._rewrite_json_model(request_body)
        target_path = self._target_path(upstream)
        headers = self._upstream_request_headers(request_body)
        port = upstream.port or (443 if upstream.scheme == "https" else 80)
        connection_type = (
            http.client.HTTPSConnection if upstream.scheme == "https" else http.client.HTTPConnection
        )
        connection = connection_type(
            upstream.hostname,
            port,
            timeout=self.config.upstream_timeout_seconds,
        )

        try:
            connection.request(self.command, target_path, body=request_body or None, headers=headers)
            response = connection.getresponse()
            content_type = response.getheader("Content-Type", "")
            is_stream = stream_requested or "text/event-stream" in content_type.lower()
            if self.command == "HEAD":
                self._send_proxy_headers(response.status, response.reason, response.getheaders(), 0)
            elif is_stream:
                self._stream_response(response)
            else:
                response_body = response.read()
                self._send_proxy_response(response, response_body)
        except (OSError, http.client.HTTPException, TimeoutError) as exc:
            if not self.wfile.closed:
                self._send_json(
                    502,
                    {
                        "error": "Polaris Preview 无法连接 v38 上游",
                        "detail": str(exc),
                        "upstream": self.config.upstream_url,
                    },
                )
        finally:
            connection.close()

    def _read_request_body(self) -> bytes:
        raw_length = self.headers.get("Content-Length")
        if not raw_length:
            return b""
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise http.client.HTTPException("无效的 Content-Length") from exc
        return self.rfile.read(max(0, length))

    def _rewrite_json_model(self, body: bytes) -> tuple[bytes, bool]:
        if not body:
            return body, False
        content_type = self.headers.get("Content-Type", "").lower()
        if "json" not in content_type:
            return body, False
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return body, False
        if not isinstance(payload, dict):
            return body, False
        stream_requested = payload.get("stream") is True
        if "model" in payload:
            payload["model"] = self.config.target_model
        return self._json_bytes(payload), stream_requested

    def _target_path(self, upstream: SplitResult) -> str:
        base_path = upstream.path.rstrip("/")
        request_path = self.path if self.path.startswith("/") else f"/{self.path}"
        return f"{base_path}{request_path}" or "/"

    def _upstream_request_headers(self, body: bytes) -> dict[str, str]:
        headers = {
            name: value
            for name, value in self.headers.items()
            if name.lower() not in REQUEST_EXCLUDED_HEADERS
        }
        if body:
            headers["Content-Length"] = str(len(body))
        headers["X-Polaris-Preview"] = VERSION
        return headers

    def _send_proxy_response(self, response: http.client.HTTPResponse, body: bytes) -> None:
        self._send_proxy_headers(response.status, response.reason, response.getheaders(), len(body))
        if body:
            self.wfile.write(body)

    def _stream_response(self, response: http.client.HTTPResponse) -> None:
        self.send_response(response.status, response.reason)
        self._copy_response_headers(response.getheaders())
        self._send_common_headers()
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            while True:
                chunk = response.read1(64 * 1024)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError, http.client.HTTPException, TimeoutError):
            pass

    def _send_proxy_headers(
        self,
        status: int,
        reason: str,
        upstream_headers: Iterable[tuple[str, str]],
        content_length: int,
    ) -> None:
        self.send_response(status, reason)
        self._copy_response_headers(upstream_headers)
        self._send_common_headers()
        self.send_header("Content-Length", str(content_length))
        self.end_headers()

    def _copy_response_headers(self, headers: Iterable[tuple[str, str]]) -> None:
        for name, value in headers:
            if name.lower() not in RESPONSE_EXCLUDED_HEADERS:
                self.send_header(name, value)

    def _send_common_headers(self) -> None:
        self.send_header("X-Polaris-Verification", VERIFICATION_HEADER)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("X-Content-Type-Options", "nosniff")

    def _send_local(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self._send_common_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD" and body:
            self.wfile.write(body)

    def _send_json(self, status: int, payload: dict[str, object]) -> None:
        self._send_local(status, self._json_bytes(payload), "application/json; charset=utf-8")

    @staticmethod
    def _json_bytes(payload: object) -> bytes:
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def create_server(
    listen_host: str = "127.0.0.1",
    listen_port: int = 8140,
    upstream_url: str = "http://127.0.0.1:8138",
    upstream_timeout_seconds: float = 600.0,
) -> PolarisHTTPServer:
    config = GatewayConfig(
        upstream_url=upstream_url.rstrip("/"),
        upstream_timeout_seconds=upstream_timeout_seconds,
    )
    config.parsed_upstream()
    return PolarisHTTPServer((listen_host, listen_port), config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Polaris v0.1 Preview 草稿代理")
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=8140)
    parser.add_argument("--upstream", default="http://127.0.0.1:8138")
    parser.add_argument("--upstream-timeout", type=float, default=600.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    server = create_server(
        listen_host=args.listen_host,
        listen_port=args.listen_port,
        upstream_url=args.upstream,
        upstream_timeout_seconds=args.upstream_timeout,
    )
    host, port = server.server_address[:2]
    print(f"Polaris v0.1 Preview 已启动：http://{host}:{port}", flush=True)
    print(f"v38 上游：{server.config.upstream_url}", flush=True)
    print("验证状态：draft-only；exact_verifier=not_ready；不包含 FullDepth/K3 能力。", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nPolaris Preview 网关已停止。", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
