"""Polaris v0.1 Preview 网关的纯标准库测试。"""

from __future__ import annotations

import http.client
import json
import shutil
import subprocess
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
from gateway import TARGET_MODEL, create_server  # noqa: E402


FAST16_DIR = Path(__file__).resolve().parents[2]
LAUNCHER_SCRIPT = FAST16_DIR / "run-polaris-v0.1-preview.ps1"
LAUNCHER_BATCH = FAST16_DIR / "run-polaris-v0.1-preview.bat"


class FakeUpstreamHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    requests: list[dict[str, object]] = []

    def log_message(self, message_format: str, *args: object) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        body = json.dumps({"status": "ok"}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length)
        payload = json.loads(raw_body.decode("utf-8"))
        self.__class__.requests.append(
            {"path": self.path, "payload": payload, "headers": dict(self.headers.items())}
        )
        if payload.get("stream") is True:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            first = {"choices": [{"delta": {"content": "草稿"}}]}
            second = {"choices": [{"delta": {"content": "完成"}}]}
            self.wfile.write(f"data: {json.dumps(first, ensure_ascii=False)}\n\n".encode("utf-8"))
            self.wfile.flush()
            time.sleep(0.4)
            self.wfile.write(f"data: {json.dumps(second, ensure_ascii=False)}\n\n".encode("utf-8"))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self.close_connection = True
            return

        body = json.dumps(
            {
                "type": "message",
                "model": payload["model"],
                "content": [{"type": "text", "text": "本地假上游"}],
            },
            ensure_ascii=False,
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("X-Fake-Upstream", "yes")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class GatewayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        FakeUpstreamHandler.requests.clear()
        cls.upstream = ThreadingHTTPServer(("127.0.0.1", 0), FakeUpstreamHandler)
        cls.upstream.daemon_threads = True
        cls.upstream_thread = threading.Thread(target=cls.upstream.serve_forever, daemon=True)
        cls.upstream_thread.start()
        upstream_port = cls.upstream.server_address[1]

        cls.gateway = create_server(
            listen_port=0,
            upstream_url=f"http://127.0.0.1:{upstream_port}",
            upstream_timeout_seconds=3,
        )
        cls.gateway_thread = threading.Thread(target=cls.gateway.serve_forever, daemon=True)
        cls.gateway_thread.start()
        cls.gateway_port = cls.gateway.server_address[1]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.gateway.shutdown()
        cls.gateway.server_close()
        cls.upstream.shutdown()
        cls.upstream.server_close()
        cls.gateway_thread.join(timeout=2)
        cls.upstream_thread.join(timeout=2)

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, object] | None = None,
    ) -> tuple[http.client.HTTPResponse, bytes]:
        connection = http.client.HTTPConnection("127.0.0.1", self.gateway_port, timeout=3)
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {} if body is None else {"Content-Type": "application/json"}
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        response_body = response.read()
        connection.close()
        return response, response_body

    def test_status_exposes_draft_only_contract(self) -> None:
        response, body = self.request("GET", "/polaris/status")
        status = json.loads(body.decode("utf-8"))
        self.assertEqual(response.status, 200)
        self.assertEqual(response.getheader("X-Polaris-Verification"), "draft-only")
        self.assertEqual(status["exact_verifier"], "not_ready")
        self.assertEqual(status["output_mode"], "draft_only")
        self.assertFalse(status["capabilities"]["full_depth_verification"])
        self.assertFalse(status["capabilities"]["k3"])

    def test_root_is_utf8_chinese_experience_page(self) -> None:
        response, body = self.request("GET", "/")
        page = body.decode("utf-8")
        self.assertEqual(response.status, 200)
        self.assertIn("Polaris v0.1 Preview", page)
        self.assertIn("草稿模式", page)
        self.assertIn("不具备 K3 能力", page)

    def test_openai_json_request_rewrites_model_alias(self) -> None:
        payload = {
            "model": "polaris-preview",
            "messages": [{"role": "user", "content": "你好"}],
            "stream": False,
        }
        response, _ = self.request("POST", "/v1/chat/completions?trace=test", payload)
        recorded = FakeUpstreamHandler.requests[-1]
        self.assertEqual(response.status, 200)
        self.assertEqual(recorded["path"], "/v1/chat/completions?trace=test")
        self.assertEqual(recorded["payload"]["model"], TARGET_MODEL)
        self.assertEqual(recorded["headers"]["X-Polaris-Preview"], "v0.1-preview")

    def test_anthropic_non_stream_json_is_proxied(self) -> None:
        payload = {
            "model": "claude-compatible-alias",
            "messages": [{"role": "user", "content": "测试"}],
            "max_tokens": 16,
        }
        response, body = self.request("POST", "/v1/messages", payload)
        result = json.loads(body.decode("utf-8"))
        self.assertEqual(response.status, 200)
        self.assertEqual(response.getheader("X-Fake-Upstream"), "yes")
        self.assertEqual(response.getheader("X-Polaris-Verification"), "draft-only")
        self.assertEqual(result["model"], TARGET_MODEL)
        self.assertEqual(FakeUpstreamHandler.requests[-1]["path"], "/v1/messages")

    def test_sse_is_forwarded_incrementally(self) -> None:
        connection = http.client.HTTPConnection("127.0.0.1", self.gateway_port, timeout=3)
        payload = json.dumps(
            {"model": "preview", "messages": [], "stream": True},
            ensure_ascii=False,
        ).encode("utf-8")
        connection.request(
            "POST",
            "/v1/chat/completions",
            body=payload,
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        started = time.monotonic()
        first_line = response.readline().decode("utf-8")
        first_latency = time.monotonic() - started
        remaining = response.read().decode("utf-8")
        connection.close()

        self.assertEqual(response.status, 200)
        self.assertIn("text/event-stream", response.getheader("Content-Type"))
        self.assertEqual(response.getheader("X-Polaris-Verification"), "draft-only")
        self.assertLess(first_latency, 0.3, "首个 SSE 事件不应等待上游结束")
        self.assertIn("草稿", first_line)
        self.assertIn("完成", remaining)
        self.assertIn("[DONE]", remaining)


@unittest.skipUnless(sys.platform == "win32", "PowerShell launcher tests require Windows")
class LauncherCompatibilityTest(unittest.TestCase):
    def run_launcher(self, executable: str, *arguments: str) -> bytes:
        result = subprocess.run(
            [executable, *arguments],
            cwd=FAST16_DIR.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=15,
            check=False,
        )
        self.assertEqual(
            result.returncode,
            0,
            result.stdout.decode("ascii", errors="replace"),
        )
        self.assertIn(b"POLARIS_SELF_TEST_OK", result.stdout)
        return result.stdout

    def test_launcher_script_is_ascii_safe_utf8_without_bom(self) -> None:
        raw = LAUNCHER_SCRIPT.read_bytes()
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
        self.assertTrue(raw.isascii(), "PS1 must remain ASCII-safe for Windows PowerShell 5")
        raw.decode("utf-8", errors="strict")

    def test_windows_powershell_5_startup_self_test(self) -> None:
        executable = shutil.which("powershell.exe")
        if executable is None:
            self.skipTest("powershell.exe is unavailable")
        output = self.run_launcher(
            executable,
            "-NoLogo",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(LAUNCHER_SCRIPT),
            "-SelfTest",
        )
        self.assertIn(b"edition=Desktop", output)

    def test_powershell_7_startup_self_test(self) -> None:
        executable = shutil.which("pwsh.exe")
        if executable is None:
            self.skipTest("pwsh.exe is unavailable")
        output = self.run_launcher(
            executable,
            "-NoLogo",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(LAUNCHER_SCRIPT),
            "-SelfTest",
        )
        self.assertIn(b"edition=Core", output)

    def test_batch_prefers_powershell_7(self) -> None:
        cmd = shutil.which("cmd.exe")
        if cmd is None:
            self.skipTest("cmd.exe is unavailable")
        output = self.run_launcher(cmd, "/d", "/c", str(LAUNCHER_BATCH), "-SelfTest")
        expected_edition = b"edition=Core" if shutil.which("pwsh.exe") else b"edition=Desktop"
        self.assertIn(expected_edition, output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
