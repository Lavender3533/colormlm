#!/usr/bin/env python3
"""run_gate 的纯 CPU、伪 HTTP 自测。"""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import run_gate
import validate


class FakeHandler(BaseHTTPRequestHandler):
    requests: list[dict[str, Any]] = []

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        type(self).requests.append(body)
        if "tools" in body:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path":"src/main.rs"}'},
                    }
                ],
            }
            finish_reason = "tool_calls"
        else:
            message = {"role": "assistant", "content": '{"answer":31}'}
            finish_reason = "stop"
        response = json.dumps(
            {
                "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
            },
            ensure_ascii=False,
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, _format: str, *_args: object) -> None:
        return


def task(task_id: str, model_input: dict[str, Any], secret: str) -> dict[str, Any]:
    return {
        "schema_version": "capability-task-v1",
        "id": task_id,
        "dimension": "reasoning",
        "split": "dev",
        "family": "selftest",
        "input": model_input,
        "reference_answer": {"secret": secret},
        "validator": {"type": "exact_text", "expected": secret},
        "critical_decision_tokens": [{"id": "secret", "token_text": secret, "rationale": secret}],
        "failure_conditions": [secret],
    }


class RunGateTest(unittest.TestCase):
    def test_fake_http_and_no_oracle_leak(self) -> None:
        secrets = ["REFERENCE_SENTINEL_31", "REFERENCE_SENTINEL_TOOL"]
        tasks = [
            task(
                "dev_reasoning_01",
                {
                    "messages": [{"role": "user", "content": "计算题"}],
                    "temperature": 0,
                    "max_output_tokens": 16,
                },
                secrets[0],
            ),
            task(
                "dev_tools_01",
                {
                    "messages": [{"role": "user", "content": "读取文件"}],
                    "temperature": 0,
                    "max_output_tokens": 24,
                    "tools": [
                        {"name": "read_file", "parameters": {"path": "string"}},
                        {"name": "list_dir", "parameters": {"path": "string"}},
                    ],
                },
                secrets[1],
            ),
        ]

        FakeHandler.requests = []
        server = HTTPServer(("127.0.0.1", 0), FakeHandler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            with tempfile.TemporaryDirectory(prefix="capability-run-gate-") as temp_dir:
                root = Path(temp_dir)
                tasks_path = root / "tasks.jsonl"
                output_path = root / "responses.jsonl"
                tasks_path.write_text(
                    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in tasks),
                    encoding="utf-8",
                    newline="\n",
                )
                count, failures = run_gate.run_tasks(
                    tasks_path=tasks_path,
                    endpoint=f"http://127.0.0.1:{server.server_port}/v1",
                    model="ColorLM-selftest",
                    run_id="selftest-run",
                    output_path=output_path,
                    seed=23,
                    timeout=5,
                )
                self.assertEqual((count, failures), (2, 0))
                rows = validate.read_jsonl(output_path)
                self.assertEqual(len(rows), 2)
                for line, row in enumerate(rows, 1):
                    validate.validate_response(row, output_path, line)
                self.assertEqual(rows[1]["tool_calls"], [{"name": "read_file", "arguments": {"path": "src/main.rs"}}])

                self.assertEqual(len(FakeHandler.requests), 2)
                serialized_requests = json.dumps(FakeHandler.requests, ensure_ascii=False)
                for secret in secrets:
                    self.assertNotIn(secret, serialized_requests)
                for forbidden_key in (
                    "reference_answer",
                    "validator",
                    "critical_decision_tokens",
                    "failure_conditions",
                ):
                    self.assertNotIn(forbidden_key, serialized_requests)
                self.assertEqual(
                    set(FakeHandler.requests[0]),
                    {"model", "messages", "temperature", "max_tokens", "seed", "stream"},
                )
                tool_schema = FakeHandler.requests[1]["tools"][0]["function"]["parameters"]
                self.assertEqual(tool_schema["required"], ["path"])
                self.assertFalse(tool_schema["additionalProperties"])
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=5)

    def test_endpoint_normalization(self) -> None:
        expected = "http://127.0.0.1:8105/v1/chat/completions"
        self.assertEqual(run_gate.normalize_endpoint("http://127.0.0.1:8105"), expected)
        self.assertEqual(run_gate.normalize_endpoint("http://127.0.0.1:8105/v1/"), expected)
        self.assertEqual(run_gate.normalize_endpoint(expected), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
