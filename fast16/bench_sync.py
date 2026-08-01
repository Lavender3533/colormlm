"""Decode-speed A/B bench + sched-prof reader for the sync-overhead audit.

Usage:
    python fast16/bench_sync.py --label baseline --n 256 --rounds 2

Talks to the native /completion endpoint (returns timings directly), then tails
the newest [sched-prof] lines from the server stderr log so one invocation
yields both the t/s number and the wait-time decomposition.
See fast16/SYNC_OVERHEAD_AUDIT.md for the experiment queue.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import urllib.request
from pathlib import Path

PORT = 8092
BASE = f"http://127.0.0.1:{PORT}"
LOG = Path(__file__).resolve().parent / "runtime" / "qwen36-server.stderr.log"

# Deliberately non-repetitive prose task: keeps ngram-mod speculation out of the
# measurement (repetitive code prompts let the drafter distort per-token cost).
PROMPT = (
    "请用中文连续写一段散文, 主题是深夜的机房与显卡风扇的声音, "
    "不要分段, 不要列表, 不要重复句式, 一直写下去。"
)


def post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.load(response)


def healthy() -> bool:
    try:
        with urllib.request.urlopen(BASE + "/health", timeout=3) as response:
            return json.load(response).get("status") == "ok"
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="run")
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()

    if not healthy():
        print("server 8092 不健康", file=sys.stderr)
        return 1

    log_offset = LOG.stat().st_size if LOG.is_file() else 0

    per_tok = []
    for r in range(args.rounds):
        result = post(
            "/completion",
            {
                "prompt": PROMPT,
                "n_predict": args.n,
                "temperature": 0,
                "cache_prompt": False,
            },
        )
        t = result.get("timings", {})
        ms = t.get("predicted_per_token_ms")
        tps = t.get("predicted_per_second")
        n = t.get("predicted_n")
        per_tok.append(ms)
        print(f"[{args.label}] round {r + 1}: {n} tok, {ms:.2f} ms/tok, {tps:.2f} t/s")

    if len(per_tok) > 1:
        print(f"[{args.label}] median {statistics.median(per_tok):.2f} ms/tok")

    # sched-prof lines emitted since we started
    if LOG.is_file():
        with LOG.open("rb") as fh:
            fh.seek(log_offset)
            tail = fh.read().decode("utf-8", errors="replace")
        prof = [line for line in tail.splitlines() if "[sched-prof]" in line]
        for line in prof[-4:]:
            print(line)
        if not prof:
            print("(无 [sched-prof] 输出 — 服务器未开 GGML_SCHED_PROFILE=1 或不足 128 token)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
