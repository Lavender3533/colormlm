"""Polaris S14 官方聊天编码的纯离线自检。"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .encoding_dsv4 import encode_messages, parse_message_from_completion_text


ROOT = Path(__file__).resolve().parent
DEFAULT_TOKENIZER = Path(r"D:\models\Polaris-S14\tokenizer.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8", errors="strict"))


def _validate_source_contract() -> int:
    contract = _load_json(ROOT / "source_contract.json")
    assert contract["repo"] == "deepseek-ai/DeepSeek-V4-Flash-0731"
    assert contract["revision"] == "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
    checked = 0
    for entry in contract["files"]:
        path = ROOT / entry["path"]
        assert path.is_file(), path
        assert _sha256(path) == entry["local_sha256"], path
        checked += 1
    return checked


def _validate_official_fixtures() -> int:
    modes = {1: "thinking", 2: "thinking", 3: "thinking", 4: "chat"}
    for case in range(1, 5):
        source = _load_json(ROOT / "tests" / f"test_input_{case}.json")
        if case == 1:
            messages = source["messages"]
            messages[0]["tools"] = source["tools"]
        else:
            messages = source
        actual = encode_messages(messages, thinking_mode=modes[case])
        expected_snapshot = (ROOT / "tests" / f"test_output_{case}.txt").read_text(
            encoding="utf-8", errors="strict"
        )
        assert expected_snapshot.endswith("\n")
        expected = expected_snapshot[:-1]
        assert actual == expected, f"official encoding fixture {case} drift"

    prompt = encode_messages(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "1+1=?"},
        ],
        thinking_mode="thinking",
        reasoning_effort="max",
    )
    assert prompt.startswith("<｜begin▁of▁sentence｜>Reasoning Effort: Beyond maximum")
    assert prompt.endswith("<｜Assistant｜><think>")

    completion = (
        "Need a file.</think>\n\n<｜DSML｜tool_calls>\n"
        '<｜DSML｜invoke name="read_file">\n'
        '<｜DSML｜parameter name="path" string="true">D:/x.txt</｜DSML｜parameter>\n'
        "</｜DSML｜invoke>\n</｜DSML｜tool_calls><｜end▁of▁sentence｜>"
    )
    parsed = parse_message_from_completion_text(completion, thinking_mode="thinking")
    assert parsed["reasoning_content"] == "Need a file."
    assert parsed["content"] == ""
    assert len(parsed["tool_calls"]) == 1
    assert parsed["tool_calls"][0]["function"]["name"] == "read_file"
    assert json.loads(parsed["tool_calls"][0]["function"]["arguments"]) == {
        "path": "D:/x.txt"
    }
    return 4


def _validate_local_tokenizer() -> dict[str, object]:
    path = Path(os.environ.get("POLARIS_S14_TOKENIZER", os.fspath(DEFAULT_TOKENIZER)))
    if not path.is_file():
        return {"checked": False, "reason": f"missing: {path}"}
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(os.fspath(path))
    prompt = encode_messages(
        [{"role": "user", "content": "你好"}],
        thinking_mode="thinking",
        reasoning_effort="low",
    )
    ids = tokenizer.encode(prompt).ids
    assert ids and ids[0] == 0
    assert tokenizer.id_to_token(0) == "<｜begin▁of▁sentence｜>"
    return {"checked": True, "token_count": len(ids), "first_token_id": ids[0]}


def main() -> int:
    report = {
        "status": "pass",
        "source_files_checked": _validate_source_contract(),
        "official_fixtures_checked": _validate_official_fixtures(),
        "tokenizer": _validate_local_tokenizer(),
        "network_accessed": False,
        "claim_limit": "只证明官方消息编码固定且可复现，不证明模型质量",
    }
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
