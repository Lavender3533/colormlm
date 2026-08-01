#!/usr/bin/env python3
"""把官方 DeepSeek-V4 消息编码编译为只读 forced-prefill token 流。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Protocol, Sequence

from .encoding_dsv4 import (
    ASSISTANT_SP_TOKEN,
    LATEST_REMINDER_SP_TOKEN,
    USER_SP_TOKEN,
    bos_token,
    dsml_token,
    encode_messages,
    eos_token,
    thinking_end_token,
    thinking_start_token,
)


INPUT_FORMAT = "polaris-s14-forced-prefill-input-v1"
OUTPUT_FORMAT = "polaris-s14-forced-prefill-v1"
CHAT_ENCODING_REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
S14_VOCAB_SIZE = 129_280
S14_BOS_ID = 0
TOKENIZER_PROFILES = {"s14", "fixture"}
VALID_ROLES = {"system", "user", "assistant", "tool", "latest_reminder", "developer"}
VALID_REASONING_EFFORTS = {"low", "high", "max"}
TOOL_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:-]{0,127}$")
RESERVED_MARKERS = (
    bos_token,
    eos_token,
    USER_SP_TOKEN,
    ASSISTANT_SP_TOKEN,
    LATEST_REMINDER_SP_TOKEN,
    thinking_start_token,
    thinking_end_token,
    dsml_token,
    "<tool_result>",
    "</tool_result>",
)


class ForcedPrefillError(ValueError):
    """输入、tokenizer 或输出违反 forced-prefill 合同时抛出。"""


class TokenizerBackend(Protocol):
    profile: str
    fingerprint: str
    vocab_size: int
    bos_token_id: int
    decoder_runtime_compatible: bool

    def encode(self, text: str) -> tuple[int, ...]: ...


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8", errors="strict")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


class LocalTokenizer:
    """只加载 tokenizer.json；不导入模型代码，也不读取权重。"""

    def __init__(self, tokenizer_path: str | Path, profile: str = "s14"):
        if profile not in TOKENIZER_PROFILES:
            raise ForcedPrefillError(f"未知 tokenizer profile: {profile!r}")
        path = Path(tokenizer_path)
        if not path.is_file():
            raise ForcedPrefillError(f"tokenizer.json 不存在: {path}")
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - 取决于运行环境
            raise ForcedPrefillError("需要 Python tokenizers 包读取 tokenizer.json") from exc
        try:
            tokenizer = Tokenizer.from_file(os.fspath(path))
        except Exception as exc:
            raise ForcedPrefillError(f"无法加载 tokenizer.json: {exc}") from exc

        vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
        bos_id = tokenizer.token_to_id(bos_token)
        if not isinstance(bos_id, int) or isinstance(bos_id, bool):
            raise ForcedPrefillError("tokenizer 缺少官方 BOS token")
        if tokenizer.id_to_token(bos_id) != bos_token:
            raise ForcedPrefillError("tokenizer 的 BOS ID 不能反解为官方 BOS token")
        bos_encoding = tokenizer.encode(bos_token, add_special_tokens=False).ids
        if bos_encoding != [bos_id]:
            raise ForcedPrefillError("官方 BOS 必须被编码为一个独立 token")
        if profile == "s14" and (vocab_size != S14_VOCAB_SIZE or bos_id != S14_BOS_ID):
            raise ForcedPrefillError(
                f"S14 tokenizer 必须是 vocab={S14_VOCAB_SIZE}, BOS={S14_BOS_ID}；"
                f"实际 vocab={vocab_size}, BOS={bos_id}"
            )

        self._tokenizer = tokenizer
        self.profile = profile
        self.fingerprint = _sha256_file(path)
        self.vocab_size = vocab_size
        self.bos_token_id = bos_id
        self.decoder_runtime_compatible = profile == "s14"
        self.path = os.fspath(path.resolve())

    def encode(self, text: str) -> tuple[int, ...]:
        try:
            token_ids = tuple(
                self._tokenizer.encode(text, add_special_tokens=False).ids
            )
        except Exception as exc:
            raise ForcedPrefillError(f"tokenizer 编码失败: {exc}") from exc
        return _validate_token_ids(token_ids, self.vocab_size)


@dataclass(frozen=True)
class ForcedPrefillInput:
    messages: tuple[dict[str, Any], ...]
    reasoning_effort: str
    tools: tuple[dict[str, Any], ...]

    def canonical_value(self) -> dict[str, Any]:
        return {
            "format": INPUT_FORMAT,
            "messages": list(self.messages),
            "reasoning_effort": self.reasoning_effort,
            "tools": list(self.tools),
        }


@dataclass(frozen=True)
class ForcedPrefillBuild:
    artifact: dict[str, Any]
    encoded_prompt: str


def _reject_unknown_keys(
    value: Mapping[str, Any], allowed: set[str], where: str
) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ForcedPrefillError(f"{where} 含未知字段: {sorted(unknown)}")


def _require_keys(value: Mapping[str, Any], required: set[str], where: str) -> None:
    missing = required - set(value)
    if missing:
        raise ForcedPrefillError(f"{where} 缺少字段: {sorted(missing)}")


def _safe_text(value: Any, where: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ForcedPrefillError(f"{where} 必须是字符串")
    if not allow_empty and not value.strip():
        raise ForcedPrefillError(f"{where} 不能为空")
    if "\x00" in value:
        raise ForcedPrefillError(f"{where} 不能包含 NUL")
    if any(0xD800 <= ord(char) <= 0xDFFF for char in value):
        raise ForcedPrefillError(f"{where} 含非法 Unicode surrogate")
    for marker in RESERVED_MARKERS:
        if marker in value:
            raise ForcedPrefillError(f"{where} 不能注入保留协议标记 {marker!r}")
    return value


def _canonical_json(value: Any, where: str) -> Any:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise ForcedPrefillError(f"{where} 不能包含 NaN/Infinity")
        return value
    if isinstance(value, str):
        return _safe_text(value, where, allow_empty=True)
    if isinstance(value, list):
        return [_canonical_json(item, f"{where}[]") for item in value]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ForcedPrefillError(f"{where} 的 JSON object key 必须是字符串")
        return {
            _safe_text(key, f"{where} key", allow_empty=True): _canonical_json(
                value[key], f"{where}.{key}"
            )
            for key in sorted(value)
        }
    raise ForcedPrefillError(f"{where} 含非 JSON 类型 {type(value).__name__}")


def _normalize_tool(tool: Any, index: int) -> dict[str, Any]:
    where = f"tools[{index}]"
    if not isinstance(tool, Mapping):
        raise ForcedPrefillError(f"{where} 必须是 object")
    _reject_unknown_keys(tool, {"type", "function"}, where)
    _require_keys(tool, {"type", "function"}, where)
    if tool["type"] != "function":
        raise ForcedPrefillError(f"{where}.type 只允许 'function'")
    function = tool["function"]
    if not isinstance(function, Mapping):
        raise ForcedPrefillError(f"{where}.function 必须是 object")
    allowed = {"name", "description", "parameters", "strict"}
    _reject_unknown_keys(function, allowed, f"{where}.function")
    _require_keys(function, {"name", "parameters"}, f"{where}.function")
    name = _safe_text(function["name"], f"{where}.function.name")
    if not TOOL_NAME_RE.fullmatch(name):
        raise ForcedPrefillError(f"{where}.function.name 非法: {name!r}")
    parameters = _canonical_json(function["parameters"], f"{where}.function.parameters")
    if not isinstance(parameters, dict):
        raise ForcedPrefillError(f"{where}.function.parameters 必须是 object")
    normalized_function: dict[str, Any] = {"name": name}
    if "description" in function:
        normalized_function["description"] = _safe_text(
            function["description"], f"{where}.function.description", allow_empty=True
        )
    normalized_function["parameters"] = parameters
    if "strict" in function:
        if not isinstance(function["strict"], bool):
            raise ForcedPrefillError(f"{where}.function.strict 必须是 bool")
        normalized_function["strict"] = function["strict"]
    return {"type": "function", "function": normalized_function}


def _normalize_tool_call(
    value: Any, message_index: int, call_index: int, known_tools: set[str]
) -> dict[str, Any]:
    where = f"messages[{message_index}].tool_calls[{call_index}]"
    if not isinstance(value, Mapping):
        raise ForcedPrefillError(f"{where} 必须是 object")
    _reject_unknown_keys(value, {"id", "type", "function"}, where)
    _require_keys(value, {"type", "function"}, where)
    if value["type"] != "function":
        raise ForcedPrefillError(f"{where}.type 只允许 'function'")
    function = value["function"]
    if not isinstance(function, Mapping):
        raise ForcedPrefillError(f"{where}.function 必须是 object")
    _reject_unknown_keys(function, {"name", "arguments"}, f"{where}.function")
    _require_keys(function, {"name", "arguments"}, f"{where}.function")
    name = _safe_text(function["name"], f"{where}.function.name")
    if not TOOL_NAME_RE.fullmatch(name) or name not in known_tools:
        raise ForcedPrefillError(f"{where} 引用了未定义工具 {name!r}")
    arguments_text = _safe_text(
        function["arguments"], f"{where}.function.arguments", allow_empty=False
    )
    try:
        arguments = _loads_json_text(arguments_text, f"{where}.function.arguments")
    except ForcedPrefillError:
        raise
    if not isinstance(arguments, Mapping):
        raise ForcedPrefillError(f"{where}.function.arguments 必须编码 JSON object")
    arguments = _canonical_json(arguments, f"{where}.function.arguments")
    call_id = value.get("id", f"call_{message_index}_{call_index}")
    call_id = _safe_text(call_id, f"{where}.id")
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(
                arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ),
        },
    }


def _normalize_messages(
    values: Any, known_tools: set[str]
) -> tuple[dict[str, Any], ...]:
    if not isinstance(values, list) or not values:
        raise ForcedPrefillError("messages 必须是非空数组")
    if len(values) > 1024:
        raise ForcedPrefillError("messages 超过 1024 条硬上限")

    normalized: list[dict[str, Any]] = []
    pending_call_ids: list[str] = []
    seen_call_ids: set[str] = set()
    previous_role: str | None = None
    for index, message in enumerate(values):
        where = f"messages[{index}]"
        if not isinstance(message, Mapping):
            raise ForcedPrefillError(f"{where} 必须是 object")
        role = message.get("role")
        if not isinstance(role, str) or role not in VALID_ROLES:
            raise ForcedPrefillError(f"{where}.role 非法: {role!r}")
        if role == "system":
            if index != 0:
                raise ForcedPrefillError("system 消息只允许位于 messages[0]")
            allowed = {"role", "content"}
        elif role in {"user", "developer", "latest_reminder"}:
            allowed = {"role", "content"}
        elif role == "assistant":
            allowed = {"role", "content", "reasoning_content", "tool_calls"}
        else:
            allowed = {"role", "content", "tool_call_id"}
        _reject_unknown_keys(message, allowed, where)
        _require_keys(message, {"role"}, where)

        if role in {"system", "user", "developer", "latest_reminder"}:
            _require_keys(message, {"content"}, where)
            if pending_call_ids:
                raise ForcedPrefillError(
                    f"{where} 出现在未闭合工具调用 {pending_call_ids!r} 之前"
                )
            normalized.append(
                {"role": role, "content": _safe_text(message["content"], f"{where}.content")}
            )
            previous_role = role
            continue

        if role == "assistant":
            if previous_role not in {"user", "developer", "tool"}:
                raise ForcedPrefillError(
                    f"{where} assistant 必须跟在 user/developer/tool 之后"
                )
            if pending_call_ids:
                raise ForcedPrefillError(
                    f"{where} 出现在未闭合工具调用 {pending_call_ids!r} 之前"
                )
            item: dict[str, Any] = {"role": "assistant"}
            content = message.get("content")
            if content is not None:
                item["content"] = _safe_text(
                    content, f"{where}.content", allow_empty=True
                )
            reasoning = message.get("reasoning_content")
            if reasoning is not None:
                item["reasoning_content"] = _safe_text(
                    reasoning, f"{where}.reasoning_content", allow_empty=True
                )
            calls_value = message.get("tool_calls")
            if calls_value is not None:
                if not isinstance(calls_value, list) or not calls_value:
                    raise ForcedPrefillError(f"{where}.tool_calls 必须是非空数组")
                calls = [
                    _normalize_tool_call(call, index, call_index, known_tools)
                    for call_index, call in enumerate(calls_value)
                ]
                call_ids = [call["id"] for call in calls]
                if len(call_ids) != len(set(call_ids)) or any(
                    call_id in seen_call_ids for call_id in call_ids
                ):
                    raise ForcedPrefillError(f"{where}.tool_calls ID 必须全局唯一")
                seen_call_ids.update(call_ids)
                pending_call_ids = call_ids
                item["tool_calls"] = calls
            if not item.get("content") and not item.get("tool_calls"):
                raise ForcedPrefillError(f"{where} 必须含非空 content 或 tool_calls")
            normalized.append(item)
            previous_role = role
            continue

        _require_keys(message, {"content"}, where)
        if not pending_call_ids:
            raise ForcedPrefillError(f"{where} 没有待接收的 assistant tool_call")
        call_id_value = message.get("tool_call_id")
        if call_id_value is None:
            if len(pending_call_ids) != 1:
                raise ForcedPrefillError(f"{where}.tool_call_id 在多调用场景中不可省略")
            call_id = pending_call_ids[0]
        else:
            call_id = _safe_text(call_id_value, f"{where}.tool_call_id")
        if call_id not in pending_call_ids:
            raise ForcedPrefillError(f"{where}.tool_call_id 未匹配待接收调用")
        pending_call_ids.remove(call_id)
        normalized.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": _safe_text(message["content"], f"{where}.content"),
            }
        )
        previous_role = role

    if pending_call_ids:
        raise ForcedPrefillError(f"消息结束时仍有未闭合工具调用: {pending_call_ids!r}")
    if normalized[0]["role"] == "assistant":
        raise ForcedPrefillError("messages 不能以 assistant 开始")
    if normalized[-1]["role"] not in {"user", "developer", "tool"}:
        raise ForcedPrefillError("forced-prefill 消息必须以 user/developer/tool 请求结束")
    return tuple(normalized)


def validate_input(value: Any) -> ForcedPrefillInput:
    if not isinstance(value, Mapping):
        raise ForcedPrefillError("输入根节点必须是 object")
    allowed = {"format", "messages", "reasoning_effort", "tools"}
    _reject_unknown_keys(value, allowed, "input")
    _require_keys(value, allowed, "input")
    if value["format"] != INPUT_FORMAT:
        raise ForcedPrefillError(f"input.format 必须是 {INPUT_FORMAT!r}")
    effort = value["reasoning_effort"]
    if not isinstance(effort, str) or effort not in VALID_REASONING_EFFORTS:
        raise ForcedPrefillError(
            f"reasoning_effort 必须是 {sorted(VALID_REASONING_EFFORTS)} 之一"
        )
    tools_value = value["tools"]
    if not isinstance(tools_value, list):
        raise ForcedPrefillError("tools 必须是数组")
    if len(tools_value) > 128:
        raise ForcedPrefillError("tools 超过 128 个硬上限")
    tools = tuple(_normalize_tool(tool, index) for index, tool in enumerate(tools_value))
    tools = tuple(sorted(tools, key=lambda tool: tool["function"]["name"]))
    names = [tool["function"]["name"] for tool in tools]
    if len(names) != len(set(names)):
        raise ForcedPrefillError("tools.function.name 必须唯一")
    messages = _normalize_messages(value["messages"], set(names))
    return ForcedPrefillInput(messages=messages, reasoning_effort=effort, tools=tools)


def _inject_tools(contract: ForcedPrefillInput) -> list[dict[str, Any]]:
    messages = copy.deepcopy(list(contract.messages))
    if not contract.tools:
        return messages
    destination = next(
        (message for message in messages if message["role"] == "system"), None
    )
    if destination is None:
        destination = next(
            (message for message in messages if message["role"] == "developer"), None
        )
    if destination is None:
        destination = {"role": "system", "content": ""}
        messages.insert(0, destination)
    destination["tools"] = copy.deepcopy(list(contract.tools))
    return messages


def _validate_token_ids(values: Sequence[int], vocab_size: int) -> tuple[int, ...]:
    result = tuple(values)
    if not result:
        raise ForcedPrefillError("tokenizer 返回了空 token 流")
    if any(not isinstance(token, int) or isinstance(token, bool) for token in result):
        raise ForcedPrefillError("token ID 必须是 int")
    if any(token < 0 or token >= vocab_size for token in result):
        raise ForcedPrefillError("token ID 越出 tokenizer vocab")
    return result


def _validate_tokenizer_backend(tokenizer: TokenizerBackend) -> None:
    profile = getattr(tokenizer, "profile", None)
    fingerprint = getattr(tokenizer, "fingerprint", None)
    vocab_size = getattr(tokenizer, "vocab_size", None)
    bos_id = getattr(tokenizer, "bos_token_id", None)
    compatible = getattr(tokenizer, "decoder_runtime_compatible", None)
    if profile not in TOKENIZER_PROFILES:
        raise ForcedPrefillError("tokenizer backend profile 非法")
    if not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise ForcedPrefillError("tokenizer backend fingerprint 必须是小写 SHA-256")
    if (
        not isinstance(vocab_size, int)
        or isinstance(vocab_size, bool)
        or vocab_size <= 0
    ):
        raise ForcedPrefillError("tokenizer backend vocab_size 非法")
    if not isinstance(bos_id, int) or isinstance(bos_id, bool) or not 0 <= bos_id < vocab_size:
        raise ForcedPrefillError("tokenizer backend bos_token_id 非法")
    if not isinstance(compatible, bool):
        raise ForcedPrefillError("tokenizer backend 兼容标记必须是 bool")
    if profile == "s14":
        if vocab_size != S14_VOCAB_SIZE or bos_id != S14_BOS_ID or not compatible:
            raise ForcedPrefillError("s14 backend 未满足 vocab/BOS/兼容性硬合同")
    elif compatible:
        raise ForcedPrefillError("fixture backend 不得冒充 Polaris S14 兼容 tokenizer")


def compile_forced_prefill(
    value: Any, tokenizer: TokenizerBackend
) -> ForcedPrefillBuild:
    _validate_tokenizer_backend(tokenizer)
    contract = validate_input(value)
    messages = _inject_tools(contract)
    try:
        prompt = encode_messages(
            messages,
            thinking_mode="thinking",
            reasoning_effort=contract.reasoning_effort,
            drop_thinking=True,
            add_default_bos_token=True,
        )
    except (AssertionError, KeyError, NotImplementedError, TypeError, ValueError) as exc:
        raise ForcedPrefillError(f"官方 DeepSeek-V4 encoding 拒绝输入: {exc}") from exc
    if not prompt.startswith(bos_token) or prompt.count(bos_token) != 1:
        raise ForcedPrefillError("编码结果必须且只能以一个官方 BOS 开始")
    if not prompt.endswith(ASSISTANT_SP_TOKEN + thinking_start_token):
        raise ForcedPrefillError("编码结果未落在 assistant thinking forced-prefill 边界")
    try:
        prompt_bytes = prompt.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ForcedPrefillError(f"编码结果不是合法 UTF-8: {exc}") from exc
    token_ids = _validate_token_ids(tokenizer.encode(prompt), tokenizer.vocab_size)
    if token_ids[0] != tokenizer.bos_token_id:
        raise ForcedPrefillError(
            f"首 token ID {token_ids[0]} 不是 tokenizer BOS {tokenizer.bos_token_id}"
        )

    token_ids_list = list(token_ids)
    artifact = {
        "format": OUTPUT_FORMAT,
        "chat_encoding": {
            "implementation": "official-deepseek-v4",
            "revision": CHAT_ENCODING_REVISION,
            "thinking_mode": "thinking",
            "reasoning_effort": contract.reasoning_effort,
        },
        "input": {
            "sha256": _sha256_json(contract.canonical_value()),
            "message_count": len(contract.messages),
            "roles": [message["role"] for message in contract.messages],
            "tool_count": len(contract.tools),
        },
        "prompt": {
            "utf8_bytes": len(prompt_bytes),
            "utf8_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
            "terminal_boundary": "<｜Assistant｜><think>",
        },
        "tokenizer": {
            "profile": tokenizer.profile,
            "sha256": tokenizer.fingerprint,
            "vocab_size": tokenizer.vocab_size,
            "bos_token": bos_token,
            "bos_token_id": tokenizer.bos_token_id,
        },
        "token_ids": token_ids_list,
        "token_count": len(token_ids_list),
        "token_ids_sha256": _sha256_json(token_ids_list),
        "decoder_consumption": {
            "mode": "sequential_forced_prefill",
            "position_base": 0,
            "position_count": len(token_ids_list),
            "position_rule": "token_ids[position]",
            "polaris_s14_compatible": tokenizer.decoder_runtime_compatible,
        },
        "execution": {
            "model_executed": False,
            "generated_token_count": 0,
        },
    }
    return ForcedPrefillBuild(artifact=artifact, encoded_prompt=prompt)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ForcedPrefillError(f"JSON 含重复 key: {key!r}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ForcedPrefillError(f"JSON 不允许常量 {value}")


def _loads_json_text(text: str, where: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_invalid_constant,
        )
    except ForcedPrefillError:
        raise
    except json.JSONDecodeError as exc:
        raise ForcedPrefillError(f"{where} 不是合法 JSON: {exc}") from exc


def load_input_bytes(data: bytes, where: str = "input") -> Any:
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ForcedPrefillError(f"{where} 不是合法 UTF-8: {exc}") from exc
    return _loads_json_text(text, where)


def _read_input(path: str) -> Any:
    if path == "-":
        return load_input_bytes(sys.stdin.buffer.read(), "stdin")
    source = Path(path)
    try:
        data = source.read_bytes()
    except OSError as exc:
        raise ForcedPrefillError(f"无法读取输入 {source}: {exc}") from exc
    return load_input_bytes(data, os.fspath(source))


def _write_output(path: str, artifact: Mapping[str, Any]) -> None:
    payload = json.dumps(
        artifact, ensure_ascii=False, sort_keys=True, indent=2
    ).encode("utf-8") + b"\n"
    if path == "-":
        sys.stdout.buffer.write(payload)
        return
    destination = Path(path)
    if not destination.parent.is_dir():
        raise ForcedPrefillError(f"输出目录不存在: {destination.parent}")
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, destination)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise ForcedPrefillError(f"无法写入输出 {destination}: {exc}") from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="把 DeepSeek-V4 messages 编译为 Polaris S14 forced-prefill token IDs"
    )
    parser.add_argument("--input", required=True, help="UTF-8 JSON 输入；'-' 表示 stdin")
    parser.add_argument("--tokenizer", required=True, help="本地 tokenizer.json")
    parser.add_argument(
        "--tokenizer-profile",
        choices=sorted(TOKENIZER_PROFILES),
        default="s14",
        help="默认 s14 会硬校验 vocab=129280/BOS=0；fixture 仅供离线测试",
    )
    parser.add_argument("--output", default="-", help="UTF-8 JSON 输出；默认 stdout")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        value = _read_input(args.input)
        tokenizer = LocalTokenizer(args.tokenizer, profile=args.tokenizer_profile)
        result = compile_forced_prefill(value, tokenizer)
        _write_output(args.output, result.artifact)
    except ForcedPrefillError as exc:
        print(f"forced-prefill 拒绝: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
