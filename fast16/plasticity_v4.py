"""ColorLM v4 local next-token plasticity trainer.

The quantized trunk stays frozen.  Training obtains the model's causal final
hidden states, projects them through fixed A, and applies a local delta rule to
the target-token rows of B.  No source text is stored in the weight file.
"""

from __future__ import annotations

import argparse
import json
import struct
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np


BASE_URL = "http://127.0.0.1:8092"
DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "runtime" / "colorlm-v4-plastic.bin"
HEADER = struct.Struct("<6I2Q")
MAGIC = 0x34504C43  # CLP4
VERSION = 1
STATE_HEADER = struct.Struct("<4I")
STATE_MAGIC = 0x34535043  # CPS4
STATE_VERSION = 1


def post_json(path: str, payload: dict[str, Any], timeout: int = 300) -> Any:
    request = urllib.request.Request(
        BASE_URL + path,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def tokenize(text: str) -> list[int]:
    response = post_json(
        "/tokenize",
        {"content": text, "add_special": True, "parse_special": True},
    )
    return [int(token) for token in response["tokens"]]


def render_question(question: str, system: str | None = None) -> str:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": question})
    response = post_json(
        "/apply-template",
        {"messages": messages},
    )
    return str(response["prompt"])


def causal_embeddings(text: str) -> np.ndarray:
    response = post_json("/embeddings", {"content": text})
    matrix = np.asarray(response[0]["embedding"], dtype=np.float32)
    if matrix.ndim != 2:
        raise RuntimeError(f"隐藏状态维度异常: {matrix.shape}")
    return matrix


def open_weights(path: Path) -> tuple[np.memmap, np.memmap, int, int, int]:
    with path.open("rb") as stream:
        raw = stream.read(HEADER.size)
    if len(raw) != HEADER.size:
        raise RuntimeError("ColorPlasticity 文件头不完整")
    magic, version, n_embd, n_vocab, rank, _reserved, a_count, b_count = HEADER.unpack(raw)
    if magic != MAGIC or version != VERSION:
        raise RuntimeError("ColorPlasticity 文件版本不匹配")
    if a_count != n_embd * rank or b_count != n_vocab * rank:
        raise RuntimeError("ColorPlasticity 张量尺寸不匹配")

    a = np.memmap(
        path,
        mode="r",
        dtype="<f4",
        offset=HEADER.size,
        shape=(rank, n_embd),
    )
    b = np.memmap(
        path,
        mode="r+",
        dtype="<f2",
        offset=HEADER.size + a_count * np.dtype("<f4").itemsize,
        shape=(n_vocab, rank),
    )
    return a, b, n_embd, n_vocab, rank


def training_state_path(weights_path: Path) -> Path:
    return weights_path.with_name(weights_path.stem + "-state.bin")


def load_training_state(weights_path: Path, rank: int) -> tuple[np.ndarray, int]:
    path = training_state_path(weights_path)
    if not path.exists():
        return np.eye(rank, dtype=np.float32) * 100.0, 0
    raw = path.read_bytes()
    expected = STATE_HEADER.size + rank * rank * np.dtype("<f4").itemsize
    if len(raw) != expected:
        raise RuntimeError("ColorPlasticity 巩固状态尺寸不匹配")
    magic, version, stored_rank, contexts = STATE_HEADER.unpack_from(raw)
    if magic != STATE_MAGIC or version != STATE_VERSION or stored_rank != rank:
        raise RuntimeError("ColorPlasticity 巩固状态版本不匹配")
    inverse_correlation = np.frombuffer(
        raw,
        dtype="<f4",
        count=rank * rank,
        offset=STATE_HEADER.size,
    ).copy().reshape(rank, rank)
    if not np.all(np.isfinite(inverse_correlation)):
        raise RuntimeError("ColorPlasticity 巩固状态包含非有限值")
    return inverse_correlation, int(contexts)


def save_training_state(
    weights_path: Path,
    inverse_correlation: np.ndarray,
    contexts: int,
) -> None:
    path = training_state_path(weights_path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    matrix = np.asarray(inverse_correlation, dtype="<f4", order="C")
    payload = STATE_HEADER.pack(
        STATE_MAGIC,
        STATE_VERSION,
        matrix.shape[0],
        contexts,
    ) + matrix.tobytes(order="C")
    temporary.write_bytes(payload)
    temporary.replace(path)


def active_token_rows(b: np.memmap, block_size: int = 4096) -> np.ndarray:
    active: list[np.ndarray] = []
    for start in range(0, b.shape[0], block_size):
        stop = min(start + block_size, b.shape[0])
        mask = np.any(np.asarray(b[start:stop]) != 0, axis=1)
        if np.any(mask):
            active.append(np.flatnonzero(mask).astype(np.int64) + start)
    if not active:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(active)


def reload_runtime() -> dict[str, Any]:
    return post_json("/slots/0?action=plastic_reload", {})


def erase_slot() -> None:
    post_json("/slots/0?action=erase", {})


def reset_weights(path: Path) -> dict[str, Any]:
    _a, b, _n_embd, _n_vocab, _rank = open_weights(path)
    b[:] = np.float16(0)
    b.flush()
    training_state_path(path).unlink(missing_ok=True)
    result = reload_runtime()
    erase_slot()
    return result


def learn(
    path: Path,
    question: str,
    answer: str,
    margin: float,
    learning_rate: float,
    epochs: int,
    system: str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    prompt = render_question(question, system=system)
    # Include the assistant turn terminator.  Without this target the final
    # answer token remains the strongest learned attractor and generation can
    # loop instead of ending the turn.
    full = prompt + answer + "<|im_end|>"
    prompt_tokens = tokenize(prompt)
    full_tokens = tokenize(full)
    if full_tokens[: len(prompt_tokens)] != prompt_tokens:
        raise RuntimeError("聊天模板前缀与训练序列没有严格对齐")
    if len(full_tokens) <= len(prompt_tokens):
        raise RuntimeError("答案没有产生可训练 token")

    hidden = causal_embeddings(full)
    if hidden.shape[0] != len(full_tokens):
        raise RuntimeError(
            f"隐藏状态与 token 未对齐: {hidden.shape[0]} != {len(full_tokens)}"
        )

    a, b, n_embd, n_vocab, rank = open_weights(path)
    if hidden.shape[1] != n_embd:
        raise RuntimeError(f"隐藏宽度不匹配: {hidden.shape[1]} != {n_embd}")

    first_target = len(prompt_tokens)
    examples: list[tuple[np.ndarray, int]] = []
    for position in range(first_target, len(full_tokens)):
        target = full_tokens[position]
        if not 0 <= target < n_vocab:
            raise RuntimeError(f"目标 token 越界: {target}")
        z = np.asarray(a @ hidden[position - 1], dtype=np.float32)
        z /= max(float(np.linalg.norm(z)), 1e-6)
        examples.append((z, target))

    # Solve the small sequence as one competitive prediction problem.  A
    # positive-only Delta rule makes all answer tokens strong on neighbouring
    # hidden states and can create a token attractor.  Here each context raises
    # its correct token while suppressing the other tokens from the same
    # learned sequence.  The result is still a continuous B matrix used by the
    # neural logits path; no text or lookup table is used during inference.
    keys = np.stack([z for z, _target in examples]).astype(np.float32)
    target_ids = np.asarray([target for _z, target in examples], dtype=np.int64)
    target_rows = np.unique(target_ids)
    touched_ids = np.union1d(active_token_rows(b), target_rows)
    token_to_row = {int(token): row for row, token in enumerate(touched_ids)}
    desired = np.full(
        (len(touched_ids), len(examples)),
        -0.5 * margin,
        dtype=np.float32,
    )
    for position, target in enumerate(target_ids):
        desired[token_to_row[int(target)], position] = margin

    # Recursive least-squares gain uses only a rank x rank inverse correlation
    # statistic from prior neural states.  It contains no source text or token
    # targets.  Directions already carrying prior knowledge receive smaller
    # updates, while novel directions remain highly plastic.
    inverse_correlation, consolidated_contexts = load_training_state(path, rank)
    projected = keys @ inverse_correlation
    innovation = projected @ keys.T
    innovation += np.eye(len(examples), dtype=np.float32) * 0.1
    gain = np.linalg.solve(innovation, projected).T
    rows = np.asarray(b[touched_ids], dtype=np.float32)
    losses: list[float] = []
    for _epoch in range(epochs):
        residual = desired - rows @ keys.T
        rows += learning_rate * (residual @ gain.T)
        np.clip(rows, -32.0, 32.0, out=rows)
        losses.append(float(np.mean(np.abs(desired - rows @ keys.T))))
    b[touched_ids] = rows.astype(np.float16)
    b.flush()

    inverse_correlation -= gain @ projected
    inverse_correlation = 0.5 * (inverse_correlation + inverse_correlation.T)
    minimum_diagonal = float(np.min(np.diag(inverse_correlation)))
    if minimum_diagonal < 1e-6:
        inverse_correlation += np.eye(rank, dtype=np.float32) * (1e-6 - minimum_diagonal)
    consolidated_contexts += len(examples)
    save_training_state(path, inverse_correlation, consolidated_contexts)

    runtime = reload_runtime()
    erase_slot()
    return {
        "question": question,
        "answer": answer,
        "rank": rank,
        "trained_tokens": len(examples),
        "target_token_rows": len(target_rows),
        "updated_token_rows": len(touched_ids),
        "consolidated_contexts": consolidated_contexts,
        "mean_absolute_error": losses,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "runtime": runtime,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="ColorLM v4 预测误差快速学习")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--question")
    parser.add_argument("--answer")
    parser.add_argument("--margin", type=float, default=18.0)
    parser.add_argument("--learning-rate", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    if args.reset:
        result: Any = reset_weights(args.weights)
    else:
        if not args.question or not args.answer:
            parser.error("学习需要同时提供 --question 和 --answer")
        if not 0.0 < args.learning_rate <= 1.0:
            parser.error("--learning-rate 必须位于 (0, 1]")
        if not 1 <= args.epochs <= 32:
            parser.error("--epochs 必须位于 [1, 32]")
        result = learn(
            args.weights,
            args.question,
            args.answer,
            args.margin,
            args.learning_rate,
            args.epochs,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
