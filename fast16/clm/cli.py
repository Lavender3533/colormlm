"""Command line interface for CLM v0."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .format import ClmReader, pack_checkpoint
from .doccompile import compile_documents
from .gpu import DirectMLCore, export_directml_graph
from .model import ZeroTrainModel


def _pack(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    manifest = pack_checkpoint(
        args.checkpoint,
        args.output,
        memory_paths=args.memory,
        storage_dtype=args.storage_dtype,
        tokenizer_mode=args.tokenizer,
        max_seq_len=args.max_seq_len,
    )
    elapsed = time.perf_counter() - started
    print(json.dumps({
        "output": str(Path(args.output).resolve()),
        "tensor_count": len(manifest["tensors"]),
        "payload_bytes": manifest["payload_bytes"],
        "elapsed_seconds": round(elapsed, 3),
    }, ensure_ascii=False, indent=2))
    return 0


def _inspect(args: argparse.Namespace) -> int:
    with ClmReader(args.model) as reader:
        print(json.dumps(reader.summary(), ensure_ascii=False, indent=2))
    return 0


def _verify(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    with ClmReader(args.model) as reader:
        bad = reader.verify_tensors()
        summary = reader.summary()
    if bad:
        print(json.dumps({"ok": False, "bad_tensors": bad}, ensure_ascii=False, indent=2))
        return 1
    model = ZeroTrainModel.from_clm(args.model)
    result = model.generate(args.prompt, new_tokens=args.new_tokens, refinement_steps=8)
    print(json.dumps({
        "ok": True,
        "model": summary["model_name"],
        "tensor_count": summary["tensor_count"],
        "sample": result.text,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }, ensure_ascii=False, indent=2))
    return 0


def _generate(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    model = ZeroTrainModel.from_clm(args.model)
    core_backend = None
    backend_name = "CPU"
    if args.device == "gpu":
        graph_path = Path(args.gpu_graph) if args.gpu_graph else Path(args.model).with_suffix(".onnx")
        core_backend = DirectMLCore(graph_path)
        backend_name = "DirectML"
    load_seconds = time.perf_counter() - started
    generated_at = time.perf_counter()
    result = model.generate(
        args.prompt,
        new_tokens=args.new_tokens,
        refinement_steps=args.steps,
        core_backend=core_backend,
    )
    generate_seconds = time.perf_counter() - generated_at
    print(result.text)
    print(json.dumps({
        "load_seconds": round(load_seconds, 4),
        "generate_seconds": round(generate_seconds, 4),
        "refinement_steps": result.steps,
        "memory_records": result.memory_records,
        "backend": backend_name,
    }, ensure_ascii=False))
    return 0


def _gpu_export(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    output = export_directml_graph(args.model, args.output)
    print(json.dumps({
        "ok": True,
        "output": str(output.resolve()),
        "file_bytes": output.stat().st_size,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }, ensure_ascii=False, indent=2))
    return 0


def _gpu_info(args: argparse.Namespace) -> int:
    core = DirectMLCore(args.graph)
    print(json.dumps(core.info(), ensure_ascii=False, indent=2))
    return 0


def _chat(args: argparse.Namespace) -> int:
    model = ZeroTrainModel.from_clm(args.model)
    graph_path = Path(args.gpu_graph) if args.gpu_graph else Path(args.model).with_suffix(".onnx")
    core = DirectMLCore(graph_path)
    print("ColorLM ZeroTrain v1 | DirectML GPU")
    while True:
        try:
            prompt = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt in {"/exit", "/quit"}:
            break
        if not prompt:
            continue
        started = time.perf_counter()
        try:
            result = model.generate(
                prompt,
                new_tokens=args.new_tokens,
                refinement_steps=args.steps,
                core_backend=core,
            )
        except ValueError as error:
            print(f"CLM> {error}")
            continue
        print(f"CLM> {result.text}")
        print(f"[{time.perf_counter() - started:.3f}s | DirectML | memory={result.memory_records}]")
    return 0


def _memory_build(args: argparse.Namespace) -> int:
    summary = compile_documents(
        args.input,
        args.output,
        max_value_bytes=args.max_value_bytes,
        max_files=args.max_files,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="clm", description="ColorLM ZeroTrain runtime")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pack = subparsers.add_parser("pack", help="build a CLM from a local checkpoint")
    pack.add_argument("--checkpoint", required=True)
    pack.add_argument("--output", required=True)
    pack.add_argument("--memory", action="append", default=[])
    pack.add_argument("--storage-dtype", choices=("f16", "f32"), default="f16")
    pack.add_argument("--tokenizer", choices=("utf8-byte", "character"), default="utf8-byte")
    pack.add_argument("--max-seq-len", type=int, default=512)
    pack.set_defaults(handler=_pack)

    inspect = subparsers.add_parser("inspect", help="show CLM metadata")
    inspect.add_argument("model")
    inspect.set_defaults(handler=_inspect)

    verify = subparsers.add_parser("verify", help="verify checksums and one forward pass")
    verify.add_argument("model")
    verify.add_argument("--prompt", default="def ")
    verify.add_argument("--new-tokens", type=int)
    verify.set_defaults(handler=_verify)

    generate = subparsers.add_parser("generate", help="generate text with a CLM")
    generate.add_argument("model")
    generate.add_argument("--prompt", default="def ")
    generate.add_argument("--new-tokens", type=int)
    generate.add_argument("--steps", type=int, default=8)
    generate.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    generate.add_argument("--gpu-graph")
    generate.set_defaults(handler=_generate)

    gpu_export = subparsers.add_parser("gpu-export", help="export CLM core to a DirectML ONNX graph")
    gpu_export.add_argument("model")
    gpu_export.add_argument("--output", required=True)
    gpu_export.set_defaults(handler=_gpu_export)

    gpu_info = subparsers.add_parser("gpu-info", help="show active DirectML graph providers")
    gpu_info.add_argument("graph")
    gpu_info.set_defaults(handler=_gpu_info)

    chat = subparsers.add_parser("chat", help="start a persistent DirectML session")
    chat.add_argument("model")
    chat.add_argument("--gpu-graph")
    chat.add_argument("--new-tokens", type=int)
    chat.add_argument("--steps", type=int, default=8)
    chat.set_defaults(handler=_chat)

    memory_build = subparsers.add_parser("memory-build", help="compile local documents to CLM memory JSONL")
    memory_build.add_argument("--input", action="append", required=True)
    memory_build.add_argument("--output", required=True)
    memory_build.add_argument("--max-value-bytes", type=int, default=384)
    memory_build.add_argument("--max-files", type=int)
    memory_build.set_defaults(handler=_memory_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    return int(args.handler(args))
