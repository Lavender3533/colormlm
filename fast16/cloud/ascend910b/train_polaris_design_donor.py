"""在 Ascend 910B4 上训练北极星 Design Genome LoRA donor。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any


SYSTEM = (
    "你是北极星前端结构规划器。先理解业务、信息层级、交互、响应式与可访问性约束，"
    "再输出紧凑 Design Genome。不要输出解释、Markdown或HTML。"
)
GENOME_KEYS = {"v", "q", "y", "l", "c", "x", "r", "a", "z"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def validate_dataset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    split_counts = {
        split: sum(str(row.get("split")) == split for row in rows)
        for split in ("train", "validation", "blind")
    }
    if split_counts != {"train": 128, "validation": 16, "blind": 0}:
        raise ValueError(f"必须是128 train + 16 internal validation: {split_counts}")
    if len({str(row.get("task_id")) for row in rows}) != len(rows):
        raise ValueError("task_id 重复")
    required_contract = {
        "forbid_dead_controls",
        "forbid_default_three_cards",
        "forbid_emoji_icons",
        "forbid_remote_assets",
        "require_form_labels",
        "require_reduced_motion",
        "require_semantic_html",
        "require_visible_focus",
        "viewports",
    }
    family_by_split: dict[str, set[str]] = {"train": set(), "validation": set()}
    for row in rows:
        prompt = row.get("prompt")
        target = row.get("target_genome")
        contract = row.get("anti_pattern_contract")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("存在空 prompt")
        if not isinstance(target, str):
            raise ValueError("存在无效 target_genome")
        genome = json.loads(target)
        if set(genome) != GENOME_KEYS:
            raise ValueError("target_genome 字段不符合 Design Genome v1")
        if not isinstance(contract, dict) or set(contract) != required_contract:
            raise ValueError("anti_pattern_contract 未完整投影九类约束")
        split = str(row["split"])
        family_by_split[split].add(str(row["source_family"]))
    return {
        "row_count": len(rows),
        "split_counts": split_counts,
        "source_families": sorted(family_by_split["train"] | family_by_split["validation"]),
        "internal_validation_only": True,
        "official_validation_or_blind_read": False,
    }


def prompt_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
                str(row["prompt"])
                + "\n只输出一行合法紧凑JSON，字段必须严格遵守Design Genome v1。"
            ),
        },
    ]


def apply_template(tokenizer: Any, messages: list[dict[str, str]], **kwargs: Any) -> Any:
    options = dict(kwargs)
    options.setdefault("enable_thinking", False)
    try:
        return tokenizer.apply_chat_template(messages, **options)
    except TypeError:
        options.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **options)


def encode_row(tokenizer: Any, row: dict[str, Any], maximum: int) -> dict[str, list[int]]:
    messages = prompt_messages(row)
    prompt_ids = apply_template(
        tokenizer,
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    full_ids = apply_template(
        tokenizer,
        [*messages, {"role": "assistant", "content": str(row["target_genome"])}],
        tokenize=True,
        add_generation_prompt=False,
    )
    prompt_ids = [int(value) for value in prompt_ids]
    full_ids = [int(value) for value in full_ids]
    if full_ids[: len(prompt_ids)] != prompt_ids:
        common = 0
        for left, right in zip(prompt_ids, full_ids):
            if left != right:
                break
            common += 1
        if common < max(1, len(prompt_ids) - 4):
            raise ValueError(f"chat template前缀无法对齐: task={row['task_id']}")
        prompt_ids = full_ids[:common]
    if len(full_ids) > maximum:
        raise ValueError(
            f"样本超过max-length，拒绝静默截断: task={row['task_id']} tokens={len(full_ids)}"
        )
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :],
    }


def collate(batch: list[dict[str, list[int]]], pad_id: int, torch: Any) -> dict[str, Any]:
    width = max(len(item["input_ids"]) for item in batch)
    input_ids = []
    attention = []
    labels = []
    for item in batch:
        padding = width - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_id] * padding)
        attention.append(item["attention_mask"] + [0] * padding)
        labels.append(item["labels"] + [-100] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def extract_genome(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) and set(value) == GENOME_KEYS else None


def flatten_genome(genome: dict[str, Any]) -> list[str]:
    return [
        str(genome["v"]),
        *map(str, genome["q"]),
        *map(str, genome["y"]),
        *map(str, genome["l"]),
        *(".".join(map(str, item)) for item in genome["c"]),
        *map(str, genome["x"]),
        *map(str, genome["r"]),
        str(genome["a"]),
        str(genome["z"]),
    ]


def evaluate(model: Any, tokenizer: Any, rows: list[dict[str, Any]], device: Any, torch: Any) -> dict[str, Any]:
    model.eval()
    exact = 0
    valid_json = 0
    correct_fields = 0
    total_fields = 0
    examples = []
    with torch.no_grad():
        for row in rows:
            messages = prompt_messages(row)
            encoded = apply_template(
                tokenizer,
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(device)
            generated = model.generate(
                input_ids=encoded,
                max_new_tokens=256,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            text = tokenizer.decode(generated[0, encoded.shape[-1] :], skip_special_tokens=True).strip()
            predicted = extract_genome(text)
            reference = json.loads(str(row["target_genome"]))
            is_exact = predicted == reference
            exact += int(is_exact)
            valid_json += int(predicted is not None)
            reference_fields = flatten_genome(reference)
            predicted_fields = flatten_genome(predicted) if predicted is not None else []
            correct_fields += sum(
                left == right for left, right in zip(reference_fields, predicted_fields)
            )
            total_fields += len(reference_fields)
            if len(examples) < 8:
                examples.append(
                    {
                        "task_id": row["task_id"],
                        "valid_genome": predicted is not None,
                        "exact": is_exact,
                        "prediction": None if predicted is None else canonical_json(predicted),
                    }
                )
    return {
        "sample_count": len(rows),
        "valid_genome_rate": valid_json / max(len(rows), 1),
        "exact_genome_rate": exact / max(len(rows), 1),
        "field_accuracy": correct_fields / max(total_fields, 1),
        "examples": examples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-8B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--wall-seconds", type=float, default=4800.0)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.dataset)
    dataset_contract = validate_dataset(rows)
    if args.selftest:
        print(
            json.dumps(
                {
                    "format": "polaris-design-donor-selftest-v1",
                    "ok": True,
                    "dataset": dataset_contract,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.output_dir.exists() or args.report.exists():
        raise FileExistsError("拒绝覆盖已有adapter目录或报告")
    if min(args.epochs, args.max_length, args.batch_size, args.gradient_accumulation) <= 0:
        raise ValueError("训练预算必须为正数")
    if args.wall_seconds < 60:
        raise ValueError("wall-seconds不得低于60秒")

    import torch
    import torch_npu  # noqa: F401
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch_npu.npu.is_available():
        raise RuntimeError("Ascend NPU不可用；拒绝CPU冒充云训练")
    device = torch.device("npu:0")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch_npu.npu.manual_seed_all(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        revision=args.revision,
        trust_remote_code=False,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.revision,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.to(device)
    lora = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)

    train_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [row for row in rows if row["split"] == "validation"]
    encoded_train = [encode_row(tokenizer, row, args.max_length) for row in train_rows]
    loader = DataLoader(
        encoded_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id, torch),
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    started = time.perf_counter()
    deadline = started + args.wall_seconds
    loss_history: list[float] = []
    optimizer_steps = 0
    completed_epochs = 0
    stopped_for_wall = False
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        for micro_step, batch in enumerate(loader, 1):
            if time.perf_counter() >= deadline:
                stopped_for_wall = True
                break
            batch = {key: value.to(device, non_blocking=False) for key, value in batch.items()}
            result = model(**batch)
            loss = result.loss / args.gradient_accumulation
            loss.backward()
            loss_history.append(float(result.loss.detach().float().cpu()))
            if micro_step % args.gradient_accumulation == 0 or micro_step == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
        if stopped_for_wall:
            break
        completed_epochs = epoch + 1

    model.config.use_cache = True
    validation = evaluate(model, tokenizer, validation_rows, device, torch)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    torch_npu.npu.synchronize()
    elapsed = time.perf_counter() - started
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    resolved_revision = getattr(model.config, "_commit_hash", None)
    report = {
        "format": "polaris-v47-design-genome-lora-donor-report-v1",
        "status": "internal_validation_only",
        "device": {
            "selected": "npu:0",
            "name": torch_npu.npu.get_device_name(0),
            "peak_memory_bytes": int(torch_npu.npu.max_memory_allocated(0)),
        },
        "base_model": args.model_id,
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256_file(args.dataset),
        "dataset_contract": dataset_contract,
        "adapter": str(args.output_dir.resolve()),
        "trainable_parameters": int(trainable),
        "total_parameters_with_adapter": int(total),
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "completed_epochs": completed_epochs,
        "optimizer_steps": optimizer_steps,
        "wall_seconds": elapsed,
        "stopped_for_wall": stopped_for_wall,
        "loss_first": loss_history[0] if loss_history else None,
        "loss_last": loss_history[-1] if loss_history else None,
        "validation": validation,
        "prototype_gate": {
            "valid_genome_rate_min": 0.90,
            "exact_genome_rate_min": 0.50,
            "field_accuracy_min": 0.85,
            "passed": bool(
                validation["valid_genome_rate"] >= 0.90
                and validation["exact_genome_rate"] >= 0.50
                and validation["field_accuracy"] >= 0.85
            ),
        },
        "claim_limit": (
            "仅为8个train家族的内部留出；必须经冻结跨家族validation、编译器网页A/B与blind，"
            "不得宣称通用前端能力或GPT/Claude级能力。"
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
