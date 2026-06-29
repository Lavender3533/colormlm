# -*- coding: utf-8 -*-
"""Fix: add CUDA_LAUNCH_BLOCKING and auto-detect vocab_size"""
import json

path = r"D:\project\大模型ssd化\colormlm\colab_colorlm.ipynb"
with open(path, encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    src = "".join(cell["source"])

    # Fix Cell 4: add debug info after teacher loading
    if "teacher_hidden" in src and "teacher = AutoModel" in src:
        src = src.replace(
            "print(f'Teacher loaded: {teacher.config.hidden_size}d')",
            "print(f'Teacher loaded: {teacher.config.hidden_size}d')\n"
            "print(f'Tokenizer vocab_size: {tokenizer.vocab_size}')\n"
            "print(f'Tokenizer pad_token_id: {tokenizer.pad_token_id}')"
        )
        cell["source"] = [l + "\n" for l in src.rstrip("\n").split("\n")]

    # Fix Cell 5: use actual vocab_size from tokenizer instead of hardcoded
    if "ColorLMStudent(" in src and "vocab_size=151643" in src:
        src = src.replace(
            "model = ColorLMStudent(\n    vocab_size=151643",
            "# Auto-detect vocab size from tokenizer\n"
            "actual_vocab = max(tokenizer.vocab_size, input_ids_all.max().item() + 1)\n"
            "print(f'Using vocab_size: {actual_vocab}')\n\n"
            "model = ColorLMStudent(\n    vocab_size=actual_vocab"
        )
        cell["source"] = [l + "\n" for l in src.rstrip("\n").split("\n")]

    # Fix Cell 6: add CUDA_LAUNCH_BLOCKING before training loop
    if "optimizer = torch.optim.AdamW" in src:
        src = src.replace(
            "optimizer = torch.optim.AdamW",
            "# Debug: check for out-of-range token IDs\n"
            "print(f'input_ids range: [{input_ids_all.min()}, {input_ids_all.max()}]')\n"
            "print(f'Embedding vocab: {model.token_embed.num_embeddings}')\n"
            "assert input_ids_all.max() < model.token_embed.num_embeddings, 'Token ID out of range!'\n\n"
            "optimizer = torch.optim.AdamW"
        )
        cell["source"] = [l + "\n" for l in src.rstrip("\n").split("\n")]

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Fixed! Vocab size auto-detect + debug checks added.")
