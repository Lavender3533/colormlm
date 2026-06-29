# -*- coding: utf-8 -*-
import json, os

path = r"D:\project\大模型ssd化\colormlm\kaggle_colorlm.py"
with open(path, encoding="utf-8") as f:
    lines = f.readlines()

cells = []
current = []
for line in lines:
    if line.strip().startswith("# %% Cell"):
        if current:
            cells.append(current)
        current = [line]
    else:
        current.append(line)
if current:
    cells.append(current)

for i in range(len(cells)):
    while cells[i] and cells[i][-1].strip() == "":
        cells[i].pop()

# Skip empty cells
cells = [c for c in cells if any(l.strip() for l in c)]

notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "accelerator": "GPU",
        "language_info": {"name": "python"}
    },
    "cells": []
}

for i, cell_lines in enumerate(cells):
    text = "".join(cell_lines)
    if "Qwen/Qwen2.5-1.5B-Instruct" in text:
        text = text.replace(
            "print('Loading Qwen2.5-1.5B-Instruct...')",
            "# Colab can access huggingface.co directly\n# If blocked, uncomment: import os; os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'\nprint('Loading Qwen2.5-1.5B-Instruct...')"
        )
    notebook["cells"].append({
        "cell_type": "code",
        "metadata": {"id": "cell_%d" % i},
        "source": [l + "\n" for l in text.rstrip("\n").split("\n")],
        "outputs": [],
        "execution_count": None
    })

out = r"D:\project\大模型ssd化\colormlm\colab_colorlm.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)
print("OK: %d cells, %d bytes" % (len(notebook["cells"]), os.path.getsize(out)))
