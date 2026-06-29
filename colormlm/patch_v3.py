# -*- coding: utf-8 -*-
"""Patch colab_colorlm_v2.ipynb -> V3: fix distill_proj to use raw hidden instead of quantized"""
import json

path = r"D:\project\大模型ssd化\colormlm_repo\colab_colorlm_v2.ipynb"
with open(path, encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    src = "".join(cell["source"])

    # Fix forward() in ColorLMStudent: distill uses raw x, not quantized
    if "distill = self.distill_proj(quantized)" in src:
        src = src.replace(
            "distill = self.distill_proj(quantized)",
            "distill = self.distill_proj(x)  # use raw hidden, not quantized (fixes cosine sim)"
        )
        cell["source"] = [l + "\n" for l in src.rstrip("\n").split("\n")]
        print("Fixed: distill_proj now uses raw x instead of quantized")

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("Saved V3 patch")
