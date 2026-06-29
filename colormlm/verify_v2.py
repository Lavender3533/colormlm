# -*- coding: utf-8 -*-
import json
path = r"D:\project\大模型ssd化\colormlm\colab_colorlm_v2.ipynb"
with open(path, encoding="utf-8") as f:
    nb = json.load(f)
print("Cells: %d" % len(nb["cells"]))
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"]).strip()
    first = src.split("\n")[0][:80]
    print("  Cell %d: %s" % (i, first))
# Verify FSQ is in Cell 2
src2 = "".join(nb["cells"][1]["source"])
assert "class FSQ" in src2, "FSQ not found in Cell 2!"
assert "class VectorQuantize" not in src2, "Old VQ still present!"
print("\nFSQ verified, VQ removed. Ready!")
