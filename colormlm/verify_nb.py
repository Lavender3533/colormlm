# -*- coding: utf-8 -*-
import json
path = r"D:\project\大模型ssd化\colormlm\colab_colorlm.ipynb"
with open(path, encoding="utf-8") as f:
    nb = json.load(f)
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"]).strip()
    first = src.split("\n")[0][:80]
    print("Cell %d: %s" % (i, first))
