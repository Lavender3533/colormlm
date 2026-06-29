# -*- coding: utf-8 -*-
import re

path = r"D:\project\大模型ssd化\colormlm\gen_kaggle.py"
with open(path, encoding="utf-8") as f:
    lines = f.readlines()

result = []
for line in lines:
    stripped = line.strip()
    # Replace single-line docstrings with comments
    if stripped.startswith('"""') and stripped.endswith('"""') and len(stripped) > 6:
        indent = line[:len(line) - len(line.lstrip())]
        doc = stripped[3:-3]
        result.append(f"{indent}# {doc}\n")
    # Skip standalone triple-quote lines (docstring boundaries)
    elif stripped == '"""':
        continue
    else:
        result.append(line)

with open(path, "w", encoding="utf-8") as f:
    f.writelines(result)
print("Fixed gen_kaggle.py successfully!")
