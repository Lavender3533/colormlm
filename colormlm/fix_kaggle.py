# -*- coding: utf-8 -*-
path = r"D:\project\大模型ssd化\colormlm\kaggle_colorlm.py"
with open(path, encoding="utf-8") as f:
    lines = f.readlines()

# Fix the broken multi-line docstring at line 86-89
# Lines 85-89 should be:
# class ColorLMStudent(nn.Module):
#     # Student Model: 70M params (vs Qwen 1543M = 22x compression)
#     # Input:  token IDs [B, S]
#     # Output: RGB codes [B, S, 4], temperature [B, S, 1], distill projection [B, S, 1536]
result = []
i = 0
while i < len(lines):
    line = lines[i]
    stripped = line.strip()
    # Detect bare docstring content (was inside multi-line docstring)
    if stripped.startswith("Student Model:") and "70M params" in stripped:
        indent = "    "
        result.append(f"{indent}# Student Model: 70M params (vs Qwen 1543M = 22x compression)\n")
        i += 1
        # Skip blank line
        if i < len(lines) and lines[i].strip() == "":
            result.append(lines[i])
            i += 1
        # Convert Input/Output lines to comments
        while i < len(lines) and (lines[i].strip().startswith("Input:") or lines[i].strip().startswith("Output:")):
            s = lines[i].strip()
            ind = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
            result.append(f"{ind}# {s}\n")
            i += 1
    else:
        result.append(line)
        i += 1

with open(path, "w", encoding="utf-8") as f:
    f.writelines(result)
print("Fixed kaggle_colorlm.py")
