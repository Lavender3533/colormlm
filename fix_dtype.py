# -*- coding: utf-8 -*-
path = r"D:\project\大模型ssd化\fsq_generate.py"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    new_lines.append(line)
    # After "return tensors[key]" in get_tensor method, add dtype cast
    if line.strip() == "return tensors[key]" and "def _dequantize" not in "".join(new_lines[-10:]):
        # Insert dtype cast before the return
        indent = line[:len(line) - len(line.lstrip())]
        new_lines.pop()  # remove the return line
        new_lines.append(indent + "t = tensors[key]\n")
        new_lines.append(indent + "if t.dtype == torch.bfloat16:\n")
        new_lines.append(indent + "    t = t.float()\n")
        new_lines.append(indent + "return t\n")

with open(path, "w", encoding="utf-8") as f:
    f.writelines(new_lines)
print("Fixed dtype issue")
