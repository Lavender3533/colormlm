# -*- coding: utf-8 -*-
import torch
path = r"D:\project\大模型ssd化\colormlm\data\student_best.pt"
sd = torch.load(path, map_location="cpu", weights_only=True)
# Find embedding size
for k, v in sd.items():
    if "token_embed" in k or "lm_head" in k:
        print("%s: %s" % (k, v.shape))
