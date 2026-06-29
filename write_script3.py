# -*- coding: utf-8 -*-
import os

lines = [
    '# -*- coding: utf-8 -*-',
    'import os, gc, sys, time',
    'from safetensors import safe_open',
    'from safetensors.torch import save_file',
    'import torch',
    '',
    'MODEL_DIR = r"D:\project\\b'\xe5\xa4\xa7\xe6\xa8\xa1\xe5\x9e\x8bssd\xe5\x8c\x96'\\models\\qwen3-coder"',
    'OUTPUT_DIR = r"D:\project\\b'\xe5\xa4\xa7\xe6\xa8\xa1\xe5\x9e\x8bssd\xe5\x8c\x96'\\models\\qwen3-coder-fsq-residual"',
]

print('test')
