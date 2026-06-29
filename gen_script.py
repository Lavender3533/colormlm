# -*- coding: utf-8 -*-
# Helper: generates transform_residual.py with hardcoded paths
import os

BASE = r"D:\project\大模型ssd化"
outpath = os.path.join(BASE, "transform_residual.py")

code = '''
# -*- coding: utf-8 -*-
import os, gc, sys, time
from safetensors import safe_open
from safetensors.torch import save_file
import torch

MODEL_DIR = os.environ["FSQ_MODEL_DIR"]
OUTPUT_DIR = os.environ["FSQ_OUTPUT_DIR"]
os.makedirs(OUTPUT_DIR, exist_ok=True)
N_LEVELS = 16

def fsq_encode(w, n=N_LEVELS):
    mn = w.min().item()
    mx = w.max().item()
    if mx - mn < 1e-10:
        return torch.zeros_like(w, dtype=torch.uint8), mn, mx
    normed = (w - mn) / (mx - mn)
    codes = torch.clamp((normed * n).long(), 0, n - 1)
    return codes.to(torch.uint8), mn, mx

def fsq_decode(codes, mn, mx, n=N_LEVELS):
    return codes.float() / float(n - 1) * (mx - mn) + mn

def is_expert(key):
    if "mlp.experts." not in key:
        return False
    return any(x in key for x in ["gate_proj.weight", "up_proj.weight", "down_proj.weight"])

def process_shard(shard):
    inp = os.path.join(MODEL_DIR, shard)
    out = os.path.join(OUTPUT_DIR, shard)
    if not os.path.exists(inp):
        print("SKIP:", shard)
        return
    t0 = time.time()
    with safe_open(inp, framework="pt", device="cpu") as f:
        all_keys = list(f.keys())
    ek = [k for k in all_keys if is_expert(k)]
    ok_keys = [k for k in all_keys if not is_expert(k)]
    print("  Keys:", len(all_keys), "total,", len(ek), "expert,", len(ok_keys), "other")
    new = {}
    for key in ok_keys:
        with safe_open(inp, framework="pt", device="cpu") as f:
            new[key] = f.get_tensor(key)
    done = 0
    for key in ek:
        with safe_open(inp, framework="pt", device="cpu") as f:
            w = f.get_tensor(key)
        wf = w.float()
        c1, mn1, mx1 = fsq_encode(wf)
        r1 = fsq_decode(c1, mn1, mx1)
        residual = wf - r1
        c2, mn2, mx2 = fsq_encode(residual)
        new[key + "._c1"] = c1
        new[key + "._c2"] = c2
        new[key + "._meta"] = torch.tensor([mn1, mx1, mn2, mx2], dtype=torch.float32)
        del w, wf, c1, r1, residual, c2
        done += 1
        if done % 100 == 0:
            print("    ", done, "/", len(ek))
            gc.collect()
    print("  Saving...")
    save_file(new, out)
    sz = os.path.getsize(out) / 1e9
    elapsed = time.time() - t0
    print("  Done: " + str(round(sz,2)) + "GB in " + str(round(elapsed)) + "s")
    del new
    gc.collect()

def main():
    print("Residual FSQ Transform")
    print("Source:", MODEL_DIR)
    print("Output:", OUTPUT_DIR)
    shards = sorted([f for f in os.listdir(MODEL_DIR) if f.endswith(".safetensors")])
    print("Found", len(shards), "shards")
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    for i, s in enumerate(shards):
        if i < start:
            continue
        print("")
        print("[" + str(i+1) + "/" + str(len(shards)) + "]", s)
        process_shard(s)
    total = sum(os.path.getsize(os.path.join(OUTPUT_DIR, f)) for f in os.listdir(OUTPUT_DIR) if f.endswith(".safetensors"))
    orig_total = sum(os.path.getsize(os.path.join(MODEL_DIR, f)) for f in os.listdir(MODEL_DIR) if f.endswith(".safetensors"))
    print("")
    print("Original:", round(orig_total/1e9, 2), "GB")
    print("Residual FSQ:", round(total/1e9, 2), "GB")
    print("Compression:", round(orig_total/total, 2), "x")

if __name__ == "__main__":
    main()
'''

with open(outpath, 'w', encoding='utf-8') as f:
    f.write(code)
print("Written to", outpath)
