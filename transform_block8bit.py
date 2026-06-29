# -*- coding: utf-8 -*-
import os, gc, sys, time, json
from safetensors import safe_open
from safetensors.torch import save_file
import torch

MODEL_DIR = os.environ["FSQ_MODEL_DIR"]
OUTPUT_DIR = os.environ["FSQ_OUTPUT_DIR"]
os.makedirs(OUTPUT_DIR, exist_ok=True)

BLOCK_SIZE = 128
N_LEVELS = 256  # 8-bit

def is_expert(key):
    if "mlp.experts." not in key:
        return False
    return any(x in key for x in ["gate_proj.weight", "up_proj.weight", "down_proj.weight"])

def block_quantize(w, block_size=BLOCK_SIZE, n_levels=N_LEVELS):
    """Block-wise quantization: group weights into blocks, normalize per block"""
    orig_shape = w.shape
    w_flat = w.flatten()
    
    # Pad to multiple of block_size
    pad = (block_size - len(w_flat) % block_size) % block_size
    if pad > 0:
        w_flat = torch.cat([w_flat, torch.zeros(pad)])
    
    blocks = w_flat.reshape(-1, block_size)
    
    # Per-block min/max
    b_min = blocks.min(dim=1, keepdim=True).values
    b_max = blocks.max(dim=1, keepdim=True).values
    
    # Normalize to [0, 1]
    normed = (blocks - b_min) / (b_max - b_min + 1e-10)
    
    # Quantize to uint8
    codes = torch.clamp((normed * (n_levels - 1)).round().long(), 0, n_levels - 1).to(torch.uint8)
    
    # Store metadata: min and max per block
    meta = torch.cat([b_min.squeeze(1), b_max.squeeze(1)], dim=0).to(torch.float32)
    
    return codes, meta, orig_shape

def block_dequantize(codes, meta, orig_shape, block_size=BLOCK_SIZE, n_levels=N_LEVELS):
    """Dequantize block-wise quantized weights"""
    n_blocks = codes.shape[0]
    b_min = meta[:n_blocks].unsqueeze(1)
    b_max = meta[n_blocks:].unsqueeze(1)
    
    # Dequantize
    recon = codes.float() / float(n_levels - 1) * (b_max - b_min) + b_min
    
    # Flatten and trim to original size
    w_flat = recon.flatten()[:orig_shape.numel()]
    return w_flat.reshape(orig_shape)

def process_shard(shard):
    inp = os.path.join(MODEL_DIR, shard)
    out = os.path.join(OUTPUT_DIR, shard)
    
    if not os.path.exists(inp):
        print("SKIP:", shard)
        return
    
    t0 = time.time()
    
    # Load all tensors
    tensors = {}
    with safe_open(inp, framework="pt", device="cpu") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    
    new = {}
    n_exp = 0
    
    for k, v in tensors.items():
        if is_expert(k):
            wf = v.float()
            codes, meta, shape = block_quantize(wf)
            
            # Store quantized data
            new[k + "._bq_codes"] = codes
            new[k + "._bq_meta"] = meta
            new[k + "._bq_shape"] = torch.tensor(list(shape), dtype=torch.int32)
            
            del wf, codes, meta
            n_exp += 1
        else:
            new[k] = v
        
        gc.collect()
    
    print("  Saving " + str(len(new)) + " tensors...")
    save_file(new, out)
    
    out_sz = os.path.getsize(out) / 1e9
    inp_sz = os.path.getsize(inp) / 1e9
    elapsed = time.time() - t0
    
    print("  " + str(n_exp) + " expert weights quantized")
    print("  " + str(round(inp_sz, 2)) + "GB -> " + str(round(out_sz, 2)) + "GB")
    print("  Compression: " + str(round(inp_sz / out_sz, 2)) + "x")
    print("  Time: " + str(round(elapsed)) + "s")
    
    del new, tensors
    gc.collect()

def main():
    print("Block-wise 8-bit Quantization")
    print("Source:", MODEL_DIR)
    print("Output:", OUTPUT_DIR)
    print("Block size:", BLOCK_SIZE)
    print("Levels:", N_LEVELS, "(8-bit)")
    print()
    
    shards = sorted([f for f in os.listdir(MODEL_DIR) if f.endswith(".safetensors")])
    print("Found", len(shards), "shards")
    
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    
    for i, s in enumerate(shards):
        if i < start:
            continue
        print("")
        print("[" + str(i+1) + "/" + str(len(shards)) + "]", s)
        process_shard(s)
    
    # Summary
    total_out = sum(os.path.getsize(os.path.join(OUTPUT_DIR, f)) for f in os.listdir(OUTPUT_DIR) if f.endswith(".safetensors"))
    total_in = sum(os.path.getsize(os.path.join(MODEL_DIR, f)) for f in os.listdir(MODEL_DIR) if f.endswith(".safetensors"))
    
    print("")
    print("=" * 50)
    print("Original:", round(total_in / 1e9, 2), "GB")
    print("Quantized:", round(total_out / 1e9, 2), "GB")
    print("Compression:", round(total_in / total_out, 2), "x")
    print("=" * 50)

if __name__ == "__main__":
    main()
