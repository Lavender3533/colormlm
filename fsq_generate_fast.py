# -*- coding: utf-8 -*-
"""FSQ Fast Inference - Optimized Block-8bit with mmap + KV cache"""
import os, torch, json, time, math
from safetensors import safe_open
from transformers import AutoTokenizer
import numpy as np

class MmapWeightStore:
    """Weight storage with lazy shard loading"""
    def __init__(self, model_dir):
        self.model_dir = model_dir
        
        with open(os.path.join(model_dir, "model.safetensors.index.json"), "r") as f:
            self.weight_map = json.load(f).get("weight_map", {})
        
        # Group keys by shard
        self.shard_keys = {}
        for key, shard in self.weight_map.items():
            if shard not in self.shard_keys:
                self.shard_keys[shard] = []
            self.shard_keys[shard].append(key)
        
        # No shard cache - open/close each time
        
        print(f"  WeightStore: {len(self.weight_map)} weights in {len(self.shard_keys)} shards")
    
    def get_tensor(self, key):
        shard = self.weight_map[key]
        path = os.path.join(self.model_dir, shard)
        f = safe_open(path, framework="pt", device="cpu")
        
        # Check if block-quantized
        all_keys = list(f.keys())
        if key + "._bq_codes" in all_keys:
            codes = f.get_tensor(key + "._bq_codes")
            meta = f.get_tensor(key + "._bq_meta")
            shape_t = f.get_tensor(key + "._bq_shape")
            shape = [shape_t[0].item(), shape_t[1].item()]
            
            n = codes.shape[0]
            b_min = meta[:n].unsqueeze(1)
            b_max = meta[n:].unsqueeze(1)
            recon = codes.float() / 255.0 * (b_max - b_min) + b_min
            return recon.flatten()[:shape[0]*shape[1]].reshape(shape)
        
        t = f.get_tensor(key)
        if t.dtype == torch.bfloat16:
            t = t.float()
        return t
    
    def get_layer_weights(self, layer_idx):
        """Load only attention/norm/router weights for a layer (not experts)"""
        prefix = f"model.layers.{layer_idx}."
        weights = {}
        for key in self.weight_map:
            if key.startswith(prefix) and "experts." not in key:
                short = key[len(prefix):]
                if "._bq" not in short:
                    weights[short] = self.get_tensor(key)
        return weights
    
    def get_expert_weight(self, layer_idx, expert_idx, proj):
        """Load a single expert weight on demand"""
        key = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{proj}.weight"
        return self.get_tensor(key)

class FastFSQModel:
    def __init__(self, model_dir, device="cpu"):
        self.model_dir = model_dir
        self.device = device
        
        with open(os.path.join(model_dir, "config.json"), "r") as f:
            self.config = json.load(f)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        
        self.num_layers = self.config["num_hidden_layers"]
        self.hidden_size = self.config["hidden_size"]
        self.num_heads = self.config["num_attention_heads"]
        self.num_kv_heads = self.config.get("num_key_value_heads", self.num_heads)
        self.head_dim = self.config.get("head_dim", self.hidden_size // self.num_heads)
        self.num_experts = self.config.get("num_experts", 128)
        self.top_k = self.config.get("num_experts_per_tok", 8)
        self.rms_norm_eps = self.config.get("rms_norm_eps", 1e-6)
        self.scaling = self.head_dim ** -0.5
        
        # RoPE frequencies
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
        self.inv_freq = inv_freq
        
        # Load weights with mmap
        print("Loading weights (mmap)...")
        self.store = MmapWeightStore(model_dir)
        
        # Load non-layer weights
        self.embed = self.store.get_tensor("model.embed_tokens.weight")
        self.norm = self.store.get_tensor("model.norm.weight")
        self.lm_head = self.store.get_tensor("lm_head.weight")
        
        # Cache for layer weights
        self.layer_cache = {}
        self.max_cached_layers = 3  # Keep 3 layers in memory
        
        # KV cache
        self.kv_cache = [None] * self.num_layers
        
        print("Model ready!")
    
    def get_layer(self, layer_idx):
        if layer_idx not in self.layer_cache:
            # Evict old layers if cache is full
            if len(self.layer_cache) >= self.max_cached_layers:
                oldest = min(self.layer_cache.keys())
                del self.layer_cache[oldest]
            self.layer_cache[layer_idx] = self.store.get_layer_weights(layer_idx)
        return self.layer_cache[layer_idx]
    
    def rms_norm(self, x, weight):
        x_f = x.float()
        var = x_f.pow(2).mean(-1, keepdim=True)
        return (weight * x_f * torch.rsqrt(var + self.rms_norm_eps)).to(x.dtype)
    
    def apply_rope(self, q, k, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float()
        pos_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ pos_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(1)
        sin = emb.sin().unsqueeze(1)
        
        half = q.shape[-1] // 2
        q_out = q * cos + torch.cat([-q[..., half:], q[..., :half]], dim=-1) * sin
        k_out = k * cos + torch.cat([-k[..., half:], k[..., :half]], dim=-1) * sin
        return q_out, k_out
    
    def forward_layer_with_cache(self, layer_idx, hidden_states, position_ids, use_cache=True):
        weights = self.get_layer(layer_idx)
        
        # Attention
        residual = hidden_states
        h = self.rms_norm(hidden_states, weights["input_layernorm.weight"])
        
        B, S, _ = h.shape
        q = h @ weights["self_attn.q_proj.weight"].T
        k = h @ weights["self_attn.k_proj.weight"].T
        v = h @ weights["self_attn.v_proj.weight"].T
        
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # QK Norm
        q = self.rms_norm(q, weights["self_attn.q_norm.weight"])
        k = self.rms_norm(k, weights["self_attn.k_norm.weight"])
        
        # RoPE
        q, k = self.apply_rope(q, k, position_ids)
        
        # KV Cache
        if use_cache and self.kv_cache[layer_idx] is not None:
            prev_k, prev_v = self.kv_cache[layer_idx]
            k = torch.cat([prev_k, k], dim=2)
            v = torch.cat([prev_v, v], dim=2)
        if use_cache:
            self.kv_cache[layer_idx] = (k.detach(), v.detach())
        
        # GQA expand
        if self.num_kv_heads < self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scaling
        
        # Causal mask (only for new tokens)
        total_len = k.shape[2]
        if total_len > S:
            # We have cached KV, mask differently
            causal = torch.zeros(S, total_len, device=hidden_states.device, dtype=torch.bool)
            for i in range(S):
                causal[i, total_len - S + i + 1:] = True
        else:
            causal = torch.triu(torch.ones(S, S, device=hidden_states.device, dtype=torch.bool), diagonal=1)
        
        attn.masked_fill_(causal.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        attn_out = (attn @ v).transpose(1, 2).reshape(B, S, -1)
        
        o_proj = weights["self_attn.o_proj.weight"]
        attn_out = (o_proj @ attn_out.transpose(-1, -2)).transpose(-1, -2)
        
        hidden_states = residual + attn_out
        
        # MoE
        residual = hidden_states
        h = self.rms_norm(hidden_states, weights["post_attention_layernorm.weight"])
        
        gate_w = weights["mlp.gate.weight"]
        router_logits = h @ gate_w.T
        router_logits = torch.softmax(router_logits.float(), dim=-1)
        topk_values, topk_indices = torch.topk(router_logits, self.top_k, dim=-1)
        topk_values /= topk_values.sum(dim=-1, keepdim=True)
        
        moe_out = torch.zeros_like(h)
        for i in range(self.top_k):
            idx = topk_indices[0, 0, i].item()
            w = topk_values[0, 0, i].item()
            
            gate_w = self.store.get_expert_weight(layer_idx, idx, "gate_proj")
            up_w = self.store.get_expert_weight(layer_idx, idx, "up_proj")
            down_w = self.store.get_expert_weight(layer_idx, idx, "down_proj")
            
            gate = h @ gate_w.T
            up = h @ up_w.T
            gate = torch.nn.functional.silu(gate)
            expert_out = (gate * up) @ down_w.T
            
            del gate_w, up_w, down_w
            moe_out = moe_out + expert_out * w
        
        hidden_states = residual + moe_out
        
        import gc
        del weights, h, moe_out
        gc.collect()
        
        return hidden_states
    
    def clear_cache(self):
        self.kv_cache = [None] * self.num_layers
    
    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=50, temperature=0.7, top_p=0.9):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        tokens = input_ids[0].tolist()
        
        self.clear_cache()
        
        print("Generating...")
        t0 = time.time()
        first_token_time = None
        
        # Prefill: process all input tokens at once
        input_t = torch.tensor([tokens], dtype=torch.long)
        hidden_states = self.embed[input_t].float()
        position_ids = torch.arange(hidden_states.shape[1]).unsqueeze(0)
        
        import psutil
        for i in range(self.num_layers):
            try:
                hidden_states = self.forward_layer_with_cache(i, hidden_states, position_ids, use_cache=True)
                if i % 10 == 0:
                    mem = psutil.virtual_memory()
                    print(f"  Layer {i}: mem={mem.percent}%")
            except Exception as e:
                print(f"  Error at layer {i}: {e}")
                import traceback
                traceback.print_exc()
                break
        
        hidden_states = self.rms_norm(hidden_states, self.norm)
        logits = hidden_states @ self.lm_head.T
        
        first_token_time = time.time() - t0
        print(f"  Prefill: {len(tokens)} tokens in {first_token_time:.2f}s")
        
        # Decode: generate tokens one by one
        decode_start = time.time()
        
        for step in range(max_new_tokens):
            next_logits = logits[0, -1, :] / temperature
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumsum = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            mask = cumsum - torch.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[mask] = float("-inf")
            probs = torch.softmax(sorted_logits, dim=-1)
            next_idx = torch.multinomial(probs, 1).item()
            next_token = sorted_indices[next_idx].item()
            
            tokens.append(next_token)
            if next_token == self.tokenizer.eos_token_id:
                break
            
            # Forward the new token
            try:
                input_t = torch.tensor([[next_token]], dtype=torch.long)
                hidden_states = self.embed[input_t].float()
                position_ids = torch.tensor([[len(tokens) - 1]])
                
                for i in range(self.num_layers):
                    hidden_states = self.forward_layer_with_cache(i, hidden_states, position_ids, use_cache=True)
            except Exception as e:
                print(f"  Decode error at step {step}: {e}")
                import traceback
                traceback.print_exc()
                break
            
            hidden_states = self.rms_norm(hidden_states, self.norm)
            logits = hidden_states @ self.lm_head.T
            
            if step % 10 == 0:
                elapsed = time.time() - decode_start
                speed = (step + 1) / elapsed if elapsed > 0 else 0
                print(f"  Step {step}: {speed:.2f} tok/s")
        
        total_time = time.time() - t0
        decode_time = time.time() - decode_start
        n_generated = len(tokens) - len(input_ids[0])
        
        print(f"\nResults:")
        print(f"  Prefill: {first_token_time:.2f}s ({len(tokens) - n_generated} tokens)")
        print(f"  Decode: {decode_time:.2f}s ({n_generated} tokens)")
        print(f"  Speed: {n_generated / decode_time:.2f} tok/s (decode)")
        print(f"  Total: {total_time:.2f}s")
        
        return self.tokenizer.decode(tokens)

def main():
    import sys
    model_dir = sys.argv[1] if len(sys.argv) > 1 else r"D:\\project\\大模型ssd化\\models\\qwen3-coder-bq8"
    
    print("=" * 50)
    print("FSQ Fast Inference (mmap + KV cache)")
    print("=" * 50)
    
    model = FastFSQModel(model_dir)
    result = model.generate("def fibonacci", max_new_tokens=50)
    
    print("\nOutput:")
    print(result)

if __name__ == "__main__":
    main()
