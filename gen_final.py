# -*- coding: utf-8 -*-
import os
BASE = r"D:\project\大模型ssd化"
outpath = os.path.join(BASE, "fsq_generate_final.py")

code = r'''# -*- coding: utf-8 -*-
"""FSQ Final Inference - All optimizations applied"""
import os, torch, json, time, math, gc
from safetensors import safe_open
from transformers import AutoTokenizer

class LayerCache:
    """Cache layer weights in float16 to avoid repeated dequantization"""
    def __init__(self, max_layers=2):
        self.cache = {}
        self.max_layers = max_layers
    
    def get(self, layer_idx):
        return self.cache.get(layer_idx)
    
    def put(self, layer_idx, weights):
        if len(self.cache) >= self.max_layers:
            oldest = min(self.cache.keys())
            del self.cache[oldest]
            gc.collect()
        self.cache[layer_idx] = weights

class FastFSQModel:
    def __init__(self, model_dir, device="cpu"):
        self.model_dir = model_dir
        
        with open(os.path.join(model_dir, "config.json"), "r") as f:
            self.config = json.load(f)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        
        with open(os.path.join(model_dir, "model.safetensors.index.json"), "r") as f:
            self.weight_map = json.load(f).get("weight_map", {})
        
        # Group keys by shard and layer
        self.shard_handles = {}
        for shard in set(self.weight_map.values()):
            path = os.path.join(model_dir, shard)
            self.shard_handles[shard] = safe_open(path, framework="pt", device="cpu")
        
        self.num_layers = self.config["num_hidden_layers"]
        self.hidden_size = self.config["hidden_size"]
        self.num_heads = self.config["num_attention_heads"]
        self.num_kv_heads = self.config.get("num_key_value_heads", self.num_heads)
        self.head_dim = self.config.get("head_dim", self.hidden_size // self.num_heads)
        self.num_experts = self.config.get("num_experts", 128)
        self.top_k = self.config.get("num_experts_per_tok", 8)
        self.rms_norm_eps = self.config.get("rms_norm_eps", 1e-6)
        self.scaling = self.head_dim ** -0.5
        
        # RoPE
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
        self.inv_freq = inv_freq
        
        # Load static weights in float16
        print("Loading static weights...")
        self.embed = self._load_tensor("model.embed_tokens.weight").half()
        self.norm = self._load_tensor("model.norm.weight").half()
        self.lm_head = self._load_tensor("lm_head.weight").half()
        
        # Layer weight cache (float16)
        self.layer_cache = LayerCache(max_layers=2)
        
        # Expert weight cache (per-layer, float16)
        self.expert_cache = {}
        self.expert_cache_layer = None
        
        # KV cache
        self.kv_cache = [None] * self.num_layers
        
        print(f"Model ready: {self.num_layers}L, {self.num_heads}H, {self.num_experts}E")
    
    def _load_tensor(self, key):
        """Load and dequantize a single tensor"""
        shard = self.weight_map[key]
        f = self.shard_handles[shard]
        
        if key + "._bq_codes" in f.keys():
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
    
    def _load_layer_weights(self, layer_idx):
        """Load all non-expert weights for a layer, cached in float16"""
        cached = self.layer_cache.get(layer_idx)
        if cached is not None:
            return cached
        
        prefix = f"model.layers.{layer_idx}."
        weights = {}
        for key in self.weight_map:
            if key.startswith(prefix) and "experts." not in key and "._bq" not in key:
                short = key[len(prefix):]
                weights[short] = self._load_tensor(key).half()
        
        self.layer_cache.put(layer_idx, weights)
        return weights
    
    def _load_expert_weight(self, layer_idx, expert_idx, proj):
        """Load a single expert weight, with caching"""
        cache_key = (layer_idx, expert_idx, proj)
        if cache_key in self.expert_cache:
            return self.expert_cache[cache_key]
        
        key = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{proj}.weight"
        w = self._load_tensor(key).half()
        
        # Cache up to 24 expert weights (8 experts × 3 projections)
        if len(self.expert_cache) < 24:
            self.expert_cache[cache_key] = w
        
        return w
    
    def _clear_expert_cache(self, layer_idx):
        """Clear expert cache when moving to new layer"""
        if self.expert_cache_layer != layer_idx:
            self.expert_cache.clear()
            self.expert_cache_layer = layer_idx
    
    def rms_norm(self, x, weight):
        x_f = x.float()
        var = x_f.pow(2).mean(-1, keepdim=True)
        return (weight.float() * x_f * torch.rsqrt(var + self.rms_norm_eps)).half()
    
    def apply_rope(self, q, k, pos):
        inv = self.inv_freq[None, :, None].float()
        pos_f = pos[:, None, :].float()
        freqs = (inv @ pos_f).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(1).half()
        sin = emb.sin().unsqueeze(1).half()
        
        half = q.shape[-1] // 2
        q_out = q * cos + torch.cat([-q[..., half:], q[..., :half]], dim=-1) * sin
        k_out = k * cos + torch.cat([-k[..., half:], k[..., :half]], dim=-1) * sin
        return q_out, k_out
    
    def forward_layer(self, layer_idx, hidden_states, pos):
        self._clear_expert_cache(layer_idx)
        weights = self._load_layer_weights(layer_idx)
        
        # === Attention ===
        residual = hidden_states
        h = self.rms_norm(hidden_states, weights["input_layernorm.weight"])
        
        B, S, _ = h.shape
        q = (h @ weights["self_attn.q_proj.weight"].T)
        k = (h @ weights["self_attn.k_proj.weight"].T)
        v = (h @ weights["self_attn.v_proj.weight"].T)
        
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        q = self.rms_norm(q, weights["self_attn.q_norm.weight"])
        k = self.rms_norm(k, weights["self_attn.k_norm.weight"])
        
        q, k = self.apply_rope(q, k, pos)
        
        # KV Cache
        if self.kv_cache[layer_idx] is not None:
            pk, pv = self.kv_cache[layer_idx]
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        self.kv_cache[layer_idx] = (k.detach(), v.detach())
        
        # GQA
        if self.num_kv_heads < self.num_heads:
            k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scaling
        total_len = k.shape[2]
        if total_len > S:
            mask = torch.zeros(S, total_len, device=hidden_states.device, dtype=torch.bool)
            for i in range(S):
                mask[i, total_len - S + i + 1:] = True
        else:
            mask = torch.triu(torch.ones(S, S, device=hidden_states.device, dtype=torch.bool), diagonal=1)
        attn.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = torch.softmax(attn.float(), dim=-1).half()
        attn_out = (attn @ v).transpose(1, 2).reshape(B, S, -1)
        
        attn_out = (weights["self_attn.o_proj.weight"] @ attn_out.transpose(-1, -2)).transpose(-1, -2)
        hidden_states = residual + attn_out
        
        # === MoE ===
        residual = hidden_states
        h = self.rms_norm(hidden_states, weights["post_attention_layernorm.weight"])
        
        router_logits = (h @ weights["mlp.gate.weight"].T).float()
        router_logits = torch.softmax(router_logits, dim=-1)
        topk_val, topk_idx = torch.topk(router_logits, self.top_k, dim=-1)
        topk_val /= topk_val.sum(dim=-1, keepdim=True)
        
        moe_out = torch.zeros_like(h)
        for i in range(self.top_k):
            idx = topk_idx[0, 0, i].item()
            w = topk_val[0, 0, i].item()
            
            gate_w = self._load_expert_weight(layer_idx, idx, "gate_proj")
            up_w = self._load_expert_weight(layer_idx, idx, "up_proj")
            down_w = self._load_expert_weight(layer_idx, idx, "down_proj")
            
            gate = torch.nn.functional.silu(h @ gate_w.T)
            expert_out = (gate * (h @ up_w.T)) @ down_w.T
            moe_out = moe_out + expert_out * w
        
        hidden_states = residual + moe_out
        return hidden_states
    
    def clear_kv_cache(self):
        self.kv_cache = [None] * self.num_layers
    
    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=50, temperature=0.7, top_p=0.9):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        tokens = input_ids[0].tolist()
        
        self.clear_kv_cache()
        
        print("Generating...")
        t0 = time.time()
        
        # Prefill
        h = self.embed[input_ids].half()
        pos = torch.arange(h.shape[1]).unsqueeze(0)
        
        for i in range(self.num_layers):
            h = self.forward_layer(i, h, pos)
            if i % 10 == 0:
                print(f"  Layer {i}...")
        
        h = self.rms_norm(h, self.norm)
        logits = (h @ self.lm_head.T).float()
        
        prefill_time = time.time() - t0
        print(f"  Prefill: {prefill_time:.2f}s")
        
        # Decode
        decode_start = time.time()
        
        for step in range(max_new_tokens):
            next_logits = logits[0, -1, :] / temperature
            sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumsum = torch.cumsum(probs, dim=-1)
            mask = cumsum - probs >= top_p
            sorted_logits[mask] = float("-inf")
            probs = torch.softmax(sorted_logits, dim=-1)
            next_token = sorted_idx[torch.multinomial(probs, 1).item()].item()
            
            tokens.append(next_token)
            if next_token == self.tokenizer.eos_token_id:
                break
            
            # Forward single token
            h = self.embed[torch.tensor([[next_token]])].half()
            pos = torch.tensor([[len(tokens) - 1]])
            
            for i in range(self.num_layers):
                h = self.forward_layer(i, h, pos)
            
            h = self.rms_norm(h, self.norm)
            logits = (h @ self.lm_head.T).float()
            
            if step % 10 == 0:
                elapsed = time.time() - decode_start
                speed = (step + 1) / elapsed if elapsed > 0 else 0
                print(f"  Step {step}: {speed:.2f} tok/s")
        
        total = time.time() - t0
        decode_time = time.time() - decode_start
        n_gen = len(tokens) - len(input_ids[0])
        
        print(f"\nResults:")
        print(f"  Prefill: {prefill_time:.2f}s")
        print(f"  Decode: {decode_time:.2f}s ({n_gen} tokens, {n_gen/decode_time:.2f} tok/s)")
        print(f"  Total: {total:.2f}s")
        
        return self.tokenizer.decode(tokens)

def main():
    import sys
    model_dir = sys.argv[1] if len(sys.argv) > 1 else r"D:\\project\\大模型ssd化\\models\\qwen3-coder-bq8"
    
    print("=" * 50)
    print("FSQ Final Inference (all optimizations)")
    print("=" * 50)
    
    model = FastFSQModel(model_dir)
    result = model.generate("def fibonacci", max_new_tokens=50)
    
    print("\nOutput:")
    print(result)

if __name__ == "__main__":
    main()
'''

with open(outpath, 'w', encoding='utf-8') as f:
    f.write(code)
print("Written to", outpath)
