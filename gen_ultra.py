# -*- coding: utf-8 -*-
import os
BASE = r"D:\project\大模型ssd化"
outpath = os.path.join(BASE, "fsq_generate_ultra.py")

code = r'''# -*- coding: utf-8 -*-
"""FSQ Ultra Inference - Maximum optimization"""
import os, torch, json, time, math, gc
from safetensors import safe_open
from transformers import AutoTokenizer

class UltraFSQModel:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        
        with open(os.path.join(model_dir, "config.json"), "r") as f:
            self.config = json.load(f)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        
        with open(os.path.join(model_dir, "model.safetensors.index.json"), "r") as f:
            self.weight_map = json.load(f).get("weight_map", {})
        
        # Shard handle (lazy)
        self.shard_handle = None
        self.shard_name = None
        
        self.num_layers = self.config["num_hidden_layers"]
        self.hidden_size = self.config["hidden_size"]
        self.num_heads = self.config["num_attention_heads"]
        self.num_kv_heads = self.config.get("num_key_value_heads", self.num_heads)
        self.head_dim = self.config.get("head_dim", self.hidden_size // self.num_heads)
        self.num_experts = self.config.get("num_experts", 128)
        self.top_k = self.config.get("num_experts_per_tok", 8)
        self.rms_norm_eps = self.config.get("rms_norm_eps", 1e-6)
        self.scaling = self.head_dim ** -0.5
        self.gqa_repeat = self.num_heads // self.num_kv_heads
        
        # RoPE (pre-compute for common lengths)
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
        self.inv_freq = inv_freq
        
        # Pre-allocate reusable tensors
        self._expert_buf = torch.zeros(1, 1, self.hidden_size, dtype=torch.float16)
        
        # Load static weights
        print("Loading...")
        self.embed = self._load("model.embed_tokens.weight").half()
        self.norm_w = self._load("model.norm.weight").half()
        self.lm_head_w = self._load("lm_head.weight").half()
        
        # Pre-load ALL non-expert weights for ALL layers (they're small)
        print("Pre-loading attention weights...")
        self.layer_weights = []
        for i in range(self.num_layers):
            w = self._load_layer(i)
            self.layer_weights.append(w)
            if i % 10 == 0:
                print(f"  Layer {i}")
        
        # KV cache (pre-allocated)
        self.kv_k = [None] * self.num_layers
        self.kv_v = [None] * self.num_layers
        
        print(f"Ready: {self.num_layers}L, {self.num_heads}H, {self.num_experts}E")
    
    def _get_shard(self, shard):
        if self.shard_name != shard:
            self.shard_handle = safe_open(os.path.join(self.model_dir, shard), framework="pt", device="cpu")
            self.shard_name = shard
        return self.shard_handle
    
    def _load(self, key):
        shard = self.weight_map[key]
        f = self._get_shard(shard)
        keys = list(f.keys())
        if key + "._bq_codes" in keys:
            codes = f.get_tensor(key + "._bq_codes")
            meta = f.get_tensor(key + "._bq_meta")
            shape_t = f.get_tensor(key + "._bq_shape")
            shape = [shape_t[0].item(), shape_t[1].item()]
            n = codes.shape[0]
            b_min = meta[:n].unsqueeze(1)
            b_max = meta[n:].unsqueeze(1)
            # Dequantize directly to float16
            scale = (b_max - b_min) / 255.0
            recon = codes.float() * scale + b_min
            return recon.flatten()[:shape[0]*shape[1]].reshape(shape)
        t = f.get_tensor(key)
        return t.float() if t.dtype == torch.bfloat16 else t
    
    def _load_layer(self, idx):
        prefix = f"model.layers.{idx}."
        w = {}
        for key in self.weight_map:
            if key.startswith(prefix) and "experts." not in key and "._bq" not in key:
                short = key[len(prefix):]
                w[short] = self._load(key).half()
        return w
    
    def _load_expert(self, layer, expert, proj):
        key = f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight"
        return self._load(key).half()
    
    def _rms(self, x, w):
        xf = x.float()
        return (w.float() * xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.rms_norm_eps)).half()
    
    def _rope(self, q, k, pos):
        inv = self.inv_freq[None, :, None].float()
        pf = pos[:, None, :].float()
        freqs = (inv @ pf).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(1).half()
        sin = emb.sin().unsqueeze(1).half()
        h = q.shape[-1] // 2
        qo = q * cos + torch.cat([-q[..., h:], q[..., :h]], dim=-1) * sin
        ko = k * cos + torch.cat([-k[..., h:], k[..., :h]], dim=-1) * sin
        return qo, ko
    
    def forward(self, hidden, pos, layers_to_run=None):
        if layers_to_run is None:
            layers_to_run = range(self.num_layers)
        
        for i in layers_to_run:
            w = self.layer_weights[i]
            
            # Attention
            res = hidden
            h = self._rms(hidden, w["input_layernorm.weight"])
            B, S, _ = h.shape
            
            q = h @ w["self_attn.q_proj.weight"].T
            k = h @ w["self_attn.k_proj.weight"].T
            v = h @ w["self_attn.v_proj.weight"].T
            
            q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
            
            q = self._rms(q, w["self_attn.q_norm.weight"])
            k = self._rms(k, w["self_attn.k_norm.weight"])
            q, k = self._rope(q, k, pos)
            
            # KV cache
            if self.kv_k[i] is not None:
                k = torch.cat([self.kv_k[i], k], dim=2)
                v = torch.cat([self.kv_v[i], v], dim=2)
            self.kv_k[i] = k.detach()
            self.kv_v[i] = v.detach()
            
            # GQA
            if self.gqa_repeat > 1:
                k = k.repeat_interleave(self.gqa_repeat, dim=1)
                v = v.repeat_interleave(self.gqa_repeat, dim=1)
            
            # Attention score
            attn = (q @ k.transpose(-2, -1)) * self.scaling
            tl = k.shape[2]
            if tl > S:
                mask = torch.ones(S, tl, device=hidden.device, dtype=torch.bool)
                for j in range(S):
                    mask[j, :tl - S + j + 1] = False
            else:
                mask = torch.triu(torch.ones(S, S, device=hidden.device, dtype=torch.bool), diagonal=1)
            attn.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            attn = torch.softmax(attn.float(), dim=-1).half()
            ao = (attn @ v).transpose(1, 2).reshape(B, S, -1)
            ao = (w["self_attn.o_proj.weight"] @ ao.transpose(-1, -2)).transpose(-1, -2)
            hidden = res + ao
            
            # MoE
            res = hidden
            h = self._rms(hidden, w["post_attention_layernorm.weight"])
            
            logits = (h @ w["mlp.gate.weight"].T).float()
            logits = torch.softmax(logits, dim=-1)
            tv, ti = torch.topk(logits, self.top_k, dim=-1)
            tv /= tv.sum(dim=-1, keepdim=True)
            
            # Batch expert computation
            moe = torch.zeros_like(h)
            for j in range(self.top_k):
                idx = ti[0, 0, j].item()
                wt = tv[0, 0, j].item()
                
                gw = self._load_expert(i, idx, "gate_proj")
                uw = self._load_expert(i, idx, "up_proj")
                dw = self._load_expert(i, idx, "down_proj")
                
                g = torch.nn.functional.silu(h @ gw.T)
                eo = (g * (h @ uw.T)) @ dw.T
                moe = moe + eo * wt
            
            hidden = res + moe
        
        return hidden
    
    def clear_kv(self):
        self.kv_k = [None] * self.num_layers
        self.kv_v = [None] * self.num_layers
    
    @torch.no_grad()
    def generate(self, prompt, max_tokens=50, temperature=0.7, top_p=0.9):
        ids = self.tokenizer.encode(prompt, return_tensors="pt")
        tokens = ids[0].tolist()
        
        self.clear_kv()
        print("Generating...")
        t0 = time.time()
        
        # Prefill
        h = self.embed[ids].half()
        pos = torch.arange(h.shape[1]).unsqueeze(0)
        h = self.forward(h, pos)
        h = self._rms(h, self.norm_w)
        logits = (h @ self.lm_head_w.T).float()
        
        t_prefill = time.time() - t0
        print(f"  Prefill: {t_prefill:.2f}s")
        
        # Decode
        t_dec = time.time()
        for step in range(max_tokens):
            lg = logits[0, -1, :] / temperature
            sl, si = torch.sort(lg, descending=True)
            probs = torch.softmax(sl, dim=-1)
            cumsum = torch.cumsum(probs, dim=-1)
            mask = cumsum - probs >= top_p
            sl[mask] = float("-inf")
            probs = torch.softmax(sl, dim=-1)
            nxt = si[torch.multinomial(probs, 1).item()].item()
            
            tokens.append(nxt)
            if nxt == self.tokenizer.eos_token_id:
                break
            
            h = self.embed[torch.tensor([[nxt]])].half()
            pos = torch.tensor([[len(tokens) - 1]])
            h = self.forward(h, pos)
            h = self._rms(h, self.norm_w)
            logits = (h @ self.lm_head_w.T).float()
            
            if step % 10 == 0:
                e = time.time() - t_dec
                s = (step + 1) / e if e > 0 else 0
                print(f"  Step {step}: {s:.2f} tok/s")
        
        total = time.time() - t0
        dec = time.time() - t_dec
        n = len(tokens) - len(ids[0])
        print(f"\nPrefill: {t_prefill:.2f}s, Decode: {dec:.2f}s ({n} tok, {n/dec:.2f} tok/s)")
        
        return self.tokenizer.decode(tokens)

def main():
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else r"D:\\project\\大模型ssd化\\models\\qwen3-coder-bq8"
    print("=" * 50)
    print("FSQ Ultra Inference")
    print("=" * 50)
    m = UltraFSQModel(d)
    r = m.generate("def fibonacci", max_tokens=50)
    print("\n" + r)

if __name__ == "__main__":
    main()
'''

with open(outpath, 'w', encoding='utf-8') as f:
    f.write(code)
print("Written to", outpath)
