# -*- coding: utf-8 -*-
"""FSQ Inference Engine - Block-wise 8-bit Quantized MoE Model"""
import os, torch, json, time
from safetensors import safe_open
from transformers import AutoTokenizer

class FSQInference:
    def __init__(self, model_dir, device="cpu"):
        self.model_dir = model_dir
        self.device = device
        self.config = None
        self.tokenizer = None
        self.shards = {}
        self.weight_index = {}
        
        self._load_config()
        self._load_tokenizer()
        self._build_index()
    
    def _load_config(self):
        config_path = os.path.join(self.model_dir, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        print("Model:", self.config.get("model_type", "unknown"))
        print("Layers:", self.config.get("num_hidden_layers", "?"))
        print("Experts:", self.config.get("num_experts", "?"))
        print("Active experts:", self.config.get("num_experts_per_tok", "?"))
    
    def _load_tokenizer(self):
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir, trust_remote_code=True
        )
        print("Tokenizer loaded, vocab size:", len(self.tokenizer))
    
    def _build_index(self):
        """Build index of which shard contains which weight"""
        index_path = os.path.join(self.model_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            self.weight_index = index.get("weight_map", {})
            self.shards = {s: None for s in set(self.weight_index.values())}
            print("Indexed", len(self.weight_index), "weights in", len(self.shards), "shards")
        else:
            # Single shard model
            shard = [f for f in os.listdir(self.model_dir) if f.endswith(".safetensors")][0]
            self.shards = {shard: None}
            print("Single shard model:", shard)
    
    def _get_shard(self, shard_name):
        """Load shard lazily"""
        if self.shards[shard_name] is None:
            path = os.path.join(self.model_dir, shard_name)
            tensors = {}
            with safe_open(path, framework="pt", device="cpu") as f:
                for k in f.keys():
                    tensors[k] = f.get_tensor(k)
            self.shards[shard_name] = tensors
            print("Loaded shard:", shard_name, "(", len(tensors), "tensors)")
        return self.shards[shard_name]
    
    def get_tensor(self, key):
        """Get a tensor by key"""
        if key in self.weight_index:
            shard = self.weight_index[key]
        else:
            # Search in loaded shards
            for s_name, s_tensors in self.shards.items():
                if s_tensors and key in s_tensors:
                    return s_tensors[key]
            # Try to find in shard files
            for s_name in self.shards:
                tensors = self._get_shard(s_name)
                if key in tensors:
                    return tensors[key]
            raise KeyError("Weight not found: " + key)
        
        tensors = self._get_shard(shard)
        
        # Check if it's block-quantized
        if key + "._bq_codes" in tensors:
            return self._dequantize(key, tensors)
        
        return tensors[key]
    
    def _dequantize(self, key, tensors):
        """Dequantize block-wise quantized weight"""
        codes = tensors[key + "._bq_codes"]
        meta = tensors[key + "._bq_meta"]
        shape_tensor = tensors[key + "._bq_shape"]
        shape = [shape_tensor[0].item(), shape_tensor[1].item()]
        
        n_blocks = codes.shape[0]
        b_min = meta[:n_blocks].unsqueeze(1)
        b_max = meta[n_blocks:].unsqueeze(1)
        
        recon = codes.float() / 255.0 * (b_max - b_min) + b_min
        w_flat = recon.flatten()[:shape[0] * shape[1]]
        return w_flat.reshape(shape)
    
    def generate(self, prompt, max_tokens=100, temperature=0.7):
        """Generate text from prompt"""
        print("Generating from prompt:", repr(prompt[:50]))
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        
        print("Input tokens:", input_ids.shape[1])
        
        # Simple greedy generation (simplified - not full model forward)
        # This is a placeholder - real implementation needs full transformer forward pass
        tokens = input_ids[0].tolist()
        
        print("Note: Full inference requires complete transformer implementation")
        print("Current version demonstrates weight loading and dequantization")
        
        return self.tokenizer.decode(tokens)

def main():
    import sys
    
    model_dir = r"D:\\project\\大模型ssd化\\models\\qwen3-coder-bq8"
    
    if len(sys.argv) > 1:
        model_dir = sys.argv[1]
    
    print("=" * 50)
    print("FSQ Inference Engine")
    print("=" * 50)
    
    engine = FSQInference(model_dir)
    
    # Test weight loading
    print("")
    print("Testing weight loading...")
    test_key = "model.layers.0.mlp.experts.0.gate_proj.weight"
    try:
        w = engine.get_tensor(test_key)
        print("Loaded:", test_key, "-> shape", w.shape, "dtype", w.dtype)
        print("Range:", w.min().item(), "to", w.max().item())
        print("Weight loading: OK")
    except Exception as e:
        print("Error:", e)
    
    print("")
    print("Engine ready for inference")

if __name__ == "__main__":
    main()
