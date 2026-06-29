# -*- coding: utf-8 -*-
path = r"D:\project\大模型ssd化\fsq_generate_fast.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

old_mmap = '''class MmapWeightStore:
    """Memory-mapped weight storage for fast access"""
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
        
        # Open all shards with mmap
        self.shard_files = {}
        for shard in set(self.weight_map.values()):
            path = os.path.join(model_dir, shard)
            self.shard_files[shard] = safe_open(path, framework="numpy", device="cpu")
        
        print(f"  MmapStore: {len(self.weight_map)} weights in {len(self.shard_files)} shards")
    
    def get_tensor(self, key):
        shard = self.weight_map[key]
        f = self.shard_files[shard]'''

new_mmap = '''class MmapWeightStore:
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
        
        # Cache for opened shards (keep only current shard open)
        self.shard_cache = {}
        self.current_shard = None
        
        print(f"  WeightStore: {len(self.weight_map)} weights in {len(self.shard_keys)} shards")
    
    def _open_shard(self, shard):
        if shard not in self.shard_cache:
            path = os.path.join(self.model_dir, shard)
            self.shard_cache[shard] = safe_open(path, framework="pt", device="cpu")
        self.current_shard = shard
        return self.shard_cache[shard]
    
    def get_tensor(self, key):
        shard = self.weight_map[key]
        f = self._open_shard(shard)'''

content = content.replace(old_mmap, new_mmap)

# Also fix the numpy issue - use pt framework instead
old_check = '''        # Check if block-quantized
        all_keys = list(f.keys())
        if key + "._bq_codes" in all_keys:
            codes = torch.from_numpy(f.get_tensor(key + "._bq_codes").copy())
            meta = torch.from_numpy(f.get_tensor(key + "._bq_meta").copy())
            shape_t = f.get_tensor(key + "._bq_shape")
            shape = [int(shape_t[0]), int(shape_t[1])]'''

new_check = '''        # Check if block-quantized
        all_keys = list(f.keys())
        if key + "._bq_codes" in all_keys:
            codes = f.get_tensor(key + "._bq_codes")
            meta = f.get_tensor(key + "._bq_meta")
            shape_t = f.get_tensor(key + "._bq_shape")
            shape = [shape_t[0].item(), shape_t[1].item()]'''

content = content.replace(old_check, new_check)

old_recon = '''            n = codes.shape[0]
            b_min = meta[:n].unsqueeze(1)
            b_max = meta[n:].unsqueeze(1)
            recon = codes.float() / 255.0 * (b_max - b_min) + b_min
            return recon.flatten()[:shape[0]*shape[1]].reshape(shape)
        
        t = torch.from_numpy(f.get_tensor(key).copy())'''

new_recon = '''            n = codes.shape[0]
            b_min = meta[:n].unsqueeze(1)
            b_max = meta[n:].unsqueeze(1)
            recon = codes.float() / 255.0 * (b_max - b_min) + b_min
            return recon.flatten()[:shape[0]*shape[1]].reshape(shape)
        
        t = f.get_tensor(key)'''

content = content.replace(old_recon, new_recon)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("Fixed mmap to lazy loading")
