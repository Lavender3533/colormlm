# -*- coding: utf-8 -*-
path = r"D:\project\大模型ssd化\fsq_generate.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Replace the get_tensor and _load_shard methods with lazy loading
old_get = '''    def get_tensor(self, key):
        shard = self.weight_map.get(key)
        if shard is None:
            for s_name in self.shard_cache:
                if key in self.shard_cache[s_name]:
                    return self.shard_cache[s_name][key]
            for s_name in set(self.weight_map.values()):
                if s_name not in self.shard_cache:
                    self._load_shard(s_name)
                if key in self.shard_cache[s_name]:
                    return self.shard_cache[s_name][key]
            raise KeyError(key)
        
        if shard not in self.shard_cache:
            self._load_shard(shard)
        
        tensors = self.shard_cache[shard]
        
        if key + "._bq_codes" in tensors:
            return self._dequantize(key, tensors)
        
        t = tensors[key]
        if t.dtype == torch.bfloat16:
            t = t.float()
        return t
    
    def _load_shard(self, shard_name):
        path = os.path.join(self.model_dir, shard_name)
        tensors = {}
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        self.shard_cache[shard_name] = tensors'''

new_get = '''    def get_tensor(self, key):
        shard = self.weight_map.get(key)
        if shard is None:
            raise KeyError(key)
        
        path = os.path.join(self.model_dir, shard)
        with safe_open(path, framework="pt", device="cpu") as f:
            # Check if this is a block-quantized weight
            all_keys = list(f.keys())
            if key + "._bq_codes" in all_keys:
                codes = f.get_tensor(key + "._bq_codes")
                meta = f.get_tensor(key + "._bq_meta")
                shape_t = f.get_tensor(key + "._bq_shape")
                shape = [shape_t[0].item(), shape_t[1].item()]
                
                n_blocks = codes.shape[0]
                b_min = meta[:n_blocks].unsqueeze(1)
                b_max = meta[n_blocks:].unsqueeze(1)
                recon = codes.float() / 255.0 * (b_max - b_min) + b_min
                return recon.flatten()[:shape[0]*shape[1]].reshape(shape)
            
            t = f.get_tensor(key)
            if t.dtype == torch.bfloat16:
                t = t.float()
            return t'''

content = content.replace(old_get, new_get)

# Remove _load_shard and _dequantize methods (no longer needed)
# Find and remove them
import re
content = re.sub(r'    def _load_shard\(self.*?\n.*?self\.shard_cache\[shard_name\] = tensors\n', '', content, flags=re.DOTALL)
content = re.sub(r'    def _dequantize\(self.*?\n.*?return recon\.flatten\(\)\[:shape\[0\]\*shape\[1\]\]\.reshape\(shape\)\n', '', content, flags=re.DOTALL)

# Remove shard_cache init
content = content.replace('        self.shard_cache = {}', '        # Lazy loading - no cache')

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("Fixed memory: lazy loading enabled")
