# -*- coding: utf-8 -*-
path = r"D:\project\大模型ssd化\fsq_generate_final.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Replace: open all shards at once -> lazy open one shard at a time
old = '''        # Group keys by shard and layer
        self.shard_handles = {}
        for shard in set(self.weight_map.values()):
            path = os.path.join(model_dir, shard)
            self.shard_handles[shard] = safe_open(path, framework="pt", device="cpu")'''

new = '''        # Lazy shard opening - only keep one shard handle at a time
        self.shard_handle = None
        self.shard_name = None'''

content = content.replace(old, new)

# Replace _load_tensor to use lazy shard opening
old_load = '''    def _load_tensor(self, key):
        """Load and dequantize a single tensor"""
        shard = self.weight_map[key]
        f = self.shard_handles[shard]'''

new_load = '''    def _load_tensor(self, key):
        """Load and dequantize a single tensor"""
        shard = self.weight_map[key]
        if self.shard_name != shard:
            self.shard_handle = safe_open(os.path.join(self.model_dir, shard), framework="pt", device="cpu")
            self.shard_name = shard
        f = self.shard_handle'''

content = content.replace(old_load, new_load)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("Fixed: lazy shard opening")
