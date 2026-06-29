# -*- coding: utf-8 -*-
path = r"D:\project\大模型ssd化\fsq_generate_fast.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Don't cache shards at all - open/read/close each time
old_shard = '''    def _open_shard(self, shard):
        if shard not in self.shard_cache:
            path = os.path.join(self.model_dir, shard)
            self.shard_cache[shard] = safe_open(path, framework="pt", device="cpu")
        self.current_shard = shard
        return self.shard_cache[shard]
    
    def get_tensor(self, key):
        shard = self.weight_map[key]
        f = self._open_shard(shard)'''

new_shard = '''    def get_tensor(self, key):
        shard = self.weight_map[key]
        path = os.path.join(self.model_dir, shard)
        f = safe_open(path, framework="pt", device="cpu")'''

content = content.replace(old_shard, new_shard)

# Remove shard_cache init
content = content.replace('        # Cache for opened shards (keep only current shard open)\n        self.shard_cache = {}\n        self.current_shard = None', '        # No shard cache - open/close each time')

# Add gc.collect after each layer
old_layer_end = '''        hidden_states = residual + moe_out
        return hidden_states'''

new_layer_end = '''        hidden_states = residual + moe_out
        
        import gc
        del weights, h, moe_out
        gc.collect()
        
        return hidden_states'''

content = content.replace(old_layer_end, new_layer_end)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("Fixed: no shard cache + gc")
