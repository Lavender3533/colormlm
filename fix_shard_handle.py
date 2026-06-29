# -*- coding: utf-8 -*-
path = r"D:\project\大模型ssd化\fsq_generate_fast.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Keep shard handles open (they don't load data, just file handles)
old = '''        # No shard cache - open/close each time
        print(f"  WeightStore: {len(self.weight_map)} weights in {len(self.shard_keys)} shards")
    
    def get_tensor(self, key):
        shard = self.weight_map[key]
        path = os.path.join(self.model_dir, shard)
        f = safe_open(path, framework="pt", device="cpu")'''

new = '''        # Keep shard handles open (lightweight, just file descriptors)
        self.shard_handles = {}
        for shard_name in set(self.weight_map.values()):
            path = os.path.join(model_dir, shard_name)
            self.shard_handles[shard_name] = safe_open(path, framework="pt", device="cpu")
        print(f"  WeightStore: {len(self.weight_map)} weights in {len(self.shard_handles)} shards")
    
    def get_tensor(self, key):
        shard = self.weight_map[key]
        f = self.shard_handles[shard]'''

content = content.replace(old, new)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("Fixed: keep shard handles open")
