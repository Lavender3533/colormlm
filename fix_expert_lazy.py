# -*- coding: utf-8 -*-
path = r"D:\project\大模型ssd化\fsq_generate_fast.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Replace get_layer_weights to only load non-expert weights
old = '''    def get_layer_weights(self, layer_idx):
        """Load all weights for a layer at once"""
        prefix = f"model.layers.{layer_idx}."
        weights = {}
        for key in self.weight_map:
            if key.startswith(prefix):
                short = key[len(prefix):]
                weights[short] = self.get_tensor(key)
        return weights'''

new = '''    def get_layer_weights(self, layer_idx):
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
        return self.get_tensor(key)'''

content = content.replace(old, new)

# Update forward_layer_with_cache to use lazy expert loading
old_expert = '''        moe_out = torch.zeros_like(h)
        for i in range(self.top_k):
            idx = topk_indices[0, 0, i].item()
            w = topk_values[0, 0, i].item()
            
            gate = h @ weights[f"mlp.experts.{idx}.gate_proj.weight"].T
            up = h @ weights[f"mlp.experts.{idx}.up_proj.weight"].T
            gate = torch.nn.functional.silu(gate)
            expert_out = (gate * up) @ weights[f"mlp.experts.{idx}.down_proj.weight"].T
            moe_out = moe_out + expert_out * w'''

new_expert = '''        moe_out = torch.zeros_like(h)
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
            moe_out = moe_out + expert_out * w'''

content = content.replace(old_expert, new_expert)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("Fixed: lazy expert loading")
