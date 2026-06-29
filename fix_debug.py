# -*- coding: utf-8 -*-
path = r"D:\project\大模型ssd化\fsq_generate_fast.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Add debug output in prefill
old_prefill = '''        # Prefill: process all input tokens at once
        input_t = torch.tensor([tokens], dtype=torch.long)
        hidden_states = self.embed[input_t].float()
        position_ids = torch.arange(hidden_states.shape[1]).unsqueeze(0)
        
        for i in range(self.num_layers):
            hidden_states = self.forward_layer_with_cache(i, hidden_states, position_ids, use_cache=True)'''

new_prefill = '''        # Prefill: process all input tokens at once
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
                break'''

content = content.replace(old_prefill, new_prefill)

# Also wrap decode in try/except
old_decode = '''            # Forward the new token
            input_t = torch.tensor([[next_token]], dtype=torch.long)
            hidden_states = self.embed[input_t].float()
            position_ids = torch.tensor([[len(tokens) - 1]])
            
            for i in range(self.num_layers):
                hidden_states = self.forward_layer_with_cache(i, hidden_states, position_ids, use_cache=True)'''

new_decode = '''            # Forward the new token
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
                break'''

content = content.replace(old_decode, new_decode)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("Added debug output")
