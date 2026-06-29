# -*- coding: utf-8 -*-
path = r"D:\project\大模型ssd化\fsq_generate.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Add verbose error catching in generate
old_gen = '''        for step in range(max_new_tokens):
            input_t = torch.tensor([tokens], dtype=torch.long)
            logits = self.forward(input_t)'''

new_gen = '''        for step in range(max_new_tokens):
            input_t = torch.tensor([tokens], dtype=torch.long)
            try:
                logits = self.forward(input_t)
            except Exception as e:
                print("Error at step", step, ":", e)
                import traceback
                traceback.print_exc()
                break'''

content = content.replace(old_gen, new_gen)

# Also add layer progress
old_layer = '''    def forward_layer(self, layer_idx, hidden_states, position):
        # RMSNorm'''
new_layer = '''    def forward_layer(self, layer_idx, hidden_states, position):
        if layer_idx % 10 == 0:
            print("  Layer", layer_idx, "...")
        # RMSNorm'''
content = content.replace(old_layer, new_layer)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("Added error catching + progress")
