"""
ColorLM 交互式 Demo

可以输入文本，查看:
  1. 温度分析（哪些 token 重要）
  2. VQ 码本编码
  3. 迭代修正过程（token 预测暂不准确，需要 MLM 微调）
"""

import os, sys
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pretrained_demo import PretrainedColorLM

def main():
    print("=" * 55)
    print("  ColorLM Interactive Demo")
    print("=" * 55)
    print("\nLoading GPT-2 + VQ + Temperature...")
    model = PretrainedColorLM('gpt2', n_codebooks=4, codebook_size=128, codebook_dim=32)

    # Quick temperature training
    print("\nTraining temperature head (20 epochs)...")
    optimizer = torch.optim.AdamW(model.get_trainable_params(), lr=1e-3)
    train_texts = [
        "def quicksort(arr):",
        "class MyStack:",
        "for i in range(n):",
        "if x > 0: return True",
        "import os, sys",
        "result = [x**2 for x in items]",
        "while not done:",
        "try: val = int(s)",
        "with open(path) as f:",
        "return self.data[key]",
    ]
    keywords = {'def', 'return', 'if', 'else', 'for', 'while', 'class',
                'import', 'from', 'try', 'with', 'as', 'in', 'not', 'and'}

    for epoch in range(20):
        for text in train_texts:
            ids = model.encode_text(text)
            tokens = model.tokenizer.convert_ids_to_tokens(ids[0])
            hidden, codes, temp, code_logits, vq_loss = model.forward(ids)

            temp_target = torch.zeros_like(temp)
            for i, tok in enumerate(tokens):
                clean = tok.strip().lower().replace('\u0120', '')
                if clean in keywords:
                    temp_target[0, i, 0] = 1.0
                elif clean in {'(', ')', ':', ',', ' ', '[', ']', '{', '}'}:
                    temp_target[0, i, 0] = 0.1
                else:
                    temp_target[0, i, 0] = 0.5
            loss = F.mse_loss(temp, temp_target) + 0.05 * vq_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print("Done!\n")
    print("-" * 55)
    print("Commands:")
    print("  输入任意文本 -> 查看温度分析 + VQ 编码")
    print("  'mask 文本' -> 演示迭代修正 (例: mask def sort(arr):)")
    print("  'quit' -> 退出")
    print("-" * 55)

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() == 'quit':
            break

        # Mask mode
        if user_input.lower().startswith('mask '):
            text = user_input[5:].strip()
            if not text:
                print("  Usage: mask <text>")
                continue

            ids = model.encode_text(text)
            n = ids.shape[1]
            tokens = model.tokenizer.convert_ids_to_tokens(ids[0])

            # Auto-mask: mask all except first token
            mask_pos = list(range(1, n))
            test_ids = ids.clone()
            original_tokens = []
            for p in mask_pos:
                original_tokens.append(model.tokenizer.decode([ids[0, p].item()]))
                test_ids[0, p] = model.tokenizer.pad_token_id or 0

            print(f"\n  Original: {model.decode_ids(ids[0])}")
            print(f"  Masked:   {model.decode_ids(test_ids[0])}")
            print(f"  Masking positions {mask_pos}")
            print(f"  Hidden tokens: {original_tokens}")

            # Iterative refine
            model.eval()
            result_ids = test_ids.clone()
            determined = set()
            history = []

            for step in range(8):
                with torch.no_grad():
                    h, c, t, cl, _ = model.forward(result_ids)
                temp_sq = t.squeeze(-1).squeeze(0)

                remaining = [p for p in mask_pos if p not in determined]
                if not remaining:
                    break

                temps = [(p, temp_sq[p].item()) for p in remaining]
                temps.sort(key=lambda x: x[1], reverse=True)

                n_fix = max(1, len(temps) // 3)
                to_fix = temps[:n_fix]

                step_preds = []
                for pos, temp_val in to_fix:
                    emb_table = model.backbone.get_input_embeddings().weight
                    adapted = model.adapt(emb_table)
                    hp = h[0, pos]
                    sim = F.cosine_similarity(adapted, hp.unsqueeze(0), dim=-1)
                    pred = int(sim.argmax().item())
                    result_ids[0, pos] = pred
                    determined.add(pos)
                    pred_char = model.tokenizer.decode([pred])
                    step_preds.append((pos, pred_char, temp_val))

                history.append(step_preds)
                print(f"\n  Step {step+1}:")
                for pos, pred_char, tv in step_preds:
                    actual = original_tokens[pos - 1] if pos - 1 < len(original_tokens) else "?"
                    marker = "OK" if pred_char.strip() == actual.strip() else "miss"
                    print(f"    pos {pos}: temp={tv:.3f} -> predicted='{pred_char}' "
                          f"(actual='{actual}') [{marker}]")

            print(f"\n  Final: {model.decode_ids(result_ids[0])}")
            print(f"  (Note: token prediction needs MLM fine-tuning)")
            continue

        # Normal mode: temperature + VQ analysis
        text = user_input
        ids = model.encode_text(text)
        tokens = model.tokenizer.convert_ids_to_tokens(ids[0])

        model.eval()
        with torch.no_grad():
            hidden, codes, temp, code_logits, _ = model.forward(ids)

        print(f"\n  Input: '{text}'")
        print(f"  Tokens: {ids.shape[1]}")
        print(f"\n  {'Token':>15} {'Temp':>6} {'Codes':>20} {'Bar'}")
        print(f"  {'-'*55}")

        for i in range(ids.shape[1]):
            tok = tokens[i]
            t_val = temp[0, i, 0].item()
            c_val = codes[0, i].tolist()
            bar = "#" * int(t_val * 15)
            c_str = str(c_val)
            print(f"  {tok:>15} {t_val:.3f} {c_str:>20} {bar}")

        # Highlight high/low temperature
        high_temp = [(tokens[i], temp[0, i, 0].item()) for i in range(ids.shape[1])]
        high_temp.sort(key=lambda x: x[1], reverse=True)

        print(f"\n  Most important (high temp):")
        for tok, t in high_temp[:3]:
            print(f"    {tok}: {t:.3f}")
        print(f"  Least important (low temp):")
        for tok, t in high_temp[-3:]:
            print(f"    {tok}: {t:.3f}")


if __name__ == "__main__":
    main()