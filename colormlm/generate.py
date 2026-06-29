"""
ColorLM 推理模块 — 迭代修正生成

核心流程:
  Step 0: 所有未知位置设为 MASK
  Step 1: 并行预测所有 MASK 位置的 token + 温度
  Step 2: 温度最高的位置 -> 高置信度 -> 确定下来
  Step 3: 剩余 MASK 位置重新预测（现在有更多已确定的上下文了）
  重复直到全部确定
"""

import torch
import torch.nn.functional as F
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from colormlm.model import ColorLM


def iterative_generate(model, prompt_ids, total_len, n_steps=8,
                       mask_id=1, device='cpu', verbose=True):
    """
    迭代修正生成

    参数:
      model: ColorLM 模型 (必须有 token_pred_head)
      prompt_ids: [prompt_len] prompt 的 token IDs
      total_len: 总输出长度（prompt + 生成）
      n_steps: 最大修正轮数
      mask_id: MASK token 的 ID
      verbose: 是否打印每步详情

    返回:
      ids: [1, total_len] 最终 token IDs
      history: 每一步的详细信息
    """
    model.eval()
    prompt_len = len(prompt_ids)

    # Step 0: 初始化 - prompt 已知，其余全 MASK
    ids = torch.full((1, total_len), mask_id, dtype=torch.long, device=device)
    ids[0, :prompt_len] = prompt_ids.clone()

    determined = torch.zeros(1, total_len, dtype=torch.bool, device=device)
    determined[0, :prompt_len] = True

    history = []

    for step in range(n_steps):
        # --- 前向传播 ---
        tok_emb = model.token_embed(ids)
        pos = torch.arange(total_len, device=device).unsqueeze(0)
        x = tok_emb + model.pos_embed(pos)

        for layer in model.layers:
            x = layer(x)
        x = model.final_norm(x)

        temperature = model.temperature_head(x)  # [1, S, 1]
        temp = temperature.squeeze(-1).squeeze(0)  # [S]

        token_logits = model.token_pred_head(x)  # [1, S, vocab_size]

        # --- 找出未确定的位置 ---
        undetermined_mask = ~determined.squeeze(0)
        n_undetermined = undetermined_mask.sum().item()

        if n_undetermined == 0:
            if verbose:
                print(f"  Step {step+1}: ALL DONE!")
            break

        # --- 温度排序: 未确定位置中，温度最高的先确定 ---
        step_temp = temp.clone()
        step_temp[~undetermined_mask] = -1

        # 每轮确定剩余位置的一定比例
        ratio = 0.3 + 0.1 * step  # 30%, 40%, 50%, ...
        n_to_fix = max(1, int(n_undetermined * min(ratio, 1.0)))

        _, top_indices = step_temp.topk(min(n_to_fix, n_undetermined))

        # --- 用模型预测填充确定的位置 ---
        pred_tokens = token_logits[0, top_indices].argmax(dim=-1)
        ids[0, top_indices] = pred_tokens
        determined[0, top_indices] = True

        # --- 记录历史 ---
        step_info = {
            "step": step + 1,
            "n_fixed": len(top_indices),
            "n_remaining": n_undetermined - len(top_indices),
            "fixed_positions": top_indices.tolist(),
            "fixed_tokens": pred_tokens.tolist(),
            "avg_temperature": temp[top_indices].mean().item(),
        }
        history.append(step_info)

        if verbose:
            print(f"  Step {step+1}: fixed {len(top_indices)} positions, "
                  f"remaining {n_undetermined - len(top_indices)}")

    return ids, history


def demo():
    """演示迭代修正生成"""
    print("=" * 60)
    print("  ColorLM Iterative Refinement Demo")
    print("=" * 60)

    # 查找最新的模型文件
    data_dir = Path(__file__).parent / "data"
    candidates = sorted(data_dir.glob("colormlm_v2_*.pt"), reverse=True)
    if not candidates:
        candidates = sorted(data_dir.glob("colormlm_*.pt"), reverse=True)
    if not candidates:
        print("No model found! Run train_v2.py first.")
        return

    ckpt_path = candidates[0]
    print(f"\nLoading: {ckpt_path.name}")

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = ckpt["config"]
    char2id = ckpt["char2id"]
    id2char = ckpt["id2char"]

    model = ColorLM(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"],
        n_codebooks=cfg["n_codebooks"],
        codebook_size=cfg["codebook_size"],
        codebook_dim=cfg["codebook_dim"],
        max_seq_len=cfg["max_seq_len"],
    )

    # 加载 token prediction head
    model.token_pred_head = torch.nn.Linear(cfg["d_model"], cfg["vocab_size"] + 1)
    model.load_state_dict(ckpt["model_state"], strict=False)

    # --- Test 1: Prompt-based generation ---
    prompt = "def "
    prompt_ids = torch.tensor([char2id.get(c, 0) for c in prompt], dtype=torch.long)
    total_len = 60

    print(f"\nTest 1: Prompt-based generation")
    print(f"  Prompt: '{prompt}'")
    print(f"  Target length: {total_len}")
    print(f"  Refinement steps: 10\n")

    ids, history = iterative_generate(
        model, prompt_ids, total_len, n_steps=10,
        mask_id=1, verbose=True
    )

    result_ids = ids[0].tolist()
    result_text = ''.join(id2char.get(tid, '') for tid in result_ids)
    print(f"\n  Result: {result_text}")

    # Show step-by-step determination
    print(f"\n  Step-by-step determination:")
    for info in history:
        positions = info["fixed_positions"]
        tokens = info["fixed_tokens"]
        chars = []
        for pos, tok in zip(positions[:5], tokens[:5]):
            ch = id2char.get(tok, '?')
            chars.append(f"'{ch}'@{pos}")
        print(f"    Step {info['step']}: avg_temp={info['avg_temperature']:.3f} | "
              f"{' '.join(chars)}")

    # --- Test 2: Full fill-in (no prompt) ---
    print(f"\n{'='*60}")
    print(f"Test 2: Full fill-in (all MASK)")
    total_len2 = 40
    empty_prompt = torch.tensor([], dtype=torch.long)

    print(f"  Target length: {total_len2}")
    print(f"  Refinement steps: 10\n")

    ids2, history2 = iterative_generate(
        model, empty_prompt, total_len2, n_steps=10,
        mask_id=1, verbose=True
    )

    result2 = ''.join(id2char.get(tid, '') for tid in ids2[0].tolist())
    print(f"\n  Result: {result2}")

    # --- Comparison stats ---
    print(f"\n{'='*60}")
    print("  Stats")
    print(f"{'='*60}")
    print(f"  Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"  Prompt gen: {len(history)} steps to fill {total_len - len(prompt)} positions")
    print(f"  Full fill:  {len(history2)} steps to fill {total_len2} positions")


if __name__ == "__main__":
    demo()