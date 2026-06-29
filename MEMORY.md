# -*- coding: utf-8 -*-
# ColorLM 项目记忆文件

## 项目目标
"大模型SSD化" - 让大模型跑在普通电脑上
- 最终目标: 8B+ 模型在 RX 5700 XT (8GB) + 32GB RAM 上运行
- 方法: FSQ-Native 架构 + MoE + 查找表推理

## 硬件环境
- GPU: RX 5700 XT (8GB VRAM, AMD, PyTorch CUDA不可用)
- RAM: 32GB
- OS: Windows 11
- Python: 3.12.5, PyTorch 2.9.1+cpu
- 代理端口: 7890

## 已验证结果
- VQ码本坍塌: 2/128 (失败)
- FSQ码本: 8/8 全满 (成功)
- 蒸馏70M学生 <- 1.5B老师: 余弦相似度 0.96 (成功)
- 温度系统: 关键词=HIGH, 标点=LOW (成功)
- FSQ权重压缩: 简单/分组都产生垃圾输出 (失败)
- Ollama 4-bit量化: qwen2.5:7b @ 5 tok/s (成功)

## FSQ-Native Transformer 架构 (V5 - 2026-06-29)
核心创新: 用FSQ码本查找替代FFN矩阵乘法
- 参数量: 79.8M (对比 Qwen 1.5B = 19x 压缩)
- FFN压缩: 16x (传统FFN 524K vs FSQ FFN 32K 参数)
- 注意力: 保持浮点 (占~20%参数, 质量保障)
- FFN: FSQ码本查找 (占~80%参数, 是优化重点)
- MoE路由: top-2 专家激活 (稀疏计算)
- 编码: 浮点向量 -> FSQ离散码 (8级, STE梯度)

### 文件位置
- architecture.py: D:\project\大模型ssd化\fsq_native\architecture.py
- train.py: D:\project\大模型ssd化\fsq_native\train.py
- Colab笔记本: D:\project\大模型ssd化\colormlm_repo\colab_fsq_native.ipynb

## MiMo模型分析
- MiMo-V2.5-Pro: ~450B MoE, ~1TB (不可能在本地跑)
- MiMo-7B-RL: 7B dense, ~4GB 4-bit (可以用ollama跑)
- MiMo-V2-Flash: MoE, 中等规模

## 下一步
1. 上传 architecture.py + train.py 到 Colab
2. 蒸馏训练验证 FSQ-Native 架构
3. 如果验证成功, 扩大规模
4. 设计 FSQ-MoE 大模型架构

## GitHub
- URL: https://github.com/Lavender3533/colormlm
