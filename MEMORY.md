# -*- coding: utf-8 -*-
# ColorLM 项目记忆文件

## 2026-07-29 当前权威方向

- 最终方向：`ColorLM Neural Bus`，详细设计见`fast16/COLORLM_NEURAL_BUS.md`。
- 当前主模型：`ColorLM-v6-Q3Router-Fused-A1.gguf`，任何新机制关闭时必须退回v6。
- 当前顺序：Fast16推理加速 → 正确专家分页 → 两站Neural Bus → 多供体残差胶囊。
- Claude Code接入已经可用但暂时冻结，不作为当前研发主线。
- 外来专家不再直接替换原生专家；采用`native + alpha * donor_delta`残差胶囊。
- 长期组件：Fast Spine、Neural Slice ABI、虚拟专家、两级路由、专家竞价、误差循环、
  隐藏状态推测、多时间尺度记忆、GPU/RAM/SSD三级分页。
- 闪念收件箱：`fast16/IDEA_INBOX.md`。用户说“闪念”时先记录，不自动改变主线。

以下内容为历史记录，其中模型、磁盘和当前状态可能已经过期；不得覆盖上面的权威方向。

## 项目目标
"大模型SSD化" - 让大模型跑在普通电脑上
- 最终目标: 30B MoE 模型在 RX 5700 XT (8GB) + 32GB RAM 上运行
- **核心方向: 架构创新** - 不是简单转格式，而是研究新压缩/推理方式
- 用户强调: 项目目的是**更改架构**，不是用现成模型

## 硬件环境
- GPU: RX 5700 XT (8GB VRAM, AMD, PyTorch CUDA不可用)
- RAM: 32GB
- CPU: i5-12400F (支持 AVX2)
- OS: Windows 11
- Python: 3.12.5, PyTorch 2.9.1+cpu
- 代理端口: 7890
- D: 盘可用: ~57GB

## 架构创新方向 (用户确认的想法)
1. **RGB+温度表示** - 用更紧凑的向量表示权重，温度表示token重要性
2. **FSQ码空间推理** - 让模型直接在离散码空间(6 dims x 8 levels)工作，而非浮点向量
3. **MoE专家路由优化** - 减少专家切换开销，当前每token需384次磁盘读取
4. **双向/并行预测** - 一次预测多个token，不对的单独重新预测
5. **分层RGB** - 顶层RGB对应大值分类，下层RGB对应细分，更精确地表示权重

## 已验证结果
### 早期探索 (Colab GPU, Qwen2.5-1.5B-Instruct 作为教师)
- VQ码本坍塌: 2/128
- FSQ码本: 8/8 全满
- 蒸馏70M学生 <- Qwen2.5-1.5B-Instruct: 余弦相似度 0.96
- 温度系统: 关键词=HIGH, 标点=LOW

### Block-8bit 量化 (Qwen3-Coder-30B-A3B)
- 原始: 56.9GB → BQ8: 31.6GB (1.8x 压缩)
- 余弦相似度: 0.999954
- 对比 Q4_K_M: 0.996794
- 脚本: transform_block8bit.py

### 自定义推理引擎 (Qwen3-Coder-30B-A3B)
- fsq_generate_ultra.py: 0.22 tok/s
- fsq_generate_v7.py: 0.12 tok/s
- 每token需要384次磁盘读取 (48层 × 8个活跃专家)

### 预缩放整数矩阵乘法 (benchmark)
- float32矩阵乘法: 0.051 ms
- uint8→float转换: 0.294 ms
- 预缩放方法: 4.24 ms / 8专家 (2.88x)

## GGUF 转换记录 (Qwen3-Coder-30B-A3B)
### convert_to_q8.py 产出的文件
- 输出: models/qwen3-coder-q8_0.gguf (32.4GB)
- Ollama 加载结果: "supplied file was not in GGUF format"

### 发现的问题
1. Magic Number: 写了 0x46475547 (文件头"GUGF")，正确应为 0x46554747 (文件头"GGUF")
2. KV类型: FLOAT32 写成了类型5(INT32)，正确应为类型6(FLOAT32)
3. 缺少 general.name 和 general.file_type 字段

### fix_gguf_full.py 修复后
- Ollama 结果: "parsing GGUF" 100% → "failed to validate GGUF with llama-quantize"
- 原因: MoE模型的张量格式与dense模型不同

### MoE vs Dense 张量格式差异 (Qwen3-Coder-30B-A3B, 128专家/层, 48层)
- convert_to_q8.py 写的: 每个专家单独tensor (18432个tensor)
  - model.layers.0.mlp.experts.0.gate_proj.weight
  - model.layers.0.mlp.experts.1.gate_proj.weight ...
- Ollama/llama.cpp 要求: 所有专家合并成3D tensor (144个tensor)
  - model.layers.0.mlp.gate_proj.weight, shape [128, n_ff, n_embd]

### 当前状态
- 原始模型 (Qwen3-Coder-30B-A3B safetensors) 已被删除
- 修复后的GGUF仍不可用 (张量格式问题，需原始模型重新处理)
- 需要重新获取模型或使用 llama.cpp 官方转换器

## 工具/脚本状态
```
D:\project\大模型ssd化\
├── models\
│   ├── qwen3-coder-q8_0.gguf    # 32.4GB (张量格式错误，不可用)
│   ├── Modelfile
│   ├── Qwen1.5B-fsq/
│   └── Qwen2.5-1.5B-Instruct/
├── convert_to_q8.py             # GGUF转换器 (magic/KV/MoE格式均有问题)
├── convert_to_q8_v2.py          # 修复了KV类型 (MoE格式仍错)
├── fix_gguf_full.py             # 修复GGUF header (可用)
├── transform_block8bit.py       # BQ8量化脚本
├── fsq_generate_v7.py           # 自定义推理引擎
├── llama.cpp/                   # GGUF工具
├── colormlm/                    # GitHub仓库
└── MEMORY.md
```

## 技术参数
- RX 5700 XT 不支持 ROCm，不能用 PyTorch GPU
- Vulkan SDK: C:\Users\Kangnaixi\scoop\apps\vulkan\current
- Ollama 已安装，qwen2.5:7b 可用 (14.37 t/s)
- safetensors mmap 在Windows页面文件不够时会OOM
- bfloat16分块转换可避免OOM

## GitHub
- URL: https://github.com/Lavender3533/colormlm
