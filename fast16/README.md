# ColorLM ZeroTrain

这是 1–15GB 快速学习模型路线的独立实验目录。当前人工试用入口是 v4：

- 独立 `colorlmv4` 架构标识和 12.651GiB 模型文件；
- 35B 总参数、约 3B 动态激活的稀疏专家知识张量；
- 30 个门控 DeltaNet 状态层和 10 个待迁移 ColorKernel 层；
- Vulkan GPU 非专家计算，MoE 权重由 CPU/mmap 承担；
- 不使用 BM25、关键词判断或项目文档提示词注入；
- 当前迁移 1 个 ColorKernel 层，并在最后 4 层执行共享权重递归精炼；
- 快速权重和剩余 ColorKernel 层继续接入中。

双击 `fast16/run-v4-gpu.bat` 试用。输入 `/exit` 退出，输入 `/clear` 清空上下文。

v3 保留为早期对照：

- 1.04GB Qwen2.5-1.5B-Instruct Q4_K_M 通用语言核心；
- llama.cpp `ggml-vulkan` 全层 GPU 推理；
- UTF-8 HTTP 对话和流式输出；
- 388 条零训练编译记忆；
- 4096 token 上下文与常驻显存服务。

## 立即试用

双击 `fast16/run-core-gpu.bat`。退出输入 `/exit`，清空上下文输入 `/clear`。

`run-gpu-demo.bat` 是早期 v2 架构实验，仅用于验证 Byte 词表、递归层和 DirectML，不再作为通用对话入口。

## v2 架构实验

v2 定义了 CLM 单文件格式，并实现：

- 本地检查点到 `.clm` 的安全打包；
- FP16 权重存储和 mmap tensor 读取；
- 独立于旧 ColorLM 模块的前向运行时；
- 最后两层共享递归；
- 编译进模型的连续关联记忆；
- tensor 校验和与结构验证。
- 数据无关的 UTF-8 Byte 权重移植；
- AMD GPU DirectML 前向运行时。
- 512 Byte 数据无关位置扩展；
- 扁平 UINT8 记忆和本地文档编译器。

当前本机只保留了 91 万参数的字符代码生成检查点，因此首个 CLM 是格式和新前向结构的 v0 实体，不代表最终核心规模。格式设计允许后续直接替换更强的本地权重。

## 构建模型

在项目根目录运行：

```powershell
python -m fast16.clm pack `
  --checkpoint colormlm/data/v3_final.pt `
  --memory fast16/data/bootstrap_memory.jsonl `
  --output fast16/models/colormlm-zerotrain-v0.clm
```

## 检查和运行

直接双击 `fast16/run-demo.bat` 可以运行已验证的代码生成示例。

```powershell
python -m fast16.clm inspect fast16/models/colormlm-zerotrain-v0.clm
python -m fast16.clm verify fast16/models/colormlm-zerotrain-v0.clm
python -m fast16.clm generate fast16/models/colormlm-zerotrain-v0.clm --prompt "def max" --new-tokens 36
```

GPU 单次生成：

```powershell
python -m fast16.clm generate fast16/models/colormlm-zerotrain-v2.clm `
  --prompt "你好" `
  --device gpu `
  --gpu-graph fast16/models/colormlm-zerotrain-v2.onnx
```

## 记忆数据格式

记忆源是 UTF-8 JSONL，每行包含一个 Key 和 Value：

```json
{"key":"def max","value":"(a,b): return a if a>b else b"}
```

它不是训练样本。打包器用冻结 embedding 将 Key 编译为向量，把 Value 编译为模型内部 token memory。
