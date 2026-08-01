# Polaris Native Sparse-Depth S14：本地数值参考原语

本目录推进首个本地 S14 token 之前的数值阻塞项，固定来源为
`deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b1d1fe23eabc4c98b48bffcef5a386062`。
源码路径、字节数、SHA-256 和行段见 `source_audit.json`。

已实现：

- MXFP4 E2M1 低/高 nibble 解包、UE8M0 scale、按输出行和 K-group 分块的 FP4 linear；
- 显式 bit-level F8_E4M3FN 解码、128×128 UE8M0 tile scale、分块 FP8 linear；
- 四流 HC pre/split/Sinkhorn/post 的官方 FP32 顺序；
- 共享 KV、index gather、attention sink 的稳定稀疏注意力；
- 未来 Vulkan shader 的机器可读 ABI 草案。
- 固定 revision 官方最终 `hc_head → RMSNorm → BF16-checkpoint head` 路径；head 按词表
  行分块提升到 FP32 计算，输出 FP32 logits，与官方 `ParallelHead` 一致。

`I8` expert weight 不会按有符号整数解释。固定样本的 header 用 I8 表示原始 byte，
每 byte 仍是两个 E2M1 nibble。scale 的 `0xFF` 与 E4M3 的 `0x7F/0xFF` 默认硬拒绝。

运行小型、零网络测试：

```powershell
$env:PYTHONUTF8='1'
python -X utf8 -m pytest -q fast16/research/polaris_meridian_v1/local_s14_primitives/tests
python -X utf8 fast16/research/polaris_meridian_v1/local_s14_primitives/selftest.py
```

外部真实样本测试会查找 `POLARIS_S14_EXPERT_ABI_SAMPLE`、
`POLARIS_S14_FP8_HC_ABI_SAMPLE`，其次尝试已知的 `D:/models/Polaris-S14/abi_samples/...`；
路径不存在时 skip。样本和权重不进入 Git，默认 CI 不依赖它们。

这些实现是小规模 CPU/PyTorch 语义 oracle，不是优化 runtime。通过测试不表示 S14 已经
forward、不表示首 token 已产生，也不证明速度或质量。

## 最终头的 dtype 边界

官方源码明确写明 checkpoint 的 `head.weight` 是 BF16，但推理类把参数保存为 FP32，且执行
`F.linear(x.float(), self.weight)`，所以官方 logits 是 FP32。`final_head.py` 接收 BF16
checkpoint 权重并按行分块转 FP32，不使用 BF16 matmul 改写数值语义。公式、源码行号与 SHA
见 `final_head_source_audit.json`。不同词表分块可能触发不同 GEMM kernel，测试采用严格 FP32
容差，不宣称跨 kernel 逐 bit 一致。生产入口默认硬校验 `[B,S,4,4096]` hidden 和
`[129280,4096]` 单卡完整 head；小形状只能显式作为测试使用。
