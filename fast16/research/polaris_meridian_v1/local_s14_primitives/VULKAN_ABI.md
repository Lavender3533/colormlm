# Polaris S14 Vulkan shader ABI v1

这是未来 RX 5700 XT Vulkan compute port 的 host/device 合同草案，不包含 shader，也不宣称性能。
机器可读版本见 `vulkan_abi_v1.json`。

## 固定字节语义

- 所有 buffer 为 little-endian、row-major；offset/stride 一律以字节计。
- host 必须先验证 `offset + 最大逻辑访问偏移 + element_bytes <= buffer_bytes`。
- SSBO dynamic offset 对齐为 `max(64, minStorageBufferOffsetAlignment)`。
- shader 从 `uint32` word 提取 8-bit/4-bit 值，不要求设备开启 8-bit storage。
- 越界不能 clamp 到别的元素。host 先拒绝，shader 仍做防御检查并在 status buffer 记录首个错误。

真实 expert safetensors header 把 packed weight 标为 `I8`，这只是物理字节容器：
逻辑 `[N,K]` 的偶数 K 位于低 nibble，奇数 K 位于高 nibble。scale 是 `[N,K/32]`
的 F8_E8M0。`0xFF` 是 NaN，dispatch 前必须拒绝。

非 expert/attention linear 的权重为 `[N,K]` F8_E4M3，权重 scale 是
`[ceil(N/128),ceil(K/128)]` F8_E8M0；activation scale 是
`[M,ceil(K/128)]`。E4M3 的 `0x7F/0xFF` 为 NaN。两类 GEMM 均使用 F32
scale-corrected 累加，最终一次舍入到输出 dtype。

## HC 与 sparse attention

HC 固定四流，`mixes=[B,T,24]`。组合矩阵执行稳定 row-softmax+eps、一次列归一，
随后 19 轮行/列归一。`comb[source,target]`，HC post 必须按 source 轴求和。

稀疏注意力共享 `kv=[B,N,D]`。合法 index 只有 `-1` padding 或 `[0,N)`；重复位置
重复计权。sink 是零 value 的虚拟 token，只进入 softmax 分母。CPU 参考把全 padding 行
稳定扩展为零输出；官方 GPU kernel 对该非生成态输入没有同样保证。
