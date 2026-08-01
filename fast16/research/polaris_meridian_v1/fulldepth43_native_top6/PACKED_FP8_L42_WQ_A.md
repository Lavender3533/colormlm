# L42 `wq_a` packed-FP8 Vulkan exact 闭环

日期：2026-08-02

## 结论

北极星已在 RX 5700 XT 上完成第一条可复用的 FullDepth43 packed-FP8 attention 投影闭环：

```text
CPU F32 activation [1,1,4096]
  → 共享 binary arena
  → 持久 Vulkan worker
  → 常驻 GPU 的 E4M3 weight + UE8M0 scale
  → exact-audit FP8 matvec
  → F32 readback + CPU BF16 RNE
  → F32 storage carrying exact BF16 values [1,1,1024]
```

真实 L42 `layers.42.attn.wq_a` 连续执行 arena epoch 0 和 1，两次输出均命中冻结 SHA-256：

```text
76469fd163f5db49de956eff9b29087afa4caa97d566be80bab9d9119facb0b8
```

两个输出逐位完全一致。这是第一条“不在 CPU 展开完整 F32 权重”的 attention 投影生产原语，但尚未接入 43 层完整 token 热路径。

## 冻结身份

| 对象 | 形状 / 字节 | SHA-256 |
|---|---:|---|
| input | `[1,1,4096]` F32 / 16,384 B | `47156935b19ca5483f0e92d2284eaa6a9417686978dc4b41ca893ee162f37577` |
| weight | `[1024,4096]` E4M3 / 4,194,304 B | `1efcea39938dfadc143c41813bc32327a9bb5369b2b612feac76d9dfb8001ce7` |
| scale | `[8,32]` UE8M0 / 256 B | `dfb4085717aa527f8affa5a1640c5f806867c5ba6e0301d170f387be8b6660cf` |
| output | `[1,1,1024]` BF16-rounded F32 / 4,096 B | `76469fd163f5db49de956eff9b29087afa4caa97d566be80bab9d9119facb0b8` |

模型 revision 固定为 `7872f01b1d1fe23eabc4c98b48bffcef5a386062`，FullDepth43 catalog SHA-256 固定为 `ca619984d4a46ad1a3701d2b4035766ea40c3a3dbedd3a474ce1df7aad4d0049`。

## 持久资源与每请求流量

- worker 启动时验证并上传一次 weight + scale：`4,194,560 B`。
- Vulkan context、descriptor、command buffer、fence、weight 和 scale VRAM 跨请求存活。
- 每次请求只上传 `16,384 B` activation，并回读 `4,096 B` 输出。
- Python 与 Rust 只通过 JSONL 传控制信息；tensor 通过同一 canonical `.bin` arena 传输。
- arena input/output 必须 4-byte 对齐、不重叠、不越界，epoch 必须严格单调。

## Fail-closed 门

- Rust request 使用 `deny_unknown_fields`；protocol、revision、profile、layer、position、投影形状、activation contract、BF16舍入合同和所有 SHA 均固定。
- worker 只接受 RX 5700 XT `0x1002:0x731f`，不允许换设备后静默扩大结论。
- 每次请求前 Python 完整 poison 输出区；返回后验证全区 SHA、BF16 低 16 位全 0、shape 和有限值。
- 任一 JSON、路径、offset、epoch、SHA、shape、NaN、半写或超时漂移都会 poison 并终止 worker。

## 验证

- Rust example tests：`16/16`。
- FullDepth43 Python tests：`51 passed, 2 subtests passed`，其中 packed-FP8 client 独立 `6/6`。
- 隔离 fresh release build：通过。
- 真实 GPU：连续 epoch `0/1` 均通过，输出 shape `[1,1,1024]`，两次逐位一致。
- 同一持久 worker 连续20次完整 Python→arena→Rust→Vulkan→arena→Python 往返全部通过
  输出SHA门；平均`3.0521 ms`、中位`2.9935 ms`、范围`2.7673--3.8129 ms`。这是完整协议
  往返时间，不是孤立kernel时间，也不能直接乘投影数外推整token速度。

## 边界与下一步

当前仅闭合 L42 `wq_a`，而且最终 BF16 RNE 仍在 CPU 完成；不能据此宣称完整 attention、完整 GPU token 或端到端加速已经完成。下一步按冻结顺序扩展 L42 `wkv/wq_b/indexer.wq_b/wo_b`，再为 `wo_a [8,1024,4096]` 实现 grouped BF16-weight 专用内核；完整 L42 对齐后才推广到 43 层并跑一次两-token A/B。
