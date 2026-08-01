# L42 标准 packed-FP8 Vulkan exact 闭环

日期：2026-08-02

## 结论

北极星已在 RX 5700 XT 上完成五条标准 FullDepth43 packed-FP8 attention 投影与一条 grouped
`wo_a` 投影闭环。最先闭合的 `wq_a` 使用如下持久执行路径，其余标准投影复用同一个可变 shape、
严格 SHA 的 GPU slot；`wo_a` 使用 8 组独立 BF16-weight matvec：

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

两个输出逐位完全一致。随后 `wkv/wq_b/indexer.wq_b/wo_b` 也全部命中各自冻结输出 SHA；五条
合计 46,592 个 BF16 元素逐位一致。这证明标准 attention 投影已经不必在 CPU 展开完整 F32
权重。六条结果已回放到完整真实 L42，最终层输出仍精确不变；尚未推广到43层完整token热路径。

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

- Rust example tests：`18/18`。
- FullDepth43 Python tests：`51 passed, 2 subtests passed`，其中 packed-FP8 client 独立 `6/6`。
- 隔离 fresh release build：通过。
- 真实 GPU：连续 epoch `0/1` 均通过，输出 shape `[1,1,1024]`，两次逐位一致。
- 同一持久 worker 连续20次完整 Python→arena→Rust→Vulkan→arena→Python 往返全部通过
  输出SHA门；平均`3.0521 ms`、中位`2.9935 ms`、范围`2.7673--3.8129 ms`。这是完整协议
  往返时间，不是孤立kernel时间，也不能直接乘投影数外推整token速度。

## Grouped `wo_a` 与完整 L42 回接

专用内核直接消费 `[8192,4096]` packed E4M3 weight、`[64,32]` UE8M0 scale 和
`[8,4096]` BF16-carrying F32 输入。每组只读取自己的 1024 行；解码权重先经过 BF16 RNE，
K=4096 exact 归约固定为 `0,1,3,4,6,2,5,7`，最终输出再经过 BF16 RNE。

- RX 5700 XT：`8192/8192` 个输出逐位一致。
- 输出 SHA-256：`2be0aa3b4b67aae58f62a77d2a255d6240b5baf3d71f37c9084fd890741d2eb9`。
- 100 次短计时平均 kernel `4.1069832 ms`；整套墙钟 `841.8368 ms`。
- Rust example 测试：`20/20`。

`verify_l42_attention_replay.py` 会在完整 L42 的每个实际调用点重新核对投影输入 SHA，再替换为
GPU 已逐位证明的输出。实际执行的 `wq_a/wq_b/wkv/wo_a/wo_b` 全部命中，最终 L42 输出仍为：

```text
853b8b947a3f7a275cf748d7e97a311ebb22323cd0c2f3e5e973f27b04388895
```

`indexer.wq_b` 已在同一标准套件独立闭合，但 L42 position0 参考不会调用 indexer，因此没有伪装成
该次完整层回放中的已执行节点。

## 边界与下一步

当前证明范围是 L42 六条 attention 投影的真实 GPU 数值闭合，以及它们在冻结完整 L42 轨迹上的
组合等价。最终 BF16 RNE仍在CPU完成，43层生产执行器尚未消费这些内核；不能据此宣称完整GPU
token或端到端加速已经完成。下一步是把同一 packed 权重路径推广到43层，再跑一次两-token A/B。

## 标准投影 fixture 与真实 GPU 结果

现已用同一完整 L42 CPU参考运行一次性冻结其余标准 packed-FP8 投影，且完整层输出仍命中
`853b8b947a3f7a275cf748d7e97a311ebb22323cd0c2f3e5e973f27b04388895`：

| 投影 | N×K | BF16元素 | GPU执行/回读/舍入/验证 | BF16输出 SHA-256 |
|---|---:|---:|---:|---|
| `wq_a` | `1024×4096` | 1,024 | `1.1247 ms` | `76469fd163f5db49de956eff9b29087afa4caa97d566be80bab9d9119facb0b8` |
| `wkv` | `512×4096` | 512 | `0.6143 ms` | `3cc7f8f4264c6448dd32f9044c0d001107f06d57209a91a80fa56bdda59dd541` |
| `wq_b` | `32768×1024` | 32,768 | `9.7557 ms` | `284391a5a45d6a5367060ecd444a21770e69fa7949455bea6823317f4fb43c04` |
| `indexer.wq_b` | `8192×1024` | 8,192 | `2.9623 ms` | `d9adda7639665267be4fac36e2a74755bb5d730a4a2a8734695198fc4f331501` |
| `wo_b` | `4096×8192` | 4,096 | `9.3225 ms` | `84ce63ca9233b07bea99741f9982accac17bc65025b0098b7017acd7dab6db10` |

`capture_l42_fp8_projections.py` 会从76个SHA校验后的本地资产重新生成五条投影的输入/输出
二进制和严格manifest；目标目录必须不存在，fixture不作为模型权重提交。

首次泛化运行在 `wq_b` 的32,768个元素中捕获了1个BF16中点漂移。原因不是权重或索引错误，而是
exact shader把 `K=4096` 的 OpenBLAS Haswell 归约顺序套给了 `K=1024`。审计内核现按 K 固定
归约：`K=1024` 使用 `0,1,2,3,4,5,7,6`，宽投影保留原顺序；重新运行后五条逐元素与SHA均精确
通过，未使用容差或单元素补丁。上述时间来自一次correctness suite，只用于数量级观测，不作为稳定
性能基准。
