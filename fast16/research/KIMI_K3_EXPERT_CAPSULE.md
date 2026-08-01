# Kimi K3 MXFP4 专家胶囊

## 当前本地证据

- K3 Safetensors 索引已缓存：497,220 个张量、96 个分片，索引声明总大小约 1.420 TiB。
- MoE 专家层为 1--92，每层 896 个专家。
- 本地已有第 92 层的完整分片 header，含 896 个专家的六个 U8 张量：
  `w1/w2/w3.weight_packed` 和 `w1/w2/w3.weight_scale`。
- `w1 -> gate`、`w3 -> up`、`w2 -> down`。解码形状分别为
  `[3072,3584]`、`[3072,3584]`、`[3584,3072]`，方向是 PyTorch Linear 的
  `[out_features,in_features]`。
- 量化布局为 32 值一组，16 个 U8 交错存放 low/high nibble，共享一个
  E8M0 scale：`E2M1[nibble] * 2**(scale_u8 - 127)`。
- 官方 `modeling_kimi_linear.py` 已确认 K3 routed expert 不是直接工作在 7168 维：
  先由 `routed_expert_down_proj` 把 7168 维压到 3584 维，专家在
  `3584 -> 3072 -> 3584` 中运行，经 RMSNorm 后再由
  `routed_expert_up_proj` 回到 7168 维。
- 官方 SiTU 为
  `4*tanh(gate/4)*sigmoid(gate) * (25*tanh(up/25))`，不能按普通
  `SiLU(gate)*up` 接入。

## 每专家与外桥成本

- 专家 MXFP4 主 Range：17,547,264 字节，即 16.734375 MiB。
- 每次远端提取先读取 823,016 字节（0.784889 MiB）分片头，把本地 header
  的 SHA-256 和同一 ETag/Last-Modified 绑定到主 Range，避免 `master` 更新后用旧偏移读新分片。
- float16 三矩阵数据：66,060,288 字节，即 63 MiB；加 `.npy` 头后略大。
- float32 三矩阵数据：132,120,576 字节，即 126 MiB。
- 保留原始胶囊和 float16 矩阵时，每专家约 79.734375 MiB。
- 若取某一层全部 896 个专家，仅原始 MXFP4 就需约 14.643 GiB，不应在未做路由筛选时批量下载。
- 第 92 层共享 latent down/norm/up 主 Range 是 102,767,616 字节，即
  98.006836 MiB；只需提取一次。折叠后两个 float16 外桥和 norm 共 28.006836 MiB。
- 第一颗完整宏胶囊需约 114.755 MiB 原始权重（共享桥 + 专家 + 一行 router）；
  同层后续专家只增加约 16.75 MiB。

## 用法

`plan` 只读本地索引和 header，不会访问网络：

```powershell
python fast16/research/kimi_k3_expert_capsule.py plan --layer 92 --expert 0
```

`extract` 先校验远端 header，再使用一个可续传主 Range，保留原始 MXFP4，并解码成 float16 的
`gate.f16.npy`、`up.f16.npy`、`down.f16.npy`：

```powershell
python fast16/research/kimi_k3_expert_capsule.py extract --layer 92 --expert 0
```

只下载原始胶囊，之后再离线解码：

```powershell
python fast16/research/kimi_k3_expert_capsule.py extract --layer 92 --expert 0 --raw-only
python fast16/research/kimi_k3_expert_capsule.py decode --capsule-dir fast16/research/biopsy_cache/moonshotai_Kimi-K3/master/expert_capsules/layer-92/expert-000
```

零网络检查 nibble 顺序和 scale 公式：

```powershell
python fast16/research/kimi_k3_expert_capsule.py self-test
```

latent bridge 同样可先做零网络计划；获准后提取共享源，再离线折叠：

```powershell
python fast16/research/kimi_k3_latent_macro_capsule.py plan --layer 92
python fast16/research/kimi_k3_latent_macro_capsule.py extract --layer 92
python fast16/research/kimi_k3_latent_macro_capsule.py build --bridge-dir fast16/research/biopsy_cache/moonshotai_Kimi-K3/master/latent_bridges/layer-92
```

候选专家 291 的真实 router 行已用本地权重运输为 2048 维；可用同一命令生成其他专家：

```powershell
python fast16/research/kimi_k3_latent_macro_capsule.py router --layer 92 --expert 291 --output-dir fast16/research/neural_bus_capsules/kimi_k3_l92_e291/router
```

专家和外桥完成后，`assemble` 会物化运行时直接消费的六个无头小端 F16：

```powershell
python fast16/research/kimi_k3_latent_macro_capsule.py assemble `
  --expert-dir <decoded-expert-dir> `
  --bridge-dir <folded-bridge-dir> `
  --router-dir fast16/research/neural_bus_capsules/kimi_k3_l92_e291/router `
  --output-dir fast16/research/neural_bus_capsules/kimi_k3_l92_e291/runtime
```

运行时文件名和精确大小已锁定：`b_in.f16` 14,680,064、`gate.f16`
22,020,096、`up.f16` 22,020,096、`down.f16` 22,020,096、`norm.f16` 7,168、
`b_out.f16` 14,680,064 字节，合计 95,427,584 字节。

## 安全约束和剩余阻塞

- 服务端必须返回 HTTP 206 且 `Content-Range` 精确匹配；否则立即中止，避免误下载约 16 GiB 的整分片。
- 远端 header 和主 Range 必须共享强 ETag 或 Last-Modified；断点续传使用 `If-Range`。
  无验证器时直接拒绝，不会在正式胶囊中降级为猜测。
- 输出目录会用 `source-plan.json` 绑定 repo、revision、layer、expert 和 Range，拒绝把旧专家文件当成新专家复用。
- 目前只缓存了第 92 层专家分片的 header。其他层先用
  `remote_neural_biopsy.py inspect --shapes` 缓存对应 header，再运行本工具。
- 本地尚无任何 K3 专家的原始六张量，因此已验证布局、Range 和已知向量解码，
  但尚未对真实专家数值分布做检查。完成第一颗宏胶囊需获准约 115.54 MiB
  网络读取（含两次 header 同版校验），不需运行 K3 模型。
- 本工具产出的是“原始 K3 latent expert”，不是可直接注入 2048 维 ColorLM 的最终胶囊。
  完整宏胶囊还需要同层 gate、`routed_expert_down_proj`、
  `routed_expert_up_proj`、`routed_expert_norm`，并把现有 7168x2048 坐标运输折叠到
  两个外桥中。
- 第一颗 K3 宏胶囊不再先砍到 512 神经元。15--30GiB 最终预算允许保留完整
  3072 神经元专家；先验证完整函数路径，确认有效后再以速度为目标做神经元分页。
