# Polaris S14 RX 5700 XT Vulkan 数值线交接

## 结论

**已达到有边界的真实 GPU parity**：在 AMD Radeon RX 5700 XT 上，使用当前冻结的
DeepSeek-V4 L42 Range payload、真实 route `126,12,205,149,227,174` 与真实 L42
前向导出的 kernel 输入，跑通了：

- 真实 top-6 routed + shared 的 GPU-resident 最小 MoE batch：同一输入顺序执行六个
  MXFP4 expert，清零 accumulator 后按真实 route weight 累加，再执行 shared FP8 expert
  并无权重累加；
- 真实 E126 packed MXFP4 的 GPU-resident 最小整链：
  `w1/w3 -> clamp-SwiGLU(limit=10) -> w2 -> route-weight mix`；
- 真实 `layers.42.attn.wq_a` packed FP8 linear；
- 两者均相对独立 F32 CPU 解码/累加 reference 做误差检查，并用 Vulkan timestamp 测量。

**未达到完整官方 expert/layer parity**：最小整链在 w1/w3 与 w2 之间保持 F32，尚未加入
官方 `_linear_fp4` 的 BF16 舍入与下一次 UE8M0/E4M3FN activation requantization。
也没有跑完整 L42、43 层 S14 首 token 或 token/s。不得扩大宣称。

## RX 5700 XT 实测

设备：`AMD Radeon RX 5700 XT`，PCI ID `0x1002:0x731f`，AMD proprietary driver
`26.7.1`，timestamp period `10 ns`。

| 路径 | dispatch 范围 | 均值 | max abs | RMSE |
|---|---|---:|---:|---:|
| L42 top-6 routed + shared 最小 MoE batch | accumulator clear + 35 个顺序 dispatch 与合法 barrier，1 次；不含上传/回读 | 1.3696000 ms | 1.096725464e-5 | 1.426471034e-6 |
| L42/E126 最小整链 | 5 个 dispatch + 合法 inter-dispatch barrier，100 次；不含上传/回读 | 0.1570380 ms | 1.072883606e-6 | 1.483723944e-7 |
| L42 `attn.wq_a` FP8 | 单 dispatch + 重复写 barrier，100 次；不含上传/回读 | 0.0838004 ms | 5.960464478e-8 | 7.205182117e-9 |

E126 route weight 为 `0.2747795581817627`。完整结构化记录在
`evidence/s14_vulkan_numeric_rx5700xt.json`。

## 真实资产与 SHA

L42 CPU capture 在执行前验证 76 个 payload、共 247,515,224 字节。关键来源：

- route manifest：`feccc1b5dde256c9ad750985b4ab5446732a1f4deec796976198e95798c64b86`
- 冻结原始 `ffn_input`：`7e2d3167e3782eca8d762c3cc92d53bb9d64a65c7b18d37d16797ff39f611ad4`
- E126 w1/w3 kernel 输入：`1a2006c1b79b31bc3db3540ef730b08547c6148e224074b0ba712e2c3b9d7c9f`
- `wq_a` kernel 输入：`47156935b19ca5483f0e92d2284eaa6a9417686978dc4b41ca893ee162f37577`
- E126 w1/w3/w2 weight：
  `2d3fc95f...285faa` / `a7cf5a4e...fc3ad` / `dfaad185...e766d`
- `wq_a` weight：`1efcea39...01ce7`

完整 weight/scale SHA 不截断地保存在 evidence JSON。runner 会重新计算，而不是只打印
manifest 自报值。

## RouteLoadBatch 缓存与 fence 合同

- `VramPool` slot 带单调 generation 与 `pin_count`；S14 descriptor metadata 必须携带并
  校验 generation，stale generation 会硬拒绝。
- `Loading` 与 pinned `Ready` 均不可淘汰；compute lease 只能在覆盖 dispatch 的 fence
  完成后释放，因此 fence 前 slot 不可复用。
- 六个 routed 页与独立 shared FP8 页组成一个 `RouteLoadBatch`。两个 pool 全部通过
  generation preflight 后才一次性 publish；容量、上传或 fence 失败会取消本批全部
  reservation，并释放已取得的 cached pins。
- routed slot 固定至少 `13,369,344 B`；shared slot 固定至少 `25,167,360 B`，二者使用
  独立 pool，避免 shared 页挤占 routed cache。

## 资产审计与路径选择

- `ssd_inference` 已有 ash device、device-local/staging buffer、descriptor、command buffer、
  fence 与 GLSL build 基础，复用后只需补 packed 数值核和最小整链调度。
- 现有 ggml bridge 仅实现 GGUF `Q4_K x Q8_K` AVX2 CPU 路径；其 block layout 与
  L42 的 E2M1+UE8M0、E4M3FN+UE8M0 ABI 不兼容，因此没有拿它伪装 L42 reference。
- 现有通用 SwiGLU 不含 DeepSeek limit-10，故新增 S14 专用 shader；route mix 也用独立
  shader，使链内中间结果保持 GPU resident。

## 复现

从仓库根目录执行，capture 目录必须尚不存在：

```powershell
$capture = Join-Path $env:TEMP "polaris-s14-l42-vulkan-run"
python -X utf8 fast16/research/polaris_meridian_v1/l42_real_reference/l42_reference.py --capture-dir $capture
$env:POLARIS_S14_L42_CAPTURE_DIR = $capture
Push-Location scheduler
cargo run --release --offline -p ssd_inference --example s14_vulkan_numeric
Pop-Location
```

该流程只读 `D:/models/Polaris-S14` 的现有资产；不下载权重，不启停模型服务。

## 硬拒绝合同

runner 会拒绝：capture 环境变量缺失；capture/manifest revision、shape、SHA 漂移；
payload 越出冻结 `range_cache`；payload 长度/SHA 漂移；量化 NaN code；设备 PCI ID
不是 `0x1002:0x731f`；top-6 ID/weight 或 shared manifest 漂移；timestamp 不可用；或
误差超过阈值。不存在合成输入回退。

## 后续最短路径

top-6 accumulation 与 FP8 shared expert 的最小 GPU batch 已完成。若要把“最小整链
parity”升级为“完整官方 MoE parity”，下一步应在 GPU 链中加入
BF16 boundary 与 activation requantization，并用 capture 中冻结的
`expert_126_w2` 输入/输出作为中间指纹。完整 L42/token 路径仍需 attention、HC 和
router 等未实现能力。
