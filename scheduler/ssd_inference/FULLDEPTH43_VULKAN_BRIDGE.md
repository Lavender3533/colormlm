# FullDepth43 -> RX 5700 XT Vulkan 单层桥

## 结论

2026-08-02 首先把 `FullDepth43/native-top6` correctness executor 在真实 `0..42`
连续前向中产生的 L42 激活、原生 route 和 42 个 routed/shared payload，送入现有
Vulkan top6+shared 最小 MoE batch。设备为 `AMD Radeon RX 5700 XT`
（PCI `0x1002:0x731f`）。

同日的第二步已补齐官方 BF16/重量化边界，并把单层 GPU MoE branch
真实写回 Python FullDepth state。这仍不是 43 层 GPU executor。当前证明：

- 输入确实来自完成 `0..41` 后的实时 FullDepth43 L42，不是旧 fixture；
- route 为 `[26,23,123,28,154,152]`，与本次 CPU FullDepth report 一致；
- 42 个真实 payload 会在 Rust 侧逐文件重新计算 SHA-256；
- 现有 bounded Vulkan packed chain 在新输入尺度下仍通过数值检查。
- 新 official-boundary graph 的 4096 个 BF16 输出与 Python CPU 官方路径逐位相等；
- executor 将 worker 返回的 tensor 本身传入 `hc_post`，不是只写一份旁路报告。

## 真实单层回写结果

| 项目 | RX 5700 XT 实测 |
|---|---:|
| official-boundary GPU 核 | `1.85204 ms` |
| worker 请求墙钟（读 105 MB + upload + pipeline bind + 回读） | `432.9413 ms` |
| CPU 官方边界参考 | `4.1784083 s` |
| BF16 对照元素 | `4096` |
| max abs / RMSE | `0 / 0` |
| CPU/GPU F32 视图 SHA-256 | `78078068c5e5a2e21141b6d459d8c26dfe209fb3d6f2ee36e6111f81280bf868` |

机器可读证据：
`evidence/fulldepth43_vulkan_writeback_rx5700xt.json`。

## 实测

FullDepth43 CPU/PyTorch 单 token 仍提交 `token_id=5`：

| 项目 | 实测 |
|---|---:|
| executor 内部总时长 | 212.585266 s |
| 命令墙钟 | 219.4 s |
| L42 完整 CPU reference 层时长 | 5.232330 s |
| L42 bridge 原始 FFN 输入 SHA-256 | `3709794755053a3e5f4e6f1f5fbe43c5b6fb9e0468a00fd2f0e60deae750766a` |
| activation-quant 后 GPU 输入 SHA-256 | `94ecacbfa5e09411c22e296c7e8e5096ae51cc0d5a1d3aef4959d394a9505e84` |

同一个实时 L42 route 在 RX 5700 XT 上：

| 项目 | 实测 |
|---|---:|
| 同语义 Rust CPU packed reference | 249.575100 ms |
| GPU 测量次数 | 20 |
| Vulkan fill + 35 dispatch + barriers 均值 | 1.422014 ms / 次 |
| 20 次 submit + readback sync 总计 | 28.713000 ms |
| 一次性 bridge 墙钟（读 105 MB、上传、建 pipeline、执行 20 次、回读） | 714.817200 ms |
| max abs / RMSE | 0.11328125 / 0.0032810247 |
| reference max abs / reference RMSE | 18711.113 / 1866.8712732 |
| relative RMSE | 1.7574992e-6 |
| CPU reference 输出 SHA-256 | `88e2c57d05de93102f8f9353ddbfb9f017c4b2a36d7c025d8ad7e218fade4efa` |
| GPU 输出 SHA-256 | `7b5f17d25048c3022ac327ce1c1eacbcf47557856cbe953bd1e616451faf58cd` |

kernel 对同语义 CPU reference 是约 `175.5x`，但当前一次性 bridge 墙钟仍是
CPU packed reference 的约 `2.86x`；必须复用 device/pipeline/VRAM payload，不能逐层启动进程。

## 为什么仍没有扩到 43 层

旧 Vulkan batch 的 reference semantics 是 F32 packed decode，routed weight 在 w2 输出后
累加。新 graph 已逐项实现官方 FullDepth expert 的：

1. w1/w3 输出 BF16 边界；
2. route weight 在 w2 前乘到 hidden；
3. weighted hidden 先 BF16，再做 E4M3FN activation requant；
4. w2 输出 BF16 后才累加；shared 路径也有相同边界。

官方边界、持久 Vulkan worker 和 BF16 branch 回传已完成；但现在每个请求仍重新读取/
upload 当层的 105 MB payload，而且验证模式仍重算 CPU reference。扩到 43 层需要
route-first payload -> `VramPool` 的跨层/跨 token generation-safe 驻留，再将已验证的
逐位 CPU 参考从生产热路径移到抽检门。因此本结果不宣称 219.76 s/token 已降低。

## 复现

先生成一个新的 capture（目录必须不存在；不加 `--download-missing`，不会联网补权重）：

```powershell
$capture = Join-Path $env:TEMP 'polaris-fulldepth43-vulkan-bridge'
python -X utf8 -m fast16.research.polaris_meridian_v1.fulldepth43_native_top6.executor run `
  --report fast16/research/polaris_meridian_v1/fulldepth43_native_top6/vulkan_bridge_full_run_report.json `
  --vulkan-bridge-capture $capture `
  --vulkan-bridge-layer 42
```

再执行 GPU：

```powershell
$env:POLARIS_FULLDEPTH43_VULKAN_BRIDGE_DIR = $capture
$env:POLARIS_FULLDEPTH43_VULKAN_EVIDENCE = `
  'scheduler/ssd_inference/evidence/fulldepth43_vulkan_bridge_rx5700xt.json'
Push-Location scheduler
cargo run --release --offline -p ssd_inference --example s14_vulkan_numeric
Pop-Location
```

仓库已固化本次 16 KiB 输入和 bridge manifest，可把第一条环境变量直接指向
`scheduler/ssd_inference/evidence/fulldepth43_vulkan_bridge_capture` 复跑 GPU。

复跑 official-boundary 逐位对照：

```powershell
Push-Location scheduler
cargo build --release --offline -p ssd_inference --example s14_vulkan_numeric
Pop-Location
python -X utf8 -m fast16.research.polaris_meridian_v1.fulldepth43_native_top6.verify_vulkan_writeback `
  --worker scheduler/target/release/examples/s14_vulkan_numeric.exe
```
