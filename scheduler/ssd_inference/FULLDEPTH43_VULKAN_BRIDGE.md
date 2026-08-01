# FullDepth43 -> RX 5700 XT Vulkan 单层桥

## 结论

2026-08-02 已把 `FullDepth43/native-top6` correctness executor 在真实 `0..42`
连续前向中产生的 L42 激活、原生 route 和 42 个 routed/shared payload，送入现有
Vulkan top6+shared 最小 MoE batch。设备为 `AMD Radeon RX 5700 XT`
（PCI `0x1002:0x731f`）。

这不是 43 层 GPU executor，也不能替换 correctness token commit。当前桥证明的是：

- 输入确实来自完成 `0..41` 后的实时 FullDepth43 L42，不是旧 fixture；
- route 为 `[26,23,123,28,154,152]`，与本次 CPU FullDepth report 一致；
- 42 个真实 payload 会在 Rust 侧逐文件重新计算 SHA-256；
- 现有 bounded Vulkan packed chain 在新输入尺度下仍通过数值检查。

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

## 为什么没有扩到 43 层

当前 Vulkan batch 的 reference semantics 是 F32 packed decode，routed weight 在 w2 输出后
累加。官方 FullDepth expert 则要求：

1. w1/w3 输出 BF16 边界；
2. route weight 在 w2 前乘到 hidden；
3. weighted hidden 先 BF16，再做 E4M3FN activation requant；
4. w2 输出 BF16 后才累加；shared 路径也有相同边界。

因此把当前 GPU 输出写回 FullDepth state 会破坏 correctness，不能用更快但不同语义的值
宣称降低 219.76 s/token。安全扩到 43 层还需要：官方 BF16/activation-quant shader、持久
Vulkan worker、route-first payload 到 `VramPool` 的逐层复用，以及 BF16 MoE branch 回传。
本提交止于单层可运行桥，并把这些缺口作为硬 claim limit。

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
