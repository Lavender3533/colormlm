# FullDepth43 / native-top6 reference

这是 DeepSeek-V4-Flash-0731 固定 revision 的完整 43 层 CPU/PyTorch
correctness 入口，不是 S14 跳层模式，也不是 top-1 近似。

硬合同：

- 层集合必须精确为 `0..42`，不允许 identity skip。
- 每层必须先执行 attention 与官方 router，再读取当前 token 的
  真实 top-6 expert 页。
- 每层都执行 top-6 routed experts + shared expert + mHC post。
- 连续 token 保留全 43 层 window KV 和 ratio-4/128 compressor remainder。
- forced-prefill 队列未耗尽时，下一个输入必须取官方 `token_ids[position]`；
  队列耗尽后才改用原生 argmax。
- 只有 43 层全部完成后，原生 HC/norm/BF16 head argmax 才能提交 token。
- 任何缺页、顺序、状态或预算错误都 fail closed，token/cursor/43 层 state
  一起原子回滚，不会伪造 token。

## 当前真实状态

本机已有完整官方 index 和 45 个固定 revision header，因此已在
`D:/models/Polaris-S14/fulldepth43_native_top6_catalog.json` 生成了不含
payload 的全 43 层 catalog。

```text
catalog ranges                 67,612
static prerequisite ranges     1,564
static candidate-ready         1,564 / 7,786,905,820 bytes
static missing                     0 /             0 bytes
one-token native top-6 cold           3,449,290,752 bytes
current cold upper bound               3,449,290,752 bytes
```

当前 preflight 为 `ready`，静态缺页已归零。2026-08-02 的首次真实执行
已通过全部 `0..42` 层，每层记录 6 个唯一原生路由专家，最终
HC/norm/BF16 head 提交了真实 `token_id=5`。机器可读证据在
`first_real_token_report.json`；该结果只证明 correctness，不是速度或质量声明。

重建 catalog 并跑离线 preflight（不下载权重）：

```powershell
python -X utf8 -m fast16.research.polaris_meridian_v1.fulldepth43_native_top6.preflight `
  --rebuild-catalog `
  --asset-root D:/models/Polaris-S14 `
  --catalog D:/models/Polaris-S14/fulldepth43_native_top6_catalog.json
```

显式运行入口（已缓存路由页命中时可直接运行）：

```powershell
python -X utf8 -m fast16.research.polaris_meridian_v1.fulldepth43_native_top6.executor run
```

首个官方聊天预览使用仓内 `first_preview_forced_prefill.json` 的 5 个 token：

```powershell
python -X utf8 -m fast16.research.polaris_meridian_v1.fulldepth43_native_top6.executor run `
  --forced-prefill `
  --token-count 5
```

`--forced-prefill` 不带路径时固定使用上述仓内产物，也可显式指定其他通过
S14 format/revision/vocab/BOS/token-hash/decoder-consumption 验收的产物。中间四次
argmax 只记录为反事实输出，不会覆盖 forced 队列的下一输入。

新 token 若命中未缓存专家页，要允许补页，必须同时显式传入
`--download-missing` 和不小于
preflight cold upper bound 的 `--download-budget-bytes`。catalog 本身永远保持
`download_authorized=false`。

`preflight_report.json` 只记录本机当前缺口；真实执行证据以
`first_real_token_report.json` 为准。

## Vulkan 单层桥

executor 可在真实 FullDepth 前向经过指定层时，固化已完成层前缀、FFN 激活、原生
top-6 route 和 42 个带 SHA proof 的 routed/shared payload：

```powershell
python -X utf8 -m fast16.research.polaris_meridian_v1.fulldepth43_native_top6.executor run `
  --vulkan-bridge-capture <fresh-dir> `
  --vulkan-bridge-layer 42
```

当前只允许单 token，capture 目录必须不存在。该入口不会改变 CPU correctness 计算；
GPU 结果不会写回 token state。真实 RX 5700 XT 数值与速度记录见
`scheduler/ssd_inference/FULLDEPTH43_VULKAN_BRIDGE.md`。
