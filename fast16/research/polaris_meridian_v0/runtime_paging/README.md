# 北极星全深度经络：专家分页 v0

本目录是一个不依赖训练、也不依赖完整模型加载的最小运行时原型：从 Safetensors/GGUF
头部建立专家到文件字节区间的目录，再用有界 RAM LRU、异步预取和 single-flight 读取
300B+ donor 的专家页。它不会把“小模型”当作北极星目标，也不声明已经获得 Kimi/DeepSeek
能力。

## 已实现

- Safetensors：识别 `layers.{L}....experts.{E}.*` 的独立专家张量；适配 Kimi K3 与
  DeepSeek V4 一类“每专家独立命名”的分片。
- GGUF：识别 `blk.{L}.ffn_{gate,up,down}_exps.weight`，将连续 packed bank 精确等分成
  单专家页。
- 同一专家的物理连续组件自动合并。Kimi K3 的 packed/scale 六组件若连续，只需一次读取。
- 默认要求所有专家的组件签名一致；单个分片若只有部分 `w1/w2/w3` 会拒绝生成运行时
  目录，防止残缺专家静默进入槽池。`--allow-incomplete` 只允许用于头部取证。
- RAM 热缓存按**字节容量** LRU，而不是按专家个数；可容纳不同 donor 的不同页尺寸。
- 相同页并发 miss 使用 single-flight，避免重复 SSD I/O。
- 多 worker 预取使用线程独立的持久文件句柄，避免共享 `seek` 锁把并发读串行化。
- 热命中返回同一个不可变 `bytes` 对象，不复制 17 MiB 级 payload。

## 快速自检

```powershell
python fast16/research/polaris_meridian_v0/runtime_paging/selftest.py
```

脚本创建临时合成 Safetensors，验证目录边界、连续区间合并、single-flight、LRU 容量与
热命中，并写出 `selftest_report.json`。默认只产生 64 MiB 临时数据，结束自动删除；它不
启动模型、不下载权重。

## 建立真实目录

Kimi/DeepSeek Safetensors 分片：

```powershell
python fast16/research/polaris_meridian_v0/runtime_paging/page_catalog.py `
  --format safetensors `
  --input D:\models\Kimi-K3\model-00093-of-000096.safetensors `
  --donor kimi-k3 `
  --output D:\models\Kimi-K3\polaris-pages.json
```

GGUF（可传单个分片或目录；专家数可显式固定）：

```powershell
python fast16/research/polaris_meridian_v0/runtime_paging/page_catalog.py `
  --format gguf `
  --input D:\models\donor.gguf `
  --donor donor-name `
  --experts 256 `
  --output D:\models\donor-pages.json
```

调用端最小接口：

```python
from pathlib import Path
from page_cache import ExpertPageCache

with ExpertPageCache(Path("polaris-pages.json"), 8 * 1024**3, workers=4) as cache:
    cache.prefetch(["kimi-k3:L00092:E00291"])
    payload = cache.get("kimi-k3:L00092:E00291")
```

## 与推理主线的接口

这个原型负责 `PageKey -> immutable bytes`。接入 `llama.cpp` 时应复用现有
`llama-slot-pool.*` 的 GPU 张量槽和 logical-to-slot LUT，但必须把页的 ready 状态放在
LUT 可见之前。推荐顺序：

1. 本目录目录器生成每层专家精确 span；
2. C++ 读取线程把 span 读入 page-aligned/pinned staging；
3. Vulkan transfer queue 上传到空闲槽并发 fence；
4. fence 完成后原子发布 `logical expert -> slot`；
5. 预测失败时等待正确页，绝不回落到其他专家。

首个速度门不是“目录能打开”，而是固定贪心输出与原模型一致、可观测 SSD/PCIe stall，
并在真实路由轨迹下接近 20 tok/s。50 tok/s 只有在活跃计算本身足够小且缓存命中极高时
才可能，不能由 SSD 吞吐测试直接推出。

已有短报告：

- `selftest_report.json`：64 MiB 合成 Safetensors 的目录、并发和 LRU 自检；
- `real_gguf_cache_report.json`：现有 13.73 GB v6 GGUF 上跨 40 层的真实随机页读取；
- `speed_budget_report.json`：把实测页吞吐换算为 Kimi K3 原生路径的 miss/显存带宽下限。
