# FullDepth43 shared-only GPU 常驻负门

## 假设

每 token 的43层 shared experts 总计 `1,082,196,480 B`（约1.008GiB），而且跨 token
必然复用。通用 GPU LRU 会被3.212GiB/token的 routed experts 顺序冲刷，所以本实验只允许
shared experts 进入2GiB严格 SHA 身份缓存；routed experts 继续使用原100.5MiB固定上传槽。

显式入口：

```powershell
$env:POLARIS_SHARED_GPU_PAYLOAD_CACHE_GIB = '2'
```

默认值为0；只接受0或2，且与旧通用 `POLARIS_GPU_PAYLOAD_CACHE_GIB` 互斥。

## 机制结果

- 第一 token：43 miss，上传 `1,082,196,480 B`。
- 第二 token：43/43 hit，0 miss，0 B shared上传。
- 43个shared身份占用 `1,082,196,480 B`，0 eviction。
- routed固定槽每层上传从 `105,399,808 B` 降为 `80,232,448 B`。
- 输出仍为 `[5,223]`；86/86 Vulkan MoE、0 fallback、所有权账本闭合。

因此“选择性常驻”机制本身成立，并且没有再次犯通用LRU顺序抖动的问题。

## 完整速度门

| 运行 | 完整两 token | 相对候选 |
| --- | ---: | ---: |
| 前置关闭基线 | 54.6110s | 候选快3.91% |
| shared-only候选 | 52.4784s | — |
| 后置关闭基线 | 50.8944s | 候选慢3.11% |

正反方向再次不一致。虽然第二 token 物理上少上传约1.08GB，保留43组独立Vulkan buffers、
descriptor绑定和当前多进程/逐层控制开销抵消了收益。该实验不晋级默认路径，也不能包装为
第三速度里程碑。

## 架构结论

当前两-token profile约49.02s，其中 attention exclusive约18.00s、Python→Rust MoE约8.89s、
Range routed获取约5.97s、capture I/O约3.82s、Range proof index约3.93s。继续增加小缓存或
控制文件优化无法接近20–50 token/s。

下一条主线必须把43层attention、router、payload调度和MoE迁入单一持久Rust/Vulkan执行块，
并预留K-token speculative verification入口，用一次权重驻留/读取服务多个候选token。

机器可读证据见 `FULLDEPTH43_SHARED_GPU_CACHE_AB.json`。这是机制通过、速度未晋级的负门。
