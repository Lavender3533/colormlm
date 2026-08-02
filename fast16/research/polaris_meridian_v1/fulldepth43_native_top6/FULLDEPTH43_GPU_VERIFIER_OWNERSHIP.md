# FullDepth43 GPU 单一验证者里程碑

## 结论

GPU 专属 Range payload 不再先由 Python 完整 SHA、再由 Rust 重复读取和 SHA。开启
`--range-gpu-verifier-ownership`后，Python 只验证缓存路径、metadata、字节数和既有 SHA 身份；
Rust Vulkan worker 在对应 GPU 计算前完成唯一一次内容校验。真实相邻两-token A/B：

| 指标 | 基线 | 候选 | 变化 |
| --- | ---: | ---: | ---: |
| 完整 execution | 60.4563s | 58.4415s | -2.0149s / -3.33% |
| 有效吞吐 | 0.03308 token/s | 0.03422 token/s | +3.45% |
| Python 完整 SHA | 4487 | 835 | -3652 |
| Python 哈希字节 | 14,297,784,540 | 1,929,210,972 | -12,368,573,568 |

输出保持`[5,223]`，两侧均为86/86层、472个Attention投影、86次Vulkan MoE、零CPU
fallback；position 1仍为236/236 attention slot hit与0 B静态权重上传。

## 不可绕过的启用合同

该模式默认关闭，只有同时满足以下条件才允许启用：

- 零下载，只接受已有 Range cache hit；cache miss仍必须由Python完整SHA。
- 43层Attention全部由Vulkan worker执行，且CPU verify关闭。
- 43层MoE全部启用fast-production Vulkan writeback。
- MoE CPU verify和CPU fallback均关闭。
- 延迟页一旦进入`_load_tensor`、`_weight_fp8`、`_weight_fp4`或`_load_i64`，立即失败。
- Attention页只能声明`vulkan_attention_worker`所有权；routed/shared页只能声明
  `vulkan_moe_worker`所有权。

Rust回执绑定排序后的`(tensor, bytes, expected_sha256)`，并返回`verified_count`、
`verified_bytes`、身份摘要与`verified_before_compute=true`。本次候选558个回执全部有效：

- Attention：Python延迟944段 / 9,551,013,120 B；Rust计算前验证完全相同。
- MoE：Python延迟3612段 / 9,062,974,464 B；Rust计算前验证完全相同。
- 合计：4556段 / 18,613,987,584 B，所有权闭合，无缺失回执。

## 为什么没有获得理论上的13--20秒

`range_fetch_routed`确实从15.4090s降至5.7654s，节省9.6436s；但Rust worker边界从
12.6388s升至17.2491s，Attention exclusive也增加2.4821s。原因是旧路径中的Python预读和
SHA顺带把数据加热进OS文件缓存；删除重复读取后，首次冷页成本转移到了真正的Rust验证者。
因此只能晋级完整墙钟实测的3.33%，不能把少读12.369GB直接换算成token收益。

这也给出了下一条更清晰的速度主线：不是恢复Python预热，而是让Rust在当前层GPU计算期间预取并
验证下一层真实payload，并把当前逐层capture manifest控制面收进持久worker。这样才能隐藏冷读，
同时保留“只读一次、只验一次”的边界。

机器可读证据见`FULLDEPTH43_GPU_VERIFIER_OWNERSHIP_AB.json`。本里程碑仅证明速度与完整性，
不构成质量、长上下文、Kimi前端能力或Claude/GPT追赶完成的证据。
