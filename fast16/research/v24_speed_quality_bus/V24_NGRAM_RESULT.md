# v24 n-gram推测解码短门

## 结论

拒绝把`ngram-mod`作为连续神经岛的默认路径，正式启动器默认改为`--spec-type none`。

这不是因为推测解码理论无效，而是当前v17含三层Gated DeltaNet recurrent state、L47 KV和
四层精确专家缓存。n-gram候选触发多token目标验证时，会走与单token decode不同的完整专家银行
路径，并要求所有私有状态正确checkpoint/rollback。当前短门证明该组合没有保持预期的贪心等价。

## 冻结契约

- 契约：`ngram_gate_v1.json`
- SHA-256：`e42a27c855825187cfb31b15b71d10f8ac10b61f00c9979f5992bb5f8f81fdaa`
- 模型：同一`ColorLM-v17-Coder-Neural-Island`运行包
- 只切换：`--spec-type none`与`ngram-mod(match=16,min=4,max=16)`
- 温度0、seed 18、四个短任务：TypeScript补丁、Rust新代码、中文自然交流、`read_file`工具调用。

## 实测

| 路径 | 总completion token | 客户端合计速度 |
|---|---:|---:|
| none | 207 | 14.2444 token/s |
| ngram-mod | 213 | 14.1415 token/s |

总体变化`-0.72%`。四题中只有TypeScript补丁输出精确一致；Rust、中文、工具三题的消息哈希、
停止原因或token数不一致。工具题速度从`11.07`降至`8.91 token/s`，下降`19.52%`。

因此它同时未过两道门：

1. 输出逐消息精确等价；
2. 总体速度至少有正向信号。

## 后续

- 不再扫描n-gram参数救该候选，避免在不安全状态回滚上浪费时间。
- 若未来实现完整Neural Island sequence checkpoint/rollback，可重新开启独立版本；在此之前
  `--spec-type`只保留研究入口。
- 下一速度线转向不改变数学输出的专家缓存策略：采集四层真实top-10路由序列，离线比较LRU、
  频率/分段缓存和跨token预取，再只实现有明确上传字节收益的候选。
