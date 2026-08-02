# FullDepth43 内联 manifest 负门

## 结论

Python 现在可以把规范 UTF-8 `manifest_json`、其精确 SHA-256 和 capture root
直接送入持久 Rust worker。Rust 先验证原始字符串摘要，再反序列化并复用原有 manifest、
输入、payload 和输出边界检查。显式入口为：

```text
--vulkan-writeback-inline-manifest
```

该开关默认关闭。实现正确，但没有获得可重复的完整墙钟收益，因此不晋级正式默认路径。

## 真实 RX 5700 XT 相邻 A/B

| 指标 | 文件基线 | inline 候选 | 变化 |
| --- | ---: | ---: | ---: |
| 完整两 token execution | 50.3515s | 52.5943s | +4.45%（候选更慢） |
| 有效吞吐 | 0.03972 token/s | 0.03803 token/s | -4.26% |
| Python IPC/响应校验 | 0.32039s | 0.25399s | -20.72% |
| `bridge_manifest.json` | 86 | 0 | -100% |

候选之后的反向文件基线为 54.6110s，候选相对它又快 3.69%。正反方向不一致，说明
约 66ms 的 IPC 子项节省被更大的 SSD、SHA、GPU 上传和系统噪声淹没。不能挑有利方向宣称加速。

## 完整性门

- 三次运行输出均为 `[5, 223]`。
- 候选 86/86 层使用 `inline_json`，86/86 Vulkan MoE，0 CPU fallback。
- GPU verifier owner、count、bytes 和身份多重集闭合。
- 候选运行目录没有生成任何 `bridge_manifest.json`。
- 最终全目录Python 96项测试与22个子测试、Rust release example 41项测试及真实一 token冒烟均通过。

## 决策

保留新协议和显式开关，方便后续把输入/输出也改为共享 arena，但默认仍走已晋级的文件路径。
下一速度主线不再继续抠几十 KiB 控制文件，而是处理两 token 仍读取并 SHA 的
`7,593,067,008 B` MoE payload，以及每 token 约 4.22GiB 的 CPU→GPU 专家流量。

机器可读数据见 `FULLDEPTH43_INLINE_MANIFEST_AB.json`。这是一项负门，不是速度里程碑、
质量提升或 Claude/GPT 能力证据。
