# DeepSeek-V4 L42 真实单层单 token 参考

这个目录把原先的一次性 L42 实验固化成了可复现脚本。它默认只读
`D:/models/Polaris-S14` 中已存在的 Range cache，不访问网络，不会在真实数据缺失时
使用合成数据或静默降级。

## 它真实执行了什么

1. 用四份相同 BF16 `sin(arange(4096)*0.013)` 构造 L42 输入。
2. 执行 attention HC pre、FP8 稀疏注意力、HC post。
3. 执行 FFN HC pre，由真实 router 重新计算 top-6，而不是强制使用 manifest 结果。
4. 执行命中的 6 个 MXFP4 专家和 1 个 FP8 共享专家，再做 FFN HC post。
5. 从实际计算得到的 tensor 重新计算摘要和 F32 little-endian SHA-256。

报告不写入墙钟耗时等非确定字段；同一受支持环境和同一组真实 payload 应得到相同 JSON 摘要。

UE8M0 激活 scale、E4M3FN 量化、MXFP4 低 nibble/高 nibble 布局和 BF16 舍入点都在
参考前向中显式实现。

## 运行

在仓库根目录执行：

```powershell
python -X utf8 fast16/research/polaris_meridian_v1/l42_real_reference/l42_reference.py
```

运行正向指纹和四项负向拒绝自检：

```powershell
python -X utf8 fast16/research/polaris_meridian_v1/l42_real_reference/selftest.py
```

自检会更新同目录的 `SELFTEST_REPORT.json`。负向测试只在系统临时目录中制作
manifest 副本，不会修改 `D:/models/Polaris-S14`。

为 Vulkan 数值 runner 导出真实前向中的量化后 kernel 输入时，目标目录必须尚不存在：

```powershell
python -X utf8 fast16/research/polaris_meridian_v1/l42_real_reference/l42_reference.py --capture-dir "$env:TEMP/polaris-s14-l42-vulkan"
```

capture manifest 会记录输入 SHA、三个来源 manifest SHA，以及本次真实 payload 校验计数；
缺失或漂移时 Vulkan runner 必须拒绝，不能回退到合成激活。

## 边界

这里的证据只能称为“真实 DeepSeek-V4 L42 单层单 token 参考”。它不是
S14/43 层首 token，不证明完整模型质量，也不是 GPU/Vulkan 性能测试。
