# Polaris Ascend S14 Adapter

这是 DeepSeek-V4-Flash-0731 原生稀深 S14 的 Ascend 910B4 **接口脚手架**。它固定官方
revision 与 14 层集合，定义单层 `load → execute → free` 生命周期，并提供 doctor、dry-run
和合成自检。

## 固定身份

- 仓库：`deepseek-ai/DeepSeek-V4-Flash-0731`
- revision：`7872f01b1d1fe23eabc4c98b48bffcef5a386062`
- 层：`0,1,2,6,7,14,15,22,23,30,31,40,41,42`
- 原生宽度：4096；mHC：4 streams；router：top-6 / 256 experts

## 当前能做与不能做

`adapter.py` 已实现：

1. `VerifiedTensorProvider`：接收上游已经完成 revision、Range 与 SHA 校验的当前层 tensor；
2. `RangePackLayerSource`：同一时刻最多一个层租约；
3. `StreamingS14Runner`：无论成功或异常都在 `finally` 释放当前层；
4. `OfficialAscendLayerExecutor`：缺任一原生语义时硬拒绝；
5. 无权重 dry-run 与纯 Python synthetic 生命周期测试。

当前明确缺少 MXFP4 解包/内核、UE8M0 scale、FP8 attention、四路 mHC、CSA/HCA state、
原生 router/shared expert、tokenizer/embedding/head 和 Ascend 数值对齐。因此
`native_forward_ready=false`。torch 或 torch_npu 暴露某个 dtype 名字不算内核已支持。

## 本地零设备验证

```powershell
python -X utf8 selftest.py --output selftest_report.json
python -X utf8 adapter.py dry-run --output dry_run_report.json
python -X utf8 doctor.py --output doctor_offline.json
```

这些命令不访问网络、不下载权重，也不会导入 torch_npu 或触碰 NPU。

## 真实接入边界

上游 range pack 只能通过一个外部 provider 接入 `iter_verified_tensors(layer_id)`；每个 tensor
必须带 `verified=true` 证明。adapter 本身不依赖或修改 `s14_range_pack/`，也不会绕过其完整性门。

真实 executor 必须注入 fixed-revision official block，并让一份经过实际测试签发的 runtime
manifest 同时满足：全部十项 `required_for_native_forward=passed` 且
`native_forward_ready=true`。仓内冻结矩阵保持 false；在此之前 `native` 命令必定拒绝。

## 声明边界

合成 hidden 数字没有模型含义。本目录不证明 S14 已运行、质量提升、达到 Claude/GPT，或能在
两小时会话内完成真实权重传输。
