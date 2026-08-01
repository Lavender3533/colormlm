# Ascend S14 交接

## 已完成

- 固定 DeepSeek 官方 revision 和冻结 S14 层集合；
- 明确支持/拒绝矩阵；
- doctor 可识别 Python 架构、torch/torch_npu、设备名、HBM，以及
  FP4/UE8M0/FP8/mHC/CSA-HCA 缺口；
- 实现单层 `load → execute → free` adapter，异常路径也释放；
- dry-run 与 synthetic selftest 均不访问网络、不下载权重、不触碰设备；
- native 命令在语义未补齐时硬拒绝。

## 下一条 GitCode 910B4 容器命令

假设项目已经位于 `$HOME/ColorLM`，先只运行 doctor，不分配测试矩阵、不下载权重：

```bash
cd "$HOME/ColorLM/fast16/research/polaris_meridian_v1/ascend_s14" && PYTHONUTF8=1 python -X utf8 doctor.py --probe-npu --require-device Ascend910B4 --output doctor_910b4.json
```

预期：能看到 `torch_npu`、`Ascend910B4` 与约 32 GiB HBM；但
`native_forward_ready` 仍应为 `false`，并列出十项语义缺口。这是正确结果，不是失败。

随后可运行零权重接口自检：

```bash
PYTHONUTF8=1 python -X utf8 selftest.py --output selftest_910b4.json
PYTHONUTF8=1 python -X utf8 adapter.py dry-run --output dry_run_910b4.json
```

## 真正 native forward 的下一工程动作

1. 固定 revision 的官方 runtime 代码做 Ascend graph port；
2. 先单独证明 MXFP4+UE8M0 expert matmul 和 FP8 attention 数值对齐；
3. 实现并对齐 mHC 四流与 CSA/HCA cache；
4. 给 range pack 实现 `VerifiedTensorProvider`，严格单层在生；
5. 用一个 token、一个层做 CUDA/官方参考对 Ascend 的 hidden/router/NLL 对齐；
6. 对齐通过后才签发单独 runtime manifest：对应 capability 全部为 `passed` 且
   `native_forward_ready=true`，再允许完整 S14；不要把仓内冻结矩阵原地自证为 true。

不得把 doctor、dry-run 或 synthetic selftest 写成真实 DeepSeek forward。
