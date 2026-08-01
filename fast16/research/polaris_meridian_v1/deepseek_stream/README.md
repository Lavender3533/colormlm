# DeepSeek-V4 原生状态流式采集

这个目录只负责一个前置里程碑：在固定的官方 DeepSeek-V4-Flash-0731 revision 上，采集 L39--L42 的真实隐藏态、mHC 支路和 router top-6，为后续“巨型供体器官移植”建立可验证接口。

它目前**不是 DeepSeek 能力移植结果**。CPU/NPU selftest 只验证采集协议、tensor 搬运和文件完整性；只有 `run_capture.py --execute-native` 通过真实适配器、完整权重来源证明和 CNOB 严格验证后，才能称为原生状态采集。

## 固定边界

- 仓库：`deepseek-ai/DeepSeek-V4-Flash-0731`
- revision：`7872f01b1d1fe23eabc4c98b48bffcef5a386062`
- 观察层：L39、L40、L41、L42
- 每个 token：26 个 CNOB v2 chunk
- 任务侧车：UTF-8 JSONL
- 采集上限：示例 8 任务 × 64 token，约 86 MiB
- 不需要 DSpark 的 3 个权重分片

## 本地自检

```powershell
python -X utf8 -m py_compile capture_io.py instrumentation.py planner.py doctor.py run_capture.py selftest.py
python -X utf8 selftest.py --device cpu --output selftest_report.json --force
python -X utf8 planner.py --output dry_run_plan.json --force
python -X utf8 doctor.py --device cpu --skip-network --output doctor_cpu_report.json --force
python -X utf8 run_capture.py --tasks capture_tasks.example.jsonl --output-dir dry-run
```

最后一条命令默认只打印计划，不创建原生采集结果。

## 910B4 容器检查

把这个目录和固定元数据上传到容器后运行：

```bash
python -X utf8 doctor.py --device npu --output doctor_npu_report.json
python -X utf8 selftest.py --device npu --output selftest_npu_report.json
```

`torch_npu` 矩阵乘成功不代表官方 CUDA/TileLang FP4 kernel 可用。当前真正缺少的是能够证明以下事项的原生 runtime adapter：

1. 使用固定 revision；
2. 读取完整 45 个 base-forward 文件，或可校验的等价远端 tensor source；
3. 实际运行官方语义的 embedding、L0--L42、final norm 和 lm head；
4. 将 `OfficialDeepSeekStepProbe` 挂在真实层上；
5. 返回 `polaris-deepseek-native-adapter-attestation-v1` 证明。

适配器具备上述条件后才运行：

```bash
python -X utf8 run_capture.py \
  --execute-native \
  --adapter your_adapter:run_native_capture \
  --tasks capture_tasks.example.jsonl \
  --output-dir output/native
```

## 文件说明

| 文件 | 用途 |
|---|---|
| `official_snapshot.json` | 官方 revision、元数据哈希、权重分片与字节边界 |
| `capture_contract.json` | CNOB v2 tensor 和 sidecar 契约 |
| `capture_io.py` | `.part` 流式写入、原子提交及严格校验 |
| `instrumentation.py` | 官方层、mHC 与 router 的只读探针 |
| `planner.py` | 网络、磁盘、RAM、HBM 和两小时窗口规划 |
| `doctor.py` | 容器设备与资源检查，不证明原生 forward |
| `run_capture.py` | 默认 dry-run；显式适配器才允许真实采集 |
| `selftest.py` | 合成接口 fixture，不是模型能力证据 |

## 当前结论

采集协议已经在 CPU 上通过。DeepSeek 原生 forward 仍未执行；Ascend 910B4 的官方语义流式执行器仍是阻塞项。不要把 `selftest_report.json` 或 `doctor_cpu_report.json` 写成供体能力已经进入北极星。
