# ColorLM v47 并行速度研究交接

## 已交付

本目录已经形成可复现的纯CPU离线闭环：`DESIGN.md`、`manifest.json`、3份JSON Schema、`short_gate_contract.json`、`cost_model.py`、`offline_gate.py`、`selftest.py`及三份报告。所有工具只使用Python标准库；证据文件只读，未读取任何模型权重正文，未启动/停止模型服务，未调用GPU，也未下载内容。

## 本轮实测

执行时间：2026-08-01（报告内另含UTC时间）。

### 自检

`selftest_report.json`：13项通过；解析512条v24路由记录，层集合为44/45/46/47，每层后半段共形成2560次专家请求。UTF-8无BOM检查通过，schema/contract可解析，成本模型可JSON往返。

### 证据固定

manifest列出的8个外部记录均存在且SHA-256匹配，包括v38运行检查、v36/v17 full16、v24 n-gram负结果、v24 LRU/LFU运行结果、v24路由dump/分析和v17岛manifest。若任一文件变化，离线门会失败。

### A：多层直接草稿头

- 最小参数：`4,464,640`；F16权重`8.516 MiB`。
- 4位置词表投影使草稿成本约`2.498 GFLOP/base-step`，在3B active参数假设下约为基座单步`41.64%`。
- 在“均匀接受率0.65、目标批验证因子1.35”假设下解析上界`1.430x`。
- 参数、权重和解析上界门通过；没有训练权重、真实接受率和逐token等价证据，`runtime_promotable=false`。

### B：动态K局部latent循环

- 最小参数：`2,228,549`；F16权重`4.251 MiB`。
- 假设K分布`[0.35,0.25,0.20,0.12,0.08]`，mean K=`1.33`、p95 K=`4`。
- 平均额外计算约`5.841 MFLOP/token`，只占假设基座单步`0.097%`；这仍是增加计算，不构成速度收益。
- K域、K0硬旁路和预算门通过；没有质量-成本曲线及动态路由校准，不能集成。

### C：工作量感知分层缓存

冻结回放使用每层前64步训练转移、后64步冷启动测试，GPU32槽/CPU128槽、每步最多预取4页，三策略容量相同：

| 策略 | GPU命中 | CPU warm命中 | 冷miss | GPU上传 | CPU执行 | 需求SSD | 预测SSD |
|---|---:|---:|---:|---:|---:|---:|---:|
| 分层LRU | 599 | 788 | 1173 | 1961 | 0 | 1979.44MiB | 0 |
| 分层LFU97 | 724 | 668 | 1168 | 1836 | 0 | 1971.00MiB | 0 |
| v47 | 694 | 723 | 1143 | 519 | 1347 | 1928.81MiB | 195.75MiB |

v47冷miss相对LRU降低`2.56%`，有效预取精度`23.28%`。它明显减少GPU上传，但额外预测SSD读取使总SSD字节高于LRU，同时大量工作转到CPU；预声明要求冷miss至少降低5%，所以`C_cold_miss_reduction=false`，总体离线决策为`reject`。这不是程序错误，而是候选没有达到门槛。

## 如何继续

1. 不要在当前后64步上扫描预取页数、promotion margin或衰减率后声称留出胜出；这段数据已经开封。另采至少两段独立route trace，一段开发、一段最终留出。
2. A先离线采v38的L12/L24/L39状态并训练直接4-token头。未达到60%接受率前不要改运行时；达到后先实现临时sequence branch原子提交，专门覆盖v24暴露的状态问题。
3. B先用固定K=0/1/2/4得到质量-成本Pareto曲线。若共享循环不能替代更贵计算或修复A/C质量，不训练动态router。
4. C优先实现只预取确定残差页的Q0并测真实overlap；Q1预测预取需要新的独立留出。运行报告必须同时包含RAM、VRAM、SSD需求/预测字节、PCIe上传、CPU执行、GPU队列、p50/p95等待。
5. 三路线单独晋级。组合会改变接受率、K和专家路由分布，必须重新冻结组合门。

## 复现命令

```powershell
python fast16/research/parallel_speed_v47/selftest.py
python fast16/research/parallel_speed_v47/cost_model.py --output fast16/research/parallel_speed_v47/cost_report.json
python fast16/research/parallel_speed_v47/offline_gate.py --output fast16/research/parallel_speed_v47/offline_gate_report.json
```

当前最后一条命令按合同返回退出码2并写出完整报告，因为C硬门失败。不得把A/B预算门通过写成运行候选通过，也不得宣称相关论文速度可直接复现。
