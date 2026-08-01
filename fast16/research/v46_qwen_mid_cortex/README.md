# ColorLM v45–v46 完整主脑与中层皮层实验

## 结论

正式最佳仍为 `ColorLM-v38-Qwen36-Shared-Sequence-Policy`。

v45先在新跨模板 validation 60题上比较v38、完整Qwen3.6与完整GLM-4.7：

```text
v38:          30/60, 196.27 s
Qwen3.6:     40/60, 438.98 s, 10胜0回归
GLM-4.7:     17/60, 224.96 s, 0胜13回归
```

但在当时未触碰的blind 60题上：

```text
v38:          22/60
Qwen3.6:     23/60, 4胜3回归, 净胜1
```

因此完整Qwen换脑不能晋级；validation中的规划增益没有跨模板稳定。

v46改为一个物理GGUF的连续中层皮层：保留v36 L0–L15、L32–L39、embedding、
final norm与output head，把Qwen3.6 L16–L31的292个完整层张量运输进来。临时
GGUF为15.767GiB，可真实Vulkan启动。

已消费validation上的开发门：

```text
v38:          30/60, 196.436 s
v46:          35/60, 196.412 s
对比:          5胜0回归，规划+4，调试+1，墙钟-0.01%
```

新建并冻结的24题blind（六类各4题）结果：

```text
v38:          15/24, 83.670 s
v46:          13/24, 94.442 s
对比:          1胜3回归，规划-3，墙钟+12.87%
```

故v46未通过一次性blind，不是正式模型。按预注册合同停止直接层拼接，不扫其他层范围。

## 保留证据

- `v45_backbone_screen/v45_backbone_screen_report.json`：validation三主脑对比。
- `v45_backbone_screen/v45_backbone_blind_report.json`：Qwen完整换脑blind。
- `v46_dev_gate_report.json`：中层皮层开发门。
- `v46_blind_tasks_v1.jsonl`及manifest：新冻结24题。
- `v46_blind_gate_report.json`：一次性晋级门。
- `ColorLM-v46-Qwen36-Mid-Cortex-L16-L31.gguf.json`：物理张量来源报告；失败GGUF可回收。

## 得到的架构结论

1. 完整换脑和中层皮层都能在单一模板族上产生大增益，但都没有跨模板稳定。
2. 问题不是模型没加载或融合没改变输出，而是供体能力未被可靠地约束到正确状态。
3. 下一架构不应继续换层范围，而应是使用跨模板数据学会稀疏进入/离开的受控序列能力岛，
   或者对完整主脑做真实任务蒸馏；不再用单token、固定alpha或无路由的直接层拼接延长实验。
