# Polaris Genesis 真实权重岛出生—修剪门

日期：2026-08-04

结论：**PASS（进程级原型）**

## 事件顺序

```text
固定初始候选
  -> 受限外部验证器 fail
  -> spawn v17 Qwen L44-L47 真实权重岛
  -> 产生修复候选
  -> AST 安全合同 + 7 个用例 pass
  -> commit 候选
  -> prune 临时岛进程
```

## 实测回执

| 项目 | 结果 |
|---|---:|
| 失败是否先于spawn | true |
| 真实岛manifest绑定 | true |
| 权重岛总字节 | 3,773,072,128 |
| 真实路由回执 | 4,096 B |
| 路由回执 SHA-256 | `567bc78f6d0d3046770bfcb052808b320e376d1c44852cf30d71cc6eed2599c0` |
| 启动岛 | 17.79 s |
| 修复请求 | 24.20 s |
| 生成 | 61 token / 7.10 s / 8.60 token/s |
| 外部验证 | 7/7 pass |
| 无关 Design IR SHA | 前后一致 |
| 终止后8117端口 | free |

岛 manifest SHA-256：
`7ea2b40a2551ba30bf4dd664043c0bc33d9c486785a62ce2793e7c3ec8ff3cba`。

## 修复候选

```python
def normalize_score(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        num = float(value)
        return max(0, min(100, num))
    except (ValueError, TypeError):
        return None
```

## 这次能证明什么

- 外部失败回执可以成为真实权重岛物理出生的先行事件。
- 岛确实执行了原始连续权重，不是固定文本、模拟输出或仅加载 manifest。
- 候选可经独立机器验证后提交，临时岛可在任务后终止。
- `--allow-mmap` 使旧 v38/v17 运行时避免强制锁定约8.34 GiB Vulkan Host
  模型缓冲；它只改变物理装载，不改变模型数学。

## 这次不能证明什么

- 初始失败候选是固定 fixture，不是 v38 当场生成。
- spawn/prune 发生在进程级，尚未在单一运行时内动态创建或删除计算图节点。
- v17 岛仍经v38固定语言主干和输出头表达；系统尚未脱离 Transformer 主干。
- 单题通过不构成能力提升、人工意识或新架构的证据。

完整机器回执见 `live_birth_prune/gate_receipt.json`。
