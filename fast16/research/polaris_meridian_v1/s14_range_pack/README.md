# Polaris S14 精确 Range 打包器

本目录把固定 revision 的 DeepSeek-V4 S14 候选转换为可审计、可恢复的 tensor Range 计划。
默认命令不下载权重；只有原生 route trace 覆盖全部 14 层、每个 Range 的哈希锁齐全并显式传入
`materialize --execute` 时，才允许传输 payload。

## 当前状态

- 源模型：`deepseek-ai/DeepSeek-V4-Flash-0731`；
- revision：`7872f01b1d1fe23eabc4c98b48bffcef5a386062`；
- S14 层：`0,1,2,6,7,14,15,22,23,30,31,40,41,42`；
- 52.231GB 是完整选层 shard 上界；最终 Range 包只保留非专家原生路径与 route trace 证明需要的专家；
- 50GiB GitCode overlay 虽能勉强容纳上界原始字节，但无法保留 2GiB 安全余量，因此上界方案拒绝
  直接写 overlay。应在真实 trace 后按精确计划复算，或使用可校验的外部输出盘/逐层 RAM 消费；
- 目前只有合成 metadata 自检，尚无真实 DeepSeek route trace、权重或质量结果。

## 顺序

```bash
python -X utf8 range_pack.py budget --force
python -X utf8 range_pack.py fetch-headers \
  --header-dir headers --execute-metadata-fetch
python -X utf8 range_pack.py plan \
  --index model.safetensors.index.json \
  --header-dir headers \
  --route-trace route_trace.approved.json \
  --hash-lock range_hash_lock.json \
  --asset-lock tokenizer_asset_lock.json \
  --output s14.plan.json
python -X utf8 range_pack.py materialize \
  --plan s14.plan.json --output-dir s14-pack --execute
```

国内环境可将 `POLARIS_HF_ENDPOINT` 指向支持相同 `resolve/<revision>/...` 和精确 HTTP Range
语义的镜像。镜像不会放宽固定 revision、`Content-Range`、文件字节数或哈希门。

## 停止门

- route trace 不是原生完整前向、缺层或不是 top-6；
- 任何 tensor 来自未冻结 shard；
- 服务未返回精确 HTTP 206/`Content-Range`；
- Range SHA-256 未预锁或恢复前缀哈希不一致；
- 原生 `tokenizer.json` / `tokenizer_config.json` 未锁定字节与 SHA-256；
- 输出盘缺少精确包大小加 2GiB 余量；
- 运行时试图把 52.231GB 全部塞入 32GiB HBM。

这些门只保证来源和传输正确，不证明 S14 聪明。质量仍需通过原生四题早停门。
