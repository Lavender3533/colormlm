# S14 Range Pack 交接

## 当前状态

实现与零网络自检已完成；真实包仍处于 **BLOCKED**。

- 固定 donor：DeepSeek-V4-Flash-0731 `7872f01...6062`。
- 固定 S14：14 层，无事后改层。
- shard-union 上界：52,231,273,716 B。
- GitCode 50 GiB overlay：原始字节勉强容纳，但达不到 2 GiB 安全余量，拒绝落盘。
- 1.5 TiB RAM：允许逐 tensor/逐层匿名 RAM 流式；`/dev/shm` 64 MiB 不可用。
- 32 GiB HBM：只允许当前层/热页，不允许整包。
- 本轮没有网络访问、权重下载或模型启动。

## 主线必须补齐

1. 固定 revision 的官方 index 文件，必须通过冻结 bytes/SHA。
2. 16 个实际 source shard 的 header cache（14 层 + 2 个边界文件；
   `norm/head` 共用 shard）。
3. 来自原生 DeepSeek forward 的完整 S14 route trace；所有保留层均有 events 和实际
   expert ID，并引用原生 capture manifest SHA-256。没有 trace 就不生成 payload 清单。
4. 11 个当前尚无本地权威 SHA 的层 shard，补齐固定 Git LFS OID；不要拿 `main`
   或首次下载结果自证。
5. 为每个 Range 生成独立 SHA lock。未锁齐时 planner 会保持
   `blocked_missing_integrity_locks`；原生 tokenizer 两项 asset 也必须锁定。
6. 若写持久包，先对真正输出路径做 `statvfs`；不能用 `df` 中 overlay 的标称大小
   代替可用空间。
7. 若走 RAM，Ascend adapter 必须按 `iter_verified_tensors()` 的“校验后消费、消费后
   释放”契约接入，并证明不会缓存全包。

## 审批后最短命令

```bash
python -X utf8 range_pack.py budget --force
python -X utf8 selftest.py
python -X utf8 range_pack.py fetch-headers --header-dir "$WORK/s14-headers" --execute-metadata-fetch
python -X utf8 range_pack.py plan --index "$WORK/model.safetensors.index.json" --header-dir "$WORK/s14-headers" --route-trace "$WORK/s14-route-trace.json" --hash-lock "$WORK/s14-range-hashes.json" --asset-lock "$WORK/s14-tokenizer-assets.json" --output "$WORK/s14-pack-plan.json"
```

不要执行 `materialize --execute`，直到 plan 状态为 ready、输出空间通过、Ascend
会话剩余时间覆盖传输与 Gate 1，并且主线明确批准真实权重传输。

## 不得宣称

- 52.231 GB 已下载或已装入 NPU；
- S14 已跑通；
- route trace 已存在；
- 精确 Range 包等于 52.231 GB；
- 北极星质量已提升或接近 Claude/GPT。
