# Polaris S14 API 适配层

本 crate 提供最小协议入口：

- OpenAI：`POST /v1/chat/completions`、`GET /v1/models`
- Ollama：`POST /api/chat`、`GET /api/tags`
- 健康检查：`GET /healthz`

路由复用 `ai-company` 已验证的 Axum 组织方式，但不包含其 provider 转发、agent 编排或自研网页。
前端直接配置 Open WebUI 连接本服务。

Open WebUI 0.11 对同一个模型 ID 同时从 OpenAI 与 Ollama 两个 provider 发现时会产生归属冲突；
`Polaris-S14` 只应配置一次。当前推荐使用 OpenAI-only：启用 OpenAI 连接
`http://127.0.0.1:11435/v1`，并在 Open WebUI 中禁用指向同一服务的 Ollama 连接。
这不影响 `polaris_api` 保留 `/api/chat` 给独立 Ollama 客户端使用。

短时流排障可在启动 `polaris_api` 前设置 `POLARIS_STREAM_TELEMETRY=1`。日志只记录协议、
内部请求 ID、帧序号、UTF-8 字节/字符数、token ID 是否存在、结束原因与 usage 是否已知；
不会记录 prompt、回复正文、stop 文本或工具参数。

## 安全边界

`polaris_api` 不生成 token，也不代理 v38。position4+ production N=8 数值真门现已通过；冻结日志为
`D:/project/大模型ssd化/.tmp-polaris-tests/n8-production-20260803-proxy.stdout.log`，其 SHA-256 为
`8096d5a8798c840fc7d7725aa3281c3d95790d834452625b52739f1e69621dc8`。门禁要求逐项精确匹配
8 tokens、344 routes、12,384 dynamic ranges、0 fallback 与 `commit_epoch=8`，旧 N=4 或任意非空
字符串不再能构造 `VerifiedS14NumericalGate`。

二进制现使用 `ResidentChatEngine`：worker 依次验证上述 N=8 日志的内容与 SHA、仓内官方
DeepSeek revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062` encoding 源码 SHA、正式
`tokenizer.json` 的 SHA/词表/协议 token，然后才加载 `S14Runtime`。任一步失败或加载未完成时，
health、模型发现和生成请求都返回 HTTP 503，Open WebUI 不会看到一个伪 ready 的
`Polaris-S14`。未知 usage 不填 0；引擎流若在 `Done` 前失败或关闭，OpenAI
流不会发送 `[DONE]`，Ollama 流不会发送 `done:true`。未验证 position 应通过
`EngineError::unsupported_position` 原样暴露。

`ResidentChatEngine` 提供有界队列和单一 OS worker；`S14RuntimeChatBackend` 在该 worker 内
独占常驻 `S14Runtime` 与官方 chat codec。只有 loader 成功且调用方持有上述冻结 N=8
`VerifiedS14NumericalGate` 后才发布 `ready=true`。持久 runtime 逐项发送：

1. `EngineEvent::Delta`：真实解码文本/token；
2. `EngineEvent::Done`：仅在模型确实完成时发送，并携带真实 usage（未知则为 `None`）；
3. `EngineError`：任何 fail-closed 边界或运行失败。

## 定向验证

```powershell
cd D:\project\大模型ssd化\scheduler
cargo check --offline -p polaris_api
cargo test --offline -p polaris_api
```

production 默认只使用本地 Range cache；显式缺页拉取必须由运维同时设置统一代理和开关：

```powershell
$env:POLARIS_API_ADDR='127.0.0.1:11435'
$env:HTTP_PROXY='http://127.0.0.1:7897'
$env:HTTPS_PROXY='http://127.0.0.1:7897'
$env:POLARIS_S14_EXPLICIT_PAGE_FETCH='1'
cargo run --offline -p polaris_api
```

当前 Rust codec 精确冻结无工具的官方 `thinking/low` forced-prefill。协议层尚未表达官方 DSML
reasoning/tool-call 结构，因此 `tools`、tool role 与 `name` 继续 fail-closed；不会套用 Qwen 模板。
