# Polaris S14 复用 Open WebUI

这里不包含自研网页，也不代理 v38/v47。`polaris_api` 同时暴露 OpenAI 与 Ollama 兼容协议，Open
WebUI 直接连接它。默认端口刻意使用 `11435`，不占用本机 Ollama 的 `11434`。

## 一键入口

先构建不含模型运行的协议适配器：

```powershell
cd D:\project\大模型ssd化\scheduler
cargo build --release --offline -p polaris_api
```

检查环境（只读，不启动 Docker、Ollama、模型或页面）：

```powershell
& D:\project\大模型ssd化\fast16\check-polaris-chat.ps1
```

一键启动 API + Open WebUI：

```powershell
& D:\project\大模型ssd化\fast16\start-polaris-chat.ps1
```

入口会按以下顺序执行，任一步失败都会停止：

1. 启动或连接 `polaris_api`；
2. 要求 `GET /healthz` 为 HTTP 200，且 JSON 同时满足
   `model == "Polaris-S14"`、`ready == true`；
3. health 未 ready 时不启动 Open WebUI；
4. 优先复用本地 `open-webui` 命令，否则使用本地 Docker 镜像；Docker 路径固定
   `--pull never`，不会隐式下载；
5. Open WebUI 和 S14 health 同时 ready 后才打开 `http://127.0.0.1:3000`。

## 当前状态（2026-08-03）

production resident loader、官方 DeepSeek-V4 chat codec、`S14Runtime`、`/v1/chat/completions` 与
`/api/chat` 已真实接通。任意输入短门已得到 HTTP 200：`你好`（5 prompt tokens）真实生成
`好的，用户`（3 completion tokens），总计 8 positions；这不是固定回复或 gateway 占位。

production K=4 加速路径也已完成两个连续 block 的真实提交：第一块从 base1 发布实际 selected
checkpoint，第二块从 base5 消费该 checkpoint，重新跑完 43 层并以 `committed=true` 结束。

当前网页仍是短上下文试用：API 的权威 token-major 数值门覆盖 `[0,8)`，默认最多生成 3 token；
超过门限会明确拒绝，不会伪装成长回答。下一阶段是把已经通过的可重复 K=4 continuation 接进
resident chat backend，随后再扩大连续文本长度和速度。

启动脚本默认只在缺页时通过 `http://127.0.0.1:7897` 执行受校验的 Range fetch；可用
`-ExplicitPageFetch:$false` 强制 local-only。脚本不会下载整模。

前端会优先复用本地 Open WebUI 命令；否则使用已有 Docker 容器或本地镜像。若两者都没有，需预先
任选其一补齐：

- 预先安装本地 `open-webui` 命令；或
- 预先把 Open WebUI 镜像放入 Docker。启动脚本只使用已有镜像，不负责下载。

原生 Ollama 不需要启动：Open WebUI 使用 `OLLAMA_BASE_URL=http://host.docker.internal:11435` 直连
Polaris 的兼容端口，因而不会覆盖现有 Ollama 配置或模型。
