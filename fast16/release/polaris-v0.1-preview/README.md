# Polaris v0.1 Preview 网关

这是一个今天即可体验的轻量入口：它把 OpenAI / Anthropic 兼容请求转发给现有
`ColorLM-v38-Qwen36-Shared-Sequence-Policy` 服务，并支持普通 JSON 与 SSE 流式响应。

> 当前只提供 **draft-only** 输出。`exact_verifier=not_ready`，不代表 FullDepth 验证结果，
> 也不声明或模拟 K3 能力。每个本地响应均携带
> `X-Polaris-Verification: draft-only`。

## 一键启动

在仓库根目录运行：

```powershell
.\fast16\run-polaris-v0.1-preview.ps1
```

也可以双击或运行：

```bat
fast16\run-polaris-v0.1-preview.bat
```

BAT 入口会优先使用 PowerShell 7（`pwsh.exe`），不存在时回退到 Windows PowerShell 5。
PS1 文件本身保持 UTF-8 无 BOM 且源码仅含 ASCII，因此两种宿主均可安全解析。可在不访问端口、
不启停任何服务的情况下验证入口：

```powershell
.\fast16\run-polaris-v0.1-preview.ps1 -SelfTest
```

脚本默认执行以下动作：

1. 检查 `http://127.0.0.1:8138/health`；
2. 若 v38 未在线，则在后台调用现有
   `fast16/run-colormlm-v38-qwen36-sequence-policy.ps1`，只等待它健康，不停止任何其他服务；
3. 在 `http://127.0.0.1:8140` 启动 Preview 网关。

浏览器访问 `http://127.0.0.1:8140/` 即可使用中文聊天体验页。

## 接口

- OpenAI 兼容：`http://127.0.0.1:8140/v1/chat/completions`
- Anthropic 兼容：`http://127.0.0.1:8140/v1/messages`
- Preview 状态：`http://127.0.0.1:8140/polaris/status`
- 体验页：`http://127.0.0.1:8140/`

客户端传入的任意现有 `model` 值都会被重写为：

```text
ColorLM-v38-Qwen36-Shared-Sequence-Policy
```

也可以跳过一键脚本，单独启动网关（此时不会自动启动模型）：

```powershell
$env:PYTHONUTF8 = '1'
python fast16/release/polaris-v0.1-preview/gateway.py `
  --listen-port 8140 `
  --upstream http://127.0.0.1:8138
```

## 测试

测试仅启动本地假上游，不会启动或访问真实模型：

```powershell
$env:PYTHONUTF8 = '1'
python -m unittest discover `
  -s fast16/release/polaris-v0.1-preview `
  -p 'test_*.py' `
  -v
```

网关与测试均只使用 Python 标准库。
