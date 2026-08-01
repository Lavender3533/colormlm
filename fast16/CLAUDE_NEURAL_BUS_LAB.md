# ColorLM Neural Bus Claude Code Lab

## 启动

在要测试的代码项目目录中运行：

```powershell
colorlab-claude
```

它会自动确保`ColorLM-v13-Causal-Sparse-L12`在`127.0.0.1:8102`运行，再启动Claude Code。

## 隔离边界

- 独立`CLAUDE_CONFIG_DIR`：`fast16/runtime/claude-neural-bus-lab/config`。
- PowerShell入口强制使用UTF-8输入、输出与Python日志编码。
- 独立会话历史、认证状态和调试状态。
- `--bare --disable-slash-commands --setting-sources ""`：不读取日常Claude Code的hooks、plugins、skills、memory、用户或项目settings。
- 只使用本地假API key和`ColorLM-v13-Causal-Sparse-L12`，不读取Claude订阅或系统钥匙链。
- 显式清除Bedrock、Vertex和Foundry提供商开关，避免继承宿主终端中的云凭据模式。
- 禁用非必要流量、telemetry和错误上报。

这里的“只连接本地”指模型API固定连接`127.0.0.1:8102`，不是操作系统级断网沙箱。
若允许`WebFetch`、`WebSearch`或`Bash`，这些工具自身仍可能访问网络。

配置与会话是隔离的，但Claude Code仍会读写启动命令时的当前代码目录。
需要文件级隔离时，在专用测试副本中运行，或使用：

```powershell
colorlab-claude --worktree neural-bus-test
```

## 不影响的入口

- 日常Claude Code：`claude`
- 旧ColorLM v6入口：`colormlm-claude`
- ColorLM v13实验入口：`colorlab-claude`
