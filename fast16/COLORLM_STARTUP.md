# ColorLM 完整启动流程

## 1. 启动模型

在`D:\project\大模型ssd化`运行：

```powershell
.\启动ColorLM.bat
```

启动器会自动等待权重加载和健康检查。看到`ColorLM v17 已就绪`才算完成。

## 2. 检查模型

```powershell
.\检查ColorLM.bat
```

它会依次验证`/health`、模型alias和一次真实`只回复 OK`生成。

正确地址：

```text
健康检查：http://127.0.0.1:8105/health
OpenAI Base URL：http://127.0.0.1:8105/v1
模型名：ColorLM-v17-Coder-Neural-Island
API Key：local
```

`http://127.0.0.1:8105/`返回404是正常的，因为8105是API服务，不是网页聊天站点。

## 3. 直接对话

无需安装聊天UI：

```powershell
.\对话ColorLM.bat
```

输入`/clear`清空上下文，输入`/exit`退出。退出对话不会停止模型服务。

## 4. 接入OpenAI兼容客户端

在Cherry Studio、Chatbox或其他OpenAI兼容客户端中填写：

```text
Base URL: http://127.0.0.1:8105/v1
API Key: local
Model: ColorLM-v17-Coder-Neural-Island
```

不要把Base URL填写成`http://127.0.0.1:8105/`。

## 5. 使用Claude Code

先进入要编辑的工程目录，再调用绝对路径：

```powershell
& 'D:\project\大模型ssd化\使用ColorLM-Claude-Code.bat'
```

该入口使用独立Claude配置，不会覆盖普通Claude Code配置，并会自动确保v17服务已启动。

## 6. 停止模型

```powershell
.\停止ColorLM.bat
```

停止脚本只会终止8105/8106上属于本项目的`llama-server.exe`，不会停止其他服务。

## 常见问题

- 根地址404：正常，请访问`/health`或在客户端使用`/v1`。
- 端口8105被占用：先运行`停止ColorLM.bat`，再重新启动。
- 第一次启动失败：Windows/Vulkan可能尚未释放pinned memory，停止残留服务后再启动一次。
- 模型加载需要时间：RX 5700 XT上通常需要几十秒，启动器会等待，不要重复双击。
- 不能同时运行v17和v18.1：两套模型会争用显存和pinned host memory。
