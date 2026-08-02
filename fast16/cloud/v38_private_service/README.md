# v38 私人云端试玩服务

此目录只部署 `ColorLM-v38-Qwen36-Shared-Sequence-Policy`，不加载、不修改、不测试 Polaris S14
主线，也不把 v46、v47 或旧网页编译器描述为更好的聊天模型。

## 固定资产

- 核心 GGUF：`ColorLM-v36-Qwen36-Global-Shared-Backbone.gguf`
  - 字节数：`13,613,857,952`
  - SHA-256：`e8e9e22f3de6844adc6ab0d1ff3b29c3721e17749741b6d6595ec2fc1fe56858`
- v29 序列策略头：本目录 `runtime-v1/`
  - `policy.json` SHA-256：`c647b614aaf9d28e2fb451df0c3e70b79cd3ae387c9407316307f9f88bbc914f`
  - `weights.bin` SHA-256：`5e649b4fbbaed446906f790c2074a9cedb953f72e6cb739b78a26806521654cc`
- 自定义 llama.cpp：上游 `b46812de78f8fbcb6cf0154947e8633ebc78d9ac` 加本目录固定 patch，
  对应本地提交 `865801f14fc53218f591714ea32dd407b15022a6`。
- chat template 直接使用 GGUF 内嵌模板，不传入任何替换模板。
- API alias：`ColorLM-v38-Qwen36-Shared-Sequence-Policy`。

## 当前 GitCode Notebook 的人工步骤

### 1. 拉部署分支并安装

在 Notebook 终端执行：

```bash
export V38_SERVICE_ROOT=/opt/atomgit/v38-private-service
mkdir -p "$V38_SERVICE_ROOT/source"
git clone --depth 1 --single-branch --branch codex/v38-private-cloud \
  https://github.com/Lavender3533/colormlm.git \
  "$V38_SERVICE_ROOT/source/colormlm"
cd "$V38_SERVICE_ROOT/source/colormlm/fast16/cloud/v38_private_service"
bash install.sh
```

安装器先核对磁盘、内存与 Ascend NPU，然后用 CANN 构建唯一的 `llama-server`，并在独立
Python 3.11/3.12 环境安装固定的 Open WebUI。模型不存在时，安装器不会下载、量化或替换模型。

### 2. 上传原始模型

先在终端创建目标目录：

```bash
mkdir -p /opt/atomgit/v38-private-service/models
```

在 Jupyter 左侧文件浏览器依次进入 `v38-private-service/models`，点击 **Upload Files**，选择本机：

`D:\project\大模型ssd化\fast16\models\ColorLM-v36-Qwen36-Global-Shared-Backbone.gguf`

上传完成后执行一次校验：

```bash
cd /opt/atomgit/v38-private-service/source/colormlm/fast16/cloud/v38_private_service
sha256sum /opt/atomgit/v38-private-service/models/ColorLM-v36-Qwen36-Global-Shared-Backbone.gguf
```

结果必须严格等于上面的模型 SHA-256。若浏览器对 12.7 GiB 单文件上传失败，应停止并改用分片
上传；不能量化或换模型冒充 v38。

### 3. 启动

```bash
cd /opt/atomgit/v38-private-service/source/colormlm/fast16/cloud/v38_private_service
bash start.sh
```

脚本会隐藏读取私人 API 密钥，密钥不回显、不写源码、配置或日志，只以进程环境变量注入。两个服务
都只监听 `127.0.0.1`：llama API 为 `8138`，Open WebUI 为 `3000`。

### 4. 唯一允许的 smoke

```bash
bash health-smoke.sh
```

该脚本只检查 `/health`、`/v1/models`，并发送一次“你好，只回答：你好。”，不会运行旧长榜或能力评测。

### 5. 访问与登录

- Open WebUI：把当前 Jupyter Lab 地址结尾的 `/lab` 替换为 `/proxy/3000/`。
- OpenAI API：同一实例前缀下的 `/proxy/8138/v1`。
- 首次打开 Open WebUI 时创建唯一管理员账号。数据库创建后执行一次停止再启动，脚本会自动设置
  `ENABLE_SIGNUP=false`；之后只允许已有账号登录。
- Notebook 代理本身还受 GitCode 登录保护。API 仍必须使用启动时输入的 Bearer 密钥。

## 管理命令

```bash
# 状态
bash status.sh

# 停止
bash stop.sh

# 重新启动（会再次隐藏读取 API 密钥）
bash start.sh

# 一键永久清理（含模型、Open WebUI 数据和日志）
bash cleanup.sh --yes
```

持久工作目录统一为 `/opt/atomgit/v38-private-service`。日志在其 `logs/`，Open WebUI 数据在
`data/open-webui/`，PID 在 `run/`。

## 费用与平台边界

当前 AtomGit/GitCode Notebook 的 910B4 32 GiB 实例为免费试玩资源，但单次会话约两小时，并非
永久在线云服务。只要免费额度有效，本次预计计算费用为 `0 元`。若要长期在线，应迁移到至少
24 GiB 显存、32 GiB 内存、30 GiB 可用磁盘的按量 GPU；不能通过量化或替换模型降低门槛后仍称 v38。

