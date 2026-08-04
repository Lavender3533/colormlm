[CmdletBinding()]
param(
    [string]$ApiBaseUrl = $(if ($env:POLARIS_API_BASE_URL) { $env:POLARIS_API_BASE_URL } else { 'http://127.0.0.1:11435' }),
    [int]$WebUiPort = $(if ($env:POLARIS_WEBUI_PORT) { [int]$env:POLARIS_WEBUI_PORT } else { 3000 }),
    [ValidateSet('Auto', 'Local', 'Docker')][string]$Backend = 'Auto',
    [bool]$ExplicitPageFetch = $true,
    [string]$RangeProxyUrl = 'http://127.0.0.1:7897',
    [ValidateRange(1, 32)][int]$RushMaxTokens = 8,
    [ValidateRange(1, 24)][int]$RushRangeWorkers = 12,
    [ValidateRange(20, 128)][int]$RushDiskReserveGiB = 24,
    [switch]$ProjectionTwinFallback,
    [int]$ApiWaitSeconds = 60,
    [int]$WebUiWaitSeconds = 120,
    [switch]$NoBrowser
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
[Console]::InputEncoding = [Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$OutputEncoding = [Text.UTF8Encoding]::new($false)

$checkScript = Join-Path $PSScriptRoot 'check-polaris-chat.ps1'
$workspace = Split-Path -Parent $PSScriptRoot
$rushEnvironmentScript = Join-Path $workspace 'scheduler\polaris_api\polaris-s14-rush-env.ps1'
. $rushEnvironmentScript
$runtimeDir = Join-Path $PSScriptRoot 'runtime\polaris-chat'
$startedApi = $null
$startedWebUi = $null
$completed = $false

function Get-EnvironmentReport {
    $json = & $checkScript -ApiBaseUrl $ApiBaseUrl -WebUiPort $WebUiPort -Json | Out-String
    return $json | ConvertFrom-Json
}

function Wait-ForCondition {
    param(
        [Parameter(Mandatory = $true)][scriptblock]$Condition,
        [Parameter(Mandatory = $true)][int]$TimeoutSeconds
    )
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    do {
        if (& $Condition) { return $true }
        Start-Sleep -Milliseconds 500
    } while ([DateTime]::UtcNow -lt $deadline)
    return $false
}

function Start-ApiAdapter {
    param([Parameter(Mandatory = $true)][string]$Binary)

    $uri = [Uri]$ApiBaseUrl
    if ($uri.Host -notin @('127.0.0.1', 'localhost', '::1')) {
        throw '自动启动 polaris_api 只允许 loopback ApiBaseUrl；远程地址请先自行启动并通过 health'
    }
    New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null
    $listenAddress = "$($uri.Host):$($uri.Port)"
    $rushProfile = Get-PolarisS14RushEnvironment `
        -ApiAddress $listenAddress `
        -MaxTokens $RushMaxTokens `
        -RangeWorkers $RushRangeWorkers `
        -DiskReserveGiB $RushDiskReserveGiB `
        -ExplicitPageFetch $ExplicitPageFetch `
        -RangeProxyUrl $RangeProxyUrl `
        -ProjectionTwinFallback ([bool]$ProjectionTwinFallback)
    return Start-Process -FilePath $Binary -PassThru -WindowStyle Hidden `
        -WorkingDirectory $workspace `
        -Environment $rushProfile.Environment `
        -RedirectStandardOutput (Join-Path $runtimeDir 'polaris_api.stdout.log') `
        -RedirectStandardError (Join-Path $runtimeDir 'polaris_api.stderr.log')
}

function Test-DockerReady {
    $docker = Get-Command docker -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -eq $docker) { return $false }
    $version = & $docker.Source info --format '{{.ServerVersion}}' 2>$null
    return $LASTEXITCODE -eq 0 -and [bool]$version
}

function Ensure-DockerReady {
    if (Test-DockerReady) { return }
    $desktopCandidates = @(
        'C:\Program Files\Docker\Docker\Docker Desktop.exe',
        (Join-Path $env:LOCALAPPDATA 'Docker\Docker Desktop.exe')
    )
    $desktop = $desktopCandidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
    if (-not $desktop) {
        throw 'Docker daemon 未运行，且未找到 Docker Desktop 启动程序'
    }
    Write-Host 'S14 已 ready，正在启动本机 Docker Desktop…'
    Start-Process -FilePath $desktop -WindowStyle Hidden | Out-Null
    if (-not (Wait-ForCondition -TimeoutSeconds 120 -Condition { Test-DockerReady })) {
        throw 'Docker Desktop 在 120 秒内未 ready'
    }
}

function Start-LocalOpenWebUi {
    param([Parameter(Mandatory = $true)][string]$Command)

    New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null
    $names = @(
        'OLLAMA_BASE_URL',
        'ENABLE_OLLAMA_API',
        'OPENAI_API_BASE_URL',
        'OPENAI_API_KEY',
        'WEBUI_NAME',
        'WEBUI_AUTH',
        'ENABLE_SIGNUP',
        'ENABLE_TITLE_GENERATION',
        'ENABLE_TAGS_GENERATION',
        'ENABLE_FOLLOW_UP_GENERATION',
        'ENABLE_SEARCH_QUERY_GENERATION',
        'ENABLE_RETRIEVAL_QUERY_GENERATION',
        'BYPASS_EMBEDDING_AND_RETRIEVAL',
        'RAG_EMBEDDING_MODEL',
        'OFFLINE_MODE'
    )
    $previous = @{}
    foreach ($name in $names) { $previous[$name] = [Environment]::GetEnvironmentVariable($name, 'Process') }
    try {
        [Environment]::SetEnvironmentVariable('OLLAMA_BASE_URL', $ApiBaseUrl.TrimEnd('/'), 'Process')
        # 同一模型 ID 同时由 Ollama/OpenAI 两个 provider 暴露时，Open WebUI 会发生路由冲突，
        # 可能把真实上游回复落成空消息。页面保持 Open WebUI，只保留已验收的 OpenAI /v1 链路。
        [Environment]::SetEnvironmentVariable('ENABLE_OLLAMA_API', 'False', 'Process')
        [Environment]::SetEnvironmentVariable('OPENAI_API_BASE_URL', "$($ApiBaseUrl.TrimEnd('/'))/v1", 'Process')
        [Environment]::SetEnvironmentVariable('OPENAI_API_KEY', 'polaris-local', 'Process')
        [Environment]::SetEnvironmentVariable('WEBUI_NAME', 'Polaris S14', 'Process')
        # 服务只绑定 loopback；首个本机试聊免登录，避免把账号初始化当成模型阻塞。
        [Environment]::SetEnvironmentVariable('WEBUI_AUTH', 'False', 'Process')
        [Environment]::SetEnvironmentVariable('ENABLE_SIGNUP', 'False', 'Process')
        # N=8 试聊阶段禁止标题/标签/追问等后台任务抢占唯一 S14 worker。
        [Environment]::SetEnvironmentVariable('ENABLE_TITLE_GENERATION', 'False', 'Process')
        [Environment]::SetEnvironmentVariable('ENABLE_TAGS_GENERATION', 'False', 'Process')
        [Environment]::SetEnvironmentVariable('ENABLE_FOLLOW_UP_GENERATION', 'False', 'Process')
        [Environment]::SetEnvironmentVariable('ENABLE_SEARCH_QUERY_GENERATION', 'False', 'Process')
        [Environment]::SetEnvironmentVariable('ENABLE_RETRIEVAL_QUERY_GENERATION', 'False', 'Process')
        # 首个 S14 试聊不使用 RAG，避免 Open WebUI 启动时下载无关的 sentence-transformers 模型。
        [Environment]::SetEnvironmentVariable('BYPASS_EMBEDDING_AND_RETRIEVAL', 'True', 'Process')
        [Environment]::SetEnvironmentVariable('RAG_EMBEDDING_MODEL', '', 'Process')
        [Environment]::SetEnvironmentVariable('OFFLINE_MODE', 'True', 'Process')
        return Start-Process -FilePath $Command -ArgumentList @('serve', '--host', '127.0.0.1', '--port', "$WebUiPort") `
            -PassThru -WindowStyle Hidden `
            -RedirectStandardOutput (Join-Path $runtimeDir 'open_webui.stdout.log') `
            -RedirectStandardError (Join-Path $runtimeDir 'open_webui.stderr.log')
    }
    finally {
        foreach ($name in $names) {
            [Environment]::SetEnvironmentVariable($name, $previous[$name], 'Process')
        }
    }
}

function Start-DockerOpenWebUi {
    Ensure-DockerReady
    $docker = (Get-Command docker -ErrorAction Stop | Select-Object -First 1).Source
    $containerName = 'polaris-open-webui'
    $existing = & $docker ps -a --filter "name=^/$containerName$" --format '{{.Names}}' 2>$null
    if ($LASTEXITCODE -ne 0) { throw '无法查询 Docker 容器' }

    $apiUri = [Uri]$ApiBaseUrl
    $dockerHost = if ($apiUri.Host -in @('127.0.0.1', 'localhost', '::1')) { 'host.docker.internal' } else { $apiUri.Host }
    $dockerApiBase = "{0}://{1}:{2}" -f $apiUri.Scheme, $dockerHost, $apiUri.Port

    if ($existing) {
        $environment = @(& $docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' $containerName 2>$null)
        $ports = [string](& $docker port $containerName 8080/tcp 2>$null)
        if ($environment -notcontains "OLLAMA_BASE_URL=$dockerApiBase") {
            throw "已有 $containerName 的 OLLAMA_BASE_URL 与当前 S14 API 不一致；为保护现有数据不自动重建"
        }
        if ($ports -notmatch ":$WebUiPort$") {
            throw "已有 $containerName 未映射到请求的 WebUiPort=$WebUiPort；不自动重建"
        }
        & $docker start $containerName | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "启动 $containerName 失败" }
        return
    }

    $images = @(& $docker image ls --format '{{.Repository}}:{{.Tag}}' 2>$null | Where-Object { $_ -match '(?i)open-?webui' })
    if ($LASTEXITCODE -ne 0 -or $images.Count -eq 0) {
        throw '本地 Docker 中没有 Open WebUI 镜像；脚本按合同不会自动 pull 或下载'
    }
    $image = [string]$images[0]
    Write-Host "使用本地镜像 $image 创建 $containerName（--pull never）…"
    & $docker run --pull never --detach --name $containerName --restart unless-stopped `
        --publish "127.0.0.1:${WebUiPort}:8080" `
        --add-host 'host.docker.internal:host-gateway' `
        --env "OLLAMA_BASE_URL=$dockerApiBase" `
        --env 'ENABLE_OLLAMA_API=true' `
        --env 'WEBUI_NAME=Polaris S14' `
        --volume 'polaris-open-webui-data:/app/backend/data' `
        $image | Out-Null
    if ($LASTEXITCODE -ne 0) { throw '创建 Open WebUI 容器失败' }
}

try {
    $report = Get-EnvironmentReport
    if ($report.s14.api_binary_stale) {
        throw 'polaris_api 二进制早于 S14 运行时依赖源码；先重新离线构建，禁止 Open WebUI 静默连接旧模型代码'
    }
    if (-not $report.s14.ready) {
        if ($report.s14.health_reachable) {
            throw "S14 health 明确未 ready：$($report.s14.detail)"
        }
        if (-not $report.s14.api_binary) {
            throw 'S14 API 未运行且 polaris_api 二进制不存在；先执行 cargo build --offline -p polaris_api'
        }
        if ($report.s14.api_binary_stale) {
            throw 'polaris_api 二进制早于源码；为避免运行旧门禁，先执行 cargo build --offline -p polaris_api'
        }
        Write-Host "启动本地 API 适配器：$($report.s14.api_binary)"
        $startedApi = Start-ApiAdapter -Binary $report.s14.api_binary
        $apiReady = Wait-ForCondition -TimeoutSeconds $ApiWaitSeconds -Condition {
            $current = Get-EnvironmentReport
            return [bool]$current.s14.ready
        }
        if (-not $apiReady) {
            $current = Get-EnvironmentReport
            throw "API 已启动但 S14 health 未 ready：$($current.s14.detail)"
        }
        $report = Get-EnvironmentReport
    }

    if (-not $report.s14.ready) {
        throw "S14 health 未 ready：$($report.s14.detail)"
    }
    Write-Host 'S14 health 已严格确认 ready，允许进入前端启动阶段。'

    if (-not $report.open_webui.ready) {
        if ($report.open_webui.port_listening) {
            throw "端口 $WebUiPort 已被非 Open WebUI 服务占用"
        }
        $selectedBackend = $Backend
        if ($selectedBackend -eq 'Auto') {
            $selectedBackend = if ($report.open_webui.local_command) { 'Local' } else { 'Docker' }
        }
        if ($selectedBackend -eq 'Local') {
            if (-not $report.open_webui.local_command) {
                throw '未发现 open-webui 本地命令'
            }
            $startedWebUi = Start-LocalOpenWebUi -Command $report.open_webui.local_command
        } else {
            Start-DockerOpenWebUi
        }
    }

    $webUiReady = Wait-ForCondition -TimeoutSeconds $WebUiWaitSeconds -Condition {
        $current = Get-EnvironmentReport
        return [bool]($current.s14.ready -and $current.open_webui.ready)
    }
    if (-not $webUiReady) {
        throw 'Open WebUI 在等待期内未 ready，或等待期间 S14 health 失效'
    }

    $completed = $true
    $url = "http://127.0.0.1:$WebUiPort"
    Write-Host "Polaris S14 试聊入口已就绪：$url"
    if (-not $NoBrowser) { Start-Process $url | Out-Null }
}
catch {
    Write-Error $_.Exception.Message -ErrorAction Continue
    exit 20
}
finally {
    if (-not $completed) {
        if ($null -ne $startedWebUi -and -not $startedWebUi.HasExited) {
            Stop-Process -Id $startedWebUi.Id -Force -ErrorAction SilentlyContinue
        }
        if ($null -ne $startedApi -and -not $startedApi.HasExited) {
            Stop-Process -Id $startedApi.Id -Force -ErrorAction SilentlyContinue
        }
    }
}
