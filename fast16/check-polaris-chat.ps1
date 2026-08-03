[CmdletBinding()]
param(
    [string]$ApiBaseUrl = $(if ($env:POLARIS_API_BASE_URL) { $env:POLARIS_API_BASE_URL } else { 'http://127.0.0.1:11435' }),
    [int]$WebUiPort = $(if ($env:POLARIS_WEBUI_PORT) { [int]$env:POLARIS_WEBUI_PORT } else { 3000 }),
    [switch]$Json,
    [switch]$RequireReady
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
[Console]::InputEncoding = [Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$OutputEncoding = [Text.UTF8Encoding]::new($false)

function Get-HttpProbe {
    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [int]$TimeoutSeconds = 2
    )

    Add-Type -AssemblyName System.Net.Http
    $handler = [System.Net.Http.HttpClientHandler]::new()
    $handler.UseProxy = $false
    $client = [System.Net.Http.HttpClient]::new($handler)
    $client.Timeout = [TimeSpan]::FromSeconds($TimeoutSeconds)
    try {
        $response = $client.GetAsync($Uri).GetAwaiter().GetResult()
        $body = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
        return [ordered]@{
            reachable = $true
            status_code = [int]$response.StatusCode
            body = $body
            error = $null
        }
    }
    catch {
        return [ordered]@{
            reachable = $false
            status_code = $null
            body = ''
            error = $_.Exception.Message
        }
    }
    finally {
        $client.Dispose()
        $handler.Dispose()
    }
}

function Get-CommandPath {
    param([Parameter(Mandatory = $true)][string]$Name)
    $command = Get-Command $Name -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -eq $command) { return $null }
    return $command.Source
}

function Test-TcpPort {
    param(
        [Parameter(Mandatory = $true)][string]$HostName,
        [Parameter(Mandatory = $true)][int]$Port
    )
    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        $task = $client.ConnectAsync($HostName, $Port)
        return $task.Wait(400) -and $client.Connected
    }
    catch {
        return $false
    }
    finally {
        $client.Dispose()
    }
}

try {
    $apiUri = [Uri]$ApiBaseUrl
}
catch {
    throw "ApiBaseUrl 不是合法 URL：$ApiBaseUrl"
}
if ($apiUri.Scheme -ne 'http' -and $apiUri.Scheme -ne 'https') {
    throw 'ApiBaseUrl 只支持 http/https'
}
if ($WebUiPort -lt 1 -or $WebUiPort -gt 65535) {
    throw 'WebUiPort 必须位于 1..65535'
}

$normalizedApiBase = $ApiBaseUrl.TrimEnd('/')
$apiPortListening = Test-TcpPort -HostName $apiUri.Host -Port $apiUri.Port
$healthProbe = if ($apiPortListening) {
    Get-HttpProbe -Uri "$normalizedApiBase/healthz"
} else {
    [ordered]@{
        reachable = $false
        status_code = $null
        body = ''
        error = "TCP $($apiUri.Host):$($apiUri.Port) 未监听"
    }
}
$healthBody = $null
if ($healthProbe.body) {
    try { $healthBody = $healthProbe.body | ConvertFrom-Json } catch { $healthBody = $null }
}
$healthReady = $healthProbe.status_code -eq 200 -and
    $null -ne $healthBody -and
    $healthBody.ready -eq $true -and
    $healthBody.model -eq 'Polaris-S14'
$healthDetail = if ($null -ne $healthBody -and $null -ne $healthBody.detail) {
    [string]$healthBody.detail
} elseif ($healthProbe.error) {
    [string]$healthProbe.error
} else {
    "HTTP $($healthProbe.status_code)"
}

$webUiPortListening = Test-TcpPort -HostName '127.0.0.1' -Port $WebUiPort
$webUiProbe = if ($webUiPortListening) {
    Get-HttpProbe -Uri "http://127.0.0.1:$WebUiPort/" -TimeoutSeconds 2
} else {
    [ordered]@{ reachable = $false; status_code = $null; body = ''; error = "TCP 127.0.0.1:$WebUiPort 未监听" }
}
$webUiReady = $webUiProbe.status_code -eq 200 -and
    $webUiProbe.body -match '(?i)(open[ -]?webui|<html|<!doctype)'

$openWebUiPath = Get-CommandPath -Name 'open-webui'
if (-not $openWebUiPath) {
    $projectOpenWebUi = Join-Path $PSScriptRoot 'runtime\polaris-chat\venv\Scripts\open-webui.exe'
    if (Test-Path -LiteralPath $projectOpenWebUi -PathType Leaf) {
        $openWebUiPath = $projectOpenWebUi
    }
}

$dockerPath = Get-CommandPath -Name 'docker'
$dockerRunning = $false
$dockerVersion = $null
$dockerContainer = $null
$dockerImage = $null
if (-not $openWebUiPath -and $dockerPath -and ((Test-Path -LiteralPath '\\.\pipe\dockerDesktopLinuxEngine') -or (Test-Path -LiteralPath '\\.\pipe\docker_engine'))) {
    $dockerVersionOutput = & $dockerPath info --format '{{.ServerVersion}}' 2>$null
    if ($LASTEXITCODE -eq 0 -and $dockerVersionOutput) {
        $dockerRunning = $true
        $dockerVersion = [string]($dockerVersionOutput | Select-Object -First 1)
        $dockerContainerOutput = & $dockerPath ps -a --filter 'name=^/polaris-open-webui$' --format '{{.Names}}|{{.Status}}|{{.Ports}}' 2>$null
        if ($LASTEXITCODE -eq 0 -and $dockerContainerOutput) {
            $dockerContainer = [string]($dockerContainerOutput | Select-Object -First 1)
        }
        $dockerImages = @(& $dockerPath image ls --format '{{.Repository}}:{{.Tag}}' 2>$null | Where-Object { $_ -match '(?i)open-?webui' })
        if ($LASTEXITCODE -eq 0 -and $dockerImages.Count -gt 0) {
            $dockerImage = [string]$dockerImages[0]
        }
    }
}

$ollamaPath = Get-CommandPath -Name 'ollama'
$ollamaPortListening = Test-TcpPort -HostName '127.0.0.1' -Port 11434
$ollamaProbe = if ($ollamaPortListening) {
    Get-HttpProbe -Uri 'http://127.0.0.1:11434/api/version'
} else {
    [ordered]@{ reachable = $false; status_code = $null; body = ''; error = 'TCP 127.0.0.1:11434 未监听' }
}
$pythonPath = Get-CommandPath -Name 'python'
$nodePath = Get-CommandPath -Name 'node'

$workspace = Split-Path -Parent $PSScriptRoot
$apiBinaryCandidates = @(
    (Join-Path $workspace 'scheduler\target\release\polaris_api.exe'),
    (Join-Path $workspace 'scheduler\target\debug\polaris_api.exe')
)
$apiBinary = $apiBinaryCandidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
$apiBinaryStale = $false
if ($apiBinary) {
    # polaris_api 静态链接 S14 runtime；只比较适配层自身会在底层 runtime 已修改后误报“二进制最新”。
    # 这里覆盖正式依赖链，避免 Open WebUI 实际启动旧模型代码。
    $apiSourceRoots = @(
        (Join-Path $workspace 'scheduler\polaris_api'),
        (Join-Path $workspace 'scheduler\ssd_inference'),
        (Join-Path $workspace 'scheduler\s14_runner')
    )
    $latestApiSource = $apiSourceRoots |
        Where-Object { Test-Path -LiteralPath $_ -PathType Container } |
        ForEach-Object { Get-ChildItem -LiteralPath $_ -Recurse -File -ErrorAction SilentlyContinue } |
        Where-Object { $_.Extension -eq '.rs' -or $_.Name -in @('Cargo.toml', 'build.rs') } |
        Sort-Object LastWriteTimeUtc -Descending |
        Select-Object -First 1
    if ($latestApiSource) {
        $apiBinaryStale = (Get-Item -LiteralPath $apiBinary).LastWriteTimeUtc -lt $latestApiSource.LastWriteTimeUtc
    }
}

$blockers = [System.Collections.Generic.List[string]]::new()
if (-not $healthReady) {
    $blockers.Add("S14 health 未 ready：$healthDetail")
}
if (-not $webUiReady -and -not $openWebUiPath -and -not $dockerContainer -and -not $dockerImage) {
    if ($dockerPath -and -not $dockerRunning) {
        $blockers.Add('Docker 已安装但 daemon 未运行，当前无法确认本地是否已有 Open WebUI 镜像')
    } else {
        $blockers.Add('未发现可直接复用的 Open WebUI 命令、容器或本地镜像')
    }
}
if (-not $apiBinary -and -not $healthProbe.reachable) {
    $blockers.Add('未发现 polaris_api 二进制；先在 scheduler 下执行 cargo build --offline -p polaris_api')
}
if ($apiBinaryStale) {
    $blockers.Add('polaris_api 二进制早于 S14 运行时依赖源码；为避免网页连接旧模型代码，必须先重新离线构建')
}

$frontendAvailable = $webUiReady -or [bool]$openWebUiPath -or [bool]$dockerContainer -or [bool]$dockerImage
$report = [ordered]@{
    schema = 'polaris-chat-environment-v1'
    checked_at = [DateTimeOffset]::Now.ToString('o')
    ready_to_launch = [bool]($healthReady -and $frontendAvailable -and -not $apiBinaryStale)
    s14 = [ordered]@{
        model = 'Polaris-S14'
        api_base_url = $normalizedApiBase
        health_reachable = [bool]$healthProbe.reachable
        health_status_code = $healthProbe.status_code
        ready = [bool]$healthReady
        detail = $healthDetail
        api_binary = $apiBinary
        api_binary_stale = [bool]$apiBinaryStale
    }
    open_webui = [ordered]@{
        url = "http://127.0.0.1:$WebUiPort"
        port_listening = [bool]$webUiPortListening
        ready = [bool]$webUiReady
        local_command = $openWebUiPath
        docker_container = $dockerContainer
        docker_image = $dockerImage
    }
    docker = [ordered]@{
        installed = [bool]$dockerPath
        command = $dockerPath
        daemon_running = [bool]$dockerRunning
        server_version = $dockerVersion
    }
    ollama = [ordered]@{
        installed = [bool]$ollamaPath
        command = $ollamaPath
        native_endpoint_ready = [bool]($ollamaProbe.status_code -eq 200)
        note = 'Polaris 使用独立 11435 兼容端口，不启动或覆盖本机 Ollama 11434'
    }
    runtimes = [ordered]@{
        python = $pythonPath
        node = $nodePath
    }
    blockers = @($blockers)
}

if ($Json) {
    $report | ConvertTo-Json -Depth 6
} else {
    Write-Host 'Polaris S14 试聊环境检查'
    Write-Host "  S14 health : $(if ($healthReady) { 'READY' } else { 'NOT READY' }) ($healthDetail)"
    Write-Host "  API         : $normalizedApiBase"
    Write-Host "  Open WebUI  : $(if ($webUiReady) { 'RUNNING' } elseif ($openWebUiPath) { 'LOCAL COMMAND FOUND' } elseif ($dockerContainer -or $dockerImage) { 'DOCKER ASSET FOUND' } else { 'NOT FOUND/UNKNOWN' })"
    Write-Host "  Docker      : $(if (-not $dockerPath) { 'MISSING' } elseif ($dockerRunning) { "RUNNING $dockerVersion" } else { 'INSTALLED, STOPPED' })"
    Write-Host "  Ollama      : $(if (-not $ollamaPath) { 'MISSING' } elseif ($ollamaProbe.status_code -eq 200) { 'RUNNING ON 11434' } else { 'INSTALLED, STOPPED' })"
    Write-Host "  Python/Node : $pythonPath / $nodePath"
    if ($blockers.Count -gt 0) {
        Write-Host '  阻塞项：'
        foreach ($blocker in $blockers) { Write-Host "    - $blocker" }
    }
}

if ($RequireReady -and -not $report.ready_to_launch) {
    exit 20
}
