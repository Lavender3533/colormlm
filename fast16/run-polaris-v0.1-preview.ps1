param(
    [string] $ListenHost = '127.0.0.1',
    [int] $ListenPort = 8140,
    [string] $UpstreamUrl = 'http://127.0.0.1:8138',
    [int] $StartupTimeoutSeconds = 900
)

$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$gatewayScript = Join-Path $root 'fast16/release/polaris-v0.1-preview/gateway.py'
$runtimeScript = Join-Path $root 'fast16/run-colormlm-v38-qwen36-sequence-policy.ps1'
$healthUrl = $UpstreamUrl.TrimEnd('/') + '/health'
$defaultUpstream = 'http://127.0.0.1:8138'

function Test-PolarisUpstream {
    try {
        $response = Invoke-WebRequest -Uri $healthUrl -UseBasicParsing -TimeoutSec 2
        return $response.StatusCode -ge 200 -and $response.StatusCode -lt 300
    }
    catch {
        return $false
    }
}

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw '未找到 python 命令。请先安装 Python 3.10 或更高版本。'
}

$runtimeJob = $null
if (-not (Test-PolarisUpstream)) {
    if ($UpstreamUrl.TrimEnd('/') -ne $defaultUpstream) {
        throw "自定义上游 $UpstreamUrl 当前不可用；为避免启动错误端口，不会自动启动默认 v38。"
    }

    Write-Host 'v38 上游尚未在线，正在后台启动现有 Sequence Policy 服务……' -ForegroundColor Yellow
    $runtimeJob = Start-Job -Name "Polaris-v38-$PID" -ScriptBlock {
        param($ScriptPath, $WorkingDirectory)
        $env:PYTHONUTF8 = '1'
        $env:PYTHONIOENCODING = 'utf-8'
        Set-Location $WorkingDirectory
        & $ScriptPath
    } -ArgumentList $runtimeScript, $root

    $deadline = [DateTime]::UtcNow.AddSeconds($StartupTimeoutSeconds)
    while (-not (Test-PolarisUpstream)) {
        if ($runtimeJob.State -in @('Completed', 'Failed', 'Stopped', 'Disconnected')) {
            $detail = (Receive-Job -Job $runtimeJob -Keep 2>&1 | Out-String).Trim()
            throw "v38 后台启动作业已提前结束（$($runtimeJob.State)）：$detail"
        }
        if ([DateTime]::UtcNow -ge $deadline) {
            throw "等待 v38 健康检查超时（$StartupTimeoutSeconds 秒）；脚本未调用任何停止服务命令。"
        }
        Write-Host '.' -NoNewline
        Start-Sleep -Seconds 2
    }
    Write-Host ''
}

Write-Host "v38 上游已就绪：$healthUrl" -ForegroundColor Green
Write-Host "Polaris v0.1 Preview：http://${ListenHost}:$ListenPort/" -ForegroundColor Cyan
Write-Warning '当前为 draft-only；exact_verifier=not_ready；不包含 FullDepth/K3 能力。'

Set-Location $root
& python $gatewayScript `
    --listen-host $ListenHost `
    --listen-port $ListenPort `
    --upstream $UpstreamUrl

exit $LASTEXITCODE
