$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::InputEncoding = $utf8
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

$root = Split-Path -Parent $PSScriptRoot
$endpoint = 'http://127.0.0.1:8105'
$model = 'ColorLM-v17-Coder-Neural-Island'

function Test-ColorLMReady {
    try {
        $health = Invoke-RestMethod -Uri "$endpoint/health" -TimeoutSec 3
        $models = Invoke-RestMethod -Uri "$endpoint/v1/models" -TimeoutSec 5
        return $health.status -eq 'ok' -and $models.data[0].id -eq $model
    }
    catch {
        return $false
    }
}

if (-not (Test-ColorLMReady)) {
    $listener = Get-NetTCPConnection -LocalPort 8105 -State Listen -ErrorAction SilentlyContinue
    if ($listener) {
        throw '端口8105已被其他服务占用。请先运行 stop-colormlm-user.ps1，再重新启动。'
    }
    Push-Location $root
    try {
        & powershell -NoProfile -ExecutionPolicy Bypass -File `
            fast16\run-colormlm-v17-coder-island.ps1
        if ($LASTEXITCODE -ne 0) {
            throw "ColorLM v17启动失败，退出码: $LASTEXITCODE"
        }
    }
    finally {
        Pop-Location
    }
}

if (-not (Test-ColorLMReady)) {
    throw 'ColorLM启动器已返回，但健康检查或模型alias不匹配。'
}

Write-Host ''
Write-Host 'ColorLM v17 已就绪' -ForegroundColor Green
Write-Host "健康检查: $endpoint/health"
Write-Host "OpenAI Base URL: $endpoint/v1"
Write-Host "模型名: $model"
Write-Host 'API Key: local（任意非空字符串均可）'
Write-Host ''
Write-Host '注意：http://127.0.0.1:8105/ 返回404是正常的；它不是网页聊天页。'
