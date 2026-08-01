$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::InputEncoding = $utf8
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

$endpoint = 'http://127.0.0.1:8105'
$expectedModel = 'ColorLM-v17-Coder-Neural-Island'

try {
    $health = Invoke-RestMethod -Uri "$endpoint/health" -TimeoutSec 5
    $models = Invoke-RestMethod -Uri "$endpoint/v1/models" -TimeoutSec 5
    $actualModel = $models.data[0].id
    if ($health.status -ne 'ok' -or $actualModel -ne $expectedModel) {
        throw "服务响应异常: health=$($health.status), model=$actualModel"
    }

    $body = @{
        model = $expectedModel
        messages = @(@{ role = 'user'; content = '只回复 OK' })
        temperature = 0
        max_tokens = 8
    } | ConvertTo-Json -Depth 6
    $reply = Invoke-RestMethod `
        -Uri "$endpoint/v1/chat/completions" `
        -Method Post `
        -ContentType 'application/json; charset=utf-8' `
        -Body ([Text.Encoding]::UTF8.GetBytes($body)) `
        -TimeoutSec 90

    Write-Host 'ColorLM 检查通过' -ForegroundColor Green
    Write-Host "health: $($health.status)"
    Write-Host "model:  $actualModel"
    Write-Host "reply:  $($reply.choices[0].message.content)"
    exit 0
}
catch {
    Write-Host 'ColorLM 当前不可用' -ForegroundColor Red
    Write-Host $_.Exception.Message
    Write-Host '请运行项目根目录的 启动ColorLM.bat'
    exit 1
}
