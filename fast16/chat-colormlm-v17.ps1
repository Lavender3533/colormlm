$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::InputEncoding = $utf8
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

& "$PSScriptRoot\start-colormlm-v17-user.ps1"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$endpoint = 'http://127.0.0.1:8105/v1/chat/completions'
$model = 'ColorLM-v17-Coder-Neural-Island'
$messages = [System.Collections.ArrayList]::new()
[void]$messages.Add(@{
    role = 'system'
    content = '你是本地编程助手。使用用户的语言，直接回答，代码必须完整，避免不必要的重复。'
})

Write-Host ''
Write-Host 'ColorLM v17 终端对话' -ForegroundColor Cyan
Write-Host '命令：/clear 清空上下文，/exit 退出'
while ($true) {
    $inputText = Read-Host '你'
    if ($inputText -eq '/exit') { break }
    if ($inputText -eq '/clear') {
        while ($messages.Count -gt 1) { $messages.RemoveAt($messages.Count - 1) }
        Write-Host '上下文已清空。'
        continue
    }
    if ([string]::IsNullOrWhiteSpace($inputText)) { continue }

    [void]$messages.Add(@{ role = 'user'; content = $inputText })
    $body = @{
        model = $model
        messages = @($messages)
        temperature = 0.2
        max_tokens = 1024
    } | ConvertTo-Json -Depth 10
    try {
        $reply = Invoke-RestMethod `
            -Uri $endpoint `
            -Method Post `
            -ContentType 'application/json; charset=utf-8' `
            -Body ([Text.Encoding]::UTF8.GetBytes($body)) `
            -TimeoutSec 600
        $content = [string]$reply.choices[0].message.content
        Write-Host "`nCLM> $content`n" -ForegroundColor Green
        [void]$messages.Add(@{ role = 'assistant'; content = $content })
    }
    catch {
        Write-Host "请求失败: $($_.Exception.Message)" -ForegroundColor Red
        $messages.RemoveAt($messages.Count - 1)
    }
}
