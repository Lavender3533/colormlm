$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::InputEncoding = $utf8
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

$stopped = 0
foreach ($port in 8105, 8106) {
    $listeners = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    foreach ($listener in $listeners) {
        $process = Get-CimInstance Win32_Process -Filter "ProcessId=$($listener.OwningProcess)"
        $command = [string]$process.CommandLine
        if ($process.Name -ne 'llama-server.exe' -or $command -notlike '*大模型ssd化*') {
            Write-Warning "端口$port不是本项目的llama-server，拒绝停止。PID=$($listener.OwningProcess)"
            continue
        }
        Stop-Process -Id $listener.OwningProcess -Force -ErrorAction SilentlyContinue
        Write-Host "已停止ColorLM: port=$port, pid=$($listener.OwningProcess)"
        $stopped++
    }
}
if ($stopped -eq 0) {
    Write-Host '没有发现正在运行的ColorLM服务。'
}
