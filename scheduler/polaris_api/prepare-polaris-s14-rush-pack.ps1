param(
    [ValidateRange(1, 4)]
    [int]$MaxPackGiB = 4,

    [string]$PythonExecutable = 'python',

    [switch]$PlanOnly,

    [switch]$Fill,

    [ValidateRange(20, 128)]
    [int]$DiskReserveGiB = 24,

    [ValidateRange(1, 64)]
    [int]$MaxPacks = 16
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$cacheRoot = 'D:\models\Polaris-S14\range_cache'
$packRoot = 'D:\models\Polaris-S14\range_cache_pack'
$writer = Join-Path $root 'fast16\research\polaris_meridian_v1\s14_range_pack\range_cache_pack_writer.py'
$index = Join-Path $packRoot 'index.v1.json'

if (-not (Test-Path -LiteralPath $cacheRoot -PathType Container)) {
    throw "缺少 Polaris-S14 Range cache: $cacheRoot"
}
if (-not (Test-Path -LiteralPath $writer -PathType Leaf)) {
    throw "缺少 Range pack writer: $writer"
}
if (-not (Get-Command $PythonExecutable -ErrorAction SilentlyContinue)) {
    throw "找不到 Python 可执行程序: $PythonExecutable"
}
$running = Get-Process -Name polaris_api -ErrorAction SilentlyContinue
if ($running -and -not $PlanOnly) {
    throw "polaris_api 正在运行（PID: $($running.Id -join ',')）；请先停止服务再更新不可变 pack index"
}

$reserveBytes = [int64]$DiskReserveGiB * 1GB
$iterationLimit = $(if ($Fill -and -not $PlanOnly) { $MaxPacks } else { 1 })
$packsCommitted = 0
$entriesAdded = 0
$lastStatus = $null

for ($iteration = 1; $iteration -le $iterationLimit; $iteration++) {
    $freeBytes = [int64](Get-PSDrive -Name ([IO.Path]::GetPathRoot($packRoot).TrimEnd(':', '\'))).Free
    $availableGiB = [int][Math]::Floor(($freeBytes - $reserveBytes) / 1GB)
    if ($availableGiB -lt 1) {
        $lastStatus = 'disk_reserve_reached'
        break
    }
    $iterationPackGiB = [Math]::Min($MaxPackGiB, $availableGiB)
    $arguments = @(
        '-X', 'utf8', $writer,
        '--cache-root', $cacheRoot,
        '--pack-root', $packRoot,
        '--max-pack-gib', [string]$iterationPackGiB
    )
    if ($PlanOnly) {
        $arguments += '--dry-run'
    }

    $writerOutput = @(& $PythonExecutable @arguments)
    $writerExit = $LASTEXITCODE
    $writerOutput | ForEach-Object { Write-Host $_ }
    if ($writerExit -ne 0) {
        throw "Range pack writer 失败，exit=$writerExit"
    }
    if ($writerOutput.Count -eq 0) {
        throw 'Range pack writer 未返回 JSON 结果'
    }
    try {
        $result = $writerOutput[-1] | ConvertFrom-Json
    } catch {
        throw "Range pack writer 末行不是 JSON：$($writerOutput[-1])"
    }
    $lastStatus = [string]$result.status
    if ($lastStatus -eq 'committed') {
        $packsCommitted++
        $entriesAdded += [int]$result.entries_added
        continue
    }
    break
}

[pscustomobject]@{
    Index = $index
    IndexReady = Test-Path -LiteralPath $index -PathType Leaf
    MaxPackGiB = $MaxPackGiB
    PlanOnly = [bool]$PlanOnly
    Fill = [bool]$Fill
    DiskReserveGiB = $DiskReserveGiB
    PacksCommitted = $packsCommitted
    EntriesAdded = $entriesAdded
    LastStatus = $lastStatus
    LooseFilesDeleted = 0
}
