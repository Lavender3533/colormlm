param(
    [ValidateRange(1, 4)]
    [int]$MaxPackGiB = 4,

    [string]$PythonExecutable = 'python',

    [switch]$PlanOnly
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
if ($running) {
    throw "polaris_api 正在运行（PID: $($running.Id -join ',')）；请先停止服务再更新不可变 pack index"
}

$arguments = @(
    '-X', 'utf8', $writer,
    '--cache-root', $cacheRoot,
    '--pack-root', $packRoot,
    '--max-pack-gib', [string]$MaxPackGiB
)
if ($PlanOnly) {
    $arguments += '--dry-run'
}

& $PythonExecutable @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Range pack writer 失败，exit=$LASTEXITCODE"
}

[pscustomobject]@{
    Index = $index
    IndexReady = Test-Path -LiteralPath $index -PathType Leaf
    MaxPackGiB = $MaxPackGiB
    PlanOnly = [bool]$PlanOnly
    LooseFilesDeleted = 0
}
