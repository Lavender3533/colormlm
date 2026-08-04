param(
    [ValidateRange(1, 32)]
    [int]$MaxTokens = 8,

    [ValidateRange(1, 24)]
    [int]$RangeWorkers = 12,

    [ValidateRange(20, 128)]
    [int]$DiskReserveGiB = 24,

    [switch]$ProjectionTwinFallback
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'polaris-s14-rush-env.ps1')
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$modelRoot = 'D:\models\Polaris-S14'
$cacheRoot = Join-Path $modelRoot 'range_cache'
$binary = Join-Path $root 'scheduler\target\release\polaris_api.exe'
$logRoot = Join-Path $root '.tmp-polaris-tests'
$stderrLog = Join-Path $logRoot 'polaris-s14-rush.stderr.log'
$stdoutLog = Join-Path $logRoot 'polaris-s14-rush.stdout.log'

if (-not (Test-Path -LiteralPath $binary -PathType Leaf)) {
    throw "缺少 release S14 API: $binary"
}
if (-not (Test-Path -LiteralPath $cacheRoot -PathType Container)) {
    throw "缺少 Polaris-S14 Range cache: $cacheRoot"
}

New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$rushProfile = Get-PolarisS14RushEnvironment `
    -MaxTokens $MaxTokens `
    -RangeWorkers $RangeWorkers `
    -DiskReserveGiB $DiskReserveGiB `
    -ProjectionTwinFallback ([bool]$ProjectionTwinFallback)
$environment = $rushProfile.Environment

$running = Get-Process -Name polaris_api -ErrorAction SilentlyContinue
if ($running) {
    throw "polaris_api 已运行（PID: $($running.Id -join ',')）；为避免双GPU实例，本脚本拒绝重复启动"
}

$process = Start-Process `
    -FilePath $binary `
    -WorkingDirectory $root `
    -Environment $environment `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -PassThru

[pscustomobject]@{
    Pid = $process.Id
    Address = 'http://127.0.0.1:11435'
    MaxTokens = $MaxTokens
    RangeWorkers = $RangeWorkers
    CacheBudgetGiB = [Math]::Round($rushProfile.CacheBudgetBytes / 1GB, 2)
    DiskReserveGiB = $DiskReserveGiB
    ProjectionTwinFallback = [bool]$ProjectionTwinFallback
    RangePackIndex = $rushProfile.RangePackIndex
    StderrLog = $stderrLog
}
