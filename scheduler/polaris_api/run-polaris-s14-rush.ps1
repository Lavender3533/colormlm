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

# Keep a real disk reserve while allowing already-fetched expert pages to stay
# resident.  The old fixed 64-GiB ceiling forced each new prompt to evict pages
# that the following block immediately needed again.
$cacheBytes = [int64]((Get-ChildItem -LiteralPath $cacheRoot -File |
    Where-Object Extension -eq '.bin' |
    Measure-Object Length -Sum).Sum)
$drive = Get-PSDrive -Name ([IO.Path]::GetPathRoot($cacheRoot).TrimEnd(':\'))
$reserveBytes = [int64]$DiskReserveGiB * 1GB
$growthBytes = [Math]::Max([int64]0, [int64]$drive.Free - $reserveBytes)
$cacheBudgetBytes = [Math]::Max([int64](64GB), $cacheBytes + $growthBytes)

$environment = @{
    POLARIS_API_ADDR = '127.0.0.1:11435'
    POLARIS_S14_EXPLICIT_PAGE_FETCH = '1'
    POLARIS_S14_PROJECTION_TWIN_FALLBACK = $(if ($ProjectionTwinFallback) { '1' } else { '0' })
    # 32-GiB hosts cannot afford the former 4-GiB packed L2 together with K8
    # checkpoint/state banks.  Two GiB keeps useful hot packets without forcing
    # Windows into 95% physical-memory pressure.
    POLARIS_S14_PACKED_L2_MIB = '2048'
    POLARIS_S14_STARFOLD_MICROTILE_MIB = '16'
    # K8 is throughput-oriented, but on an 8-GiB RX 5700 XT its checkpoint
    # arena falls back to host memory and starves the GPU.  K4 keeps the active
    # working set device-local; logical prompt semantics remain unchanged.
    POLARIS_S14_PREFILL_MAX_K = '4'
    POLARIS_S14_DEFAULT_MAX_TOKENS = [string]$MaxTokens
    POLARIS_S14_REQUEST_DEADLINE_SECS = '7200'
    S14_DYNAMIC_PAGE_FETCH_WORKERS = [string]$RangeWorkers
    S14_DYNAMIC_PAGE_CACHE_BUDGET_BYTES = [string]$cacheBudgetBytes
    S14_DYNAMIC_PAGE_DISK_RESERVE_BYTES = [string]$reserveBytes
    HTTP_PROXY = 'http://127.0.0.1:7897'
    HTTPS_PROXY = 'http://127.0.0.1:7897'
    S14_DYNAMIC_PAGE_FETCH_MODELSCOPE_ENDPOINT = 'https://www.modelscope.cn/models'
    S14_DYNAMIC_PAGE_FETCH_LFS_SNAPSHOT = (Join-Path $modelRoot 'hub_blobs_snapshot.json')
}

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
    CacheBudgetGiB = [Math]::Round($cacheBudgetBytes / 1GB, 2)
    DiskReserveGiB = $DiskReserveGiB
    ProjectionTwinFallback = [bool]$ProjectionTwinFallback
    StderrLog = $stderrLog
}
