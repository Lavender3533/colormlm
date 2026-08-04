Set-StrictMode -Version Latest

function Get-PolarisS14RushEnvironment {
    [CmdletBinding()]
    param(
        [string]$ApiAddress = '127.0.0.1:11435',

        [ValidateRange(1, 32)]
        [int]$MaxTokens = 8,

        [ValidateRange(1, 24)]
        [int]$RangeWorkers = 12,

        [ValidateRange(20, 128)]
        [int]$DiskReserveGiB = 24,

        [bool]$ExplicitPageFetch = $true,

        [string]$RangeProxyUrl = 'http://127.0.0.1:7897',

        [bool]$ProjectionTwinFallback = $false,

        [string]$ModelRoot = 'D:\models\Polaris-S14'
    )

    $cacheRoot = Join-Path $ModelRoot 'range_cache'
    $packIndex = Join-Path $ModelRoot 'range_cache_pack\index.v1.json'
    if (-not (Test-Path -LiteralPath $cacheRoot -PathType Container)) {
        throw "缺少 Polaris-S14 Range cache: $cacheRoot"
    }
    if ($ExplicitPageFetch -and $RangeProxyUrl.TrimEnd('/') -ne 'http://127.0.0.1:7897') {
        throw '当前 production Range 策略只允许 http://127.0.0.1:7897'
    }

    # 已下载的热页不能被固定 64 GiB 上限反复驱逐；预算只受实际磁盘保留线约束。
    $cacheBytes = [int64]((Get-ChildItem -LiteralPath $cacheRoot -File -Filter '*.bin' |
        Measure-Object Length -Sum).Sum)
    $driveName = [IO.Path]::GetPathRoot($cacheRoot).TrimEnd(':', '\')
    $drive = Get-PSDrive -Name $driveName
    $reserveBytes = [int64]$DiskReserveGiB * 1GB
    $growthBytes = [Math]::Max([int64]0, [int64]$drive.Free - $reserveBytes)
    $cacheBudgetBytes = [Math]::Max([int64](64GB), $cacheBytes + $growthBytes)

    $environment = @{
        POLARIS_API_ADDR = $ApiAddress
        POLARIS_S14_EXPLICIT_PAGE_FETCH = $(if ($ExplicitPageFetch) { '1' } else { '0' })
        POLARIS_S14_PROJECTION_TWIN_FALLBACK = $(if ($ProjectionTwinFallback) { '1' } else { '0' })
        POLARIS_S14_PACKED_L2_MIB = '1024'
        POLARIS_S14_STARFOLD_MICROTILE_MIB = '16'
        # RX 5700 XT 上 K8 会把 checkpoint arena 推入 host memory；急行版固定物理 K4。
        POLARIS_S14_PREFILL_MAX_K = '4'
        POLARIS_S14_DEFAULT_MAX_TOKENS = [string]$MaxTokens
        POLARIS_S14_REQUEST_DEADLINE_SECS = '7200'
        S14_DYNAMIC_PAGE_FETCH_WORKERS = [string]$RangeWorkers
        S14_DYNAMIC_PAGE_CACHE_BUDGET_BYTES = [string]$cacheBudgetBytes
        S14_DYNAMIC_PAGE_DISK_RESERVE_BYTES = [string]$reserveBytes
        S14_DYNAMIC_PAGE_FETCH_MODELSCOPE_ENDPOINT = 'https://www.modelscope.cn/models'
        S14_DYNAMIC_PAGE_FETCH_LFS_SNAPSHOT = (Join-Path $ModelRoot 'hub_blobs_snapshot.json')
    }
    if ($ExplicitPageFetch) {
        $environment.HTTP_PROXY = $RangeProxyUrl
        $environment.HTTPS_PROXY = $RangeProxyUrl
    }
    if (Test-Path -LiteralPath $packIndex -PathType Leaf) {
        $environment.POLARIS_S14_RANGE_PACK_INDEX = (Resolve-Path -LiteralPath $packIndex).Path
    }

    return [pscustomobject]@{
        Environment = $environment
        CacheBudgetBytes = $cacheBudgetBytes
        DiskReserveBytes = $reserveBytes
        RangePackIndex = $(if ($environment.ContainsKey('POLARIS_S14_RANGE_PACK_INDEX')) {
            $environment.POLARIS_S14_RANGE_PACK_INDEX
        } else {
            $null
        })
    }
}
