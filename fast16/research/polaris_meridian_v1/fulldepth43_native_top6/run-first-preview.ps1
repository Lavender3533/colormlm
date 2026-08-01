param(
    [string] $ProxyUri = 'http://127.0.0.1:7897',
    [long] $DownloadBudgetBytes = 24000000000,
    [int] $RangeWorkers = 8
)

$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
if ($ProxyUri) {
    $env:HTTP_PROXY = $ProxyUri
    $env:HTTPS_PROXY = $ProxyUri
}

$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..\..')).Path
Set-Location $root
$started = [DateTimeOffset]::UtcNow

& python -X utf8 -m fast16.research.polaris_meridian_v1.fulldepth43_native_top6.executor run `
    --asset-root 'D:/models/Polaris-S14' `
    --catalog 'D:/models/Polaris-S14/fulldepth43_native_top6_catalog.json' `
    --report 'fast16/research/polaris_meridian_v1/fulldepth43_native_top6/first_preview_real_report.json' `
    --endpoint 'https://huggingface.co' `
    --download-missing `
    --download-budget-bytes $DownloadBudgetBytes `
    --token-count 5 `
    --forced-prefill 'fast16/research/polaris_meridian_v1/fulldepth43_native_top6/first_preview_forced_prefill.json' `
    --head-chunk-size 4096 `
    --range-attempts 4 `
    --range-workers $RangeWorkers

$exitCode = $LASTEXITCODE
[pscustomobject]@{
    exit_code = $exitCode
    elapsed_seconds = ([DateTimeOffset]::UtcNow - $started).TotalSeconds
    report = 'fast16/research/polaris_meridian_v1/fulldepth43_native_top6/first_preview_real_report.json'
} | ConvertTo-Json -Compress
exit $exitCode
