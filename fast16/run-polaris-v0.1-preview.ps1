param(
    [string] $ListenHost = '127.0.0.1',
    [int] $ListenPort = 8140,
    [string] $UpstreamUrl = 'http://127.0.0.1:8138',
    [int] $StartupTimeoutSeconds = 900,
    [switch] $SelfTest
)

$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$gatewayScript = Join-Path $root 'fast16/release/polaris-v0.1-preview/gateway.py'
$runtimeScript = Join-Path $root 'fast16/run-colormlm-v38-qwen36-sequence-policy.ps1'
$healthUrl = $UpstreamUrl.TrimEnd('/') + '/health'
$defaultUpstream = 'http://127.0.0.1:8138'

if ($SelfTest) {
    foreach ($requiredPath in @($gatewayScript, $runtimeScript)) {
        if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
            throw "Required file is missing: $requiredPath"
        }
    }
    if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
        throw 'The python command was not found.'
    }
    Write-Output (
        'POLARIS_SELF_TEST_OK edition={0} version={1}' -f `
            $PSVersionTable.PSEdition, $PSVersionTable.PSVersion.ToString()
    )
    exit 0
}

function Test-PolarisUpstream {
    try {
        $response = Invoke-WebRequest -Uri $healthUrl -UseBasicParsing -TimeoutSec 2
        return $response.StatusCode -ge 200 -and $response.StatusCode -lt 300
    }
    catch {
        return $false
    }
}

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw 'The python command was not found. Install Python 3.10 or newer.'
}

$runtimeJob = $null
if (-not (Test-PolarisUpstream)) {
    if ($UpstreamUrl.TrimEnd('/') -ne $defaultUpstream) {
        throw "Custom upstream $UpstreamUrl is unavailable; the default v38 service will not be started."
    }

    Write-Host 'v38 is offline. Starting the existing Sequence Policy service in a background job...' -ForegroundColor Yellow
    $runtimeJob = Start-Job -Name "Polaris-v38-$PID" -ScriptBlock {
        param($ScriptPath, $WorkingDirectory)
        $env:PYTHONUTF8 = '1'
        $env:PYTHONIOENCODING = 'utf-8'
        Set-Location $WorkingDirectory
        & $ScriptPath
    } -ArgumentList $runtimeScript, $root

    $deadline = [DateTime]::UtcNow.AddSeconds($StartupTimeoutSeconds)
    while (-not (Test-PolarisUpstream)) {
        if ($runtimeJob.State -in @('Completed', 'Failed', 'Stopped', 'Disconnected')) {
            $detail = (Receive-Job -Job $runtimeJob -Keep 2>&1 | Out-String).Trim()
            throw "The v38 startup job exited early ($($runtimeJob.State)): $detail"
        }
        if ([DateTime]::UtcNow -ge $deadline) {
            throw "Timed out after $StartupTimeoutSeconds seconds waiting for v38. No stop command was issued."
        }
        Write-Host '.' -NoNewline
        Start-Sleep -Seconds 2
    }
    Write-Host ''
}

Write-Host "v38 upstream is ready: $healthUrl" -ForegroundColor Green
Write-Host "Polaris v0.1 Preview: http://${ListenHost}:$ListenPort/" -ForegroundColor Cyan
Write-Warning 'draft-only; exact_verifier=not_ready; FullDepth and K3 are unavailable.'

Set-Location $root
& python $gatewayScript `
    --listen-host $ListenHost `
    --listen-port $ListenPort `
    --upstream $UpstreamUrl

exit $LASTEXITCODE
