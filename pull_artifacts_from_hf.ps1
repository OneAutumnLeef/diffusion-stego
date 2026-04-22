param(
    [string]$TargetDir = "."
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Get-Command python -ErrorAction SilentlyContinue
if ($python) {
    & $python.Source "$scriptDir\pull_artifacts_from_hf.py" $TargetDir
    exit $LASTEXITCODE
}

$py = Get-Command py -ErrorAction SilentlyContinue
if ($py) {
    & $py.Source -3 "$scriptDir\pull_artifacts_from_hf.py" $TargetDir
    exit $LASTEXITCODE
}

Write-Error "python or py launcher not found on PATH."
exit 1