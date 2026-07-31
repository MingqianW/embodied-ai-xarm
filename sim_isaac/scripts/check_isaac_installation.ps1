[CmdletBinding()]
param(
    [string]$OutputPath = ""
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path (Join-Path $ScriptDir "..\..")).Path
$Diagnostic = Join-Path $ScriptDir "check_isaac_installation.py"

$Candidates = [System.Collections.Generic.List[string]]::new()
$IsaacRoots = [System.Collections.Generic.List[string]]::new()
if ($env:ISAAC_SIM_PATH) {
    $IsaacRoots.Add($env:ISAAC_SIM_PATH)
}
if ($env:LOCALAPPDATA) {
    $IsaacRoots.Add((Join-Path $env:LOCALAPPDATA "NVIDIA Corporation\Isaac Sim"))
    $IsaacRoots.Add((Join-Path $env:LOCALAPPDATA "ov\pkg"))
}
if ($env:ProgramFiles) {
    $IsaacRoots.Add((Join-Path $env:ProgramFiles "NVIDIA Corporation\Isaac Sim"))
}
foreach ($Root in $IsaacRoots) {
    $Candidates.Add((Join-Path $Root "python.bat"))
    $Candidates.Add((Join-Path $Root "python.exe"))
    if (Test-Path -LiteralPath $Root -PathType Container) {
        foreach ($Child in Get-ChildItem -LiteralPath $Root -Directory -ErrorAction SilentlyContinue) {
            $Candidates.Add((Join-Path $Child.FullName "python.bat"))
            $Candidates.Add((Join-Path $Child.FullName "python.exe"))
        }
    }
}
$Candidates.Add((Join-Path $ProjectRoot ".venv311\Scripts\python.exe"))

$PathPython = Get-Command python -ErrorAction SilentlyContinue
if ($PathPython) {
    $Candidates.Add($PathPython.Source)
}

$Python = $null
foreach ($Candidate in $Candidates) {
    if ($Candidate -and (Test-Path -LiteralPath $Candidate)) {
        $Python = $Candidate
        break
    }
}

if (-not $Python) {
    throw "No Python interpreter was found. Set ISAAC_SIM_PATH or install Python 3.11."
}

Write-Host "Using Python launcher: $Python"
$Arguments = @($Diagnostic)
if ($OutputPath) {
    $Arguments += @("--output", $OutputPath)
}
& $Python @Arguments
exit $LASTEXITCODE
