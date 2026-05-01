<#
.SYNOPSIS
    End-to-end Windows release build for the GHG Predictor desktop app.

.DESCRIPTION
    1. Bakes the runtime assets (vocab + materials + model copy).
    2. Runs PyInstaller in one-folder mode -> desktop_app\build\out\GHGPredictor\.
    3. (Optional) Compiles the Inno Setup installer if iscc is on PATH or in
       the standard install location.

    Run from the repository root with the project's venv on PATH (or pass
    -PythonExe to point at it):

        .\desktop_app\build\build.ps1
        .\desktop_app\build\build.ps1 -Version 1.0.0
        .\desktop_app\build\build.ps1 -SkipInstaller

.PARAMETER Version
    Version string baked into the installer file name and metadata.

.PARAMETER PythonExe
    Path to the python.exe that should drive bake_assets.py and PyInstaller.
    Defaults to .\.venv\Scripts\python.exe relative to the repo root.

.PARAMETER SkipInstaller
    Skip the Inno Setup compile step (still produces the unpackaged dist tree).

.PARAMETER SkipBake
    Skip the bake_assets.py step. Use this when the assets/ folder is already
    populated (e.g. on CI, where the committed assets are used as-is).
#>
[CmdletBinding()]
param(
    [string] $Version       = "0.1.0",
    [string] $PythonExe     = ".\.venv\Scripts\python.exe",
    [switch] $SkipInstaller,
    [switch] $SkipBake
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $RepoRoot

if (-not (Test-Path $PythonExe)) {
    # Allow bare command names ("python", "py", ...) by resolving via PATH.
    $resolved = Get-Command $PythonExe -ErrorAction SilentlyContinue
    if ($resolved) {
        $PythonExe = $resolved.Source
    } else {
        throw "Python executable not found at '$PythonExe'. Pass -PythonExe to override."
    }
}

Write-Host "==> Repo root:   $RepoRoot"
Write-Host "==> Python:      $PythonExe"
Write-Host "==> Version:     $Version"

# ── 1. Bake assets ───────────────────────────────────────────────────────────
if ($SkipBake) {
    Write-Host "`n[1/3] Skipping bake_assets (--SkipBake)." -ForegroundColor Yellow
    foreach ($f in "ghg_model.pt","vocab.npz","materials.json") {
        $p = Join-Path "desktop_app\assets" $f
        if (-not (Test-Path $p)) { throw "Missing committed asset: $p" }
    }
} else {
    Write-Host "`n[1/3] Baking assets ..." -ForegroundColor Cyan
    & $PythonExe desktop_app\tools\bake_assets.py
    if ($LASTEXITCODE -ne 0) { throw "bake_assets.py exited with $LASTEXITCODE" }
}

# Write _version.py so the frozen app knows its own version for update checks.
$versionFile = Join-Path $RepoRoot "desktop_app\_version.py"
[System.IO.File]::WriteAllText(
    $versionFile,
    "__version__ = `"$Version`"`n",
    [System.Text.Encoding]::ASCII
)
Write-Host "    Wrote desktop_app\_version.py ($Version)"

# ── 2. PyInstaller one-folder build ──────────────────────────────────────────
Write-Host "`n[2/3] Running PyInstaller ..." -ForegroundColor Cyan
$pyDir = Split-Path $PythonExe -Parent
$pyInstaller = $null
foreach ($candidate in @(
    (Join-Path $pyDir "pyinstaller.exe"),                  # venv layout
    (Join-Path $pyDir "Scripts\pyinstaller.exe"),          # hosted python layout
    (Get-Command pyinstaller -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source)
)) {
    if ($candidate -and (Test-Path $candidate)) { $pyInstaller = $candidate; break }
}
if (-not $pyInstaller) {
    throw "pyinstaller.exe not found near $PythonExe — install desktop_app\requirements.txt first."
}

$distPath = "desktop_app\build\out"
$workPath = "desktop_app\build\work"
& $pyInstaller "desktop_app\build\ghg_predictor.spec" `
    --noconfirm --clean --distpath $distPath --workpath $workPath
if ($LASTEXITCODE -ne 0) { throw "pyinstaller exited with $LASTEXITCODE" }

$bundleSizeMB = "{0:N1}" -f ((Get-ChildItem "$distPath\GHGPredictor" -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB)
Write-Host "    Bundle size: $bundleSizeMB MB"

# ── 3. Inno Setup ────────────────────────────────────────────────────────────
if ($SkipInstaller) {
    Write-Host "`n[3/3] Skipping Inno Setup (--SkipInstaller)." -ForegroundColor Yellow
    Write-Host "Done. Unpackaged dist: $distPath\GHGPredictor"
    exit 0
}

Write-Host "`n[3/3] Compiling Inno Setup installer ..." -ForegroundColor Cyan
$iscc = $null
foreach ($candidate in @(
    (Get-Command iscc -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source),
    "C:\Program Files (x86)\Inno Setup 6\iscc.exe",
    "C:\Program Files\Inno Setup 6\iscc.exe"
)) {
    if ($candidate -and (Test-Path $candidate)) { $iscc = $candidate; break }
}
if (-not $iscc) {
    Write-Warning "Inno Setup (iscc.exe) not found. Install it from https://jrsoftware.org/isinfo.php or pass -SkipInstaller."
    exit 0
}

& $iscc "/DMyAppVersion=$Version" "desktop_app\build\installer.iss"
if ($LASTEXITCODE -ne 0) { throw "iscc exited with $LASTEXITCODE" }

$installer = Join-Path $distPath ("GHGPredictorSetup-$Version.exe")
if (Test-Path $installer) {
    $installerSizeMB = "{0:N1}" -f ((Get-Item $installer).Length / 1MB)
    Write-Host "`nDone. Installer: $installer ($installerSizeMB MB)" -ForegroundColor Green
} else {
    Write-Host "`nDone, but installer not found at expected path: $installer" -ForegroundColor Yellow
}
