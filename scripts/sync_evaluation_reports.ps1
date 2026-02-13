# Sync Evaluation Reports from Polonez server to local Windows machine
# Downloads only new/modified files (one-way: Polonez -> Local)
#
# Usage:
#   # Download new reports from Polonez
#   .\scripts\sync_evaluation_reports.ps1
#
#   # Upload local reports to Polonez (reverse sync)
#   .\scripts\sync_evaluation_reports.ps1 -Upload
#
#   # Two-way sync (download then upload)
#   .\scripts\sync_evaluation_reports.ps1 -TwoWay
#
#   # Dry run (show what would be transferred)
#   .\scripts\sync_evaluation_reports.ps1 -DryRun
#
# Prerequisites:
#   - SSH key auth configured for Polonez
#   - scp/ssh available (Windows 10+ built-in OpenSSH)

param(
    [switch]$Upload,
    [switch]$TwoWay,
    [switch]$DryRun
)

# --- Configuration ---
$SSHHost = "ksopyla@79.191.142.144"
$SSHPort = 2205
$RemotePath = "/home/ksopyla/dev/MrCogito/Cache/Evaluation_reports"
$LocalPath = "C:\Users\krzys\Dev Projects\MrCogito\Cache\Evaluation_reports"

# Ensure local directory exists
if (-not (Test-Path $LocalPath)) {
    New-Item -ItemType Directory -Path $LocalPath -Force | Out-Null
    Write-Host "Created local directory: $LocalPath"
}

function Get-RemoteFiles {
    Write-Host "Fetching file list from Polonez..." -ForegroundColor Cyan
    $remoteFiles = ssh -p $SSHPort $SSHHost "ls -1 $RemotePath/*.csv 2>/dev/null" 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $remoteFiles) {
        Write-Host "No CSV files found on Polonez or SSH failed." -ForegroundColor Yellow
        return @()
    }
    return $remoteFiles | ForEach-Object { Split-Path $_ -Leaf }
}

function Get-LocalFiles {
    $localFiles = Get-ChildItem -Path $LocalPath -Filter "*.csv" -ErrorAction SilentlyContinue
    if (-not $localFiles) { return @() }
    return $localFiles | ForEach-Object { $_.Name }
}

function Download-NewReports {
    Write-Host ""
    Write-Host "=== Downloading new reports from Polonez ===" -ForegroundColor Green
    
    $remoteFiles = Get-RemoteFiles
    $localFiles = Get-LocalFiles
    
    if ($remoteFiles.Count -eq 0) {
        Write-Host "No remote files found." -ForegroundColor Yellow
        return
    }
    
    # Find files that exist on remote but not locally
    $newFiles = $remoteFiles | Where-Object { $_ -notin $localFiles }
    
    if ($newFiles.Count -eq 0) {
        Write-Host "All files are already synced. Nothing to download." -ForegroundColor Green
        Write-Host "  Remote: $($remoteFiles.Count) files | Local: $($localFiles.Count) files"
        return
    }
    
    Write-Host "Found $($newFiles.Count) new file(s) to download:" -ForegroundColor Cyan
    $newFiles | ForEach-Object { Write-Host "  + $_" -ForegroundColor White }
    
    if ($DryRun) {
        Write-Host "`n[DRY RUN] Would download $($newFiles.Count) files." -ForegroundColor Yellow
        return
    }
    
    # Download new files using scp
    $downloaded = 0
    $failed = 0
    foreach ($file in $newFiles) {
        $remoteSrc = "${SSHHost}:${RemotePath}/${file}"
        $localDst = Join-Path $LocalPath $file
        
        Write-Host "  Downloading: $file" -ForegroundColor Gray -NoNewline
        scp -P $SSHPort $remoteSrc $localDst 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host " OK" -ForegroundColor Green
            $downloaded++
        } else {
            Write-Host " FAILED" -ForegroundColor Red
            $failed++
        }
    }
    
    Write-Host "`nDownload complete: $downloaded succeeded, $failed failed." -ForegroundColor Green
}

function Upload-NewReports {
    Write-Host ""
    Write-Host "=== Uploading new reports to Polonez ===" -ForegroundColor Green
    
    $remoteFiles = Get-RemoteFiles
    $localFiles = Get-LocalFiles
    
    if ($localFiles.Count -eq 0) {
        Write-Host "No local files found." -ForegroundColor Yellow
        return
    }
    
    # Find files that exist locally but not on remote
    $newFiles = $localFiles | Where-Object { $_ -notin $remoteFiles }
    
    if ($newFiles.Count -eq 0) {
        Write-Host "All files are already synced. Nothing to upload." -ForegroundColor Green
        Write-Host "  Local: $($localFiles.Count) files | Remote: $($remoteFiles.Count) files"
        return
    }
    
    Write-Host "Found $($newFiles.Count) new file(s) to upload:" -ForegroundColor Cyan
    $newFiles | ForEach-Object { Write-Host "  + $_" -ForegroundColor White }
    
    if ($DryRun) {
        Write-Host "`n[DRY RUN] Would upload $($newFiles.Count) files." -ForegroundColor Yellow
        return
    }
    
    # Upload new files using scp
    $uploaded = 0
    $failed = 0
    foreach ($file in $newFiles) {
        $localSrc = Join-Path $LocalPath $file
        $remoteDst = "${SSHHost}:${RemotePath}/${file}"
        
        Write-Host "  Uploading: $file" -ForegroundColor Gray -NoNewline
        scp -P $SSHPort $localSrc $remoteDst 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host " OK" -ForegroundColor Green
            $uploaded++
        } else {
            Write-Host " FAILED" -ForegroundColor Red
            $failed++
        }
    }
    
    Write-Host "`nUpload complete: $uploaded succeeded, $failed failed." -ForegroundColor Green
}

# --- Main ---
Write-Host "Evaluation Reports Sync" -ForegroundColor Cyan
Write-Host "  Local:  $LocalPath"
Write-Host "  Remote: ${SSHHost}:${RemotePath} (port $SSHPort)"
if ($DryRun) { Write-Host "  Mode:   DRY RUN" -ForegroundColor Yellow }

if ($TwoWay) {
    Download-NewReports
    Upload-NewReports
} elseif ($Upload) {
    Upload-NewReports
} else {
    # Default: download from Polonez
    Download-NewReports
}

Write-Host ""
Write-Host "Done." -ForegroundColor Green
