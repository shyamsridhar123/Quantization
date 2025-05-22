#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Cleans up the workspace by removing generated files and directories
.DESCRIPTION
    This script removes cached Python files, generated model files, and test results
    to help maintain a clean repository.
#>

$ErrorActionPreference = "Stop"
$SCRIPT_DIR = $PSScriptRoot

# Define directories and files to clean
$CLEAN_TARGETS = @(
    "$SCRIPT_DIR\__pycache__",
    "$SCRIPT_DIR\inference_results",
    "$SCRIPT_DIR\deepseek_quantization_finetuning.egg-info"
)

# Define model directories that should be preserved if requested
$MODEL_DIRS = @(
    "$SCRIPT_DIR\quantized_model",
    "$SCRIPT_DIR\onnx_exported",
    "$SCRIPT_DIR\test_onnx_export",
    "$SCRIPT_DIR\test_quant"
)

function Show-Prompt {
    param (
        [string]$message,
        [string]$default = "y"
    )
    
    $validResponses = @("y", "n")
    $defaultUpper = if ($default -eq "y") { "Y" } else { "N" }
    $otherOption = if ($default -eq "y") { "n" } else { "Y" }
    $prompt = "$message [$defaultUpper/$otherOption]: "
    
    $response = Read-Host -Prompt $prompt
    if ([string]::IsNullOrEmpty($response)) {
        $response = $default
    }
    
    return $response.ToLower()
}

# Display banner
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "DeepSeek Quantization Workspace Clean Tool" -ForegroundColor Cyan
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host ""

# Ask about model directories
$cleanModels = Show-Prompt "Remove quantized models and exported ONNX files? (This will free up significant disk space)"

if ($cleanModels -eq "y") {
    $CLEAN_TARGETS += $MODEL_DIRS
}

# List all targets
Write-Host "`nThe following directories/files will be removed:" -ForegroundColor Yellow
foreach ($target in $CLEAN_TARGETS) {
    if (Test-Path $target) {
        $size = (Get-ChildItem $target -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
        Write-Host "  - $target ($([math]::Round($size, 2)) MB)" -ForegroundColor Yellow
    }
}

# Confirm action
$confirm = Show-Prompt "`nAre you sure you want to proceed with cleanup?"

if ($confirm -eq "y") {
    foreach ($target in $CLEAN_TARGETS) {
        if (Test-Path $target) {
            try {
                Remove-Item -Path $target -Recurse -Force
                Write-Host "✓ Removed: $target" -ForegroundColor Green
            }
            catch {
                Write-Host "✗ Failed to remove: $target" -ForegroundColor Red
                Write-Host "  Error: $_" -ForegroundColor Red
            }
        }
        else {
            Write-Host "- Skipped (not found): $target" -ForegroundColor Gray
        }
    }
    
    # Clean Python cache files
    $pycFiles = Get-ChildItem -Path $SCRIPT_DIR -Filter "*.pyc" -Recurse -File
    if ($pycFiles) {
        foreach ($file in $pycFiles) {
            try {
                Remove-Item -Path $file.FullName -Force
                Write-Host "✓ Removed: $($file.FullName)" -ForegroundColor Green
            }
            catch {
                Write-Host "✗ Failed to remove: $($file.FullName)" -ForegroundColor Red
            }
        }
    }

    # Clean up temporary files
    Write-Host "`nCleaning temporary files..." -ForegroundColor Cyan
    $tempFiles = Get-ChildItem -Path $SCRIPT_DIR -Filter "*.tmp" -Recurse | Where-Object { -not $_.PSIsContainer }
    if ($tempFiles) {
        foreach ($file in $tempFiles) {
            try {
                Remove-Item -Path $file.FullName -Force
                Write-Host "✓ Removed: $($file.FullName)" -ForegroundColor Green
            }
            catch {
                Write-Host "✗ Failed to remove: $($file.FullName)" -ForegroundColor Red
            }
        }
    }

    # Clean Jupyter notebook output cells (optional)
    $cleanJupyterOutputs = Show-Prompt "`nClean Jupyter notebook outputs? (This will clear outputs but keep code)"
    if ($cleanJupyterOutputs -eq "y") {
        $notebookFiles = Get-ChildItem -Path $SCRIPT_DIR -Filter "*.ipynb" -Recurse -File
        if ($notebookFiles) {
            foreach ($file in $notebookFiles) {
                try {
                    # Backup notebook first
                    $backupPath = Join-Path -Path (Split-Path $file.FullName -Parent) -ChildPath "$($file.BaseName)_backup$($file.Extension)"
                    Copy-Item -Path $file.FullName -Destination $backupPath -Force
                    
                    # Read notebook content
                    $notebookContent = Get-Content -Path $file.FullName -Raw | ConvertFrom-Json
                    
                    # Clear cell outputs
                    foreach ($cell in $notebookContent.cells) {
                        if ($cell.cell_type -eq "code") {
                            $cell.outputs = @()
                            $cell.execution_count = $null
                        }
                    }
                    
                    # Write modified notebook
                    $notebookContent | ConvertTo-Json -Depth 100 | Set-Content -Path $file.FullName
                    Write-Host "✓ Cleared outputs in: $($file.FullName)" -ForegroundColor Green
                }
                catch {
                    Write-Host "✗ Failed to clean notebook: $($file.FullName)" -ForegroundColor Red
                    Write-Host "  Error: $_" -ForegroundColor Red
                }
            }
        }
        else {
            Write-Host "- No Jupyter notebooks found to clean" -ForegroundColor Gray
        }
    }

    # Show summary 
    Write-Host "`nSummary of Remaining Storage:" -ForegroundColor Cyan
    $modelDirsRemaining = @()
    foreach ($dir in $MODEL_DIRS) {
        if (Test-Path $dir) {
            $size = (Get-ChildItem $dir -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
            $modelDirsRemaining += @{
                Directory = $dir
                Size = [math]::Round($size, 2)
            }
        }
    }

    if ($modelDirsRemaining.Count -gt 0) {
        $modelDirsRemaining | ForEach-Object {
            Write-Host "  - $($_.Directory): $($_.Size) MB" -ForegroundColor Yellow
        }
        Write-Host "`nTo clean these directories, run this script again and choose to remove model directories." -ForegroundColor Yellow
    }

    Write-Host "`nCleanup completed successfully!" -ForegroundColor Green
}
else {
    Write-Host "`nCleanup cancelled." -ForegroundColor Yellow
}
