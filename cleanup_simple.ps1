#!/usr/bin/env pwsh
# Simple cleanup script for DeepSeek Quantization Framework

$SCRIPT_DIR = $PSScriptRoot

# Define directories to clean
$CLEAN_TARGETS = @(
    "$SCRIPT_DIR\__pycache__",
    "$SCRIPT_DIR\inference_results",
    "$SCRIPT_DIR\deepseek_quantization_finetuning.egg-info"
)

# Define model directories
$MODEL_DIRS = @(
    "$SCRIPT_DIR\quantized_model",
    "$SCRIPT_DIR\onnx_exported",
    "$SCRIPT_DIR\test_onnx_export",
    "$SCRIPT_DIR\test_quant"
)

Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "DeepSeek Quantization Workspace Clean Tool" -ForegroundColor Cyan
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host ""

# Ask about model directories
Write-Host "Remove quantized models and exported ONNX files? (This will free up significant disk space) [Y/n]: " -NoNewline -ForegroundColor Yellow
$cleanModels = Read-Host
if ($cleanModels -eq "" -or $cleanModels -eq "Y" -or $cleanModels -eq "y") {
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
Write-Host "`nAre you sure you want to proceed with cleanup? [Y/n]: " -NoNewline -ForegroundColor Yellow
$confirm = Read-Host
if ($confirm -eq "" -or $confirm -eq "Y" -or $confirm -eq "y") {
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

    Write-Host "`nCleanup completed successfully!" -ForegroundColor Green
}
else {
    Write-Host "`nCleanup cancelled." -ForegroundColor Yellow
}
