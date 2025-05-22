#!/usr/bin/env pwsh
# Simple script to organize notebooks for DeepSeek Quantization Framework

$SCRIPT_DIR = $PSScriptRoot
$ARCHIVE_DIR = "$SCRIPT_DIR\archive\notebooks"
$NOTEBOOK_DIR = "$SCRIPT_DIR\notebooks"

# Create notebooks directory if it doesn't exist
if (-not (Test-Path $NOTEBOOK_DIR)) {
    Write-Host "Creating notebooks directory..." -ForegroundColor Cyan
    New-Item -Path $NOTEBOOK_DIR -ItemType Directory | Out-Null
}

# Find all notebooks in the root directory
$notebooks = Get-ChildItem -Path $SCRIPT_DIR -Filter "*.ipynb" -File | Where-Object { $_.DirectoryName -eq $SCRIPT_DIR }

if ($notebooks.Count -eq 0) {
    Write-Host "No notebooks found in the root directory." -ForegroundColor Yellow
    exit
}

Write-Host "Found $($notebooks.Count) notebooks in the root directory:" -ForegroundColor Yellow
foreach ($notebook in $notebooks) {
    Write-Host "  - $($notebook.Name)" -ForegroundColor Yellow
}

Write-Host "`nCopying notebooks to organized directory..." -ForegroundColor Cyan
foreach ($notebook in $notebooks) {
    # Copy to notebooks directory
    $notebookPath = Join-Path -Path $NOTEBOOK_DIR -ChildPath $notebook.Name
    Copy-Item -Path $notebook.FullName -Destination $notebookPath -Force
    Write-Host "✓ Copied to notebooks/: $($notebook.Name)" -ForegroundColor Green
    
    # Copy to archive directory if it doesn't exist there
    $archivePath = Join-Path -Path $ARCHIVE_DIR -ChildPath $notebook.Name
    if (-not (Test-Path $archivePath)) {
        Copy-Item -Path $notebook.FullName -Destination $archivePath -Force
        Write-Host "✓ Archived to archive/notebooks/: $($notebook.Name)" -ForegroundColor Green
    }
    else {
        Write-Host "- Already archived: $($notebook.Name)" -ForegroundColor Gray
    }
}

# Create a simple README in the notebooks directory
$readmePath = Join-Path -Path $NOTEBOOK_DIR -ChildPath "README.md"
$readmeContent = @"
# DeepSeek Quantization Notebooks

This directory contains Jupyter notebooks for the DeepSeek-R1-Distill-Qwen-1.5B quantization framework.

## Notebooks

"@

foreach ($notebook in $notebooks) {
    $readmeContent += "`n- `$($notebook.Name)` - Quantization notebook"
}

$readmeContent += @"

## Usage

To use these notebooks:

1. Ensure you have Jupyter installed:
   ```
   pip install jupyter
   ```

2. Start Jupyter:
   ```
   jupyter notebook
   ```

3. Navigate to the desired notebook

## Notes

- These notebooks are also available in the root directory for reference
- Archive copies are stored in `archive/notebooks/`
"@

Set-Content -Path $readmePath -Value $readmeContent
Write-Host "✓ Created README index in notebooks/" -ForegroundColor Green

Write-Host "`nNotebook organization completed successfully!" -ForegroundColor Green
