# Cleanup and Maintenance

This script helps maintain a clean development environment by removing cache files, temporary outputs, and optionally model directories.

## Usage

```powershell
# Run with default options (remove cache and temp files)
.\cleanup_workspace.ps1

# Or execute it directly from PowerShell
powershell -ExecutionPolicy Bypass -File .\cleanup_workspace.ps1
```

## What Gets Cleaned

By default, the script will clean:
- Python cache files (`__pycache__/` and `*.pyc`)
- Inference results directory
- Python package build files

Optionally, you can also clean:
- Quantized model directories
- ONNX exported model files
- Test directories

## Notes

- Large model files can occupy significant disk space (several GB)
- The script will show the size of each directory before deletion
- All operations require confirmation before proceeding

For more details on the project structure, refer to the main README.md file.