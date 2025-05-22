#!/bin/pwsh
<#
.SYNOPSIS
    Master script for testing the quantized DeepSeek-R1-Distill-Qwen-1.5B model
.DESCRIPTION
    This script runs the complete testing suite, including diagnostics, performance tests,
    and generates a comprehensive HTML report
#>

$ErrorActionPreference = "Stop"
$SCRIPT_DIR = $PSScriptRoot
$MODEL_PATH = Join-Path $SCRIPT_DIR "quantized_model/onnx_int4"
$MODEL_FILE = Join-Path $MODEL_PATH "model_quantized.onnx"

# Create a timestamp for this test run
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RESULTS_DIR = Join-Path $SCRIPT_DIR "inference_results\run_$timestamp"
if (-not (Test-Path $RESULTS_DIR)) {
    New-Item -ItemType Directory -Path $RESULTS_DIR | Out-Null
}

# Set up logging
$log_file = Join-Path $RESULTS_DIR "master_test.log"
Write-Output "Starting master test at $(Get-Date)" | Tee-Object -FilePath $log_file -Append
Write-Output "Model path: $MODEL_PATH" | Tee-Object -FilePath $log_file -Append
Write-Output "Results directory: $RESULTS_DIR" | Tee-Object -FilePath $log_file -Append

# Check if model exists
if (-not (Test-Path $MODEL_FILE)) {
    Write-Error "Model file not found: $MODEL_FILE"
    exit 1
}

# Step 1: Run environment check
Write-Output "`n=== STEP 1: Environment Check ===" | Tee-Object -FilePath $log_file -Append
$env_check_file = Join-Path $RESULTS_DIR "environment_check.log"
python $SCRIPT_DIR/environment_check.py | Tee-Object -FilePath $env_check_file -Append
Get-Content $env_check_file | Tee-Object -FilePath $log_file -Append

# Step 2: Run model diagnostics
Write-Output "`n=== STEP 2: Model Diagnostics ===" | Tee-Object -FilePath $log_file -Append
$diag_file = Join-Path $RESULTS_DIR "diagnostics.log"
python $SCRIPT_DIR/enhanced_onnx_inference.py --model-path $MODEL_FILE --diagnose | Tee-Object -FilePath $diag_file -Append
Get-Content $diag_file | Tee-Object -FilePath $log_file -Append

# Step 3: Direct ONNX inference test
Write-Output "`n=== STEP 3: Direct ONNX Inference Test ===" | Tee-Object -FilePath $log_file -Append
$direct_file = Join-Path $RESULTS_DIR "direct_inference.log"
python $SCRIPT_DIR/enhanced_onnx_inference.py --model-path $MODEL_FILE --prompt "What is the advantage of quantizing language models to Int4?" --max-tokens 100 | Tee-Object -FilePath $direct_file -Append
Get-Content $direct_file | Tee-Object -FilePath $log_file -Append

# Step 4: Optimum-based standard test
Write-Output "`n=== STEP 4: Optimum-based Test ===" | Tee-Object -FilePath $log_file -Append
$optimum_file = Join-Path $RESULTS_DIR "optimum_test.log"
try {
    python $SCRIPT_DIR/test_quantized_model.py --model-path $MODEL_PATH --model-type onnx --prompt "How does quantization affect model performance?" | Tee-Object -FilePath $optimum_file -Append
    Get-Content $optimum_file | Tee-Object -FilePath $log_file -Append
} catch {
    Write-Output "Optimum-based test failed with error: $_" | Tee-Object -FilePath $log_file -Append
}

# Step 5: Performance benchmarking
Write-Output "`n=== STEP 5: Performance Benchmark ===" | Tee-Object -FilePath $log_file -Append
$benchmark_file = Join-Path $RESULTS_DIR "benchmark.json"
python $SCRIPT_DIR/benchmark_model.py --model-path $MODEL_FILE --output-file $benchmark_file --num-threads 4 --warmup-runs 1 --benchmark-runs 3 --max-tokens 50 | Tee-Object -FilePath $log_file -Append

# Step 6: Generate HTML report
Write-Output "`n=== STEP 6: Generate Report ===" | Tee-Object -FilePath $log_file -Append
$report_file = Join-Path $RESULTS_DIR "model_report.html"
python $SCRIPT_DIR/generate_report.py --results-dir $RESULTS_DIR --model-path $MODEL_PATH --output-file $report_file | Tee-Object -FilePath $log_file -Append

# Complete
Write-Output "`n=== Testing Complete ===" | Tee-Object -FilePath $log_file -Append
Write-Output "Results saved to: $RESULTS_DIR" | Tee-Object -FilePath $log_file -Append

if (Test-Path $report_file) {
    Write-Output "Opening HTML report in browser..." | Tee-Object -FilePath $log_file -Append
    Start-Process $report_file
} else {
    explorer $RESULTS_DIR
}

Write-Output "Test completed successfully at $(Get-Date)" | Tee-Object -FilePath $log_file -Append
