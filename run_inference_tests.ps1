#!/bin/pwsh
<#
.SYNOPSIS
    This script runs a series of inference tests on the quantized model
.DESCRIPTION
    Tests the ONNX Int4 quantized model with various prompts and provides performance metrics
#>

$SCRIPT_DIR = $PSScriptRoot
$MODEL_PATH = Join-Path $SCRIPT_DIR "quantized_model/onnx_int4"
$MODEL_FILE = Join-Path $MODEL_PATH "model_quantized.onnx"
$PROMPTS_FILE = Join-Path $SCRIPT_DIR "sample_prompts.txt"
$ORIGINAL_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
$NUM_THREADS = 4 # Adjust based on your CPU

# Verify paths
if (-not (Test-Path $MODEL_PATH)) {
    Write-Error "Model path not found: $MODEL_PATH"
    exit 1
}

if (-not (Test-Path $PROMPTS_FILE)) {
    Write-Error "Prompts file not found: $PROMPTS_FILE"
    exit 1
}

# Create results directory
$RESULTS_DIR = Join-Path $SCRIPT_DIR "inference_results"
if (-not (Test-Path $RESULTS_DIR)) {
    New-Item -ItemType Directory -Path $RESULTS_DIR | Out-Null
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$log_file = Join-Path $RESULTS_DIR "inference_test_$timestamp.log"

# Start logging
Write-Output "Starting inference tests at $(Get-Date)" | Tee-Object -FilePath $log_file -Append
Write-Output "Model path: $MODEL_PATH" | Tee-Object -FilePath $log_file -Append
Write-Output "=======================================" | Tee-Object -FilePath $log_file -Append

# Run diagnostics first
Write-Output "Running model diagnostics..." | Tee-Object -FilePath $log_file -Append
python $SCRIPT_DIR/diagnose_onnx_model.py --model-path $MODEL_FILE | Tee-Object -FilePath $log_file -Append

# Run performance benchmark
Write-Output "`n=======================================" | Tee-Object -FilePath $log_file -Append
Write-Output "Running performance benchmarks..." | Tee-Object -FilePath $log_file -Append
$benchmark_file = Join-Path $RESULTS_DIR "benchmark_$timestamp.json"
python $SCRIPT_DIR/benchmark_model.py --model-path $MODEL_FILE --model-id $ORIGINAL_MODEL_ID --output-file $benchmark_file --num-threads $NUM_THREADS | Tee-Object -FilePath $log_file -Append

# Run the enhanced direct ONNX inference test
Write-Output "`n=======================================" | Tee-Object -FilePath $log_file -Append
Write-Output "Running direct ONNX inference test..." | Tee-Object -FilePath $log_file -Append
python $SCRIPT_DIR/enhanced_onnx_inference.py --model-path $MODEL_FILE --prompt "What is the advantage of quantizing language models?" --max-tokens 100 | Tee-Object -FilePath $log_file -Append

# Run standard test with default prompt
Write-Output "`n=======================================" | Tee-Object -FilePath $log_file -Append
Write-Output "Running direct interactive test with default prompt..." | Tee-Object -FilePath $log_file -Append
python $SCRIPT_DIR/direct_interactive_test.py --model-path $MODEL_PATH --prompt "What is the capital of France?" | Tee-Object -FilePath $log_file -Append

# Run tests with each prompt from the file
$prompts = Get-Content $PROMPTS_FILE
$prompt_number = 1

foreach ($prompt in $prompts) {    if ([string]::IsNullOrWhiteSpace($prompt)) {
        continue
    }
    
    Write-Output "`n=======================================" | Tee-Object -FilePath $log_file -Append
    Write-Output "Running test #$($prompt_number) - $prompt" | Tee-Object -FilePath $log_file -Append
    
    # Escape quotes in the prompt
    $escaped_prompt = $prompt.Replace('"','\"')
      # Run the test
    python $SCRIPT_DIR/direct_interactive_test.py --model-path $MODEL_PATH --prompt "$escaped_prompt" | Tee-Object -FilePath $log_file -Append
    
    $prompt_number++
}

# Run comparison test with the original model (optional - commented out because it will download the model)
# Write-Output "`n=======================================" | Tee-Object -FilePath $log_file -Append
# Write-Output "Running comparison test with original model..." | Tee-Object -FilePath $log_file -Append
# python $SCRIPT_DIR/test_quantized_model.py --model-path $MODEL_PATH --model-type onnx --verify | Tee-Object -FilePath $log_file -Append

# Try the FP16 conversion for comparison if needed
# Write-Output "`n=======================================" | Tee-Object -FilePath $log_file -Append
# Write-Output "Converting model to FP16 for comparison..." | Tee-Object -FilePath $log_file -Append
# $fp16_model = Join-Path $RESULTS_DIR "model_fp16_$timestamp.onnx"
# python $SCRIPT_DIR/convert_to_fp16.py --input-model $MODEL_FILE --output-model $fp16_model --test | Tee-Object -FilePath $log_file -Append

Write-Output "`n=======================================" | Tee-Object -FilePath $log_file -Append
Write-Output "Inference tests completed at $(Get-Date)" | Tee-Object -FilePath $log_file -Append
Write-Output "Results saved to: $log_file" | Tee-Object -FilePath $log_file -Append

# Generate HTML report from results
Write-Output "`n=======================================" | Tee-Object -FilePath $log_file -Append
Write-Output "Generating HTML report..." | Tee-Object -FilePath $log_file -Append

$report_file = Join-Path $RESULTS_DIR "model_report_$timestamp.html"
python $SCRIPT_DIR/generate_report.py --results-dir $RESULTS_DIR --model-path $MODEL_PATH --output-file $report_file | Tee-Object -FilePath $log_file -Append

# Open results folder
if (Test-Path $report_file) {
    Write-Output "Report generated: $report_file" | Tee-Object -FilePath $log_file -Append
    # Open the report in the default browser
    Start-Process $report_file
} else {
    explorer $RESULTS_DIR
}
