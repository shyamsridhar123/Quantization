# DeepSeek-R1-Distill-Qwen-1.5B Model Testing Framework

This repository contains tools for testing, benchmarking, and troubleshooting the quantized DeepSeek-R1-Distill-Qwen-1.5B model, specifically for evaluating its performance after ONNX conversion and Int4 quantization.

## Testing Framework Overview

The framework includes multiple complementary approaches to testing the quantized model:

1. **Standard Testing** (`test_quantized_model.py`): Loads and evaluates the model using the Optimum/ONNX Runtime integration.
2. **Interactive Testing** (`interactive_test.py`): Provides a chat-like interface for interactive evaluation.
3. **Direct ONNX Inference** (`direct_onnx_inference.py` and `enhanced_onnx_inference.py`): Bypasses Optimum and uses ONNX Runtime directly.
4. **Troubleshooting** (`troubleshoot_onnx_model.py`): Provides diagnostic information about the model and attempts different loading configurations.
5. **Performance Benchmarking** (`benchmark_model.py`): Gathers detailed performance metrics for various types of inputs.
6. **Format Conversion** (`convert_to_fp16.py`): Converts the model between formats for comparative testing.
7. **Batch Testing** (`run_inference_tests.ps1`): Runs multiple tests in sequence and logs the results.
8. **Report Generation** (`generate_report.py`): Creates a comprehensive HTML report from test results.

## Usage Instructions

### Prerequisites

Make sure you have the following packages installed:

```
pip install torch transformers optimum[onnxruntime] onnx onnxruntime numpy matplotlib seaborn beautifulsoup4 pandas tabulate
```

### Testing the Quantized Model

#### 1. Basic Tests

To run a simple test of the quantized model with the default prompt:

```
python test_quantized_model.py --model-path ./quantized_model/onnx_int4
```

#### 2. Interactive Testing

For interactive chat-style testing:

```
python interactive_test.py --model-path ./quantized_model/onnx_int4
```

#### 3. Direct ONNX Runtime Testing

To bypass Optimum and test using ONNX Runtime directly:

```
python enhanced_onnx_inference.py --model-path ./quantized_model/onnx_int4/model_quantized.onnx
```

#### 4. Direct Interactive Testing

If you encounter generation errors with the standard interactive testing script, use the direct ONNX runtime implementation:

```
python direct_interactive_test.py --model-path ./quantized_model/onnx_int4
```

This implementation directly uses ONNX Runtime for inference, bypassing any issues with the Optimum integration while still providing an interactive chat interface.

#### 5. Diagnostics

To run diagnostics that help identify issues with the model:

```
python enhanced_onnx_inference.py --model-path ./quantized_model/onnx_int4/model_quantized.onnx --diagnose
```

#### 6. Performance Benchmarking

To measure the model's performance with various input lengths:

```
python benchmark_model.py --model-path ./quantized_model/onnx_int4/model_quantized.onnx
```

#### 7. Batch Testing

To run multiple tests with different prompts and generate a comprehensive report:

```
./run_inference_tests.ps1
```

#### 8. Generate Report

To generate an HTML report from test results:

```
python generate_report.py --results-dir ./inference_results --model-path ./quantized_model/onnx_int4
```

## Common Issues and Solutions

### 1. Model Loading Issues

If the model fails to load with the default parameters, try these alternatives:

```python
# Disable KV cache and IO binding (for Qwen2 models)
model = ORTModelForCausalLM.from_pretrained(
    model_path,
    use_cache=False,
    use_io_binding=False
)
```

### 2. Generation Errors

If the model loads but fails during generation with a 'logits' error, try:

- Using direct ONNX Runtime inference (`enhanced_onnx_inference.py` or `direct_interactive_test.py`)
- Converting the model to FP16 (`convert_to_fp16.py`)
- Re-exporting the model with position_ids input: 
  ```
  python reexport_with_position_ids.py --model-id deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --output-dir ./onnx_fixed --quantize
  ```

The 'logits' error typically occurs with Qwen2-based models that were exported without the position_ids input, which is required for proper generation. The `reexport_with_position_ids.py` script fixes this issue by creating a new export with the necessary inputs.

### 3. Memory Issues

If you encounter memory errors, try:

- Reducing the input length
- Using `torch.cuda.empty_cache()` between runs
- Setting `intra_op_num_threads` and `inter_op_num_threads` to reduce parallelism

## Tips for Optimal Performance

1. **Thread Count**: Adjust the `num_threads` parameter based on your CPU. Generally, setting it to the number of physical cores (not hyperthreaded) works best.

2. **First Run Warmup**: The first inference run is typically much slower due to model compilation and optimization. Always perform a warmup run before measuring performance.

3. **Batch Size**: For maximum throughput, consider batching multiple prompts together if your use case allows it.

4. **Input Length**: Keep in mind that inference time scales with input length. Very long prompts will be slower to process.

## Resources

- [ONNX Runtime Documentation](https://onnxruntime.ai/)
- [Optimum Documentation](https://huggingface.co/docs/optimum/index)
- [DeepSeek-R1 Model Card](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
