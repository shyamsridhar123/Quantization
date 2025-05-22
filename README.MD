# 🧠 DeepSeek Model Quantization and Testing Framework 🚀

```
 _____                 _____           _      ___  _   _ _   _ _____ ___ ________  ______ 
|  __ \               / ____|         | |    / _ \| \ | | \ | |_   _/ _ \__  / _ \|___  /
| |  | | ___  ___ ___| (___   ___  ___| | __| | | |  \| |  \| | | || | | |/ / | | |  / / 
| |  | |/ _ \/ _ \_  /\___ \ / _ \/ _ \ |/ /| | | | . ` | . ` | | || | | / /| | | | / /  
| |__| |  __/  __// / ____) |  __/  __/   < | |_| | |\  | |\  |_| || |_| / /_| |_| / /__ 
|_____/ \___|\___/___|_____/ \___|\___|_|\_(_)___/|_| \_|_| \_|_____\___/____|\___/_____|
                                                                                        
```

This project provides tools for quantizing and testing large language models from the DeepSeek family, specifically focusing on the DeepSeek-R1-Distill-Qwen-1.5B model. The primary goal is to make these models more accessible by reducing their memory footprint through quantization to Int4 precision and ONNX format conversion.

## 🔥 What is Model Quantization & Distillation? 🔥

### 📊 Quantization Explained

**Quantization** reduces the precision of numerical values in neural networks:

```
FP32 (32-bit) ➡️ INT8 (8-bit) ➡️ INT4 (4-bit)
   ┌─────────┐      ┌─────┐       ┌───┐
   │111010101│  →   │10101│  →    │1010│
   └─────────┘      └─────┘       └───┘
  More Precise       Smaller      Tiny!
  (~6GB model)      (~1.5GB model)  (~1.7GB model)
```

Quantization **dramatically reduces** model size and memory footprint while attempting to preserve model quality!

### 🧪 Distillation Explained 

**Distillation** is where a smaller model (student) learns from a larger model (teacher):

```
       Teacher Model              Student Model
       (Large, Slow)            (Small, Fast)
       ┌───────────┐              ┌─────┐
       │           │     →        │     │
       │  7B-175B  │  Knowledge   │ 1-3B│
       │ Parameters│  Transfer    │Params│
       └───────────┘              └─────┘
```

DeepSeek-R1-**Distill**-Qwen-1.5B is already a distilled model that learned from larger models!

## 🚀 Quick Start Guide

### 1. 🛠️ Setup Environment

```powershell
# Clone the repository (if you haven't already)
git clone https://github.com/yourusername/VibeCollection.git
cd VibeCollection/quantization

# Create and activate a virtual environment (recommended)
python -m venv .venv
.\.venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. 🌟 Export and Quantize the Model

```powershell
# Export the model to ONNX with position_ids support (fixes common 'logits' errors)
python reexport_with_position_ids.py --model-id deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --output-dir ./onnx_fixed --quantize

# Alternatively, use the Jupyter notebooks for more control:
jupyter notebook notebooks/quantize_int4.ipynb
```

### 3. 🔍 Test the Quantized Model

```powershell
# Interactive testing with direct ONNX Runtime approach
python direct_interactive_test.py --model-path ./quantized_model/onnx_int4

# Or run the full test suite
.\run_inference_tests.ps1
```

### 4. 📊 Benchmark Performance

```powershell
# Run performance benchmarks
python benchmark_model.py --model-path ./quantized_model/onnx_int4/model_quantized.onnx --num-threads 4
```

### 5. 🔧 Troubleshooting

If you encounter issues:

```powershell
# Run diagnostics
python diagnose_onnx_model.py --model-path ./quantized_model/onnx_int4/model_quantized.onnx

# Try direct ONNX inference if Optimum integration fails
python enhanced_onnx_inference.py --model-path ./quantized_model/onnx_int4/model_quantized.onnx
```

## 🌈 Why Quantize Models? 🌈

Quantizing models provides these amazing benefits:

```
┌─────────────────────────────────────┐
│ 🚀 8x Smaller File Size             │
│ 💾 ~70-75% Less Memory Usage        │
│ ⚡ Faster Inference Speed            │
│ 🖥️ Run on Consumer Hardware         │
│ 🔋 Lower Energy Consumption         │
│ 🏠 Enable Edge & Local Deployment   │
└─────────────────────────────────────┘
```

## 📝 Complete Workflow

### Step 1: 🧰 Prepare Your Environment

1. **Install Required Dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

2. **Verify Environment Setup**:
   ```powershell
   python environment_check.py
   ```

### Step 2: 🔄 Export and Quantize the Model

1. **Re-Export Model with Position IDs** (Recommended for Qwen2-based models):
   ```powershell
   python reexport_with_position_ids.py --model-id deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --output-dir ./onnx_fixed
   ```

2. **Quantize the Model to Int4**:
   ```powershell
   # Either use the re-exported model with quantization flag:
   python reexport_with_position_ids.py --model-id deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --output-dir ./onnx_fixed --quantize
   
   # Or run the dedicated quantization script:
   python run_quantization.py --model-id deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --output-dir ./quantized_model/onnx_int4
   
   # For more control, use the Jupyter notebook:
   jupyter notebook notebooks/quantize_int4.ipynb
   ```

### Step 3: 🧪 Test the Quantized Model

1. **Run Diagnostics**:
   ```powershell
   python diagnose_onnx_model.py --model-path ./quantized_model/onnx_int4/model_quantized.onnx
   ```

2. **Interactive Testing**:
   ```powershell
   # Recommended approach (handles common issues):
   python direct_interactive_test.py --model-path ./quantized_model/onnx_int4
   
   # Alternative approach using Optimum:
   python interactive_test.py --model-path ./quantized_model/onnx_int4
   ```

3. **Batch Testing with Multiple Prompts**:
   ```powershell
   .\run_inference_tests.ps1
   ```

### Step 4: 📈 Benchmark and Compare

1. **Performance Benchmarking**:
   ```powershell
   python benchmark_model.py --model-path ./quantized_model/onnx_int4/model_quantized.onnx --num-threads 4
   ```

2. **Compare with Original Model** (Optional):
   ```powershell
   python compare_models.py --quantized-path ./quantized_model/onnx_int4 --model-id deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
   ```

3. **Generate HTML Report**:
   ```powershell
   python generate_report.py --results-dir ./inference_results --model-path ./quantized_model/onnx_int4 --output-file ./model_report.html
   ```

### Step 5: 🔄 Format Conversion (Optional)

1. **Convert to FP16 for Troubleshooting**:
   ```powershell
   python convert_to_fp16.py --input-model ./quantized_model/onnx_int4/model_quantized.onnx --output-model ./quantized_model/onnx_fp16/model_fp16.onnx
   ```

## 🔍 Quantization Deep Dive

Quantization in this repository follows this process:

```
                 Original Model (FP32)
                         ⬇️
┌─────────────────────────────────────────────┐
│ 1. Export to ONNX                           │
│    - Convert model architecture to ONNX     │
│    - Add position_ids input (required fix)  │
└─────────────────────────────────────────────┘
                         ⬇️
┌─────────────────────────────────────────────┐
│ 2. Calibration                              │
│    - Analyze weight distributions           │
│    - Determine optimal scaling factors      │
└─────────────────────────────────────────────┘
                         ⬇️
┌─────────────────────────────────────────────┐
│ 3. INT4 Quantization                        │
│    - Apply scaling to weights               │
│    - Convert FP32 values to INT4 values     │
│    - Store quantization parameters          │
└─────────────────────────────────────────────┘
                         ⬇️
┌─────────────────────────────────────────────┐
│ 4. Optimization                             │
│    - Apply ONNX Runtime optimizations       │
│    - Fuse operations where possible         │
└─────────────────────────────────────────────┘
                         ⬇️
                 Quantized Model (INT4)
```

## 🏗️ Project Structure

### 🧩 Key Components

- **🔧 Quantization Tools**: Scripts and notebooks for exporting and quantizing models
- **🧪 Testing Framework**: Tools for testing, benchmarking, and troubleshooting the quantized models
- **🧹 Maintenance Utilities**: Scripts for project organization and cleanup

### 📄 Important Scripts

#### 🔄 Quantization and Export
- `reexport_with_position_ids.py` - 🛠️ Exports the model with position_ids (fixes 'logits' errors)
- `run_quantization.py` - 🔢 Dedicated quantization script
- `quantize_int4.ipynb` - 📓 Jupyter notebook for Int4 quantization

#### 🧪 Testing and Inference
- `direct_interactive_test.py` - 💬 Interactive testing using direct ONNX Runtime (most reliable)
- `interactive_test.py` - 🤖 Interactive testing using Optimum integration
- `diagnose_onnx_model.py` - 🔍 Detailed diagnostic checks for model issues
- `enhanced_onnx_inference.py` - ⚡ Direct ONNX Runtime inference with diagnostics
- `benchmark_model.py` - 📊 Performance benchmarking across different input sizes

#### 🔄 Execution Scripts
- `run_inference_tests.ps1` - 🧪 Comprehensive testing suite
- `run_master_test.ps1` - 🚀 Runs all tests in sequence

#### 🧹 Maintenance
- `maintain.bat` - 🛠️ Central maintenance dashboard with menu options
- `cleanup_simple.ps1` - 🧹 Removes temporary files and Python cache

## ❓ Common Issues and Solutions

### 1. 'Logits' Error During Generation ❌

**Symptom**: `KeyError: 'logits'` or similar errors during model inference.

**Solution**: The model was likely exported without position_ids input. Re-export the model with:
```powershell
python reexport_with_position_ids.py --model-id deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --output-dir ./onnx_fixed
```

### 2. Model Loading Failures ❌

**Symptom**: Model fails to load or initialize.

**Solution**: Try using different loading parameters:
```python
# For Qwen2-based models
session_options = ort.SessionOptions()
session_options.intra_op_num_threads = 4
session = ort.InferenceSession(model_path, session_options)
```

### 3. Performance Issues ⏱️

**Symptom**: Slow inference or high memory usage.

**Solution**: Adjust thread count to match your CPU cores:
```powershell
python benchmark_model.py --model-path ./model_path --num-threads 4  # Adjust to your CPU
```

## ⚙️ Configuration Options

- **🧵 Thread Count**: Adjust `--num-threads` based on your CPU. Generally, setting it to the number of physical cores works best.
- **🔢 Token Limit**: Use `--max-tokens` to control generation length.
- **📁 Model Path**: All scripts accept `--model-path` to specify the model location.

## 📚 The Technical Significance of This Project

This project demonstrates several important techniques:

```
┌─────────────────────────────────────────────────┐
│ 🚀 ONNX Conversion                              │
│   - Hardware-agnostic deployment                │
│   - Runtime optimizations for any hardware      │
├─────────────────────────────────────────────────┤
│ 🧮 INT4 Quantization                            │
│   - Aggressive compression (8x smaller)         │
│   - 1.7GB model size (vs ~6GB for FP32)        │
│   - 1.5B parameters with vocabulary of 151,936  │
│   - Carefully preserves model quality           │
├─────────────────────────────────────────────────┤
│ 🔧 Architecture-Specific Fixes                  │
│   - Fixes for Qwen2 architecture               │
│   - Custom position_ids handling               │
├─────────────────────────────────────────────────┤
│ 📊 Comprehensive Benchmarking                   │
│   - Performance across different inputs         │
│   - 2-4 tokens/sec generation speed            │
│   - 147-320ms first token latency              │
│   - Quality comparison with original            │
└─────────────────────────────────────────────────┘
```

## 🔗 Resources

- [ONNX Runtime Documentation](https://onnxruntime.ai/) 📚
- [Optimum Documentation](https://huggingface.co/docs/optimum/index) 🤗
- [DeepSeek-R1 Model Card](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) 🧠

## 📜 License

This project is provided for research and educational purposes. The DeepSeek models have their own licenses which should be respected when using them.

## 👥 Contributing

Contributions are welcome! See `CONTRIBUTING.md` for guidelines. 🌟

## ✨ Why This Matters

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  🖥️  Run Advanced AI locally on your own hardware               │
│  🔒  Keep your data private - no cloud required                 │
│  💸  No subscription costs or API fees                          │
│  🌱  Lower environmental impact than cloud inference            │
│  🛠️  Full control over inference parameters                     │
│  🚀  Deploy in resource-constrained environments                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Complete Workflow

### Step 1: Prepare Your Environment

1. **Install Required Dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

2. **Verify Environment Setup**:
   ```powershell
   python environment_check.py
   ```

### Step 2: Export and Quantize the Model

1. **Re-Export Model with Position IDs** (Recommended for Qwen2-based models):
   ```powershell
   python reexport_with_position_ids.py --model-id deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --output-dir ./onnx_fixed
   ```

2. **Quantize the Model to Int4**:
   ```powershell
   # Either use the re-exported model with quantization flag:
   python reexport_with_position_ids.py --model-id deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --output-dir ./onnx_fixed --quantize
   
   # Or run the dedicated quantization script:
   python run_quantization.py --model-id deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --output-dir ./quantized_model/onnx_int4
   
   # For more control, use the Jupyter notebook:
   jupyter notebook notebooks/quantize_int4.ipynb
   ```

### Step 3: Test the Quantized Model

1. **Run Diagnostics**:
   ```powershell
   python diagnose_onnx_model.py --model-path ./quantized_model/onnx_int4/model_quantized.onnx
   ```

2. **Interactive Testing**:
   ```powershell
   # Recommended approach (handles common issues):
   python direct_interactive_test.py --model-path ./quantized_model/onnx_int4
   
   # Alternative approach using Optimum:
   python interactive_test.py --model-path ./quantized_model/onnx_int4
   ```

3. **Batch Testing with Multiple Prompts**:
   ```powershell
   .\run_inference_tests.ps1
   ```

### Step 4: Benchmark and Compare

1. **Performance Benchmarking**:
   ```powershell
   python benchmark_model.py --model-path ./quantized_model/onnx_int4/model_quantized.onnx --num-threads 4
   ```

2. **Compare with Original Model** (Optional):
   ```powershell
   python compare_models.py --quantized-path ./quantized_model/onnx_int4 --model-id deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
   ```

3. **Generate HTML Report**:
   ```powershell
   python generate_report.py --results-dir ./inference_results --model-path ./quantized_model/onnx_int4 --output-file ./model_report.html
   ```

### Step 5: Format Conversion (Optional)

1. **Convert to FP16 for Troubleshooting**:
   ```powershell
   python convert_to_fp16.py --input-model ./quantized_model/onnx_int4/model_quantized.onnx --output-model ./quantized_model/onnx_fp16/model_fp16.onnx
   ```

## Project Structure

### Key Components

- **Quantization Tools**: Scripts and notebooks for exporting and quantizing models
- **Testing Framework**: Tools for testing, benchmarking, and troubleshooting the quantized models
- **Maintenance Utilities**: Scripts for project organization and cleanup

### Important Scripts

#### Quantization and Export
- `reexport_with_position_ids.py` - Exports the model with position_ids (fixes 'logits' errors)
- `run_quantization.py` - Dedicated quantization script
- `quantize_int4.ipynb` - Jupyter notebook for Int4 quantization

#### Testing and Inference
- `direct_interactive_test.py` - Interactive testing using direct ONNX Runtime (most reliable)
- `interactive_test.py` - Interactive testing using Optimum integration
- `diagnose_onnx_model.py` - Detailed diagnostic checks for model issues
- `enhanced_onnx_inference.py` - Direct ONNX Runtime inference with diagnostics
- `benchmark_model.py` - Performance benchmarking across different input sizes

#### Execution Scripts
- `run_inference_tests.ps1` - Comprehensive testing suite
- `run_master_test.ps1` - Runs all tests in sequence

#### Maintenance
- `maintain.bat` - Central maintenance dashboard with menu options
- `cleanup_simple.ps1` - Removes temporary files and Python cache

## Common Issues and Solutions

### 1. 'Logits' Error During Generation

**Symptom**: `KeyError: 'logits'` or similar errors during model inference.

**Solution**: The model was likely exported without position_ids input. Re-export the model with:
```powershell
python reexport_with_position_ids.py --model-id deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --output-dir ./onnx_fixed
```

### 2. Model Loading Failures

**Symptom**: Model fails to load or initialize.

**Solution**: Try using different loading parameters:
```python
# For Qwen2-based models
session_options = ort.SessionOptions()
session_options.intra_op_num_threads = 4
session = ort.InferenceSession(model_path, session_options)
```

### 3. Performance Issues

**Symptom**: Slow inference or high memory usage.

**Solution**: Adjust thread count to match your CPU cores:
```powershell
python benchmark_model.py --model-path ./model_path --num-threads 4  # Adjust to your CPU
```

## Configuration Options

- **Thread Count**: Adjust `--num-threads` based on your CPU. Generally, setting it to the number of physical cores works best.
- **Token Limit**: Use `--max-tokens` to control generation length.
- **Model Path**: All scripts accept `--model-path` to specify the model location.

## Resources

- [ONNX Runtime Documentation](https://onnxruntime.ai/)
- [Optimum Documentation](https://huggingface.co/docs/optimum/index)
- [DeepSeek-R1 Model Card](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)

## License

This project is provided for research and educational purposes. The DeepSeek models have their own licenses which should be respected when using them.

## Contributing

Contributions are welcome! See `CONTRIBUTING.md` for guidelines.