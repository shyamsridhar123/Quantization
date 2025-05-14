# DeepSeek Model Quantization and Fine-Tuning

This project explores techniques for quantizing and fine-tuning large language models from the DeepSeek family, specifically focusing on models around the 1.5B parameter range. The primary goal is to make these models more accessible by reducing their memory footprint and enabling efficient fine-tuning on consumer-grade hardware.

## Project Overview

The repository contains Jupyter notebooks and Python scripts demonstrating:

1.  **Model Quantization (`quant1.ipynb`, `load_quantized_model.py`):**
    *   Loading pre-trained DeepSeek models (e.g., `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, `deepseek-ai/deepseek-coder-1.3b-instruct`) directly in 4-bit or 8-bit precision using the `bitsandbytes` library.
    *   Options for running on CUDA-enabled GPUs or CPU (default precision for CPU).
    *   Basic inference tests to verify model loading and generation capabilities.

2.  **Parameter-Efficient Fine-Tuning with QLoRA (`fine_tune_deepseek_qlora.ipynb`):**
    *   Fine-tuning the `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` model using QLoRA (Quantized Low-Rank Adaptation).
    *   Loading the base model in 4-bit precision.
    *   Configuring and applying LoRA adapters using the `peft` library.
    *   Training the model on a custom (dummy) instruction dataset using the `SFTTrainer` from the `trl` library.
    *   Saving the fine-tuned LoRA adapter and performing inference with the adapted model.

## Key Technologies Used

*   **Python 3.8+**
*   **PyTorch:** Core deep learning framework.
*   **Hugging Face `transformers`:** For accessing pre-trained models, tokenizers, and training utilities.
*   **Hugging Face `datasets`:** For loading and processing datasets.
*   **Hugging Face `peft` (Parameter-Efficient Fine-Tuning):** For LoRA implementation.
*   **Hugging Face `trl` (Transformer Reinforcement Learning):** For `SFTTrainer`.
*   **`bitsandbytes`:** For 4-bit/8-bit quantization (NF4, FP4) and QLoRA support.
*   **`accelerate`:** For efficient hardware utilization and distributed training.
*   **Jupyter Notebooks:** For interactive development and experimentation.

## Prerequisites

Before running the notebooks or scripts, ensure you have the following:

1.  **Python Environment:** Python 3.8 or newer. It's highly recommended to use a virtual environment (e.g., `venv`, `conda`).
    ```bash
    python -m venv .venv
    # On Windows
    .\.venv\Scripts\activate
    # On macOS/Linux
    # source .venv/bin/activate
    ```

2.  **NVIDIA GPU with CUDA (Recommended for Quantization & Fine-tuning):**
    *   Ensure you have an NVIDIA GPU with a compatible CUDA toolkit version installed.
    *   QLoRA and `bitsandbytes` 4-bit/8-bit quantization are primarily designed for and most effective on NVIDIA GPUs.

3.  **Required Python Libraries:**
    You can install the necessary libraries by running the installation cells within the Jupyter notebooks or by using pip:
    ```bash
    # For PyTorch (adjust CUDA version if needed, e.g., cu118, cu121)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

    # For transformers, PEFT, TRL, bitsandbytes, etc.
    pip install transformers datasets peft accelerate bitsandbytes trl sentencepiece scipy
    ```
    *Note: `bitsandbytes` installation can sometimes be tricky, especially on Windows. Ensure your PyTorch and CUDA versions are compatible.*

## How to Use

### 1. Model Quantization and Basic Inference

*   **Jupyter Notebook: `quant1.ipynb`**
    *   Open and run the cells sequentially.
    *   Configure `MODEL_ID`, `USE_CUDA`, and `QUANTIZATION_BITS` in the configuration cell.
    *   This notebook demonstrates loading a model (e.g., `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`) with 4-bit or 8-bit quantization and performing a simple inference test.

*   **Python Script: `load_quantized_model.py`**
    *   Modify the `MODEL_ID` variable in the script if needed.
    *   Run from the terminal:
        ```bash
        python load_quantized_model.py
        ```
    *   This script loads a specified model in 4-bit precision and performs a basic inference.

### 2. Fine-tuning with QLoRA

*   **Jupyter Notebook: `fine_tune_deepseek_qlora.ipynb`**
    *   Open and run the cells sequentially.
    *   **Configuration:**
        *   Verify/update `BASE_MODEL_ID`.
        *   Modify LoRA parameters (`LORA_R`, `LORA_ALPHA`, `LORA_TARGET_MODULES`) and training arguments (`BATCH_SIZE`, `LEARNING_RATE`, `NUM_TRAIN_EPOCHS`, etc.) as needed.
    *   **Dataset:** The notebook uses a small, dummy instruction dataset created in the script. For actual fine-tuning, replace this with your custom dataset. Ensure your dataset is formatted appropriately (e.g., JSONL with a "text" field containing formatted instruction-response pairs, or separate instruction/output columns for use with a `formatting_func`).
    *   The notebook guides through:
        *   Loading the base model in 4-bit.
        *   Setting up LoRA configuration.
        *   Training using `SFTTrainer`.
        *   Saving the trained LoRA adapter.
        *   Performing inference with the fine-tuned model.

## Models

The primary models explored in this project include:

*   `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
*   `deepseek-ai/deepseek-coder-1.3b-instruct` (or similar 1.3B/1.5B DeepSeek variants)

Please ensure you have accepted any necessary terms and conditions on the Hugging Face Hub for these models if required.

## Future Exploration

*   **Experiment with different quantization methods:** Explore other PTQ techniques (e.g., GPTQ, AWQ) if direct `bitsandbytes` loading doesn't meet accuracy/performance needs.
*   **Advanced PEFT techniques:** Investigate other PEFT methods beyond LoRA.
*   **Comprehensive evaluation:** Implement robust evaluation metrics for both the quantized and fine-tuned models on relevant benchmarks or tasks.
*   **CPU optimization:** For CPU-bound scenarios, explore quantization techniques like `torch.quantization` or ONNX Runtime quantization.
*   **Larger datasets for fine-tuning:** Use more extensive and diverse datasets for more impactful fine-tuning.
*   **Hyperparameter optimization:** Systematically tune hyperparameters for both quantization and fine-tuning to achieve optimal results.

## Troubleshooting

*   **CUDA/`bitsandbytes` errors:**
    *   Ensure your NVIDIA drivers, CUDA toolkit, and PyTorch (with CUDA support) versions are compatible.
    *   `bitsandbytes` often requires specific versions to work correctly. Check its documentation for compatibility.
    *   On Windows, `bitsandbytes` can be particularly challenging. Look for pre-compiled binaries or community-provided solutions if you encounter issues.
*   **Out Of Memory (OOM) errors:**
    *   Reduce `BATCH_SIZE` or `MICRO_BATCH_SIZE`.
    *   Decrease `MAX_SEQ_LENGTH`.
    *   Use gradient accumulation (`GRADIENT_ACCUMULATION_STEPS`).
    *   Ensure you are using 4-bit quantization correctly for QLoRA.
*   **Model ID not found:** Double-check the model identifiers on Hugging Face Hub.

Contributions and suggestions are welcome!