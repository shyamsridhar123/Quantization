# DeepSeek/Qwen model special handling for ONNX export
import torch
import os
import gc
from transformers import AutoConfig, AutoModelForCausalLM

def prepare_deepseek_for_onnx_export(model_id, use_float16=False):
    """
    Prepare DeepSeek or Qwen model for ONNX export with special handling
    
    This function addresses common issues with DeepSeek and Qwen models
    during ONNX export by properly configuring them and cleaning up 
    temporary resources.
    
    Args:
        model_id (str): The Hugging Face model ID
        use_float16 (bool): Whether to use float16 for initial loading
        
    Returns:
        tuple: (model, config) - The prepared model and its config
    """
    print(f"Preparing {model_id} for ONNX export...")
    
    # Free memory before loading model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # First load the model config
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    
    # Crucial: Disable KV cache which causes problems with past_key_values during export
    if hasattr(config, 'use_cache'):
        config.use_cache = False
        print("✓ Disabled KV cache in model config to prevent export issues")
    
    # Special handling for model-specific parameters
    if 'qwen' in model_id.lower() or 'deepseek' in model_id.lower():
        print(f"✓ Detected {model_id} as Qwen/DeepSeek model, applying special configuration")
        
        # Set torch dtype appropriately
        torch_dtype = torch.float16 if use_float16 else torch.float32
        print(f"✓ Using {torch_dtype} precision for initial model loading")
        
        # Load the model with optimal settings for export
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            # Use device_map="auto" only if CUDA is available and we have enough VRAM
            device_map="auto" if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 6 * (1024**3) else None
        )
        
        # Force model to eval mode
        model.eval()
        
        return model, config
    else:
        # Standard loading for other models
        print("Using standard model loading...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            trust_remote_code=True
        )
        model.eval()
        return model, config

# Example usage
# model, config = prepare_deepseek_for_onnx_export("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
