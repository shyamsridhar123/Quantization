"""
Specialized script for exporting Qwen2 models to ONNX format
This script addresses the specific issues with DeepSeek-R1-Distill-Qwen-1.5B
using a direct approach tailored specifically for these models
"""

import os
import sys
import torch
import gc
import shutil
import tempfile
from pathlib import Path
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

def export_qwen2_model(model_id, output_dir="./onnx_exported", temp_dir="./temp_export"):
    """
    Export a Qwen2 model to ONNX format with special handling for its unique requirements
    
    Args:
        model_id: HuggingFace model ID
        output_dir: Directory to save the final ONNX model
        temp_dir: Directory for temporary files
        
    Returns:
        bool: True if export succeeds, False otherwise
    """
    try:
        import transformers
        print(f"Using transformers version: {transformers.__version__}")
        
        # Make sure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Clean up temp directory if it exists
        if os.path.exists(temp_dir):
            print(f"Cleaning up existing temp directory: {temp_dir}")
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Warning: Failed to clean temp dir: {e}")
                
        # Create a fresh temp directory
        os.makedirs(temp_dir, exist_ok=True)
        
        # Free memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Step 1: Load model config
        print("Loading model configuration...")
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        
        # Make sure use_cache is set to False in config
        if hasattr(config, "use_cache"):
            config.use_cache = False
            print("✓ Disabled KV cache in model config")
            
        # Step 2: Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Step 3: Load the PyTorch model without any cache params for Qwen2
        print("Loading Qwen2 model in PyTorch format...")
        torch_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,  # Config already has use_cache=False set
            trust_remote_code=True,
            torch_dtype=torch.float32  # Using float32 for better compatibility
        )
        
        # Make sure model is in evaluation mode
        torch_model.eval()
          # Step 4: Save model to temp directory
        print(f"Saving model to temporary directory: {temp_dir}")
        torch_model.save_pretrained(temp_dir)
        
        # Step 5: Use direct ONNX export with torch.onnx
        print("\nUsing direct PyTorch ONNX export for Qwen2 models...")
        try:
            # Create a dummy input
            dummy_input = tokenizer("Hello, I am a language model", return_tensors="pt")
            
            # Get the name of the model class to check if it's Qwen2
            model_class = torch_model.__class__.__name__
            print(f"Model class: {model_class}")
            
            # Create export path
            onnx_path = os.path.join(output_dir, "model.onnx")
            
            # Export directly with PyTorch
            with torch.no_grad():
                # Define dynamic axes for the inputs
                dynamic_axes = {
                    'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                    'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                    'output': {0: 'batch_size', 1: 'sequence_length'}
                }
                
                # Define a forward function that doesn't use past_key_values
                def forward_without_past(input_ids, attention_mask):
                    outputs = torch_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True
                    )
                    return outputs.logits
                
                # Export to ONNX format
                print("Exporting model to ONNX format...")
                torch.onnx.export(
                    forward_without_past,
                    (dummy_input['input_ids'], dummy_input['attention_mask']),
                    onnx_path,
                    export_params=True,
                    opset_version=15,
                    input_names=['input_ids', 'attention_mask'],
                    output_names=['output'],
                    dynamic_axes=dynamic_axes,
                    do_constant_folding=True,
                    verbose=False
                )
                
                print(f"✅ Successfully exported ONNX model to: {onnx_path}")
                
                # Also save the tokenizer
                tokenizer.save_pretrained(output_dir)
                
                # Save additional model configuration
                config_path = os.path.join(output_dir, "config.json")
                if not os.path.exists(config_path):
                    config.save_pretrained(output_dir)
                
                # Clean up
                del torch_model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Create a simple generation model_config.json to help optimum
                try:
                    import json
                    model_config = {
                        "model_type": config.model_type,
                        "vocab_size": config.vocab_size,
                        "onnx_config": {
                            "with_past": False,
                            "use_cache": False
                        }
                    }
                    with open(os.path.join(output_dir, "model_config.json"), "w") as f:
                        json.dump(model_config, f, indent=2)
                except Exception as e:
                    print(f"Warning: Couldn't save model_config.json: {e}")
                
                return True
                
        except Exception as export_err:
            print(f"Error during direct PyTorch ONNX export: {export_err}")
            
            # Try a last resort approach - use torch.onnx directly with JIT tracing
            try:
                print("\nTrying PyTorch JIT-based ONNX export...")
                
                # Reload the model
                model = AutoModelForCausalLM.from_pretrained(
                    temp_dir,
                    trust_remote_code=True,
                    torch_dtype=torch.float32
                )
                model.eval()
                
                # Create a dummy input
                dummy_input = tokenizer("Hello, I am a language model", return_tensors="pt")
                
                # Create a scripted model
                class OnnxWrapper(torch.nn.Module):
                    def __init__(self, model):
                        super().__init__()
                        self.model = model
                    
                    def forward(self, input_ids, attention_mask):
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            return_dict=False
                        )
                        return outputs[0]  # Return logits only
                
                onnx_wrapper = OnnxWrapper(model)
                onnx_path = os.path.join(output_dir, "model.onnx")
                
                # Export to ONNX format with explicit tracing
                print("Exporting with JIT tracing...")
                torch.onnx.export(
                    onnx_wrapper,
                    (dummy_input['input_ids'], dummy_input['attention_mask']),
                    onnx_path,
                    export_params=True,
                    opset_version=15,
                    input_names=['input_ids', 'attention_mask'],
                    output_names=['output'],
                    dynamic_axes={
                        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                        'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                        'output': {0: 'batch_size', 1: 'sequence_length'}
                    },
                    do_constant_folding=True
                )
                
                print(f"✅ Successfully exported ONNX model to: {onnx_path}")
                
                # Also save the tokenizer
                tokenizer.save_pretrained(output_dir)
                
                # Save config
                config.save_pretrained(output_dir)
                
                return True
                
            except Exception as jit_err:
                print(f"Error during JIT-based ONNX export: {jit_err}")
                print("\nAll export methods failed.")
                print("Try using bitsandbytes quantization instead.")
                return False
                
    except Exception as e:
        print(f"Unexpected error during export process: {e}")
        return False

if __name__ == "__main__":
    # Allow command line arguments or use defaults
    model_id = sys.argv[1] if len(sys.argv) > 1 else "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./onnx_exported"
    
    print(f"\n===== Exporting {model_id} to ONNX format =====\n")
    success = export_qwen2_model(model_id, output_dir)
    
    if success:
        print("\n✅ ONNX EXPORT COMPLETED SUCCESSFULLY")
        print(f"Model exported to: {output_dir}")
    else:
        print("\n❌ ONNX EXPORT FAILED")
        print("Consider using bitsandbytes quantization for this model instead.")
