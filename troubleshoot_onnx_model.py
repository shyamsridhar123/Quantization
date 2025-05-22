"""
Troubleshooting script for the quantized DeepSeek-R1-Distill-Qwen-1.5B model
This script will help diagnose issues with the quantized model
"""

import os
import torch
import time
from transformers import AutoTokenizer

# Model and tokenizer configuration
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
ONNX_MODEL_PATH = "./quantized_model/onnx_int4"

# Check if model path exists
if not os.path.exists(ONNX_MODEL_PATH):
    raise ValueError(f"Model path not found: {ONNX_MODEL_PATH}")

print("=== Troubleshooting Quantized Model ===")

# Step 1: Load tokenizer
print("\nStep 1: Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("Set pad_token to eos_token")

print(f"Tokenizer loaded - Model: {tokenizer.name_or_path}")

# Step 2: Prepare simple input
prompt = "Hello, how are you today?"
print(f"\nStep 2: Preparing input with prompt: '{prompt}'")
encoded_input = tokenizer(prompt, return_tensors="pt")
print(f"Input shape: {encoded_input.input_ids.shape}")
print(f"Input tokens: {encoded_input.input_ids[0].tolist()}")

# Step 3: Try to load the ONNX model with different options
print("\nStep 3: Loading ONNX model...")
try:
    from optimum.onnxruntime import ORTModelForCausalLM
    
    # Try different loading configurations
    configs = [
        {"name": "Default", "params": {}},
        {"name": "No cache, no binding", "params": {"use_cache": False, "use_io_binding": False}},
        {"name": "With cache", "params": {"use_cache": True, "use_io_binding": False}},
        {"name": "With binding", "params": {"use_cache": False, "use_io_binding": True}},
    ]
    
    loaded_model = None
    successful_config = None
    
    for config in configs:
        try:
            print(f"\nTrying to load model with config: {config['name']}")
            model = ORTModelForCausalLM.from_pretrained(
                ONNX_MODEL_PATH,
                **config["params"]
            )
            print(f"✓ Successfully loaded model with config: {config['name']}")
            loaded_model = model
            successful_config = config
            break
        except Exception as e:
            print(f"✗ Error: {e}")
    
    if loaded_model is None:
        print("\n❌ Failed to load the model with any configuration")
        exit(1)
        
    # Step 4: Try simple generation with the model
    print("\nStep 4: Testing generation...")
    try:
        attention_mask = torch.ones_like(encoded_input.input_ids)
        outputs = loaded_model.generate(
            input_ids=encoded_input.input_ids,
            attention_mask=attention_mask,
            max_new_tokens=20,
            do_sample=False
        )
        
        print("✓ Generation successful!")
        print(f"Output shape: {outputs.shape}")
        
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated text: {generated_text}")
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        
        try:
            print("\nTrying with explicit configuration...")
            # Try with more explicit input handling
            inputs = {
                "input_ids": encoded_input.input_ids,
                "attention_mask": attention_mask
            }
            
            # Try to run a forward pass directly
            start_time = time.time()
            with torch.no_grad():
                outputs = loaded_model(**inputs)
            end_time = time.time()
            
            print(f"✓ Forward pass successful! Time: {end_time - start_time:.3f} sec")
            print(f"Output keys: {list(outputs.keys())}")
            print(f"Output logits shape: {outputs.logits.shape}")
            
            # Try to get the next token prediction
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1)
            next_token = tokenizer.decode(next_token_id)
            print(f"Next token prediction: {next_token}")
            
        except Exception as e2:
            print(f"❌ Forward pass also failed: {e2}")
        
    # Print summary
    print("\n=== Troubleshooting Summary ===")
    print(f"- ONNX Model path: {ONNX_MODEL_PATH}")
    print(f"- Tokenizer: {tokenizer.name_or_path}")
    if successful_config:
        print(f"- Successful loading config: {successful_config['name']}")
    else:
        print("- No successful loading configuration found")
        
    # Check the model files
    print("\n=== Model Files ===")
    model_files = os.listdir(ONNX_MODEL_PATH)
    for file in model_files:
        file_path = os.path.join(ONNX_MODEL_PATH, file)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
        print(f"- {file}: {file_size:.2f} MB")
    
except ImportError:
    print("❌ Required packages not installed")
    print("Please install: pip install optimum[onnxruntime] transformers>=4.30.0")
    
except Exception as e:
    print(f"❌ Unexpected error: {e}")
