"""
Model diagnostics script for ONNX models - Uses ASCII characters for compatibility
"""

import os
import json
import onnx
import onnxruntime as ort
import argparse
from transformers import AutoTokenizer
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="ONNX Model Diagnostics")
    parser.add_argument("--model-path", type=str, 
                      default="./quantized_model/onnx_int4/model_quantized.onnx",
                      help="Path to the ONNX model file")
    parser.add_argument("--model-id", type=str,
                      default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                      help="Original model ID for tokenizer")
    return parser.parse_args()

def run_diagnostics(model_path, tokenizer):
    """Run diagnostics to identify issues with the model"""
    print("\n=== Running Model Diagnostics ===")
    
    # 1. Check if the model file exists and get its size
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        return False
    
    model_size_bytes = os.path.getsize(model_path)
    model_size_mb = model_size_bytes / (1024 * 1024)
    print(f"PASS: Model file exists: {model_path} ({model_size_mb:.2f} MB)")
    
    # 2. Check configuration files
    model_dir = os.path.dirname(model_path)
    config_path = os.path.join(model_dir, "config.json")
    ort_config_path = os.path.join(model_dir, "ort_config.json")
    
    if os.path.exists(config_path):
        print(f"PASS: Config file exists: {config_path}")
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                print(f"  - Model type: {config.get('model_type', 'unknown')}")
                print(f"  - Vocab size: {config.get('vocab_size', 'unknown')}")
        except Exception as e:
            print(f"  - Error reading config: {e}")
    else:
        print(f"ERROR: Config file not found: {config_path}")
    
    if os.path.exists(ort_config_path):
        print(f"PASS: ONNX Runtime config exists: {ort_config_path}")
        try:
            with open(ort_config_path, "r") as f:
                ort_config = json.load(f)
                print(f"  - Optimization level: {ort_config.get('optimization_level', 'unknown')}")
                print(f"  - Execution providers: {ort_config.get('execution_providers', 'unknown')}")
        except Exception as e:
            print(f"  - Error reading ONNX Runtime config: {e}")
    else:
        print(f"ERROR: ONNX Runtime config not found: {ort_config_path}")
    
    # 3. Try to load the model and get metadata
    try:
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        session = ort.InferenceSession(model_path, session_options)
        print("PASS: Model loaded successfully")
        
        # Print model inputs and outputs
        print("\nModel inputs:")
        for i, input_info in enumerate(session.get_inputs()):
            print(f"  {i}: {input_info.name} - Shape: {input_info.shape} - Type: {input_info.type}")
        
        print("\nModel outputs:")
        for i, output_info in enumerate(session.get_outputs()):
            print(f"  {i}: {output_info.name} - Shape: {output_info.shape} - Type: {output_info.type}")
            
        # Check for position_ids input (required for Qwen2 models)
        has_position_ids = False
        for input_info in session.get_inputs():
            if input_info.name == "position_ids":
                has_position_ids = True
                break
        
        if has_position_ids:
            print("PASS: Model has position_ids input (required for Qwen2 models)")
        else:
            print("WARNING: Model does not have position_ids input (required for Qwen2 batched inference)")
            print("         Consider re-exporting the model with optimum>=1.14")
        
        # 4. Try a very simple inference with a short input
        print("\nTesting inference with a simple input...")
        test_prompt = "Hello, world!"
        inputs = tokenizer(test_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].numpy()
        attention_mask = inputs["attention_mask"].numpy()
        
        onnx_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        
        try:
            print("Running model.run() with simple input...")
            outputs = session.run(None, onnx_inputs)
            print(f"PASS: Basic inference successful")
            print(f"  - Output shape: {outputs[0].shape}")
            
            # Check if output dimensions make sense
            if len(outputs[0].shape) == 3:
                batch_size, seq_len, vocab_size = outputs[0].shape
                print(f"  - Batch size: {batch_size}, Sequence length: {seq_len}, Vocabulary size: {vocab_size}")
            else:
                print(f"  - Unexpected output shape: {outputs[0].shape}")
                
            print("\nPASS: Model appears to be valid and can run inference")
            return True
        
        except Exception as e:
            print(f"ERROR: Error during basic inference: {e}")
            return False
        
    except Exception as e:
        print(f"ERROR: Error loading model: {e}")
        return False

def main():
    args = parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        return
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.model_id}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return
    
    # Run diagnostics
    run_diagnostics(args.model_path, tokenizer)

if __name__ == "__main__":
    main()
