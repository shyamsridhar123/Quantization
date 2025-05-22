"""
Utility to convert a quantized model back to FP16 for troubleshooting
This script helps diagnose if quantization is the source of inference issues
"""

import os
import sys
import torch
import time
import numpy as np
import onnx
import onnxruntime as ort
import argparse
from transformers import AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Convert quantized ONNX model to FP16")
    parser.add_argument("--input-model", type=str, 
                       default="./quantized_model/onnx_int4/model_quantized.onnx",
                       help="Path to input quantized ONNX model")
    parser.add_argument("--output-model", type=str, 
                       default="./model_fp16.onnx",
                       help="Path to save the converted model")
    parser.add_argument("--model-id", type=str,
                       default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                       help="Original model ID for tokenizer")
    parser.add_argument("--test", action="store_true",
                       help="Test the model after conversion")
    return parser.parse_args()

def load_and_convert_model(input_model_path, output_model_path):
    """Load a quantized ONNX model and convert it to FP16"""
    print(f"Loading ONNX model from {input_model_path}...")
    
    try:
        # Load the model
        model = onnx.load(input_model_path)
        print("Model loaded successfully")
        
        # Check if model is already in FP16
        is_already_fp16 = True
        for tensor in model.graph.initializer:
            if tensor.data_type != onnx.TensorProto.FLOAT16:
                is_already_fp16 = False
                break
        
        if is_already_fp16:
            print("Model is already in FP16 format")
            return model
        
        print("Converting model to FP16...")
        from onnxmltools.utils.float16_converter import convert_float_to_float16
        model_fp16 = convert_float_to_float16(model)
        
        print(f"Saving converted model to {output_model_path}...")
        onnx.save(model_fp16, output_model_path)
        
        print("Model converted and saved successfully")
        return model_fp16
    
    except Exception as e:
        print(f"Error converting model: {e}")
        return None

def test_model(model_path, tokenizer):
    """Test the model with a simple prompt"""
    print(f"\nTesting model: {model_path}")
    
    try:
        # Create ONNX Runtime session
        session_options = ort.SessionOptions()
        session = ort.InferenceSession(model_path, session_options)
        
        # Test prompt
        prompt = "Hello, how are you today?"
        print(f"Test prompt: \"{prompt}\"")
        
        # Tokenize prompt
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].numpy()
        attention_mask = inputs["attention_mask"].numpy()
        
        onnx_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        
        # Run inference
        print("Running inference...")
        start_time = time.time()
        outputs = session.run(None, onnx_inputs)
        end_time = time.time()
        
        # Print results
        print(f"Inference time: {end_time - start_time:.4f} seconds")
        print(f"Output shape: {outputs[0].shape}")
        
        # Get next token prediction
        logits = outputs[0]
        next_token_logits = logits[:, -1, :]
        next_token = np.argmax(next_token_logits, axis=-1)
        next_token_id = next_token.item(0) if next_token.size == 1 else next_token[0]
        
        # Decode token
        next_token_text = tokenizer.decode([next_token_id])
        print(f"Predicted next token: {next_token_id} (\"{next_token_text}\")")
        
        print("Model test successful!")
        return True
    
    except Exception as e:
        print(f"Error testing model: {e}")
        return False

def main():
    args = parse_args()
    
    # Set up output directory
    output_dir = os.path.dirname(args.output_model)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load and convert model
    converted_model = load_and_convert_model(args.input_model, args.output_model)
    
    if converted_model is None:
        print("Model conversion failed")
        return
    
    # Test the model if requested
    if args.test:
        # Load tokenizer
        print(f"Loading tokenizer from {args.model_id}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Test the model
            test_model(args.output_model, tokenizer)
            
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            print("Skipping model test")
    
    print("\nModel conversion complete")

if __name__ == "__main__":
    main()
