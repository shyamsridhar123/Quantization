"""
Inference testing script for the quantized DeepSeek-R1-Distill-Qwen-1.5B model
This script will load the quantized model and run inference with it
"""

import os
import sys
import time
import argparse
import torch
from transformers import AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Test quantized model inference")
    parser.add_argument("--model-path", type=str, default="./quantized_model/onnx_int4",
                       help="Path to the quantized model directory")
    parser.add_argument("--model-type", type=str, default="onnx",
                       choices=["onnx", "bnb"],
                       help="Type of quantized model (onnx or bitsandbytes)")
    parser.add_argument("--original-model-id", type=str, 
                       default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                       help="Original HuggingFace model ID for tokenizer")
    parser.add_argument("--prompt", type=str, 
                       default="What is the capital of France?",
                       help="Prompt for inference")
    parser.add_argument("--max-tokens", type=int, default=50,
                       help="Maximum number of tokens to generate")
    parser.add_argument("--verify", action="store_true",
                       help="Compare against the original model (will download it)")
    return parser.parse_args()

def load_model(args):
    model_path = args.model_path
    model_type = args.model_type
    
    print(f"Loading {model_type} model from {model_path}...")
    
    if model_type == "onnx":
        try:
            # Try to import optimum for ONNX Runtime
            from optimum.onnxruntime import ORTModelForCausalLM
            
            # First try with default parameters
            try:
                model = ORTModelForCausalLM.from_pretrained(
                    model_path,
                    use_io_binding=True
                )
                print("Successfully loaded ONNX model with default parameters")
                return model
            except Exception as e:
                print(f"Error loading model with default parameters: {e}")
                
            # Try with Qwen2-specific parameters
            try:
                print("Trying alternative loading approach for Qwen2 models...")
                model = ORTModelForCausalLM.from_pretrained(
                    model_path,
                    use_cache=False,
                    use_io_binding=False
                )
                print("Successfully loaded ONNX model with Qwen2-specific parameters")
                return model
            except Exception as e2:
                print(f"Error loading model with Qwen2-specific parameters: {e2}")
                raise RuntimeError("Failed to load ONNX model")
        
        except ImportError:
            print("Error: optimum.onnxruntime not installed. Please install it with: pip install optimum[onnxruntime]")
            sys.exit(1)
            
    elif model_type == "bnb":
        try:
            # Try to import required libraries for bitsandbytes
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig
            
            # Configure 4-bit quantization
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            
            # Load the model with the config
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
            print("Successfully loaded bitsandbytes model")
            return model
            
        except ImportError:
            print("Error: bitsandbytes not installed. Please install it with: pip install bitsandbytes transformers>=4.30.0")
            sys.exit(1)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def run_inference(model, tokenizer, prompt, max_tokens):
    print(f"\nRunning inference with prompt: \"{prompt}\"")
    
    # Prepare input
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Move inputs to the right device if the model is on a device
    if hasattr(model, "device"):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Measure inference time
    start_time = time.time()
    
    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,  # Deterministic generation
            num_return_sequences=1
        )
    
    end_time = time.time()
    inference_time = end_time - start_time
    
    # Convert output to text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Calculate tokens per second
    output_length = len(outputs[0])
    tokens_per_second = output_length / inference_time
    
    # Print results
    print("\n=== Inference Results ===")
    print(f"Generated text: {generated_text}")
    print(f"Inference time: {inference_time:.2f} seconds")
    print(f"Output length: {output_length} tokens")
    print(f"Generation speed: {tokens_per_second:.2f} tokens/sec")
    
    return generated_text, inference_time, tokens_per_second

def compare_with_original(args, quantized_output, quantized_time):
    """Compare quantized model with original model"""
    try:
        print("\n=== Comparing with original model ===")
        from transformers import AutoModelForCausalLM
        
        # Get tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.original_model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Load original model with FP16 to save memory
        print(f"Loading original model {args.original_model_id} (this might take a while)...")
        original_model = AutoModelForCausalLM.from_pretrained(
            args.original_model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Run inference
        print(f"Running inference with prompt: \"{args.prompt}\"")
        inputs = tokenizer(args.prompt, return_tensors="pt").to(original_model.device)
        
        # Measure time
        start_time = time.time()
        with torch.no_grad():
            outputs = original_model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=False,
                num_return_sequences=1
            )
        end_time = time.time()
        original_time = end_time - start_time
        
        # Decode output
        original_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Print results
        print("\n=== Original Model Results ===")
        print(f"Generated text: {original_output}")
        print(f"Inference time: {original_time:.2f} seconds")
        print(f"Speed comparison: Original is {quantized_time/original_time:.2f}x {'slower' if quantized_time > original_time else 'faster'} than quantized")
        
        # Compare outputs
        print("\n=== Output Comparison ===")
        if original_output == quantized_output:
            print("✅ Outputs are identical!")
        else:
            print("⚠️ Outputs differ:")
            print(f"Original: {original_output}")
            print(f"Quantized: {quantized_output}")
            
        return original_output, original_time
        
    except Exception as e:
        print(f"Error comparing with original model: {e}")
        return None, None

def main():
    args = parse_args()
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.original_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.original_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load the quantized model
    try:
        model = load_model(args)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Run inference
    quantized_output, quantized_time, tokens_per_second = run_inference(
        model, tokenizer, args.prompt, args.max_tokens
    )
    
    # Compare with original model if requested
    if args.verify:
        compare_with_original(args, quantized_output, quantized_time)
    
    print("\nInference test completed successfully!")

if __name__ == "__main__":
    main()
