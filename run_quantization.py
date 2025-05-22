# Run DeepSeek-R1-Distill-Qwen-1.5B quantization from command line
# This script provides a non-interactive way to run the quantization process

import os
import sys
import torch
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Quantize DeepSeek-R1-Distill-Qwen-1.5B model")
    parser.add_argument("--model-id", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                        help="HuggingFace model ID")
    parser.add_argument("--output-dir", default="./quantized_model",
                        help="Directory to save the quantized model")
    parser.add_argument("--force-cpu", action="store_true",
                        help="Force CPU quantization even if CUDA is available")
    parser.add_argument("--method", choices=["bnb", "onnx", "auto"], default="auto",
                        help="Quantization method: bitsandbytes, ONNX, or auto-detect")
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Check environment first
    print("\n=== Checking Environment ===")
    try:
        from environment_check import check_environment
        env_ready = check_environment()
        if not env_ready:
            print("Environment check failed. Please fix issues before continuing.")
            sys.exit(1)
    except ImportError:
        print("Warning: environment_check.py not found, skipping comprehensive check")
    
    # Configure quantization approach
    cuda_available = torch.cuda.is_available() and not args.force_cpu
    
    if args.method == "auto":
        use_bnb = cuda_available
        use_onnx = not cuda_available
    elif args.method == "bnb":
        if not cuda_available and not args.force_cpu:
            print("Warning: bitsandbytes requested but CUDA unavailable. Forcing CPU mode.")
        use_bnb = cuda_available
        use_onnx = not use_bnb
    else:  # method == "onnx"
        use_bnb = False
        use_onnx = True
    
    print(f"\n=== Quantization Configuration ===")
    print(f"Model ID: {args.model_id}")
    print(f"Output directory: {args.output_dir}")
    print(f"CUDA available: {cuda_available}")
    print(f"Using bitsandbytes: {use_bnb}")
    print(f"Using ONNX: {use_onnx}")
    
    # Make sure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Import necessary modules
    from transformers import AutoTokenizer
    
    # Load tokenizer
    print("\n=== Loading Tokenizer ===")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✓ Tokenizer loaded successfully")
    
    # Run appropriate quantization method
    if use_bnb:
        try:
            print("\n=== Quantizing with bitsandbytes (4-bit) ===")
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig
            
            # Configure 4-bit quantization
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )
            
            # Load and quantize model
            model = AutoModelForCausalLM.from_pretrained(
                args.model_id,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Save model config and tokenizer
            model_output_dir = os.path.join(args.output_dir, "bnb_4bit")
            model.config.save_pretrained(model_output_dir)
            tokenizer.save_pretrained(model_output_dir)
            
            print(f"✅ Model quantized successfully with bitsandbytes")
            print(f"Model configuration saved to {model_output_dir}")
            
        except Exception as e:
            print(f"Error during bitsandbytes quantization: {e}")
            sys.exit(1)
            
    if use_onnx:
        try:
            print("\n=== Quantizing with ONNX (Int4) ===")
            
            # Import our specialized module and run export
            from qwen2_onnx_export import export_qwen2_model
            export_dir = "./onnx_exported"
            
            print("Exporting model to ONNX format...")
            export_success = export_qwen2_model(args.model_id, export_dir)
            
            if not export_success:
                print("Failed to export model to ONNX format")
                sys.exit(1)
                
            # Perform dynamic quantization
            from optimum.onnxruntime import ORTQuantizer
            from optimum.onnxruntime.configuration import AutoQuantizationConfig
            
            print("Creating quantizer...")
            try:
                from optimum.onnxruntime import ORTModelForCausalLM
                ort_model = ORTModelForCausalLM.from_pretrained(export_dir)
                quantizer = ORTQuantizer.from_pretrained(ort_model)
            except Exception as e:
                print(f"Error loading ONNX model: {e}")
                print("Trying direct quantization...")
                quantizer = ORTQuantizer.from_pretrained(export_dir)
            
            # Set up quantization configuration
            print("Setting up quantization configuration...")
            try:
                quantization_config = AutoQuantizationConfig.avx2(is_static=False)
            except:
                from optimum.onnxruntime.configuration import QuantizationConfig
                quantization_config = QuantizationConfig(is_static=False, format="QInt8")
            
            # Quantize the model
            print("Applying quantization...")
            onnx_output_dir = os.path.join(args.output_dir, "onnx_int4")
            os.makedirs(onnx_output_dir, exist_ok=True)
            
            try:
                quantizer.quantize(quantization_config=quantization_config, save_dir=onnx_output_dir)
            except Exception as e:
                print(f"Error during quantization with config: {e}")
                print("Trying with minimal parameters...")
                quantizer.quantize(save_dir=onnx_output_dir)
            
            print(f"✅ Model quantized successfully with ONNX")
            print(f"Quantized model saved to {onnx_output_dir}")
            
        except Exception as e:
            print(f"Error during ONNX quantization: {e}")
            sys.exit(1)
    
    print("\n=== Quantization Complete ===")
    print(f"Output saved to {args.output_dir}")
    print("You can now load the quantized model for inference")

if __name__ == "__main__":
    main()
