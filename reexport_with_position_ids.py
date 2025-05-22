"""
Re-export the DeepSeek-R1-Distill-Qwen-1.5B model to ONNX with position_ids support
This fixes the 'logits' error during generation with the ONNX model
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from optimum.onnxruntime import ORTModelForCausalLM
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Re-export ONNX model with position_ids")
    parser.add_argument("--model-id", type=str, 
                       default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                       help="HuggingFace model ID to export")
    parser.add_argument("--output-dir", type=str, 
                       default="./onnx_fixed",
                       help="Directory to save the exported model")
    parser.add_argument("--quantize", action="store_true",
                       help="Quantize the model to INT4 after export")
    return parser.parse_args()

def export_model(model_id, output_dir, quantize=False):
    """Export the model to ONNX with position_ids support"""
    print(f"Exporting model {model_id} to {output_dir}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # For Qwen2 models, we need to set use_cache=False initially
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        use_cache=False,
        trust_remote_code=True,
        device_map="auto"
    )
      # Export the model to ONNX
    print("Exporting to ONNX...")
    onnx_model = ORTModelForCausalLM.from_pretrained(
        model_id,
        export=True,
        provider="CPUExecutionProvider",
        trust_remote_code=True,
        use_io_binding=False,
        use_cache=False,
    )
    
    # Save the exported model
    onnx_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"Model exported to {output_dir}")
    
    # Quantize if requested
    if quantize:
        print("Quantizing model to INT4...")
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        # Find the ONNX model file
        onnx_path = os.path.join(output_dir, "model.onnx")
        quant_path = os.path.join(output_dir, "model_quantized.onnx")
        
        # Quantize the model
        quantize_dynamic(
            onnx_path,
            quant_path,
            weight_type=QuantType.QInt4
        )
        
        print(f"Quantized model saved to {quant_path}")
    
    return output_dir

def main():
    args = parse_args()
    
    export_model(args.model_id, args.output_dir, args.quantize)
    
    print("\nRe-export complete! The model now includes position_ids support.")
    print("This should fix the 'logits' error during generation.")

if __name__ == "__main__":
    main()
