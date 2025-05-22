"""
Interactive testing script for the quantized DeepSeek-R1-Distill-Qwen-1.5B model
This script provides a simple chat interface to test the model interactively
"""

import os
import torch
import argparse
from transformers import AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Interactive testing for quantized model")
    parser.add_argument("--model-path", type=str, default="./quantized_model/onnx_int4",
                       help="Path to the quantized model directory")
    parser.add_argument("--model-type", type=str, default="onnx",
                       choices=["onnx", "bnb"],
                       help="Type of quantized model (onnx or bitsandbytes)")
    parser.add_argument("--original-model-id", type=str, 
                       default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                       help="Original HuggingFace model ID for tokenizer")
    parser.add_argument("--max-tokens", type=int, default=200,
                       help="Maximum number of tokens to generate")
    return parser.parse_args()

def load_model_and_tokenizer(args):
    # Load tokenizer
    print(f"Loading tokenizer from {args.original_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.original_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model based on type
    if args.model_type == "onnx":
        try:
            from optimum.onnxruntime import ORTModelForCausalLM
            
            # Try with multiple loading configurations
            try:
                model = ORTModelForCausalLM.from_pretrained(
                    args.model_path,
                    use_io_binding=True
                )
            except Exception as e:
                print(f"Error with default parameters: {e}")
                print("Trying alternative loading for Qwen2 models...")
                model = ORTModelForCausalLM.from_pretrained(
                    args.model_path,
                    use_cache=False,
                    use_io_binding=False
                )
        except ImportError:
            print("Error: optimum.onnxruntime not installed.")
            print("Please install it with: pip install optimum[onnxruntime]")
            return None, None
            
    elif args.model_type == "bnb":
        try:
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig
            
            # Configure 4-bit quantization
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
        except ImportError:
            print("Error: bitsandbytes not installed.")
            print("Please install it with: pip install bitsandbytes transformers>=4.30.0")
            return None, None
    else:
        print(f"Unknown model type: {args.model_type}")
        return None, None
        
    print(f"Model loaded successfully ({args.model_type})!")
    return model, tokenizer

def chat_loop(model, tokenizer, max_tokens):
    """Interactive chat loop with the model"""
    history = []
    
    print("\n" + "="*50)
    print("Interactive Chat with Quantized DeepSeek-R1-Distill-Qwen-1.5B")
    print("="*50)
    print("Type 'exit', 'quit', or press Ctrl+C to end the conversation")
    print("Type 'clear' to clear conversation history")
    print("-"*50)
    
    while True:
        try:
            user_input = input("\nYou: ")
            
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting chat...")
                break
                
            if user_input.lower() == "clear":
                history = []
                print("Conversation history cleared")
                continue
                
            if not user_input.strip():
                continue
                
            # Add input to history
            history.append({"role": "user", "content": user_input})
            
            # Format conversation for the model
            # Check if the tokenizer has a chat template
            if hasattr(tokenizer, "apply_chat_template"):
                input_ids = tokenizer.apply_chat_template(history, return_tensors="pt")
            else:
                # Manual formatting as fallback
                conversation = "\n".join([f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}" 
                                         for msg in history])
                input_ids = tokenizer(conversation + "\nAssistant:", return_tensors="pt").input_ids
            
            # Move to device if needed
            if hasattr(model, "device"):
                input_ids = input_ids.to(model.device)
            
            # Generate response
            print("Model is thinking...")
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )
            
            # Decode response
            if hasattr(tokenizer, "apply_chat_template"):
                # Get only the new tokens
                new_tokens = outputs[0][input_ids.shape[1]:]
                response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            else:
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Extract just the assistant's response
                if "Assistant:" in response:
                    response = response.split("Assistant:")[-1].strip()
            
            # Print response
            print(f"\nAssistant: {response}")
            
            # Add to history
            history.append({"role": "assistant", "content": response})
            
        except KeyboardInterrupt:
            print("\nExiting chat...")
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    args = parse_args()
    
    # Check if model path exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model path not found: {args.model_path}")
        return
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args)
    
    if model is None or tokenizer is None:
        print("Failed to load model or tokenizer. Exiting.")
        return
    
    # Start chat loop
    chat_loop(model, tokenizer, args.max_tokens)

if __name__ == "__main__":
    main()
