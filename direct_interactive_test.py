"""
Interactive testing script for the quantized DeepSeek-R1-Distill-Qwen-1.5B model
Uses direct ONNX Runtime approach to fix common generation issues
"""

import os
import numpy as np
import torch
import argparse
import onnxruntime as ort
from transformers import AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Interactive testing for quantized model")
    parser.add_argument("--model-path", type=str, default="./quantized_model/onnx_int4",
                       help="Path to the quantized model directory")
    parser.add_argument("--model-id", type=str, 
                       default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                       help="Original HuggingFace model ID for tokenizer")
    parser.add_argument("--max-tokens", type=int, default=200,
                       help="Maximum number of tokens to generate")
    parser.add_argument("--num-threads", type=int, default=4,
                      help="Number of threads for inference")
    parser.add_argument("--prompt", type=str, default=None,
                      help="Single prompt to use instead of interactive mode")
    return parser.parse_args()

def load_model_and_tokenizer(args):
    # Load tokenizer
    print(f"Loading tokenizer from {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine the full path to the ONNX model file
    if os.path.isdir(args.model_path):
        # Check if this is a directory with model_quantized.onnx
        potential_model_path = os.path.join(args.model_path, "model_quantized.onnx")
        if os.path.exists(potential_model_path):
            model_path = potential_model_path
        else:
            # Look for any .onnx file in the directory
            onnx_files = [f for f in os.listdir(args.model_path) if f.endswith('.onnx')]
            if onnx_files:
                model_path = os.path.join(args.model_path, onnx_files[0])
            else:
                print(f"No ONNX model found in {args.model_path}")
                return None, None
    else:
        model_path = args.model_path

    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return None, None

    # Load ONNX model with optimized settings
    print(f"Loading ONNX model from {model_path}...")
    try:
        # Configure session options
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = args.num_threads
        session_options.inter_op_num_threads = args.num_threads
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Create session
        session = ort.InferenceSession(model_path, session_options)
        
        # Print model inputs and outputs for debugging
        print("\nModel inputs:")
        for i, input_info in enumerate(session.get_inputs()):
            print(f"  {i}: {input_info.name} - Shape: {input_info.shape} - Type: {input_info.type}")
        
        print("Model loaded successfully!")
        return session, tokenizer
        
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return None, None

def prepare_inputs(tokenizer, prompt):
    """Prepare inputs for the ONNX model"""
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Convert to numpy arrays (ONNX runtime expects numpy inputs)
    input_ids = inputs["input_ids"].cpu().numpy()
    attention_mask = inputs["attention_mask"].cpu().numpy() if "attention_mask" in inputs else np.ones_like(input_ids)
    
    return input_ids, attention_mask

def greedy_generate(session, tokenizer, input_ids, attention_mask, max_new_tokens=50):
    """
    Implementation of a simple greedy search for ONNX models with improved error handling
    """
    # Start with the input sequence
    curr_input_ids = input_ids.copy()
    curr_attention_mask = attention_mask.copy()
    initial_length = curr_input_ids.shape[1]
    
    # Keep track of the generated tokens
    generated_tokens = []
    
    # Greedy search loop
    for i in range(max_new_tokens):
        try:
            # Run the model
            onnx_inputs = {
                "input_ids": curr_input_ids,
                "attention_mask": curr_attention_mask
            }
            
            # Check if position_ids is an input
            has_position_ids = any(inp.name == "position_ids" for inp in session.get_inputs())
            if has_position_ids:
                position_ids = np.arange(curr_input_ids.shape[1], dtype=np.int64)
                position_ids = position_ids.reshape(1, -1)
                onnx_inputs["position_ids"] = position_ids
            
            outputs = session.run(None, onnx_inputs)
            
            # The first output is usually the logits tensor
            logits = outputs[0]
            
            # Get the last token's logits
            next_token_logits = logits[:, -1, :]
            
            # Get the token with the highest probability
            next_token = np.argmax(next_token_logits, axis=-1)
            next_token_id = next_token.item(0) if next_token.size == 1 else next_token[0]
            
            # Add the token to our list
            generated_tokens.append(next_token_id)
            
            # Check if we've generated an EOS token
            if next_token_id == tokenizer.eos_token_id:
                break
                
            # Append the new token to the current input_ids
            next_token_as_array = np.array([[next_token_id]])
            curr_input_ids = np.concatenate([curr_input_ids, next_token_as_array], axis=1)
            
            # Extend the attention mask too
            curr_attention_mask = np.concatenate([curr_attention_mask, np.ones_like(next_token_as_array)], axis=1)
            
        except Exception as e:
            print(f"\nError during generation at token {i}: {e}")
            break
    
    # Combine the original input with the generated tokens for the full response
    full_tokens = input_ids[0, :].tolist() + generated_tokens
    
    # Get only the generated part
    result_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return result_text

def chat_loop(session, tokenizer, max_tokens):
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
            if hasattr(tokenizer, "apply_chat_template"):
                formatted_prompt = tokenizer.apply_chat_template(history, tokenize=False)
            else:
                # Manual formatting as fallback
                formatted_prompt = "\n".join([f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}" 
                                             for msg in history])
                formatted_prompt += "\nAssistant:"
            
            # Prepare inputs
            print("Model is thinking...")
            input_ids, attention_mask = prepare_inputs(tokenizer, formatted_prompt)
            
            # Generate response
            response = greedy_generate(
                session,
                tokenizer,
                input_ids,
                attention_mask,
                max_new_tokens=max_tokens
            )
            
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
    session, tokenizer = load_model_and_tokenizer(args)
    
    if session is None or tokenizer is None:
        print("Failed to load model or tokenizer. Exiting.")
        return
    
    # If a prompt is provided, run inference for that prompt
    if args.prompt:
        print(f"Using prompt: {args.prompt}")
        input_ids, attention_mask = prepare_inputs(tokenizer, args.prompt)
        
        print("Model is thinking...")
        response = greedy_generate(
            session,
            tokenizer,
            input_ids,
            attention_mask,
            max_new_tokens=args.max_tokens
        )
        
        print(f"\nResponse: {response}")
        return
    
    # Start chat loop for interactive mode
    chat_loop(session, tokenizer, args.max_tokens)

if __name__ == "__main__":
    main()
