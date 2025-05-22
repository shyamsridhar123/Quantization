"""
Direct ONNX Runtime inference script for the quantized DeepSeek-R1-Distill-Qwen-1.5B model
This script bypasses the optimum library and uses onnxruntime directly
"""

import os
import numpy as np
import time
import onnxruntime as ort
from transformers import AutoTokenizer

def prepare_model_inputs(tokenizer, prompt, device="cpu"):
    """Prepare inputs for the ONNX model"""
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Convert to numpy arrays (ONNX runtime expects numpy inputs)
    input_ids = inputs["input_ids"].cpu().numpy()
    attention_mask = inputs["attention_mask"].cpu().numpy() if "attention_mask" in inputs else np.ones_like(input_ids)
    
    # Create a dictionary of inputs
    onnx_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }
    
    return onnx_inputs, input_ids.shape[1]

def greedy_search(session, tokenizer, input_ids, attention_mask, max_new_tokens=50):
    """
    Implementation of a simple greedy search for ONNX models
    """
    # Start with the input sequence
    curr_input_ids = input_ids.copy()
    curr_attention_mask = attention_mask.copy()
    initial_length = curr_input_ids.shape[1]
    
    # Keep track of the generated tokens
    generated_tokens = []
    
    # Greedy search loop
    for _ in range(max_new_tokens):
        # Run the model
        onnx_inputs = {
            "input_ids": curr_input_ids,
            "attention_mask": curr_attention_mask
        }
        
        outputs = session.run(None, onnx_inputs)
        
        # The first output is usually the logits tensor
        logits = outputs[0]
        
        # Get the last token's logits
        next_token_logits = logits[:, -1, :]
        
        # Get the token with the highest probability
        next_token = np.argmax(next_token_logits, axis=-1)
        
        # Add the token to our list
        generated_tokens.append(int(next_token))
        
        # Check if we've generated an EOS token
        if int(next_token) == tokenizer.eos_token_id:
            break
            
        # Append the new token to the current input_ids
        next_token_as_array = np.array([[int(next_token)]])
        curr_input_ids = np.concatenate([curr_input_ids, next_token_as_array], axis=1)
        
        # Extend the attention mask too
        curr_attention_mask = np.concatenate([curr_attention_mask, np.ones_like(next_token_as_array)], axis=1)
    
    # Combine the original input with the generated tokens
    result_tokens = input_ids[0, :initial_length].tolist() + generated_tokens
    
    # Convert tokens to text
    result_text = tokenizer.decode(result_tokens, skip_special_tokens=True)
    
    return result_text, result_tokens

def main():
    # Configuration
    MODEL_PATH = "./quantized_model/onnx_int4/model_quantized.onnx"
    ORIGINAL_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return
    
    print("=== Direct ONNX Runtime Inference ===")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set up ONNX Runtime session
    print("\nLoading ONNX model...")
    try:
        # Create session
        session_options = ort.SessionOptions()
        # Configure to use minimal resources
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Load session
        start_time = time.time()
        session = ort.InferenceSession(MODEL_PATH, session_options)
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")
        
        # Print model inputs and outputs
        print("\nModel inputs:")
        for i, input_info in enumerate(session.get_inputs()):
            print(f"  {i}: {input_info.name} - Shape: {input_info.shape} - Type: {input_info.type}")
        
        print("\nModel outputs:")
        for i, output_info in enumerate(session.get_outputs()):
            print(f"  {i}: {output_info.name} - Shape: {output_info.shape} - Type: {output_info.type}")
        
        # Interactive inference loop
        print("\n" + "="*50)
        print("Enter prompts to test the model (type 'exit' to quit)")
        print("="*50)
        
        while True:
            prompt = input("\nPrompt: ")
            if prompt.lower() in ["exit", "quit"]:
                break
            
            if not prompt.strip():
                continue
                
            # Prepare inputs
            print("\nTokenizing input...")
            onnx_inputs, input_length = prepare_model_inputs(tokenizer, prompt)
            
            # Run inference
            print("Generating response...")
            start_time = time.time()
            try:
                # Use our custom greedy search implementation
                generated_text, _ = greedy_search(
                    session, 
                    tokenizer, 
                    onnx_inputs["input_ids"], 
                    onnx_inputs["attention_mask"],
                    max_new_tokens=50
                )
                inference_time = time.time() - start_time
                print(f"\nGenerated text (in {inference_time:.2f} seconds):")
                print("-"*50)
                print(generated_text)
                print("-"*50)
            except Exception as e:
                print(f"Inference error: {e}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
