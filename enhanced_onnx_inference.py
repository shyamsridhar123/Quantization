"""
Enhanced Direct ONNX Runtime inference script for the quantized DeepSeek-R1-Distill-Qwen-1.5B model
This script provides improved error handling and diagnostics for ONNX models
"""

import os
import numpy as np
import time
import json
import argparse
import onnxruntime as ort
from transformers import AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Enhanced ONNX inference")
    parser.add_argument("--model-path", type=str, 
                      default="./quantized_model/onnx_int4/model_quantized.onnx",
                      help="Path to the ONNX model file")
    parser.add_argument("--model-id", type=str,
                      default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                      help="Original model ID for tokenizer")
    parser.add_argument("--max-tokens", type=int, default=100,
                      help="Maximum tokens to generate")
    parser.add_argument("--prompt", type=str,
                      help="Use a specific prompt instead of interactive mode")
    parser.add_argument("--num-threads", type=int, default=4,
                      help="Number of threads for inference")
    parser.add_argument("--diagnose", action="store_true",
                      help="Run diagnostics on the model")
    return parser.parse_args()

def get_model_metadata(model_path):
    """Get metadata about the ONNX model"""
    try:
        session_options = ort.SessionOptions()
        sess = ort.InferenceSession(model_path, session_options)
        
        metadata = {
            "inputs": [],
            "outputs": [],
            "metadata": {}
        }
        
        # Get input info
        for i, input_info in enumerate(sess.get_inputs()):
            metadata["inputs"].append({
                "name": input_info.name,
                "shape": input_info.shape,
                "type": input_info.type
            })
        
        # Get output info
        for i, output_info in enumerate(sess.get_outputs()):
            metadata["outputs"].append({
                "name": output_info.name,
                "shape": output_info.shape,
                "type": output_info.type
            })
            
        # Get model metadata if available
        if hasattr(sess, "get_modelmeta"):
            model_meta = sess.get_modelmeta()
            if hasattr(model_meta, "custom_metadata_map"):
                metadata["metadata"] = model_meta.custom_metadata_map
        
        return metadata
    except Exception as e:
        print(f"Error getting model metadata: {e}")
        return None

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
            
            # Print progress for long generations
            if (i+1) % 10 == 0:
                print(f"Generated {i+1} tokens...", end="\r")
                
        except Exception as e:
            print(f"\nError during generation at token {i}: {e}")
            break
    
    # Combine the original input with the generated tokens
    result_tokens = input_ids[0, :initial_length].tolist() + generated_tokens
    
    # Convert tokens to text
    result_text = tokenizer.decode(result_tokens, skip_special_tokens=True)
    
    return result_text, result_tokens, (i+1)

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
        print("✓ Model loaded successfully")
        
        # Print model inputs and outputs
        print("\nModel inputs:")
        for i, input_info in enumerate(session.get_inputs()):
            print(f"  {i}: {input_info.name} - Shape: {input_info.shape} - Type: {input_info.type}")
        
        print("\nModel outputs:")
        for i, output_info in enumerate(session.get_outputs()):
            print(f"  {i}: {output_info.name} - Shape: {output_info.shape} - Type: {output_info.type}")
        
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
            print(f"✓ Basic inference successful")
            print(f"  - Output shape: {outputs[0].shape}")
            
            # Check if output dimensions make sense
            if len(outputs[0].shape) == 3:
                batch_size, seq_len, vocab_size = outputs[0].shape
                print(f"  - Batch size: {batch_size}, Sequence length: {seq_len}, Vocabulary size: {vocab_size}")
            else:
                print(f"  - Unexpected output shape: {outputs[0].shape}")
                
            print("\n✅ Model appears to be valid and can run inference")
            return True
        
        except Exception as e:
            print(f"❌ Error during basic inference: {e}")
            return False
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Configuration
    model_path = args.model_path
    original_model_id = args.model_id
    max_tokens = args.max_tokens
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    print("=== Enhanced ONNX Runtime Inference ===")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(original_model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Error loading tokenizer from HuggingFace: {e}")
        print("Trying to load tokenizer from local files...")
        model_dir = os.path.dirname(model_path)
        tokenizer_dir = os.path.dirname(model_dir)
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            print("Successfully loaded tokenizer from local files")
        except Exception as e2:
            print(f"Error loading local tokenizer: {e2}")
            print("Please make sure the tokenizer files are available")
            return
    
    # Run diagnostics if requested
    if args.diagnose:
        run_diagnostics(model_path, tokenizer)
        return
    
    # Set up ONNX Runtime session
    print("\nLoading ONNX model...")
    try:
        # Create session
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = args.num_threads
        session_options.inter_op_num_threads = args.num_threads
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Load session
        start_time = time.time()
        session = ort.InferenceSession(model_path, session_options)
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")
        
        # Print model inputs and outputs
        print("\nModel inputs:")
        for i, input_info in enumerate(session.get_inputs()):
            print(f"  {i}: {input_info.name} - Shape: {input_info.shape} - Type: {input_info.type}")
        
        print("\nModel outputs:")
        for i, output_info in enumerate(session.get_outputs()):
            print(f"  {i}: {output_info.name} - Shape: {output_info.shape} - Type: {output_info.type}")
        
        # If a specific prompt was provided, use it and exit
        if args.prompt:
            print(f"\nUsing provided prompt: {args.prompt}")
            onnx_inputs, _ = prepare_model_inputs(tokenizer, args.prompt)
            
            print("Generating response...")
            start_time = time.time()
            generated_text, tokens, tokens_generated = greedy_search(
                session, 
                tokenizer, 
                onnx_inputs["input_ids"], 
                onnx_inputs["attention_mask"],
                max_new_tokens=max_tokens
            )
            inference_time = time.time() - start_time
            tokens_per_second = tokens_generated / inference_time
            
            print(f"\nGenerated text (in {inference_time:.2f} seconds, {tokens_per_second:.2f} tokens/sec):")
            print("-"*50)
            print(generated_text)
            print("-"*50)
            return
        
        # Otherwise, start interactive mode
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
            print("Tokenizing input...")
            onnx_inputs, _ = prepare_model_inputs(tokenizer, prompt)
            
            # Run inference
            print("Generating response...")
            start_time = time.time()
            generated_text, tokens, tokens_generated = greedy_search(
                session, 
                tokenizer, 
                onnx_inputs["input_ids"], 
                onnx_inputs["attention_mask"],
                max_new_tokens=max_tokens
            )
            inference_time = time.time() - start_time
            tokens_per_second = tokens_generated / inference_time
            
            print(f"\nGenerated text (in {inference_time:.2f} seconds, {tokens_per_second:.2f} tokens/sec):")
            print("-"*50)
            print(generated_text)
            print("-"*50)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
