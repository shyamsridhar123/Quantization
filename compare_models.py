"""
Compare the quantized model against the original model
This script performs a side-by-side comparison of the quantized model with the original
"""

import os
import time
import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import onnxruntime as ort

def parse_args():
    parser = argparse.ArgumentParser(description="Compare quantized model with original")
    parser.add_argument("--onnx-model", type=str, 
                       default="./quantized_model/onnx_int4/model_quantized.onnx",
                       help="Path to the quantized ONNX model file")
    parser.add_argument("--original-model-id", type=str, 
                       default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                       help="HuggingFace ID of the original model")
    parser.add_argument("--prompts-file", type=str,
                       default="./sample_prompts.txt",
                       help="File containing sample prompts for testing")
    parser.add_argument("--max-tokens", type=int, default=50,
                       help="Maximum tokens to generate")
    parser.add_argument("--output-file", type=str,
                       default="./comparison_results.txt",
                       help="File to save comparison results")
    return parser.parse_args()

def load_original_model(model_id):
    """Load the original model from HuggingFace"""
    print(f"Loading original model from {model_id}...")
    
    # First try to load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model in half precision to save memory
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"Original model loaded successfully")
    return model, tokenizer

def load_onnx_model(model_path):
    """Load the quantized ONNX model"""
    print(f"Loading ONNX model from {model_path}...")
    
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 4
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Load session
    session = ort.InferenceSession(model_path, session_options)
    
    print(f"ONNX model loaded successfully")
    return session

def run_original_model(model, tokenizer, prompt, max_tokens):
    """Run inference with the original model"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Measure time
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            num_return_sequences=1
        )
    end_time = time.time()
    
    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return {
        "text": generated_text,
        "time": end_time - start_time,
        "tokens": len(outputs[0]) - len(inputs.input_ids[0])
    }

def greedy_search_onnx(session, tokenizer, prompt, max_tokens):
    """Run greedy search with the ONNX model"""
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].numpy()
    attention_mask = inputs["attention_mask"].numpy()
    
    # Start timing
    start_time = time.time()
    
    # Initialize variables
    curr_input_ids = input_ids.copy()
    curr_attention_mask = attention_mask.copy()
    initial_length = curr_input_ids.shape[1]
    generated_tokens = []
    
    # Generate tokens
    for i in range(max_tokens):
        try:
            # Prepare inputs
            onnx_inputs = {
                "input_ids": curr_input_ids,
                "attention_mask": curr_attention_mask
            }
            
            # Run inference
            outputs = session.run(None, onnx_inputs)
            
            # Get next token
            logits = outputs[0]
            next_token_logits = logits[:, -1, :]
            next_token = np.argmax(next_token_logits, axis=-1)
            next_token_id = next_token.item(0) if next_token.size == 1 else next_token[0]
            
            # Add to generated tokens
            generated_tokens.append(next_token_id)
            
            # Check for EOS
            if next_token_id == tokenizer.eos_token_id:
                break
                
            # Update inputs for next iteration
            next_token_as_array = np.array([[next_token_id]])
            curr_input_ids = np.concatenate([curr_input_ids, next_token_as_array], axis=1)
            curr_attention_mask = np.concatenate([curr_attention_mask, np.ones_like(next_token_as_array)], axis=1)
            
        except Exception as e:
            print(f"Error during ONNX generation: {e}")
            break
    
    end_time = time.time()
    
    # Combine tokens and decode
    result_tokens = input_ids[0, :initial_length].tolist() + generated_tokens
    result_text = tokenizer.decode(result_tokens, skip_special_tokens=True)
    
    return {
        "text": result_text,
        "time": end_time - start_time,
        "tokens": len(generated_tokens)
    }

def compute_similarity(text1, text2):
    """Compute similarity between two texts"""
    # Simple character-level Jaccard similarity
    set1 = set(text1.lower())
    set2 = set(text2.lower())
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    jaccard = intersection / union if union > 0 else 0
    
    # Word-level similarity
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    word_intersection = len(words1.intersection(words2))
    word_union = len(words1.union(words2))
    
    word_jaccard = word_intersection / word_union if word_union > 0 else 0
    
    return {
        "char_similarity": jaccard,
        "word_similarity": word_jaccard
    }

def load_prompts(prompts_file):
    """Load prompts from a file"""
    if not os.path.exists(prompts_file):
        # Return some default prompts
        return [
            "What is the capital of France?",
            "Explain quantum computing in simple terms",
            "Write a short poem about artificial intelligence"
        ]
    
    with open(prompts_file, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    return prompts

def main():
    args = parse_args()
    
    # Check if files exist
    if not os.path.exists(args.onnx_model):
        print(f"ONNX model not found: {args.onnx_model}")
        return
    
    # Load prompts
    prompts = load_prompts(args.prompts_file)
    print(f"Loaded {len(prompts)} prompts for testing")
    
    # Choose a subset of prompts if there are many
    if len(prompts) > 5:
        import random
        random.seed(42)  # For reproducibility
        prompts = random.sample(prompts, 5)
        print(f"Selected 5 random prompts for testing")
    
    results = []
    
    try:
        # Load original model
        original_model, tokenizer = load_original_model(args.original_model_id)
        
        # Load ONNX model
        onnx_session = load_onnx_model(args.onnx_model)
        
        # Run comparison for each prompt
        for i, prompt in enumerate(prompts):
            print(f"\nTesting prompt {i+1}/{len(prompts)}: {prompt[:50]}...")
            
            # Run original model
            print("Running original model...")
            original_result = run_original_model(original_model, tokenizer, prompt, args.max_tokens)
            
            # Run ONNX model
            print("Running ONNX model...")
            onnx_result = greedy_search_onnx(onnx_session, tokenizer, prompt, args.max_tokens)
            
            # Compute similarity
            similarity = compute_similarity(original_result["text"], onnx_result["text"])
            
            # Calculate performance metrics
            speedup = original_result["time"] / onnx_result["time"] if onnx_result["time"] > 0 else 0
            
            # Save results
            result = {
                "prompt": prompt,
                "original": original_result,
                "onnx": onnx_result,
                "similarity": similarity,
                "speedup": speedup
            }
            
            results.append(result)
            
            # Print summary
            print(f"Time - Original: {original_result['time']:.2f}s, ONNX: {onnx_result['time']:.2f}s")
            print(f"Speedup: {speedup:.2f}x")
            print(f"Character similarity: {similarity['char_similarity']:.2f}")
            print(f"Word similarity: {similarity['word_similarity']:.2f}")
            
        # Write results to file
        with open(args.output_file, 'w') as f:
            f.write("=== Quantized Model Comparison Results ===\n\n")
            f.write(f"Original Model: {args.original_model_id}\n")
            f.write(f"Quantized Model: {args.onnx_model}\n\n")
            
            for i, result in enumerate(results):
                f.write(f"=== Prompt {i+1} ===\n")
                f.write(f"Prompt: {result['prompt']}\n\n")
                
                f.write("Original Model Output:\n")
                f.write(f"{result['original']['text']}\n")
                f.write(f"Time: {result['original']['time']:.2f}s, Tokens: {result['original']['tokens']}\n\n")
                
                f.write("Quantized Model Output:\n")
                f.write(f"{result['onnx']['text']}\n")
                f.write(f"Time: {result['onnx']['time']:.2f}s, Tokens: {result['onnx']['tokens']}\n\n")
                
                f.write("Comparison:\n")
                f.write(f"Speedup: {result['speedup']:.2f}x\n")
                f.write(f"Character similarity: {result['similarity']['char_similarity']:.2f}\n")
                f.write(f"Word similarity: {result['similarity']['word_similarity']:.2f}\n\n")
                f.write("-" * 80 + "\n\n")
            
            # Overall summary
            avg_speedup = sum(r["speedup"] for r in results) / len(results)
            avg_char_sim = sum(r["similarity"]["char_similarity"] for r in results) / len(results)
            avg_word_sim = sum(r["similarity"]["word_similarity"] for r in results) / len(results)
            
            f.write("=== Overall Summary ===\n")
            f.write(f"Average Speedup: {avg_speedup:.2f}x\n")
            f.write(f"Average Character Similarity: {avg_char_sim:.2f}\n")
            f.write(f"Average Word Similarity: {avg_word_sim:.2f}\n")
        
        print(f"\nComparison results saved to {args.output_file}")
        
    except Exception as e:
        print(f"Error during comparison: {e}")

if __name__ == "__main__":
    main()
