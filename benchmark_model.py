"""
Performance benchmarking script for the quantized DeepSeek-R1-Distill-Qwen-1.5B model
This script tests the model with various input lengths and generates performance metrics
"""

import os
import time
import json
import argparse
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from tabulate import tabulate

def parse_args():
    parser = argparse.ArgumentParser(description="Performance benchmarking for quantized models")
    parser.add_argument("--model-path", type=str, 
                       default="./quantized_model/onnx_int4/model_quantized.onnx",
                       help="Path to the ONNX model file")
    parser.add_argument("--model-id", type=str,
                       default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                       help="Original model ID for tokenizer")
    parser.add_argument("--output-file", type=str,
                       default="./benchmark_results.json",
                       help="File to save benchmark results")
    parser.add_argument("--num-threads", type=int, default=4,
                       help="Number of threads for inference")
    parser.add_argument("--max-tokens", type=int, default=50,
                       help="Maximum tokens to generate per prompt")
    parser.add_argument("--warmup-runs", type=int, default=2,
                       help="Number of warmup runs before benchmarking")
    parser.add_argument("--benchmark-runs", type=int, default=5,
                       help="Number of runs for each benchmark")
    return parser.parse_args()

def prepare_model_inputs(tokenizer, prompt):
    """Prepare inputs for the ONNX model"""
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Convert to numpy arrays
    input_ids = inputs["input_ids"].numpy()
    attention_mask = inputs["attention_mask"].numpy()
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }

def run_inference(session, inputs):
    """Run a single inference pass and return output and time"""
    start_time = time.time()
    outputs = session.run(None, inputs)
    end_time = time.time()
    return outputs, end_time - start_time

def greedy_search(session, tokenizer, input_ids, attention_mask, max_new_tokens=50):
    """Run greedy search and measure time and tokens generated"""
    curr_input_ids = input_ids.copy()
    curr_attention_mask = attention_mask.copy()
    
    total_time = 0
    total_tokens = 0
    
    # For each new token
    for i in range(max_new_tokens):
        # Prepare inputs
        onnx_inputs = {
            "input_ids": curr_input_ids,
            "attention_mask": curr_attention_mask
        }
        
        # Run inference
        start_time = time.time()
        try:
            outputs = session.run(None, onnx_inputs)
            inference_time = time.time() - start_time
            total_time += inference_time
            
            # Get next token
            logits = outputs[0]
            next_token_logits = logits[:, -1, :]
            next_token = np.argmax(next_token_logits, axis=-1)
            next_token_id = next_token.item(0) if next_token.size == 1 else next_token[0]
            
            # Append to tokens
            total_tokens += 1
            
            # Check for EOS
            if next_token_id == tokenizer.eos_token_id:
                break
                
            # Prepare for next iteration
            next_token_as_array = np.array([[next_token_id]])
            curr_input_ids = np.concatenate([curr_input_ids, next_token_as_array], axis=1)
            curr_attention_mask = np.concatenate([curr_attention_mask, np.ones_like(next_token_as_array)], axis=1)
            
        except Exception as e:
            print(f"Error during generation: {e}")
            break
    
    return total_time, total_tokens

def benchmark_model(model_path, tokenizer, num_threads, max_tokens, warmup_runs, benchmark_runs):
    """Run a complete benchmark of the model"""
    # Test prompts of different lengths
    test_prompts = {
        "short": "What is the capital of France?",
        "medium": "Can you explain quantum computing in simple terms that a high school student would understand?",
        "long": "Write a summary of the advantages and disadvantages of different deep learning frameworks, including PyTorch, TensorFlow, and JAX. Consider their ease of use, performance, community support, and deployment options."
    }
    
    # Initialize results
    results = {
        "model_path": model_path,
        "num_threads": num_threads,
        "max_tokens": max_tokens,
        "warmup_runs": warmup_runs,
        "benchmark_runs": benchmark_runs,
        "prompts": {},
        "generation": {}
    }
    
    # Create session
    print(f"Creating ONNX session with {num_threads} threads...")
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = num_threads
    session_options.inter_op_num_threads = num_threads
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    try:
        session = ort.InferenceSession(model_path, session_options)
        print("Session created successfully")
        
        # Benchmark inference for each prompt length
        for prompt_type, prompt in test_prompts.items():
            print(f"\nBenchmarking {prompt_type} prompt: {prompt}")
            
            # Prepare inputs
            inputs = prepare_model_inputs(tokenizer, prompt)
            input_length = inputs["input_ids"].shape[1]
            
            # Warmup runs
            print(f"Running {warmup_runs} warmup iterations...")
            for i in range(warmup_runs):
                _, _ = run_inference(session, inputs)
            
            # Benchmark runs for first token generation
            print(f"Running {benchmark_runs} benchmark iterations for first token...")
            first_token_times = []
            for i in range(benchmark_runs):
                _, inference_time = run_inference(session, inputs)
                first_token_times.append(inference_time)
                print(f"Run {i+1}: {inference_time:.4f} seconds")
            
            avg_time = np.mean(first_token_times)
            std_time = np.std(first_token_times)
            
            # Save results for this prompt
            results["prompts"][prompt_type] = {
                "text": prompt,
                "tokens": input_length,
                "first_token_time": {
                    "mean": float(avg_time),
                    "std": float(std_time),
                    "min": float(np.min(first_token_times)),
                    "max": float(np.max(first_token_times)),
                    "runs": first_token_times
                }
            }
            
            print(f"Average time: {avg_time:.4f} ± {std_time:.4f} seconds")
        
        # Benchmark token generation
        print("\nBenchmarking token generation...")
        for prompt_type, prompt in test_prompts.items():
            print(f"\nTesting generation for {prompt_type} prompt")
            
            # Prepare inputs
            inputs = prepare_model_inputs(tokenizer, prompt)
            
            # Run generation
            try:
                generation_times = []
                generation_tokens = []
                
                for i in range(benchmark_runs):
                    print(f"Generation run {i+1}/{benchmark_runs}...")
                    total_time, total_tokens = greedy_search(
                        session, 
                        tokenizer, 
                        inputs["input_ids"],
                        inputs["attention_mask"],
                        max_new_tokens=max_tokens
                    )
                    
                    generation_times.append(total_time)
                    generation_tokens.append(total_tokens)
                    
                    tokens_per_second = total_tokens / total_time if total_time > 0 else 0
                    print(f"Generated {total_tokens} tokens in {total_time:.4f} seconds ({tokens_per_second:.2f} tokens/sec)")
                
                avg_time = np.mean(generation_times)
                avg_tokens = np.mean(generation_tokens)
                avg_tokens_per_second = avg_tokens / avg_time if avg_time > 0 else 0
                
                # Save results for generation
                results["generation"][prompt_type] = {
                    "mean_time": float(avg_time),
                    "mean_tokens": float(avg_tokens),
                    "tokens_per_second": float(avg_tokens_per_second),
                    "runs": [
                        {"time": float(t), "tokens": int(n)} 
                        for t, n in zip(generation_times, generation_tokens)
                    ]
                }
                
                print(f"Average: {avg_tokens:.1f} tokens in {avg_time:.4f} seconds ({avg_tokens_per_second:.2f} tokens/sec)")
                
            except Exception as e:
                print(f"Error during generation benchmark: {e}")
                results["generation"][prompt_type] = {"error": str(e)}
        
        return results
    
    except Exception as e:
        print(f"Error creating session: {e}")
        return {"error": str(e)}

def print_benchmark_summary(results):
    """Print a nicely formatted summary of benchmark results"""
    print("\n==== Benchmark Summary ====\n")
    
    # Model info
    print(f"Model: {results['model_path']}")
    print(f"Threads: {results['num_threads']}")
    print(f"Max tokens: {results['max_tokens']}")
    print(f"Runs: {results['benchmark_runs']} (with {results['warmup_runs']} warmup runs)")
    print("\n")
    
    # First token latency table
    first_token_rows = []
    for prompt_type, data in results["prompts"].items():
        first_token_rows.append([
            prompt_type,
            data["tokens"],
            f"{data['first_token_time']['mean']:.4f} ± {data['first_token_time']['std']:.4f}",
            f"{data['first_token_time']['min']:.4f}",
            f"{data['first_token_time']['max']:.4f}"
        ])
    
    print("First Token Latency:")
    print(tabulate(
        first_token_rows,
        headers=["Prompt Type", "Input Tokens", "Avg Time (s)", "Min Time (s)", "Max Time (s)"],
        tablefmt="pretty"
    ))
    print("\n")
    
    # Generation throughput table
    if "generation" in results:
        gen_rows = []
        for prompt_type, data in results["generation"].items():
            if "error" in data:
                gen_rows.append([
                    prompt_type,
                    "Error",
                    "Error",
                    "Error"
                ])
            else:
                gen_rows.append([
                    prompt_type,
                    f"{data['mean_tokens']:.1f}",
                    f"{data['mean_time']:.4f}",
                    f"{data['tokens_per_second']:.2f}"
                ])
        
        print("Text Generation Performance:")
        print(tabulate(
            gen_rows,
            headers=["Prompt Type", "Tokens Generated", "Time (s)", "Tokens/Second"],
            tablefmt="pretty"
        ))

def main():
    # Parse args
    args = parse_args()
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.model_id}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Error loading tokenizer from HuggingFace: {e}")
        print("Trying to load tokenizer from local files...")
        model_dir = os.path.dirname(args.model_path)
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
    
    # Run benchmarks
    print(f"Running benchmarks on {args.model_path}")
    results = benchmark_model(
        args.model_path,
        tokenizer,
        args.num_threads,
        args.max_tokens,
        args.warmup_runs,
        args.benchmark_runs
    )
    
    # Print summary
    print_benchmark_summary(results)
    
    # Save results to file
    if args.output_file:
        output_dir = os.path.dirname(args.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nBenchmark results saved to {args.output_file}")

if __name__ == "__main__":
    main()
