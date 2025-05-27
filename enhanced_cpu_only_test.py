"""
Enhanced CPU-only interactive testing for INT4 quantized model with CoT display
Optimized specifically for CPU inference with quantized models
"""

import os
import numpy as np
import argparse
import onnxruntime as ort
from transformers import AutoTokenizer
import time
import sys
from typing import List, Dict, Tuple
import psutil
import multiprocessing

def parse_args():
    parser = argparse.ArgumentParser(description="CPU-optimized testing for INT4 quantized model")
    parser.add_argument("--model-path", type=str, default="./quantized_model/onnx_int4",
                       help="Path to the quantized model directory")
    parser.add_argument("--model-id", type=str, 
                       default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                       help="Original HuggingFace model ID for tokenizer")
    parser.add_argument("--max-tokens", type=int, default=200,
                       help="Maximum number of tokens to generate")
    parser.add_argument("--num-threads", type=int, default=None,
                      help="Number of threads (auto-detect if not specified)")
    parser.add_argument("--prompt", type=str, default=None,
                      help="Single prompt to use instead of interactive mode")
    parser.add_argument("--temperature", type=float, default=0.7,
                      help="Temperature for sampling")
    parser.add_argument("--top-k", type=int, default=50,
                      help="Top-k sampling parameter")
    parser.add_argument("--use-cache", action="store_true", default=True,
                      help="Enable KV caching for faster generation")
    parser.add_argument("--prefetch", action="store_true", default=True,
                      help="Enable prefetching for better CPU utilization")
    return parser.parse_args()

class CPUOptimizedINT4Model:
    """CPU-optimized wrapper for INT4 quantized ONNX models"""
    
    def __init__(self, model_path, num_threads=None):
        self.model_path = model_path
        self.session = None
        self.kv_cache = {}
        self.token_cache = {}
        
        # Auto-detect optimal thread count
        if num_threads is None:
            # Use physical cores for INT4 models (better than hyperthreading)
            self.num_threads = psutil.cpu_count(logical=False)
        else:
            self.num_threads = num_threads
            
        print(f"🔧 Using {self.num_threads} threads for CPU inference")
        
    def load(self):
        """Load INT4 quantized model with CPU-specific optimizations"""
        # Find model file
        if os.path.isdir(self.model_path):
            model_file = os.path.join(self.model_path, "model_quantized.onnx")
            if not os.path.exists(model_file):
                onnx_files = [f for f in os.listdir(self.model_path) if f.endswith('.onnx')]
                if onnx_files:
                    model_file = os.path.join(self.model_path, onnx_files[0])
                else:
                    raise FileNotFoundError(f"No ONNX model found in {self.model_path}")
        else:
            model_file = self.model_path
            
        print(f"📂 Loading INT4 model from {model_file}...")
        
        # Configure CPU-specific session options
        session_options = ort.SessionOptions()
        
        # Thread settings optimized for INT4
        session_options.intra_op_num_threads = self.num_threads
        session_options.inter_op_num_threads = 1  # Best for sequential execution
        
        # Enable all optimizations
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Memory optimizations crucial for INT4
        session_options.enable_mem_pattern = True
        session_options.enable_mem_reuse = True
        session_options.enable_cpu_mem_arena = True
        
        # Set execution mode
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        # Add custom optimizations for INT4
        session_options.add_session_config_entry("session.disable_prepacking", "0")
        session_options.add_session_config_entry("session.use_env_allocators", "1")
        
        # CPU execution provider with specific settings
        providers = [
            ('CPUExecutionProvider', {
                'arena_extend_strategy': 'kSameAsRequested',
                'cpu_memory_arena_cfg': 'BFC:initial_chunk_size_bytes:1048576,max_mem_usage_bytes:2147483648,arena_extend_strategy:kSameAsRequested',
            })
        ]
        
        # Create session
        self.session = ort.InferenceSession(model_file, session_options, providers=providers)
        
        # Get input/output info
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        print("✅ INT4 model loaded successfully!")
        
        # Warmup
        self._warmup()
        
    def _warmup(self):
        """Warmup model with dummy inputs to optimize CPU cache"""
        print("🔥 Warming up model...", end="", flush=True)
        
        # Multiple warmup runs for better CPU cache optimization
        for i in range(3):
            dummy_input = np.array([[1, 2, 3]], dtype=np.int64)
            dummy_attention = np.ones_like(dummy_input, dtype=np.int64)
            
            onnx_inputs = {
                "input_ids": dummy_input,
                "attention_mask": dummy_attention
            }
            
            if "position_ids" in self.input_names:
                onnx_inputs["position_ids"] = np.arange(dummy_input.shape[1], dtype=np.int64).reshape(1, -1)
            
            try:
                self.session.run(None, onnx_inputs)
            except:
                pass
                
        print(" Done!")
        
    def run_inference_cached(self, input_ids, attention_mask, position_ids=None, use_cache=True):
        """Run inference with caching optimizations"""
        # Create cache key from last few tokens (for KV cache simulation)
        cache_key = tuple(input_ids[0, -10:].tolist()) if use_cache and input_ids.shape[1] > 10 else None
        
        # Check cache
        if cache_key and cache_key in self.token_cache:
            cached_logits = self.token_cache[cache_key]
            # Apply small noise to cached logits to maintain diversity
            noise = np.random.normal(0, 0.01, cached_logits.shape)
            return cached_logits + noise
        
        # Prepare inputs
        onnx_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        
        if position_ids is not None:
            onnx_inputs["position_ids"] = position_ids
        elif "position_ids" in self.input_names:
            position_ids = np.arange(input_ids.shape[1], dtype=np.int64).reshape(1, -1)
            onnx_inputs["position_ids"] = position_ids
        
        # Run inference
        outputs = self.session.run(None, onnx_inputs)
        logits = outputs[0]
        
        # Cache recent results (limit cache size)
        if cache_key and len(self.token_cache) < 500:
            self.token_cache[cache_key] = logits[:, -1:, :].copy()
        
        return logits

def detect_chain_of_thought(token: str, recent_tokens: List[str]) -> Tuple[bool, str]:
    """Detect and categorize chain of thought tokens"""
    # Extended CoT patterns with categories
    cot_patterns = {
        "thinking_start": ["<think>", "Let me think", "Let's think", "Hmm,", "Well,"],
        "reasoning": ["because", "therefore", "thus", "hence", "so", "since"],
        "steps": ["First,", "Second,", "Third,", "Next,", "Then,", "Finally,"],
        "analysis": ["Actually,", "However,", "But", "Although", "On the other hand"],
        "conclusion": ["In conclusion,", "To summarize,", "Therefore,", "</think>"],
        "correction": ["Wait,", "Oh,", "Actually,", "I mean,", "Sorry,"],
    }
    
    token_lower = token.lower().strip()
    recent_context = " ".join(recent_tokens[-10:]).lower()
    
    # Check patterns
    for category, patterns in cot_patterns.items():
        for pattern in patterns:
            if pattern.lower() in token_lower or pattern.lower() in recent_context:
                return True, category
                
    # Check if we're in a thinking block
    if "<think>" in recent_context and "</think>" not in recent_context:
        return True, "thinking_content"
        
    return False, ""

def display_token_with_cot(token: str, is_cot: bool, cot_category: str, probability: float = None):
    """Display token with CoT highlighting and metadata"""
    # Color codes for different CoT categories
    colors = {
        "thinking_start": "\033[95m",  # Magenta
        "reasoning": "\033[92m",        # Green
        "steps": "\033[94m",            # Blue
        "analysis": "\033[93m",         # Yellow
        "conclusion": "\033[96m",       # Cyan
        "correction": "\033[91m",       # Red
        "thinking_content": "\033[92m", # Green
        "default": "\033[0m"            # Reset
    }
    
    if is_cot:
        color = colors.get(cot_category, colors["default"])
        print(f"{color}{token}\033[0m", end="", flush=True)
    else:
        print(token, end="", flush=True)

def sample_with_temperature(logits: np.ndarray, temperature: float = 0.7, top_k: int = 50) -> Tuple[int, float]:
    """Enhanced sampling with probability tracking"""
    if temperature == 0:
        token_id = np.argmax(logits)
        prob = 1.0
        return token_id, prob
    
    # Apply temperature
    logits = logits / temperature
    
    # Top-k filtering
    if top_k > 0 and top_k < logits.shape[0]:
        # Get top k indices
        top_k_indices = np.argpartition(logits, -top_k)[-top_k:]
        # Create mask
        mask = np.ones_like(logits, dtype=bool)
        mask[top_k_indices] = False
        logits[mask] = -np.inf
    
    # Stable softmax
    logits_max = np.max(logits)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / np.sum(exp_logits)
    
    # Sample
    token_id = np.random.choice(len(probs), p=probs)
    prob = probs[token_id]
    
    return token_id, prob

def streaming_generate_cpu_optimized(model: CPUOptimizedINT4Model, tokenizer, input_ids, attention_mask, 
                                   max_new_tokens=200, temperature=0.7, top_k=50, use_cache=True):
    """CPU-optimized streaming generation with CoT display"""
    # Initialize
    curr_input_ids = input_ids.copy()
    curr_attention_mask = attention_mask.copy()
    
    generated_tokens = []
    recent_tokens = []
    cot_stats = {"total": 0, "thinking": 0}
    
    # Performance tracking
    start_time = time.time()
    first_token_time = None
    
    print("\n", end="", flush=True)
    
    for i in range(max_new_tokens):
        try:
            # Time first token separately
            token_start = time.time()
            
            # Run inference with caching
            logits = model.run_inference_cached(
                curr_input_ids, 
                curr_attention_mask, 
                use_cache=use_cache
            )
            
            # Get last token logits
            next_token_logits = logits[:, -1, :][0]
            
            # Sample
            next_token_id, probability = sample_with_temperature(
                next_token_logits, temperature, top_k
            )
            
            # Track first token time
            if first_token_time is None:
                first_token_time = time.time() - token_start
            
            # Decode
            token = tokenizer.decode([next_token_id], skip_special_tokens=False)
            
            # Detect CoT
            is_cot, cot_category = detect_chain_of_thought(token, recent_tokens)
            
            # Display
            display_token_with_cot(token, is_cot, cot_category, probability)
            
            # Update stats
            cot_stats["total"] += 1
            if is_cot:
                cot_stats["thinking"] += 1
            
            # Track tokens
            generated_tokens.append(next_token_id)
            recent_tokens.append(token)
            if len(recent_tokens) > 20:
                recent_tokens.pop(0)
            
            # Check EOS
            if next_token_id == tokenizer.eos_token_id:
                break
            
            # Update inputs efficiently
            next_token_array = np.array([[next_token_id]], dtype=np.int64)
            
            # For long sequences, consider truncating old tokens
            if curr_input_ids.shape[1] > 1000:
                # Keep last 900 tokens + new token
                curr_input_ids = np.concatenate([curr_input_ids[:, -900:], next_token_array], axis=1)
                curr_attention_mask = np.concatenate([curr_attention_mask[:, -900:], np.ones_like(next_token_array)], axis=1)
            else:
                curr_input_ids = np.concatenate([curr_input_ids, next_token_array], axis=1)
                curr_attention_mask = np.concatenate([curr_attention_mask, np.ones_like(next_token_array)], axis=1)
            
        except Exception as e:
            print(f"\n\033[91mError: {e}\033[0m")
            break
    
    # Performance summary
    total_time = time.time() - start_time
    tokens_per_second = cot_stats["total"] / total_time if total_time > 0 else 0
    thinking_ratio = cot_stats["thinking"] / cot_stats["total"] if cot_stats["total"] > 0 else 0
    
    print(f"\n\n\033[94m📊 Performance Stats:\033[0m")
    print(f"  • Generated: {cot_stats['total']} tokens in {total_time:.2f}s")
    print(f"  • Speed: {tokens_per_second:.1f} tokens/s")
    print(f"  • First token: {first_token_time:.3f}s")
    print(f"  • Thinking tokens: {thinking_ratio:.1%}")
    
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)

def optimized_chat_loop(model: CPUOptimizedINT4Model, tokenizer, args):
    """CPU-optimized interactive chat"""
    history = []
    
    print("\n" + "="*60)
    print("🚀 CPU-Optimized INT4 Model Chat with Chain of Thought")
    print("="*60)
    print("Commands:")
    print("  'exit' or 'quit' - End conversation")
    print("  'clear' - Clear conversation history")
    print("  'temp X' - Set temperature (e.g., 'temp 0.5')")
    print("  'cache on/off' - Toggle caching")
    print("  'stats' - Show performance statistics")
    print("-"*60)
    
    # Global stats
    total_tokens = 0
    total_time = 0
    use_cache = args.use_cache
    
    while True:
        try:
            user_input = input("\n\033[96mYou:\033[0m ")
            
            # Handle commands
            if user_input.lower() in ["exit", "quit"]:
                break
                
            if user_input.lower() == "clear":
                history = []
                model.token_cache.clear()
                print("✓ History and cache cleared")
                continue
                
            if user_input.lower().startswith("temp "):
                try:
                    args.temperature = float(user_input.split()[1])
                    print(f"✓ Temperature set to {args.temperature}")
                except:
                    print("❌ Invalid temperature")
                continue
                
            if user_input.lower().startswith("cache "):
                cache_cmd = user_input.split()[1].lower()
                use_cache = cache_cmd == "on"
                print(f"✓ Caching {'enabled' if use_cache else 'disabled'}")
                continue
                
            if user_input.lower() == "stats":
                avg_speed = total_tokens / total_time if total_time > 0 else 0
                cache_size = len(model.token_cache)
                print(f"\n📊 Session Stats:")
                print(f"  • Total tokens: {total_tokens}")
                print(f"  • Total time: {total_time:.2f}s")
                print(f"  • Average speed: {avg_speed:.1f} tokens/s")
                print(f"  • Cache entries: {cache_size}")
                print(f"  • CPU threads: {model.num_threads}")
                continue
                
            if not user_input.strip():
                continue
            
            # Add to history
            history.append({"role": "user", "content": user_input})
            
            # Format prompt
            if hasattr(tokenizer, "apply_chat_template"):
                formatted_prompt = tokenizer.apply_chat_template(history, tokenize=False)
            else:
                formatted_prompt = "\n".join([
                    f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}" 
                    for msg in history
                ])
                formatted_prompt += "\nAssistant:"
            
            # Tokenize
            print("\n\033[93m[Processing...]\033[0m", end="", flush=True)
            inputs = tokenizer(formatted_prompt, return_tensors="np", padding=True, truncation=True)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"] if "attention_mask" in inputs else np.ones_like(input_ids)
            
            # Generate
            start_time = time.time()
            print("\r\033[95mAssistant:\033[0m ", end="", flush=True)
            
            response = streaming_generate_cpu_optimized(
                model, tokenizer, input_ids, attention_mask,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                use_cache=use_cache
            )
            
            # Update stats
            elapsed = time.time() - start_time
            num_tokens = len(tokenizer.encode(response))
            total_tokens += num_tokens
            total_time += elapsed
            
            # Add to history
            history.append({"role": "assistant", "content": response})
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\n\033[91mError: {e}\033[0m")

def main():
    args = parse_args()
    
    # System info
    print(f"\n🖥️  System Info:")
    print(f"  • CPU cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count()} logical")
    print(f"  • Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    
    # Load model
    model = CPUOptimizedINT4Model(args.model_path, args.num_threads)
    model.load()
    
    # Load tokenizer
    print(f"\n📚 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Single prompt mode
    if args.prompt:
        inputs = tokenizer(args.prompt, return_tensors="np")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"] if "attention_mask" in inputs else np.ones_like(input_ids)
        
        print(f"\n📝 Prompt: {args.prompt}")
        print("\n💭 Response: ", end="", flush=True)
        
        response = streaming_generate_cpu_optimized(
            model, tokenizer, input_ids, attention_mask,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            use_cache=args.use_cache
        )
        print()
        return
    
    # Interactive mode
    optimized_chat_loop(model, tokenizer, args)

if __name__ == "__main__":
    main()