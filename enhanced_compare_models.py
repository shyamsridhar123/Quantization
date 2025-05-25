"""
Enhanced model comparison with full output capture and accuracy metrics (no NLTK)
"""

import os
import time
import json
import argparse
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
from difflib import SequenceMatcher
import re
from collections import Counter

def parse_args():
    parser = argparse.ArgumentParser(description="Enhanced comparison of ONNX models")
    parser.add_argument("--original-path", type=str, 
                       default="./onnx_fixed",
                       help="Path to original ONNX model directory")
    parser.add_argument("--quantized-path", type=str,
                       default="./quantized_model/onnx_int4", 
                       help="Path to quantized model directory")
    parser.add_argument("--model-id", type=str,
                       default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                       help="Model ID for tokenizer")
    parser.add_argument("--output-dir", type=str,
                       default="./comparison_results_enhanced",
                       help="Directory to save comparison results")
    parser.add_argument("--num-samples", type=int, default=5,
                       help="Number of test samples")
    parser.add_argument("--max-length", type=int, default=200,
                       help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--min-length", type=int, default=50,
                       help="Minimum generation length")
    return parser.parse_args()

# Simple text processing functions to replace NLTK
def simple_word_tokenize(text):
    """Simple word tokenizer"""
    # Remove punctuation and split by whitespace
    words = re.findall(r'\b\w+\b', text.lower())
    return words

def simple_sent_tokenize(text):
    """Simple sentence tokenizer"""
    # Split by sentence-ending punctuation
    sentences = re.split(r'[.!?]+', text)
    # Clean up and filter empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

class EnhancedModelComparator:
    def __init__(self, original_path, quantized_path, model_id):
        self.original_path = original_path
        self.quantized_path = quantized_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize both models
        print("Loading original model...")
        self.original_session = self.load_model(original_path)
        
        print("Loading quantized model...")
        self.quantized_session = self.load_model(quantized_path)
        
        # Test prompts with expected content for accuracy measurement
        self.test_prompts = [
            {
                "prompt": "What is artificial intelligence?",
                "type": "definition",
                "expected_keywords": ["computer", "machine", "learning", "data", "algorithm", "intelligence", "system", "human"],
                "expected_concepts": ["problem solving", "decision making", "pattern recognition", "automation"],
                "min_sentences": 2
            },
            {
                "prompt": "Explain how photosynthesis works in plants.",
                "type": "explanation",
                "expected_keywords": ["sunlight", "chlorophyll", "carbon dioxide", "oxygen", "glucose", "water", "energy", "leaves"],
                "expected_concepts": ["light energy", "chemical energy", "CO2", "H2O"],
                "min_sentences": 3
            },
            {
                "prompt": "What are the main causes of climate change?",
                "type": "explanation",
                "expected_keywords": ["greenhouse", "gases", "carbon", "emissions", "fossil", "fuels", "temperature", "warming"],
                "expected_concepts": ["human activities", "industrial", "deforestation", "methane"],
                "min_sentences": 2
            },
            {
                "prompt": "Describe the process of machine learning model training.",
                "type": "technical",
                "expected_keywords": ["data", "training", "model", "algorithm", "parameters", "optimization", "loss", "validation"],
                "expected_concepts": ["gradient descent", "backpropagation", "overfitting", "accuracy"],
                "min_sentences": 3
            },
            {
                "prompt": "Write a brief introduction to Python programming.",
                "type": "introduction",
                "expected_keywords": ["Python", "programming", "language", "syntax", "code", "easy", "versatile", "libraries"],
                "expected_concepts": ["high-level", "interpreted", "readability", "beginner-friendly"],
                "min_sentences": 2
            }
        ]
    
    def load_model(self, model_path):
        """Load ONNX model"""
        model_file = None
        
        if os.path.isdir(model_path):
            for file in os.listdir(model_path):
                if file.endswith('.onnx'):
                    model_file = os.path.join(model_path, file)
                    break
        else:
            model_file = model_path
            
        if not model_file or not os.path.exists(model_file):
            raise FileNotFoundError(f"No ONNX model found in {model_path}")
            
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 4
        
        return ort.InferenceSession(model_file, session_options)
    
    def sample_from_logits(self, logits, temperature=0.7, top_k=50, top_p=0.9):
        """Sample next token from logits with temperature and top-k/top-p filtering"""
        if temperature == 0:
            return np.argmax(logits)
        
        # Apply temperature
        logits = logits / temperature
        
        # Top-k filtering
        if top_k > 0:
            indices = np.argsort(logits)[-top_k:]
            logits_filtered = np.full_like(logits, -np.inf)
            logits_filtered[indices] = logits[indices]
            logits = logits_filtered
        
        # Convert to probabilities
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        # Top-p filtering
        if top_p < 1.0:
            sorted_indices = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_indices]
            cumsum_probs = np.cumsum(sorted_probs)
            
            cutoff_idx = np.searchsorted(cumsum_probs, top_p) + 1
            indices = sorted_indices[:cutoff_idx]
            
            probs_filtered = np.zeros_like(probs)
            probs_filtered[indices] = probs[indices]
            probs = probs_filtered / np.sum(probs_filtered)
        
        return np.random.choice(len(probs), p=probs)
    
    def generate_tokens(self, session, input_ids, attention_mask, max_length=200, min_length=50, temperature=0.7):
        """Generate tokens using the model until a natural stopping point"""
        generated_ids = input_ids.copy()
        generation_times = []
        
        # Get input requirements
        input_names = [input.name for input in session.get_inputs()]
        has_gather = 'onnx::Gather_3' in input_names
        
        # Track for repetition and stopping
        repetition_penalty = 1.2
        generated_tokens = []
        
        # Get sentence end tokens
        period_token = self.tokenizer.encode('.', add_special_tokens=False)[0] if self.tokenizer.encode('.', add_special_tokens=False) else None
        exclaim_token = self.tokenizer.encode('!', add_special_tokens=False)[0] if self.tokenizer.encode('!', add_special_tokens=False) else None
        question_token = self.tokenizer.encode('?', add_special_tokens=False)[0] if self.tokenizer.encode('?', add_special_tokens=False) else None
        sentence_end_tokens = [t for t in [period_token, exclaim_token, question_token] if t is not None]
        
        for step in range(max_length):
            start_time = time.time()
            
            # Prepare inputs
            inputs = {
                'input_ids': generated_ids.astype(np.int64),
                'attention_mask': attention_mask.astype(np.int64)
            }
            
            if 'position_ids' in input_names:
                position_ids = np.arange(generated_ids.shape[1], dtype=np.int64).reshape(1, -1)
                inputs['position_ids'] = position_ids
            
            if has_gather:
                inputs['onnx::Gather_3'] = np.array(generated_ids.shape[1] - 1, dtype=np.int64)
            
            # Run inference
            outputs = session.run(None, inputs)
            logits = outputs[0]
            
            # Get logits for next token
            if len(logits.shape) == 3:
                next_token_logits = logits[0, -1, :]
            else:
                next_token_logits = logits[0, :]
            
            # Apply repetition penalty
            for token_id in generated_tokens[-20:]:
                next_token_logits[token_id] /= repetition_penalty
            
            # Sample next token
            next_token_id = self.sample_from_logits(next_token_logits, temperature)
            
            generation_times.append(time.time() - start_time)
            generated_tokens.append(next_token_id)
            
            # Append token
            generated_ids = np.concatenate([generated_ids, [[next_token_id]]], axis=1)
            attention_mask = np.concatenate([attention_mask, [[1]]], axis=1)
            
            # Check for natural stopping points after minimum length
            if step >= min_length:
                # Stop at EOS
                if next_token_id == self.tokenizer.eos_token_id:
                    break
                
                # Stop at sentence end if we have enough content
                if next_token_id in sentence_end_tokens and step >= min_length * 1.5:
                    # Check if we have multiple sentences
                    current_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                    sentences = simple_sent_tokenize(current_text)
                    if len(sentences) >= 2:
                        break
                
                # Stop if repetitive
                if len(generated_tokens) > 30:
                    last_10 = generated_tokens[-10:]
                    prev_10 = generated_tokens[-20:-10]
                    if last_10 == prev_10:
                        break
        
        return generated_ids, generation_times
    
    def calculate_accuracy_metrics(self, text, prompt_info):
        """Calculate comprehensive accuracy metrics"""
        metrics = {}
        
        # Clean text
        text = text.strip()
        if text.startswith(prompt_info["prompt"]):
            text = text[len(prompt_info["prompt"]):].strip()
        
        # Basic metrics using simple tokenizers
        words = simple_word_tokenize(text)
        sentences = simple_sent_tokenize(text)
        
        metrics['word_count'] = len(words)
        metrics['sentence_count'] = len(sentences)
        metrics['avg_sentence_length'] = len(words) / max(len(sentences), 1)
        
        # 1. Keyword Coverage
        keywords_found = 0
        keyword_positions = []
        for keyword in prompt_info["expected_keywords"]:
            if keyword.lower() in text.lower():
                keywords_found += 1
                # Find position of first occurrence
                pos = text.lower().find(keyword.lower())
                keyword_positions.append(pos / max(len(text), 1))
        
        metrics['keyword_coverage'] = keywords_found / len(prompt_info["expected_keywords"])
        metrics['keywords_found'] = keywords_found
        metrics['keyword_distribution'] = np.std(keyword_positions) if keyword_positions else 0
        
        # 2. Concept Coverage
        concepts_found = 0
        for concept in prompt_info["expected_concepts"]:
            concept_words = concept.lower().split()
            if all(word in text.lower() for word in concept_words):
                concepts_found += 1
        
        metrics['concept_coverage'] = concepts_found / len(prompt_info["expected_concepts"])
        
        # 3. Relevance Score (based on topic words)
        prompt_words = set(simple_word_tokenize(prompt_info["prompt"]))
        text_words = set(words)
        common_words = prompt_words.intersection(text_words)
        metrics['relevance_score'] = len(common_words) / max(len(prompt_words), 1)
        
        # 4. Coherence Metrics
        # Check for proper sentence structure
        proper_sentences = 0
        for sentence in sentences:
            sentence_words = simple_word_tokenize(sentence)
            if len(sentence_words) >= 3:
                proper_sentences += 1
        
        metrics['sentence_coherence'] = proper_sentences / max(len(sentences), 1)
        
        # 5. Information Density
        unique_words = len(set(words))
        metrics['lexical_diversity'] = unique_words / max(len(words), 1)
        
        # 6. Response Completeness
        meets_min_sentences = len(sentences) >= prompt_info.get('min_sentences', 2)
        has_introduction = len(sentences) > 0 and len(simple_word_tokenize(sentences[0])) >= 5
        has_conclusion = len(sentences) > 1
        
        metrics['completeness_score'] = (
            (0.4 if meets_min_sentences else 0) +
            (0.3 if has_introduction else 0) +
            (0.3 if has_conclusion else 0)
        )
        
        # 7. Factual Accuracy Score (simplified - checks for contradictions)
        factual_score = 1.0
        common_errors = {
            "artificial intelligence": ["human brain", "biological"],
            "photosynthesis": ["animals", "darkness"],
            "climate change": ["natural only", "cooling"],
            "machine learning": ["no data", "magic"],
            "python": ["compiled only", "low-level only"]
        }
        
        for topic, errors in common_errors.items():
            if topic in prompt_info["prompt"].lower():
                for error in errors:
                    if error in text.lower():
                        factual_score -= 0.2
        
        metrics['factual_score'] = max(factual_score, 0)
        
        # 8. Overall Accuracy Score
        metrics['overall_accuracy'] = (
            metrics['keyword_coverage'] * 0.25 +
            metrics['concept_coverage'] * 0.20 +
            metrics['relevance_score'] * 0.15 +
            metrics['sentence_coherence'] * 0.15 +
            metrics['completeness_score'] * 0.15 +
            metrics['factual_score'] * 0.10
        )
        
        return metrics
    
    def compare_outputs(self, prompt_info, max_length=200, min_length=50, temperature=0.7):
        """Compare outputs from both models with full generation"""
        prompt = prompt_info["prompt"]
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="np", padding=True)
        input_ids = inputs['input_ids'].astype(np.int64)
        attention_mask = inputs['attention_mask'].astype(np.int64)
        
        # Generate with original model
        print(f"\nGenerating with original model (up to {max_length} tokens)...")
        start_time = time.time()
        original_ids, original_times = self.generate_tokens(
            self.original_session, input_ids, attention_mask, max_length, min_length, temperature
        )
        original_total_time = time.time() - start_time
        original_text = self.tokenizer.decode(original_ids[0], skip_special_tokens=True)
        
        # Generate with quantized model
        print(f"Generating with quantized model (up to {max_length} tokens)...")
        start_time = time.time()
        quantized_ids, quantized_times = self.generate_tokens(
            self.quantized_session, input_ids, attention_mask, max_length, min_length, temperature
        )
        quantized_total_time = time.time() - start_time
        quantized_text = self.tokenizer.decode(quantized_ids[0], skip_special_tokens=True)
        
        # Calculate accuracy metrics
        original_metrics = self.calculate_accuracy_metrics(original_text, prompt_info)
        quantized_metrics = self.calculate_accuracy_metrics(quantized_text, prompt_info)
        
        # Calculate similarity
        similarity = SequenceMatcher(None, original_text.lower(), quantized_text.lower()).ratio()
        
        # Token counts
        original_tokens = len(original_ids[0]) - len(input_ids[0])
        quantized_tokens = len(quantized_ids[0]) - len(input_ids[0])
        
        return {
            'prompt': prompt,
            'prompt_type': prompt_info['type'],
            'original_output': original_text,
            'quantized_output': quantized_text,
            'original_tokens': original_tokens,
            'quantized_tokens': quantized_tokens,
            'original_time': original_total_time,
            'quantized_time': quantized_total_time,
            'original_metrics': original_metrics,
            'quantized_metrics': quantized_metrics,
            'similarity': similarity,
            'speedup': original_total_time / quantized_total_time if quantized_total_time > 0 else 0,
            'tokens_per_second_original': original_tokens / original_total_time if original_total_time > 0 else 0,
            'tokens_per_second_quantized': quantized_tokens / quantized_total_time if quantized_total_time > 0 else 0
        }
    
    def run_comparison(self, num_samples=5, max_length=200, min_length=50, temperature=0.7):
        """Run full comparison"""
        results = []
        
        print(f"\nRunning enhanced comparison with {num_samples} samples...")
        print(f"Generation parameters: max_length={max_length}, min_length={min_length}, temperature={temperature}")
        
        for i in range(min(num_samples, len(self.test_prompts))):
            print(f"\n{'='*60}")
            print(f"Sample {i+1}/{num_samples}: {self.test_prompts[i]['prompt']}")
            print(f"{'='*60}")
            
            result = self.compare_outputs(self.test_prompts[i], max_length, min_length, temperature)
            results.append(result)
            
            # Print quick summary
            print(f"\nOriginal: {result['original_tokens']} tokens, Accuracy: {result['original_metrics']['overall_accuracy']:.2%}")
            print(f"Quantized: {result['quantized_tokens']} tokens, Accuracy: {result['quantized_metrics']['overall_accuracy']:.2%}")
            print(f"Speedup: {result['speedup']:.2f}x, Similarity: {result['similarity']:.2%}")
        
        return results
    
    def generate_report(self, results, output_dir):
        """Generate comprehensive comparison report"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate aggregate metrics
        metrics = {
            'avg_speedup': np.mean([r['speedup'] for r in results]),
            'avg_similarity': np.mean([r['similarity'] for r in results]),
            'avg_original_accuracy': np.mean([r['original_metrics']['overall_accuracy'] for r in results]),
            'avg_quantized_accuracy': np.mean([r['quantized_metrics']['overall_accuracy'] for r in results]),
            'accuracy_retention': np.mean([r['quantized_metrics']['overall_accuracy'] for r in results]) / 
                                 np.mean([r['original_metrics']['overall_accuracy'] for r in results]) if np.mean([r['original_metrics']['overall_accuracy'] for r in results]) > 0 else 0,
            'avg_original_tokens': np.mean([r['original_tokens'] for r in results]),
            'avg_quantized_tokens': np.mean([r['quantized_tokens'] for r in results]),
            'avg_original_time': np.mean([r['original_time'] for r in results]),
            'avg_quantized_time': np.mean([r['quantized_time'] for r in results]),
            'avg_original_tps': np.mean([r['tokens_per_second_original'] for r in results]),
            'avg_quantized_tps': np.mean([r['tokens_per_second_quantized'] for r in results]),
        }
        
        # Model sizes
        original_size = self.get_model_size(self.original_path)
        quantized_size = self.get_model_size(self.quantized_path)
        metrics['size_reduction'] = (1 - quantized_size / original_size) * 100
        metrics['original_size_mb'] = original_size / (1024 * 1024)
        metrics['quantized_size_mb'] = quantized_size / (1024 * 1024)
        
        # Create visualizations
        self.create_visualizations(results, metrics, output_dir)
        
        # Generate detailed report
        report_path = os.path.join(output_dir, 'enhanced_comparison_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*100 + "\n")
            f.write("ENHANCED MODEL COMPARISON REPORT WITH ACCURACY METRICS\n")
            f.write("="*100 + "\n\n")
            
            # Summary section
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-"*50 + "\n")
            f.write(f"Model Size Reduction: {metrics['size_reduction']:.1f}% ")
            f.write(f"({metrics['original_size_mb']:.1f}MB → {metrics['quantized_size_mb']:.1f}MB)\n")
            f.write(f"Average Speedup: {metrics['avg_speedup']:.2f}x\n")
            f.write(f"Accuracy Retention: {metrics['accuracy_retention']:.1%}\n")
            f.write(f"Average Accuracy - Original: {metrics['avg_original_accuracy']:.2%}, ")
            f.write(f"Quantized: {metrics['avg_quantized_accuracy']:.2%}\n")
            f.write(f"Output Similarity: {metrics['avg_similarity']:.2%}\n")
            f.write(f"Tokens/Second - Original: {metrics['avg_original_tps']:.1f}, ")
            f.write(f"Quantized: {metrics['avg_quantized_tps']:.1f}\n")
            f.write("\n")
            
            # Detailed metrics table
            f.write("DETAILED METRICS BY SAMPLE\n")
            f.write("-"*50 + "\n")
            table_data = []
            for i, result in enumerate(results):
                table_data.append([
                    f"S{i+1}",
                    result['prompt_type'],
                    f"{result['original_metrics']['overall_accuracy']:.1%}",
                    f"{result['quantized_metrics']['overall_accuracy']:.1%}",
                    f"{result['similarity']:.1%}",
                    f"{result['speedup']:.1f}x",
                    f"{result['original_tokens']}",
                    f"{result['quantized_tokens']}"
                ])
            
            table_str = tabulate(
                table_data,
                headers=["#", "Type", "Orig Acc", "Quant Acc", "Similar", "Speed", "O-Tok", "Q-Tok"],
                tablefmt="grid"
            )
            f.write(table_str + "\n\n")
            
            # Detailed results
            f.write("DETAILED RESULTS WITH FULL OUTPUTS\n")
            f.write("-"*50 + "\n\n")
            
            for i, result in enumerate(results):
                f.write(f"{'='*80}\n")
                f.write(f"SAMPLE {i+1}: {result['prompt']}\n")
                f.write(f"Type: {result['prompt_type']}\n")
                f.write(f"{'='*80}\n\n")
                
                # Performance metrics
                f.write("PERFORMANCE METRICS:\n")
                f.write(f"  Speedup: {result['speedup']:.2f}x\n")
                f.write(f"  Generation Time - Original: {result['original_time']:.2f}s, Quantized: {result['quantized_time']:.2f}s\n")
                f.write(f"  Tokens Generated - Original: {result['original_tokens']}, Quantized: {result['quantized_tokens']}\n")
                f.write(f"  Tokens/Second - Original: {result['tokens_per_second_original']:.1f}, Quantized: {result['tokens_per_second_quantized']:.1f}\n")
                f.write(f"  Output Similarity: {result['similarity']:.2%}\n")
                f.write("\n")
                
                # Accuracy metrics comparison
                f.write("ACCURACY METRICS:\n")
                f.write("  Original Model:\n")
                for metric, value in result['original_metrics'].items():
                    if isinstance(value, float):
                        f.write(f"    - {metric}: {value:.3f}\n")
                    else:
                        f.write(f"    - {metric}: {value}\n")
                
                f.write("\n  Quantized Model:\n")
                for metric, value in result['quantized_metrics'].items():
                    if isinstance(value, float):
                        f.write(f"    - {metric}: {value:.3f}\n")
                    else:
                        f.write(f"    - {metric}: {value}\n")
                
                f.write("\n")
                
                # Full outputs
                f.write("ORIGINAL MODEL OUTPUT:\n")
                f.write("-"*40 + "\n")
                f.write(result['original_output'] + "\n")
                f.write("\n")
                
                f.write("QUANTIZED MODEL OUTPUT:\n")
                f.write("-"*40 + "\n")
                f.write(result['quantized_output'] + "\n")
                f.write("\n\n")
        
        # Save JSON results
        json_path = os.path.join(output_dir, 'enhanced_results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'summary_metrics': metrics,
                'detailed_results': results
            }, f, indent=2, ensure_ascii=False)
        
        # Generate accuracy comparison CSV
        csv_path = os.path.join(output_dir, 'accuracy_comparison.csv')
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write("Sample,Type,Original_Accuracy,Quantized_Accuracy,Accuracy_Diff,Speedup,Similarity\n")
            for i, result in enumerate(results):
                orig_acc = result['original_metrics']['overall_accuracy']
                quant_acc = result['quantized_metrics']['overall_accuracy']
                f.write(f"{i+1},{result['prompt_type']},{orig_acc:.3f},{quant_acc:.3f},")
                f.write(f"{quant_acc-orig_acc:.3f},{result['speedup']:.2f},{result['similarity']:.3f}\n")
        
        print(f"\nReports saved to: {output_dir}")
        print(f"  - Enhanced report: {report_path}")
        print(f"  - JSON results: {json_path}")
        print(f"  - Accuracy CSV: {csv_path}")
        
        return metrics
    
    def get_model_size(self, model_path):
        """Get total size of model files"""
        total_size = 0
        
        if os.path.isdir(model_path):
            for root, _, files in os.walk(model_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    total_size += os.path.getsize(file_path)
        else:
            total_size = os.path.getsize(model_path)
            
        return total_size
    
    def create_visualizations(self, results, metrics, output_dir):
        """Create comprehensive visualization charts"""
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Accuracy Comparison Radar Chart
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
        
        # Metrics to compare
        accuracy_metrics = ['keyword_coverage', 'concept_coverage', 'relevance_score', 
                           'sentence_coherence', 'completeness_score', 'factual_score']
        metric_labels = ['Keyword\nCoverage', 'Concept\nCoverage', 'Relevance', 
                        'Coherence', 'Completeness', 'Factual\nAccuracy']
        
        # Calculate average for each metric
        original_scores = []
        quantized_scores = []
        
        for metric in accuracy_metrics:
            original_scores.append(np.mean([r['original_metrics'][metric] for r in results]))
            quantized_scores.append(np.mean([r['quantized_metrics'][metric] for r in results]))
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metric_labels), endpoint=False).tolist()
        original_scores += original_scores[:1]
        quantized_scores += quantized_scores[:1]
        angles += angles[:1]
        
        ax.plot(angles, original_scores, 'o-', linewidth=2, label='Original Model', color='blue')
        ax.fill(angles, original_scores, alpha=0.25, color='blue')
        ax.plot(angles, quantized_scores, 'o-', linewidth=2, label='Quantized Model', color='orange')
        ax.fill(angles, quantized_scores, alpha=0.25, color='orange')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels)
        ax.set_ylim(0, 1)
        ax.set_title('Accuracy Metrics Comparison\n(Average Across All Samples)', size=16, y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'accuracy_radar_chart.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Overall Accuracy by Sample
        fig, ax = plt.subplots(figsize=(10, 6))
        
        samples = [f"S{i+1}\n{r['prompt_type']}" for i, r in enumerate(results)]
        original_acc = [r['original_metrics']['overall_accuracy'] for r in results]
        quantized_acc = [r['quantized_metrics']['overall_accuracy'] for r in results]
        
        x = np.arange(len(samples))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, original_acc, width, label='Original', color='blue', alpha=0.7)
        bars2 = ax.bar(x + width/2, quantized_acc, width, label='Quantized', color='orange', alpha=0.7)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1%}', ha='center', va='bottom', fontsize=9)
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1%}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Sample', fontsize=12)
        ax.set_ylabel('Overall Accuracy Score', fontsize=12)
        ax.set_title('Overall Accuracy Comparison by Sample', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(samples, fontsize=10)
        ax.legend()
        ax.set_ylim(0, 1.1)
        
        # Add average lines
        ax.axhline(y=metrics['avg_original_accuracy'], color='blue', linestyle='--', alpha=0.5, 
                  label=f'Original Avg: {metrics["avg_original_accuracy"]:.1%}')
        ax.axhline(y=metrics['avg_quantized_accuracy'], color='orange', linestyle='--', alpha=0.5,
                  label=f'Quantized Avg: {metrics["avg_quantized_accuracy"]:.1%}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'accuracy_by_sample.png'), dpi=300)
        plt.close()
        
        # 3. Performance vs Accuracy Scatter
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, result in enumerate(results):
            # Original model point
            ax.scatter(result['tokens_per_second_original'], 
                      result['original_metrics']['overall_accuracy'],
                      s=200, color='blue', marker='o', alpha=0.7,
                      label='Original' if i == 0 else '')
            
            # Quantized model point
            ax.scatter(result['tokens_per_second_quantized'], 
                      result['quantized_metrics']['overall_accuracy'],
                      s=200, color='orange', marker='s', alpha=0.7,
                      label='Quantized' if i == 0 else '')
            
            # Connect with arrow
            ax.annotate('', xy=(result['tokens_per_second_quantized'], 
                               result['quantized_metrics']['overall_accuracy']),
                       xytext=(result['tokens_per_second_original'], 
                              result['original_metrics']['overall_accuracy']),
                       arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))
            
            # Add sample number
            ax.text(result['tokens_per_second_quantized'], 
                   result['quantized_metrics']['overall_accuracy'],
                   f'S{i+1}', fontsize=9, ha='left', va='bottom')
        
        ax.set_xlabel('Tokens per Second', fontsize=12)
        ax.set_ylabel('Overall Accuracy Score', fontsize=12)
        ax.set_title('Performance vs Accuracy Trade-off', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_vs_accuracy.png'), dpi=300)
        plt.close()
        
        # 4. Detailed Metrics Heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data for heatmap
        metrics_list = ['keyword_coverage', 'concept_coverage', 'relevance_score', 
                       'sentence_coherence', 'completeness_score', 'factual_score', 'overall_accuracy']
        
        heatmap_data = []
        row_labels = []
        
        for i, result in enumerate(results):
            # Original model metrics
            row_data = [result['original_metrics'][m] for m in metrics_list]
            heatmap_data.append(row_data)
            row_labels.append(f'S{i+1} Original')
            
            # Quantized model metrics
            row_data = [result['quantized_metrics'][m] for m in metrics_list]
            heatmap_data.append(row_data)
            row_labels.append(f'S{i+1} Quantized')
        
        # Create heatmap
        sns.heatmap(heatmap_data, 
                   xticklabels=['Keyword', 'Concept', 'Relevance', 'Coherence', 
                               'Complete', 'Factual', 'Overall'],
                   yticklabels=row_labels,
                   annot=True, fmt='.2f', cmap='RdYlGn',
                   cbar_kws={'label': 'Score'},
                   vmin=0, vmax=1)
        
        ax.set_title('Detailed Accuracy Metrics Heatmap', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'accuracy_heatmap.png'), dpi=300)
        plt.close()

def main():
    args = parse_args()
    
    # Check if models exist
    if not os.path.exists(args.original_path):
        print(f"Error: Original model not found at {args.original_path}")
        return
    
    if not os.path.exists(args.quantized_path):
        print(f"Error: Quantized model not found at {args.quantized_path}")
        return
    
    try:
        comparator = EnhancedModelComparator(args.original_path, args.quantized_path, args.model_id)
        results = comparator.run_comparison(
            args.num_samples, 
            args.max_length, 
            args.min_length, 
            args.temperature
        )
        metrics = comparator.generate_report(results, args.output_dir)
        
        # Print final summary
        print("\n" + "="*80)
        print("ENHANCED COMPARISON SUMMARY")
        print("="*80)
        print(f"Model Size Reduction: {metrics['size_reduction']:.1f}%")
        print(f"Average Speedup: {metrics['avg_speedup']:.2f}x")
        print(f"Accuracy Retention: {metrics['accuracy_retention']:.1%}")
        print(f"Original Model Accuracy: {metrics['avg_original_accuracy']:.2%}")
        print(f"Quantized Model Accuracy: {metrics['avg_quantized_accuracy']:.2%}")
        print(f"Output Similarity: {metrics['avg_similarity']:.2%}")
        print("="*80)
        
    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()