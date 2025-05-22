"""
Generate a comprehensive HTML report for the quantized model tests
This script processes the benchmark results and test logs to create a visual report
"""

import os
import json
import argparse
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from bs4 import BeautifulSoup

def parse_args():
    parser = argparse.ArgumentParser(description="Generate HTML report from test results")
    parser.add_argument("--results-dir", type=str, 
                       default="./inference_results",
                       help="Directory containing test results")
    parser.add_argument("--output-file", type=str, 
                       default="./model_report.html",
                       help="Output HTML report file")
    parser.add_argument("--model-path", type=str,
                       default="./quantized_model/onnx_int4",
                       help="Path to the model directory")
    return parser.parse_args()

def extract_info_from_logs(log_files):
    """Extract key information from test log files"""
    info = {
        "errors": [],
        "successes": [],
        "diagnostics": {},
        "prompts": {}
    }
    
    for log_file in log_files:
        try:
            # First try UTF-8
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                # Fall back to latin-1 which is more permissive
                with open(log_file, 'r', encoding='latin-1') as f:
                    content = f.read()
            except Exception as e:
                print(f"Warning: Could not read {log_file}: {e}")
                continue
            
            # Extract errors
            error_matches = re.findall(r'Error:?\s(.+?)(?:\n|$)', content)
            for error in error_matches:
                if error not in info["errors"]:
                    info["errors"].append(error)
            
            # Extract successful operations
            success_matches = re.findall(r'✓\s(.+?)(?:\n|$)', content)
            for success in success_matches:
                if success not in info["successes"]:
                    info["successes"].append(success)
            
            # Extract diagnostics info
            if "Running model diagnostics" in content:
                diag_section = re.search(r'=== Running Model Diagnostics ===(.+?)(?:====|$)', content, re.DOTALL)
                if diag_section:
                    diag_text = diag_section.group(1)
                    # Extract model size
                    size_match = re.search(r'Model file exists: .+? \(([\d\.]+) MB\)', diag_text)
                    if size_match:
                        info["diagnostics"]["model_size_mb"] = float(size_match.group(1))
                    
                    # Extract model type
                    type_match = re.search(r'Model type: (\w+)', diag_text)
                    if type_match:
                        info["diagnostics"]["model_type"] = type_match.group(1)
                    
                    # Extract vocab size
                    vocab_match = re.search(r'Vocab size: (\d+)', diag_text)
                    if vocab_match:
                        info["diagnostics"]["vocab_size"] = int(vocab_match.group(1))
            
            # Extract prompt results
            prompt_sections = re.findall(r'Prompt: "([^"]+)"(.+?)(?:Prompt:|$)', content, re.DOTALL)
            for prompt, section in prompt_sections:
                if prompt not in info["prompts"]:
                    info["prompts"][prompt] = {}
                
                # Extract generation time
                time_match = re.search(r'Generated text \(in ([\d\.]+) seconds', section)
                if time_match:
                    info["prompts"][prompt]["time_seconds"] = float(time_match.group(1))
                
                # Extract tokens per second
                tps_match = re.search(r'([\d\.]+) tokens/sec', section)
                if tps_match:
                    info["prompts"][prompt]["tokens_per_second"] = float(tps_match.group(1))
                
                # Extract generated text
                text_match = re.search(r'-{10,}([\s\S]+?)-{10,}', section)
                if text_match:
                    info["prompts"][prompt]["generated_text"] = text_match.group(1).strip()
    
    return info

def parse_benchmark_files(benchmark_files):
    """Parse benchmark JSON files and return combined data"""
    all_data = []
    
    for benchmark_file in benchmark_files:
        try:
            # First try UTF-8
            with open(benchmark_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data.append(data)
        except UnicodeDecodeError:
            try:
                # Fall back to latin-1 which is more permissive
                with open(benchmark_file, 'r', encoding='latin-1') as f:
                    data = json.load(f)
                    all_data.append(data)
            except Exception as e:
                print(f"Error parsing benchmark file {benchmark_file}: {e}")
        except Exception as e:
            print(f"Error parsing benchmark file {benchmark_file}: {e}")
    
    return all_data

def create_performance_charts(benchmark_data, output_dir):
    """Create performance charts from benchmark data"""
    chart_files = []
    
    if not benchmark_data:
        return chart_files
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set the style
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    for i, data in enumerate(benchmark_data):
        try:
            # First token latency chart
            if "prompts" in data:
                plt.figure(figsize=(10, 6))
                prompt_types = []
                latencies = []
                errors = []
                
                for prompt_type, prompt_data in data["prompts"].items():
                    if "first_token_time" in prompt_data:
                        prompt_types.append(prompt_type)
                        latencies.append(prompt_data["first_token_time"]["mean"])
                        errors.append(prompt_data["first_token_time"]["std"])
                
                if prompt_types:
                    print(f"[DEBUG] prompt_types: {prompt_types}")
                    print(f"[DEBUG] latencies: {latencies} (len={len(latencies)})")
                    print(f"[DEBUG] errors: {errors} (len={len(errors)})")
                    ax = sns.barplot(x=prompt_types, y=latencies, color='b')
                    # Add error bars if appropriate
                    if len(errors) == len(latencies) and len(latencies) > 1:
                        import numpy as np
                        x_coords = np.arange(len(prompt_types))
                        plt.errorbar(x=x_coords, y=latencies, yerr=errors, fmt='none', ecolor='black', capsize=5)
                    elif len(latencies) == 1:
                        print(f"Warning: Only one bar in First Token Latency chart, skipping error bars.")
                    else:
                        print(f"Warning: Skipping error bars for First Token Latency chart (data shape mismatch)")
                    ax.set_ylabel("Latency (seconds)")
                    ax.set_xlabel("Prompt Type")
                    plt.title("First Token Latency by Prompt Type")
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.tight_layout()
                    
                    # Save the chart
                    first_token_file = os.path.join(output_dir, f"first_token_latency_{i}.png")
                    plt.savefig(first_token_file)
                    plt.close()
                    chart_files.append(("First Token Latency", first_token_file))
            
            # Generation throughput chart
            if "generation" in data:
                plt.figure(figsize=(10, 6))
                prompt_types = []
                throughputs = []
                
                for prompt_type, gen_data in data["generation"].items():
                    if "tokens_per_second" in gen_data:
                        prompt_types.append(prompt_type)
                        throughputs.append(gen_data["tokens_per_second"])
                
                if prompt_types:
                    ax = sns.barplot(x=prompt_types, y=throughputs)
                    ax.set_ylabel("Tokens per Second")
                    ax.set_xlabel("Prompt Type")
                    plt.title("Generation Throughput by Prompt Type")
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    # Save the chart
                    throughput_file = os.path.join(output_dir, f"generation_throughput_{i}.png")
                    plt.savefig(throughput_file)
                    plt.close()
                    chart_files.append(("Generation Throughput", throughput_file))
                    
        except Exception as e:
            print(f"Error creating charts for benchmark {i}: {e}")
    
    return chart_files

def get_model_info(model_path):
    """Get information about the model"""
    info = {
        "path": model_path,
        "exists": os.path.exists(model_path),
        "type": "ONNX Int4"
    }
    
    if info["exists"]:
        # Check if it's a directory or file
        if os.path.isdir(model_path):
            # Count files and get total size
            info["file_count"] = 0
            info["total_size_bytes"] = 0
            
            for root, _, files in os.walk(model_path):
                for file in files:
                    info["file_count"] += 1
                    file_path = os.path.join(root, file)
                    info["total_size_bytes"] += os.path.getsize(file_path)
            
            # Look for specific files
            config_file = os.path.join(model_path, "config.json")
            if os.path.exists(config_file):
                info["has_config"] = True
                try:
                    # First try UTF-8
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                        info["config"] = config
                except UnicodeDecodeError:
                    try:
                        # Fall back to latin-1 which is more permissive
                        with open(config_file, 'r', encoding='latin-1') as f:
                            config = json.load(f)
                            info["config"] = config
                    except Exception as e:
                        info["config_error"] = f"Could not parse config.json: {e}"
                except Exception as e:
                    info["config_error"] = f"Could not parse config.json: {e}"
            
            # Convert size to MB
            info["total_size_mb"] = info["total_size_bytes"] / (1024 * 1024)
        else:
            # It's a single file
            info["file_count"] = 1
            info["total_size_bytes"] = os.path.getsize(model_path)
            info["total_size_mb"] = info["total_size_bytes"] / (1024 * 1024)
            
            # Try to determine the type
            if model_path.endswith(".onnx"):
                info["filetype"] = "ONNX"
            elif model_path.endswith(".bin"):
                info["filetype"] = "Binary"
            else:
                info["filetype"] = "Unknown"
    
    return info

def generate_html_report(log_info, benchmark_data, model_info, chart_files, output_file):
    """Generate HTML report from all collected data"""
    # Get current date and time
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create HTML content
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Quantized Model Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; line-height: 1.6; color: #333; }}
        h1, h2, h3 {{ color: #0066cc; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .section {{ margin-bottom: 30px; border: 1px solid #ddd; padding: 20px; border-radius: 5px; }}
        .summary {{ background-color: #f9f9f9; }}
        table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #0066cc; color: white; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .chart {{ margin: 20px 0; text-align: center; }}
        .error {{ color: #cc0000; }}
        .success {{ color: #007700; }}
        .prompt-box {{ border-left: 4px solid #0066cc; padding-left: 15px; margin: 15px 0; }}
        .response-box {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin: 10px 0; }}
        .metrics {{ display: flex; justify-content: space-around; flex-wrap: wrap; }}
        .metric-card {{ background-color: #f0f7ff; border-radius: 5px; padding: 15px; margin: 10px; min-width: 200px; text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #0066cc; }}
        .footer {{ text-align: center; margin-top: 30px; color: #777; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>DeepSeek-R1-Distill-Qwen-1.5B Quantized Model Evaluation</h1>
        <p>Report generated on {now}</p>
        
        <div class="section summary">
            <h2>Model Summary</h2>
            <div class="metrics">
                <div class="metric-card">
                    <h3>Model Size</h3>
                    <div class="metric-value">{model_info.get("total_size_mb", 0):.2f} MB</div>
                    <p>Quantized ONNX Model</p>
                </div>
                <div class="metric-card">
                    <h3>Precision</h3>
                    <div class="metric-value">Int4</div>
                    <p>Quantization Level</p>
                </div>
                <div class="metric-card">
                    <h3>Files</h3>
                    <div class="metric-value">{model_info.get("file_count", 0)}</div>
                    <p>Total Model Files</p>
                </div>
            </div>
"""
    
    # Add performance metrics if available
    if benchmark_data:
        # Find average tokens per second across all benchmarks
        tps_values = []
        for data in benchmark_data:
            if "generation" in data:
                for prompt_type, gen_data in data["generation"].items():
                    if "tokens_per_second" in gen_data:
                        tps_values.append(gen_data["tokens_per_second"])
        
        avg_tps = sum(tps_values) / len(tps_values) if tps_values else 0
        
        # Find average latency
        latency_values = []
        for data in benchmark_data:
            if "prompts" in data:
                for prompt_type, prompt_data in data["prompts"].items():
                    if "first_token_time" in prompt_data:
                        latency_values.append(prompt_data["first_token_time"]["mean"])
        
        avg_latency = sum(latency_values) / len(latency_values) if latency_values else 0
        
        html += f"""
            <div class="metrics">
                <div class="metric-card">
                    <h3>Generation Speed</h3>
                    <div class="metric-value">{avg_tps:.2f}</div>
                    <p>Tokens per Second</p>
                </div>
                <div class="metric-card">
                    <h3>First Token Latency</h3>
                    <div class="metric-value">{avg_latency:.4f}s</div>
                    <p>Average Response Time</p>
                </div>
            </div>
"""
    
    html += """
        </div>
        
        <div class="section">
            <h2>Diagnostic Results</h2>
"""
    
    # Add success and error information
    if log_info["successes"]:
        html += """
            <h3>Successful Operations</h3>
            <ul>
"""
        for success in log_info["successes"]:
            html += f"                <li class='success'>{success}</li>\n"
        html += "            </ul>\n"
    
    if log_info["errors"]:
        html += """
            <h3>Errors and Warnings</h3>
            <ul>
"""
        for error in log_info["errors"]:
            html += f"                <li class='error'>{error}</li>\n"
        html += "            </ul>\n"
    
    html += """
        </div>
        
        <div class="section">
            <h2>Performance Analysis</h2>
"""
    
    # Add charts
    for title, chart_file in chart_files:
        # Get relative path
        rel_path = os.path.relpath(chart_file, os.path.dirname(output_file))
        html += f"""
            <div class="chart">
                <h3>{title}</h3>
                <img src="{rel_path}" alt="{title}" style="max-width: 100%;">
            </div>
"""
    
    # Add benchmark tables if available
    if benchmark_data:
        for i, data in enumerate(benchmark_data):
            html += f"""
            <h3>Benchmark Run {i+1}</h3>
"""
            
            # First token latency table
            if "prompts" in data:
                html += """
            <h4>First Token Latency</h4>
            <table>
                <tr>
                    <th>Prompt Type</th>
                    <th>Input Tokens</th>
                    <th>Avg Time (s)</th>
                    <th>Min Time (s)</th>
                    <th>Max Time (s)</th>
                </tr>
"""
                for prompt_type, prompt_data in data["prompts"].items():
                    if "first_token_time" in prompt_data:
                        ft = prompt_data["first_token_time"]
                        html += f"""
                <tr>
                    <td>{prompt_type}</td>
                    <td>{prompt_data["tokens"]}</td>
                    <td>{ft["mean"]:.4f} ± {ft["std"]:.4f}</td>
                    <td>{ft["min"]:.4f}</td>
                    <td>{ft["max"]:.4f}</td>
                </tr>"""
                html += """
            </table>
"""
            
            # Generation throughput table
            if "generation" in data:
                html += """
            <h4>Text Generation Performance</h4>
            <table>
                <tr>
                    <th>Prompt Type</th>
                    <th>Tokens Generated</th>
                    <th>Time (s)</th>
                    <th>Tokens/Second</th>
                </tr>
"""
                for prompt_type, gen_data in data["generation"].items():
                    if "error" in gen_data:
                        html += f"""
                <tr>
                    <td>{prompt_type}</td>
                    <td colspan="3" class="error">Error: {gen_data["error"]}</td>
                </tr>"""
                    else:
                        html += f"""
                <tr>
                    <td>{prompt_type}</td>
                    <td>{gen_data["mean_tokens"]:.1f}</td>
                    <td>{gen_data["mean_time"]:.4f}</td>
                    <td>{gen_data["tokens_per_second"]:.2f}</td>
                </tr>"""
                html += """
            </table>
"""
      # Add sample generation results
    if log_info["prompts"]:
        html += """
        <div class="section">
            <h2>Sample Generation Results</h2>
"""
        
        for prompt, prompt_data in log_info["prompts"].items():
            if "generated_text" in prompt_data:
                # Pre-process the text replacement outside the f-string
                response_text = prompt_data["generated_text"].replace("\n", "<br>")
                html += f"""
            <div>
                <div class="prompt-box">
                    <strong>Prompt:</strong> {prompt}
                </div>
                <div class="response-box">
                    {response_text}
                </div>
                <p>
                    <strong>Generation Time:</strong> {prompt_data.get("time_seconds", "N/A")} seconds
                    <strong>Speed:</strong> {prompt_data.get("tokens_per_second", "N/A")} tokens/sec
                </p>
            </div>
"""
        
        html += """
        </div>
"""
    
    # Model configuration details
    if model_info.get("has_config", False) and "config" in model_info:
        html += """
        <div class="section">
            <h2>Model Configuration</h2>
            <table>
                <tr>
                    <th>Parameter</th>
                    <th>Value</th>
                </tr>
"""
        
        config = model_info["config"]
        for key, value in config.items():
            # Skip complex nested objects
            if not isinstance(value, (dict, list)):
                html += f"""
                <tr>
                    <td>{key}</td>
                    <td>{value}</td>
                </tr>"""
        
        html += """
            </table>
        </div>
"""
    
    # Finish the HTML
    html += """
        <div class="footer">
            <p>Generated with the DeepSeek-R1-Distill-Qwen-1.5B Model Evaluation Framework</p>
        </div>
    </div>
</body>
</html>
"""
    
    # Write the HTML to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    return output_file

def main():
    args = parse_args()
    
    # Find log and benchmark files
    log_files = glob.glob(os.path.join(args.results_dir, "*.log"))
    benchmark_files = glob.glob(os.path.join(args.results_dir, "*.json"))
    
    if not log_files and not benchmark_files:
        print(f"No log or benchmark files found in {args.results_dir}")
        return
    
    print(f"Found {len(log_files)} log files and {len(benchmark_files)} benchmark files")
    
    # Extract info from logs
    log_info = extract_info_from_logs(log_files)
    
    # Parse benchmark data
    benchmark_data = parse_benchmark_files(benchmark_files)
    
    # Get model info
    model_info = get_model_info(args.model_path)
    
    # Create output directory for charts
    charts_dir = os.path.join(os.path.dirname(args.output_file), "charts")
    
    # Create performance charts
    chart_files = create_performance_charts(benchmark_data, charts_dir)
    
    # Generate HTML report
    report_file = generate_html_report(log_info, benchmark_data, model_info, chart_files, args.output_file)
    
    print(f"Report generated: {report_file}")

if __name__ == "__main__":
    main()
