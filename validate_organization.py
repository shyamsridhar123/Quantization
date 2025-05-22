#!/usr/bin/env python
"""
Project Organization Validation Script

This script checks that the DeepSeek quantization project follows the organizational
structure and best practices defined in the project guidelines.
"""

import os
import sys
from pathlib import Path
import datetime

# Define expected directory structure
EXPECTED_DIRS = [
    "archive",
    "archive/development",
    "archive/notebooks",
    "archive/notebook_cells",
    "archive/setup",
    "notebooks"
]

# Define core files that should be in the main directory
CORE_FILES = [
    "benchmark_model.py",
    "compare_models.py",
    "convert_to_fp16.py",
    "deepseek_onnx_helper.py",
    "direct_onnx_inference.py",
    "enhanced_onnx_inference.py",
    "environment_check.py",
    "generate_report.py",
    "interactive_test.py",
    "qwen2_onnx_export.py",
    "run_inference_tests.ps1",
    "run_master_test.ps1",
    "run_quantization.py",
    "sample_prompts.txt",
    "test_quantized_model.py",
    "troubleshoot_onnx_model.py"
]

# Define documentation files
DOC_FILES = [
    "README.MD",
    "TESTING_README.md",
    "CLEANUP_README.md",
    "CONTRIBUTING.md",
    "PROJECT_MANAGEMENT.md"
]

def check_directory_structure(base_dir):
    """Check if the expected directory structure exists."""
    missing_dirs = []
    for dir_path in EXPECTED_DIRS:
        full_path = base_dir / dir_path
        if not full_path.exists():
            missing_dirs.append(dir_path)
    
    return missing_dirs

def check_core_files(base_dir):
    """Check if core files exist in the main directory."""
    missing_files = []
    for file_name in CORE_FILES:
        full_path = base_dir / file_name
        if not full_path.exists():
            missing_files.append(file_name)
    
    return missing_files

def check_documentation(base_dir):
    """Check if documentation files exist."""
    missing_docs = []
    for doc_file in DOC_FILES:
        full_path = base_dir / doc_file
        if not full_path.exists():
            missing_docs.append(doc_file)
    
    return missing_docs

def main():
    """Main validation function."""
    base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    print(f"Validating project organization in: {base_dir}")
    print("-" * 50)
    
    # Check directory structure
    missing_dirs = check_directory_structure(base_dir)
    if missing_dirs:
        print("Warning: Missing directories:")
        for dir_path in missing_dirs:
            print(f"  - {dir_path}")
    else:
        print("PASS: Directory structure is valid")
    
    # Check core files
    missing_files = check_core_files(base_dir)
    if missing_files:
        print("\nWarning: Missing core files:")
        for file_name in missing_files:
            print(f"  - {file_name}")
    else:
        print("PASS: All core files are present")
    
    # Check documentation
    missing_docs = check_documentation(base_dir)
    if missing_docs:
        print("\nWarning: Missing documentation:")
        for doc_file in missing_docs:
            print(f"  - {doc_file}")
    else:
        print("PASS: All documentation is present")
    
    # Print summary
    print("\n" + "-" * 50)
    if not (missing_dirs or missing_files or missing_docs):
        print("PASS: Project organization is valid!")
    else:
        print("Warning: Project organization needs attention")
          # Save validation report
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = base_dir / f"organization_validation_{timestamp}.txt"
    
    with open(report_path, "w") as f:
        f.write(f"Project Organization Validation Report - {datetime.datetime.now()}\n")
        f.write(f"Base directory: {base_dir}\n")
        f.write("-" * 50 + "\n\n")
        
        f.write("Directory Structure:\n")
        if missing_dirs:
            f.write("WARNING: Missing directories:\n")
            for dir_path in missing_dirs:
                f.write(f"  - {dir_path}\n")
        else:
            f.write("PASS: Directory structure is valid\n")
        
        f.write("\nCore Files:\n")
        if missing_files:
            f.write("WARNING: Missing core files:\n")
            for file_name in missing_files:
                f.write(f"  - {file_name}\n")
        else:
            f.write("PASS: All core files are present\n")
        
        f.write("\nDocumentation:\n")
        if missing_docs:
            f.write("WARNING: Missing documentation:\n")
            for doc_file in missing_docs:
                f.write(f"  - {doc_file}\n")
        else:
            f.write("PASS: All documentation is present\n")
        
        f.write("\n" + "-" * 50 + "\n")
        if not (missing_dirs or missing_files or missing_docs):
            f.write("PASS: Project organization is valid!")
        else:
            f.write("WARNING: Project organization needs attention")
    
    print(f"Report saved to: {report_path}")

if __name__ == "__main__":
    main()
