"""
Environment check for DeepSeek-R1-Distill-Qwen-1.5B quantization
This script verifies that all required packages are installed and configured correctly
"""
import importlib
import sys
import platform
import os

def check_package(package_name, min_version=None):
    """Check if a package is installed and meets minimum version requirements"""
    try:
        module = importlib.import_module(package_name)
        if hasattr(module, '__version__'):
            version = module.__version__
        elif hasattr(module, 'version'):
            version = module.version
        else:
            # Try to get version from pkg_resources
            try:
                import pkg_resources
                version = pkg_resources.get_distribution(package_name).version
            except:
                version = "Unknown"
                
        if min_version and version != "Unknown":
            from packaging import version as version_parser
            if version_parser.parse(version) < version_parser.parse(min_version):
                return False, f"{package_name} {version} installed, but {min_version}+ recommended"
                
        return True, f"{package_name} {version} installed"
    except ImportError:
        return False, f"{package_name} not installed"

def check_environment():
    """Perform full environment check for quantization requirements"""
    print(f"Python: {platform.python_version()} ({platform.architecture()[0]})")
    print(f"OS: {platform.system()} {platform.release()}")
    
    # Essential packages
    essential_packages = [
        ("torch", "2.0.0"),
        ("transformers", "4.30.0"),
        ("optimum", "1.10.0"),
        ("onnx", "1.14.0"),
        ("onnxruntime", "1.15.0"),
    ]
    
    # CUDA specific packages
    cuda_packages = [
        ("bitsandbytes", "0.41.0"),
        ("accelerate", "0.20.0"),
    ]
    
    # Check essential packages
    essential_missing = []
    for package, min_version in essential_packages:
        installed, message = check_package(package, min_version)
        print(f"✓ {message}" if installed else f"✗ {message}")
        if not installed:
            essential_missing.append(package)
            
    # Check CUDA support
    cuda_available = False
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"✓ CUDA available: {torch.version.cuda}")
            print(f"✓ GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
            
            # Check CUDA-specific packages
            cuda_missing = []
            for package, min_version in cuda_packages:
                installed, message = check_package(package, min_version)
                print(f"✓ {message}" if installed else f"✗ {message}")
                if not installed:
                    cuda_missing.append(package)
        else:
            print(f"✗ CUDA not available, will use CPU only")
    except:
        print(f"✗ Failed to check CUDA availability")
    
    # Summary
    if essential_missing:
        print("\n⚠️ The following essential packages are missing or outdated:")
        for package in essential_missing:
            print(f"  - {package}")
        print("\nInstall missing packages with:")
        print(f"pip install " + " ".join(essential_missing))
        
    if cuda_available and cuda_missing:
        print("\n⚠️ The following CUDA-specific packages are missing or outdated:")
        for package in cuda_missing:
            print(f"  - {package}")
        print("\nInstall missing packages with:")
        print(f"pip install " + " ".join(cuda_missing))
    
    # Overall readiness
    if not essential_missing:
        print("\n✅ Environment ready for basic quantization!")
        if cuda_available and not cuda_missing:
            print("✅ Environment ready for GPU-accelerated quantization!")
        elif cuda_available:
            print("⚠️ Environment partially ready for GPU-accelerated quantization.")
            print("   Some GPU-specific packages need to be installed.")
        else:
            print("ℹ️ Using CPU-only quantization (slower, but works without GPU)")
    else:
        print("\n❌ Environment not ready for quantization.")
        print("   Please install missing essential packages.")
    
    return not essential_missing
    
if __name__ == "__main__":
    check_environment()
