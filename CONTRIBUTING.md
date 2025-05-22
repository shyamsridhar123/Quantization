# Contributing to the DeepSeek Quantization Framework

Thank you for considering contributing to the DeepSeek-R1-Distill-Qwen-1.5B quantization framework! This document provides guidelines to streamline the contribution process.

## Table of Contents

1. [Code Organization](#code-organization)
2. [Development Workflow](#development-workflow)
3. [Code Style](#code-style)
4. [Documentation](#documentation)
5. [Pull Request Process](#pull-request-process)

## Code Organization

The project follows a specific organization structure to maintain clarity:

* **Core Testing Scripts**: Keep these in the main directory for easy access
* **Development Files**: Store in `archive/development/` 
* **Notebook Experiments**: Store notebook backups in `archive/notebooks/`
* **Setup Utilities**: Store in `archive/setup/`

## Development Workflow

1. **Set up your environment**:
   ```powershell
   # Create a virtual environment
   python -m venv .venv
   .\.venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Feature development**:
   * Create experimental scripts in a development branch
   * Move to `archive/development/` when working on a new version
   * Keep the main directory clean with only essential files

3. **Testing**:
   * Use the testing framework with `run_master_test.ps1`
   * Create new test cases for new functionality
   * Document any special testing requirements

## Code Style

* Follow PEP 8 guidelines for Python code
* Use meaningful variable and function names
* Add comments for complex logic
* Include type hints and docstrings for all functions

Example:
```python
def process_model_output(logits: torch.Tensor, attention_mask: torch.Tensor = None) -> dict:
    """
    Process the raw model output to generate text.
    
    Args:
        logits: Raw logits from the model
        attention_mask: Optional attention mask
        
    Returns:
        Dictionary containing processed outputs
    """
    # Implementation
```

## Documentation

* Update the relevant README files when adding new features
* Document any changes to the model quantization process
* Keep troubleshooting sections updated with new issues and solutions
* Maintain clear installation and usage instructions

## Pull Request Process

1. Fork the repository and create a branch for your feature
2. Implement the feature, following the code style guidelines
3. Run tests to ensure your changes work correctly
4. Document your changes in the relevant README files
5. Submit a pull request with a clear description of the changes
6. Address any feedback in code review

Thank you for your contributions!
