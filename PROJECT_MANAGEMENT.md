# DeepSeek Quantization Project Management

This document outlines the organization and management practices for the DeepSeek-R1-Distill-Qwen-1.5B quantization framework project.

## Directory Structure

The project follows a structured organization:

```
qunatization/
├── README.MD                 # Main project documentation
├── TESTING_README.md         # Testing framework documentation
├── CLEANUP_README.md         # Cleanup script documentation
├── .gitignore                # Git ignore patterns
├── archive/                  # Archived development files
│   ├── development/          # Development and experimental scripts
│   ├── notebooks/            # Notebook copies
│   ├── notebook_cells/       # Individual code cells
│   └── setup/                # Environment setup scripts
├── inference_results/        # Test outputs (gitignored)
├── quantized_model/          # Quantized model files (gitignored)
├── onnx_exported/            # ONNX model exports (gitignored)
└── [core Python scripts]     # Main functionality
```

## File Management Guidelines

1. **Core Files**: 
   - Keep essential scripts in the main directory
   - Maintain clean, well-documented code
   - Add proper docstrings and type hints

2. **Development Files**:
   - Move experimental scripts to `archive/development/`
   - Keep notebook cell extracts in `archive/notebook_cells/`
   - Store environment setup scripts in `archive/setup/`

3. **Generated Files**:
   - Keep model files in designated directories
   - Store test results in `inference_results/`
   - Exclude these from source control using `.gitignore`

## Maintenance

Regular maintenance helps keep the project organized:

1. **Cleanup Script**:
   - Run `cleanup_workspace.ps1` to remove temporary files
   - Optionally clean model directories to save space
   - Clean Python cache files

2. **Documentation**:
   - Keep README files updated
   - Document new features and changes
   - Maintain clear usage instructions

3. **Version Control**:
   - Commit only essential code
   - Follow `.gitignore` patterns
   - Include meaningful commit messages

## Best Practices

1. **Code Organization**:
   - Modularize code into logical components
   - Maintain separation of concerns
   - Use consistent naming conventions

2. **Testing**:
   - Write tests for new functionality
   - Document test procedures
   - Use the testing framework for validation

3. **Performance**:
   - Profile and optimize critical paths
   - Benchmark after significant changes
   - Document performance characteristics

By following these practices, the project will remain maintainable, well-documented, and easy to navigate.
