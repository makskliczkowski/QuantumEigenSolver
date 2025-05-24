# QES Package Structure and Setup

## Overview
This document describes the complete package structure and setup for the QES (Quantum Eigen Solver) Python package.

## Package Structure
```
QES/
├── QES/                          # Main package directory
│   ├── __init__.py              # Package initialization with version info
│   ├── Algebra/                 # Algebraic operations and models
│   ├── NQS/                     # Neural Quantum States implementation
│   ├── Solver/                  # Solver framework
│   └── general_python/          # General utilities
├── tests/                       # Test files (if any at package level)
├── requirements/                # Requirements files for different scenarios
│   ├── requirements.txt         # Core dependencies
│   ├── requirements-jax.txt     # JAX ecosystem
│   ├── requirements-ml.txt      # ML utilities
│   ├── requirements-hdf5.txt    # HDF5 support
│   ├── requirements-dev.txt     # Development tools
│   └── requirements-docs.txt    # Documentation tools
├── .dev/                        # Development configuration guides
├── setup.py                     # Setup script (legacy, but still useful)
├── pyproject.toml              # Modern Python packaging configuration
├── MANIFEST.in                 # Files to include in distribution
├── CHANGELOG.md                # Version history
├── INSTALL.md                  # Installation instructions
├── Makefile                    # Development automation
├── tox.ini                     # Testing across Python versions
├── .flake8                     # Linting configuration
└── .pre-commit-config.yaml     # Git hooks configuration
```

## Installation Methods

### 1. Basic Installation
```bash
pip install QES
```

### 2. Development Installation
```bash
git clone https://github.com/makskliczkowski/QuantumEigenSolver.git
cd QuantumEigenSolver/Python/QES
pip install -e ".[dev]"
```

### 3. Specific Feature Sets
```bash
pip install "QES[jax]"     # JAX support
pip install "QES[ml]"      # ML utilities
pip install "QES[all]"     # Everything
```

## Development Workflow

### Setup
```bash
make dev-setup           # Sets up development environment
```

### Daily Development
```bash
make dev                 # Format, lint, and test
make check              # Full pre-commit checks
```

### Testing
```bash
make test               # Quick tests
make test-all           # Test across Python versions
```

### Documentation
```bash
make docs               # Build documentation
```

### Release
```bash
make build              # Build distribution packages
make upload             # Upload to PyPI (test)
```

## Key Features

### 1. Modern Python Packaging
- Uses `pyproject.toml` for configuration
- Setuptools with SCM for version management
- Proper dependency management with extras

### 2. Code Quality Tools
- Black for code formatting
- Flake8 for linting
- MyPy for type checking
- isort for import sorting
- Pre-commit hooks for automation

### 3. Testing Infrastructure
- Pytest for testing
- Tox for multi-Python testing
- Coverage reporting
- Test discovery for multiple test directories

### 4. Documentation
- Sphinx for documentation generation
- Read the Docs theme
- Jupyter notebook integration via nbsphinx

### 5. Development Automation
- Makefile for common tasks
- Pre-commit hooks
- Continuous integration ready

## Optional Dependencies Explanation

- **jax**: High-performance numerical computing, GPU/TPU support
- **ml**: Machine learning utilities (scikit-learn, scikit-image)
- **hdf5**: HDF5 file format support for large datasets
- **dev**: Development tools (testing, linting, formatting)
- **docs**: Documentation generation tools
- **all**: All optional dependencies combined

## Version Management
Version is automatically managed through:
1. Git tags for releases
2. `setuptools_scm` for automatic version detection
3. `_version.py` file generation

## Best Practices Implemented

1. **Semantic Versioning**: Following semver.org
2. **Keep a Changelog**: Structured changelog format
3. **PEP 517/518**: Modern build system
4. **PEP 621**: Project metadata in pyproject.toml
5. **Type Hints**: MyPy configuration for type safety
6. **Code Formatting**: Black and isort for consistency
7. **Security**: Bandit for security linting
8. **Documentation**: Comprehensive docs and examples

This setup provides a professional, maintainable, and scalable Python package structure suitable for scientific computing and research applications.
