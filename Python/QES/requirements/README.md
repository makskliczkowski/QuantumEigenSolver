# QES Requirements Files

This directory contains various requirements files for different installation scenarios:

## Core Requirements
- `requirements.txt` - Core dependencies for basic functionality

## Optional Requirements
- `requirements-jax.txt` - JAX ecosystem dependencies for GPU/TPU acceleration
- `requirements-ml.txt` - Machine learning utilities
- `requirements-hdf5.txt` - HDF5 file format support
- `requirements-dev.txt` - Development tools and testing
- `requirements-docs.txt` - Documentation generation tools

## Installation Examples

### Basic installation:
```bash
pip install -r requirements.txt
```

### Development environment:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Full installation with all features:
```bash
pip install -r requirements.txt
pip install -r requirements-jax.txt
pip install -r requirements-ml.txt
pip install -r requirements-hdf5.txt
```
