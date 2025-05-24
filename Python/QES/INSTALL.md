# QES (Quantum Eigen Solver)

## Basic Installation
```bash
pip install QES
```

## Development Installation
```bash
# Clone the repository
git clone https://github.com/makskliczkowski/QuantumEigenSolver.git
cd QuantumEigenSolver/Python/QES

# Install in development mode with all dependencies
pip install -e ".[all]"
```

## Optional Dependencies

### JAX Support (GPU/CPU acceleration)
```bash
pip install "QES[jax]"
```

### Machine Learning Utilities
```bash
pip install "QES[ml]"
```

### HDF5 File Support
```bash
pip install "QES[hdf5]"
```

### Development Tools
```bash
pip install "QES[dev]"
```

### Documentation Tools
```bash
pip install "QES[docs]"
```

### All Optional Dependencies
```bash
pip install "QES[all]"
```

## Quick Start

```python
import QES

# Your quantum eigenvalue solving code here
```

## Requirements

- Python ≥ 3.8
- NumPy ≥ 1.20.0
- SciPy ≥ 1.7.0
- See setup.py for full dependency list

## License

This project is licensed under the Creative Commons Attribution 4.0 International License.

## Contributing

Please see the main repository for contributing guidelines.
