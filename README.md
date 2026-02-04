# Quantum EigenSolver

This is the top-level repository for the Quantum EigenSolver project, encompassing both the modern Python framework and the C++ implementation.

## 🚀 Start Here: Python Framework

The **[pyqusolver](pyqusolver/)** submodule is the primary component of this project. It provides a comprehensive Python framework for simulating quantum systems.

### Python Quickstart

```bash
# Clone the repository with submodules
git clone --recursive https://github.com/makskliczkowski/QuantumEigenSolver.git
cd QuantumEigenSolver/pyqusolver

# Install
pip install -e "Python/[all,dev]"

# Run a test
pytest Python/test/
```

## 🛠️ C++ Solver

The C++ implementation is located in `cpp/`. It provides high-performance solvers using Intel MKL and Armadillo.

### C++ Quickstart

```bash
cd cpp/library
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

For more details on inputs and dependencies, see [Subrepos and Workflows](docs/SUBREPOS_AND_WORKFLOWS.md).

## 📚 Documentation

*   [**Repository Overview**](docs/REPO_OVERVIEW.md): Map of directories and submodules.
*   [**Workflows**](docs/SUBREPOS_AND_WORKFLOWS.md): Detailed guides for Python, C++, and HPC usage.
