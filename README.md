# QuantumEigenSolver

This is the top-level umbrella repository for the Quantum EigenSolver project. It integrates the Python simulation framework and the high-performance C++ implementation.

**Note:** This repository uses Git submodules. Please clone recursively:
```bash
git clone --recursive https://github.com/makskliczkowski/QuantumEigenSolver.git
```

## 🚀 Start Here: Python Framework

The **[pyqusolver](pyqusolver/)** submodule is the **primary** component of this project. It provides a comprehensive Python framework for simulating quantum systems.

### Python Quickstart

```bash
# 1. Enter the submodule
cd pyqusolver

# 2. Install
pip install -e "Python/[all,dev]"

# 3. Run a test
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

## 📚 Documentation

For detailed information on the repository structure and workflows, please refer to the documentation in `docs/`:

*   [**Repository Overview**](docs/REPO_OVERVIEW.md): Map of top-level directories and submodules.
*   [**Workflows & Subrepos**](docs/SUBREPOS_AND_WORKFLOWS.md): Detailed guides for:
    *   **Python-first workflow** (Recommended)
    *   **C++ workflow** (Secondary)
    *   **HPC workflow** (Slurm scripts)
