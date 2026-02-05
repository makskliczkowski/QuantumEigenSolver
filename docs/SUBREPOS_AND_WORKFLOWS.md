# Subrepos and Workflows

This document outlines the workflows for the different components of the `QuantumEigenSolver` repository.

## 1. Python-First Workflow (Recommended)

The Python framework is the primary entry point for most users. It provides a flexible environment for setting up and running quantum simulations.

### Installation

1.  **Clone the repository recursively** to ensure all submodules are downloaded:
    ```bash
    git clone --recursive https://github.com/makskliczkowski/QuantumEigenSolver.git
    cd QuantumEigenSolver
    ```

2.  **Install the Python package:**
    Navigate to the `pyqusolver` directory and install in editable mode:
    ```bash
    cd pyqusolver
    pip install -e "Python/[all,dev]"
    ```
    *Note: This installation automatically handles the internal dependency on `general_python`.*

### Verification

Return to the repository root and run the verification script:
```bash
python3 test_import.py
```
This script checks that the `QES` package and its submodules (like `QES.general_python`) can be imported correctly.

You can also run the full test suite:
```bash
cd pyqusolver
pytest Python/test/
```

## 2. C++ Workflow (Secondary)

The C++ implementation offers high-performance solvers.

### Prerequisites

You will need the following dependencies installed on your system:
*   **CMake** (Build system)
*   **C++20 Compiler** (e.g., GCC 10+, Clang 10+)
*   **Intel MKL** (Math Kernel Library)
*   **Armadillo** (C++ Linear Algebra Library)
*   **HDF5** (Data format)

### Building

The build process uses CMake and is located in `cpp/library`.

1.  **Create a build directory:**
    ```bash
    cd cpp/library
    mkdir build
    cd build
    ```

2.  **Configure and Build:**
    ```bash
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)
    ```
    *Note: If CMake cannot find MKL, you may need to set the `MKL_INCL_DIR` environment variable.*

### Running

Input files are expected to be in `cpp/library/inputs`. Ensure your configuration files are placed there before running the executable produced in the `build/` directory.

## 3. HPC Workflow

The `slurm/` directory contains scripts to simplify job submission on Slurm-based clusters.

### Python Jobs

Use `slurm/run_py.sh` to submit Python scripts.

```bash
./slurm/run_py.sh --script=path/to/script.py --time=02:00:00 --mem=8gb
```

*   **Assumptions:**
    *   It manages a virtual environment (default location: `./venv`).
    *   If a `requirements.txt` is present, it will install dependencies automatically.

### C++ Jobs

Use `slurm/run_cpp.sh` to submit compiled C++ executables.

```bash
./slurm/run_cpp.sh --exe=./cpp/library/build/qsolver --time=01:00:00 -- "arg1 arg2"
```

*   **Assumptions:**
    *   It assumes the executable is already built.
    *   It loads standard environment modules (`intel`, `hdf5`).
