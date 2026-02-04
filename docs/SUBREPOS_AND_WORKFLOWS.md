# Subrepos and Workflows

## 1. Python-First Workflow (Primary)

The Python framework (`pyqusolver`) is the primary interface for the project.

### Installation
1.  Navigate to the submodule:
    ```bash
    cd pyqusolver
    ```
2.  Install the package in editable mode with dependencies:
    ```bash
    pip install -e "Python/[all,dev]"
    ```

### Workflow
*   **Development**: The code lives in `pyqusolver/Python/QES`.
*   **Utilities**: Core physics and math utilities are provided by `general_python` (submodule at `pyqusolver/Python/QES/general_python`).
*   **Running**:
    *   Run examples: `python3 pyqusolver/Python/examples/example_entanglement_entropy.py` (adjust path relative to your location)
    *   Run tests: `pytest pyqusolver/Python/test`

## 2. C++ Workflow (Secondary)

The C++ implementation offers high-performance exact diagonalization and solvers.

### Prerequisites
*   **Build System**: CMake
*   **Libraries**: Intel MKL, Armadillo, HDF5

### Build Instructions
```bash
cd cpp/library
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

### Inputs and Usage
*   **Input Files**: Example inputs are located in `cpp/library/INPUTS/`.
*   **Execution**: Run the compiled binary (e.g., `./QES`) passing the input file as an argument.

## 3. HPC Workflow

For High-Performance Computing (HPC) environments using SLURM.

### SLURM Scripts
*   Located in `slurm/`.
*   **`slurm/scripts/slurm_lib.sh`**: A library of bash functions for creating and submitting jobs.

### Usage
Source the library in your submission scripts to use its functions:
```bash
source slurm/scripts/slurm_lib.sh

# Example: Create a SLURM header and submit
create_slurm_header <cpu> <mem> <time> <output_dir> <job_name>
submit_slurm_job <script_file>
```
