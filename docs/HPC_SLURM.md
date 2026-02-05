# HPC Workflow & SLURM Usage

This document outlines the workflows for running C++ and Python jobs on HPC clusters using SLURM.

## Overview

The `slurm/` directory contains helper scripts and templates:
- `slurm/run_cpp.sh`: Helper for C++ jobs.
- `slurm/run_py.sh`: Helper for Python jobs.
- `slurm/template_job.sh`: Minimal template for manual submission.
- `slurm/utils/`: Shared utilities.

## 1. C++ Jobs

### 1.1 Prerequisites
- **C++20** compatible compiler (e.g., modern GCC or Intel OneAPI).
- **CMake** (3.23 or newer).
- **HDF5** libraries.
- **Armadillo** libraries.
- **MKL** (Math Kernel Library) or OpenBLAS.

### 1.2 Building the Executable

1.  **Load Modules**:
    Depending on your cluster, load the necessary modules.
    ```bash
    module load intel hdf5 cmake
    ```

2.  **Configure and Build**:
    We assume the C++ source is in `cpp/library`.

    *Note: On Ubuntu or if MKL is not automatically found, you may need to set `MKL_INCL_DIR` (e.g., `export MKL_INCL_DIR=/usr/include/mkl`).*

    ```bash
    mkdir -p build
    cd build
    cmake ../cpp/library/
    make -j$(nproc)
    ```
    This generates the `qsolver` executable (or similar name depending on CMake configuration) in `build/`.

### 1.3 Running Jobs (`run_cpp.sh`)

Use the `slurm/run_cpp.sh` wrapper to generate and submit SLURM scripts.

**Usage:**
```bash
./slurm/run_cpp.sh --exe=<PATH_TO_EXE> [OPTIONS] -- "<EXECUTABLE_ARGS>"
```

**Common Options:**
- `--time=HH:MM:SS`: Walltime limit.
- `--mem=X`: Memory in GB (e.g., `16`).
- `--cpu=N`: Number of CPU cores.
- `--partition=NAME`: SLURM partition.

**Example 1: Small ED Run**
```bash
./slurm/run_cpp.sh \
    --job-name=ed_small \
    --time=00:30:00 \
    --cpu=1 \
    --mem=4 \
    --exe=./build/qsolver \
    -- "mode=ed L=10 J=1.0"
```

**Example 2: Larger Run**
```bash
./slurm/run_cpp.sh \
    --job-name=heisenberg_large \
    --time=12:00:00 \
    --cpu=16 \
    --mem=64 \
    --exe=./build/qsolver \
    -- "mode=dmrg Lx=8 Ly=8 J=1.0 chi=1000"
```

The script will create a submission script in `./logs/` and submit it.

## 2. Python Jobs

The Python codebase is located in the `pyqusolver` submodule. Please refer to `pyqusolver/README.md` (or equivalent) for specific installation and usage details of the Python package.

### 2.1 Setup

You should prepare a virtual environment before running jobs, although the helper script can attempt to create one.

```bash
# Example setup
python3 -m venv venv
source venv/bin/activate
pip install -r pyqusolver/Python/requirements/requirements.txt
# Install the package in editable mode if needed
pip install -e pyqusolver/Python/
```

### 2.2 Running Jobs (`run_py.sh`)

Use `slurm/run_py.sh` to submit Python scripts.

**Usage:**
```bash
./slurm/run_py.sh --script=<PATH_TO_SCRIPT> [OPTIONS] -- "<SCRIPT_ARGS>"
```

**Options:**
- `--venv=PATH`: Path to virtual environment (default: `./venv`).
- `--requirements=PATH`: Path to requirements file (default: `requirements.txt`).
- `--time`, `--mem`, `--cpu`: Same as C++ jobs.

**Example:**
```bash
./slurm/run_py.sh \
    --job-name=py_bench \
    --time=02:00:00 \
    --mem=16 \
    --script=pyqusolver/examples/benchmark.py \
    --venv=./venv \
    -- "--system_size 20 --iterations 100"
```

## 3. Manual Submission

For complex workflows not covered by the helper scripts, use the template:

1.  Copy the template:
    ```bash
    cp slurm/template_job.sh my_experiment.sh
    ```
2.  Edit `my_experiment.sh`:
    -   Replace `<PARTITION>`, `<TIME>`, `<THREADS>`, `<MEMORY>` with your requirements.
    -   Add your modules and commands in the designated sections.
3.  Submit:
    ```bash
    sbatch my_experiment.sh
    ```

## 4. Troubleshooting

-   **Modules not found**: The helper scripts try to load common modules (`intel`, `hdf5`). If your cluster uses different names (e.g., `compiler/gcc`, `lib/hdf5`), you may need to edit the generated script or `slurm/utils/slurm_utils.sh`.
-   **Executables/Scripts not found**: Ensure paths provided to `--exe` or `--script` are correct. The helper scripts convert them to absolute paths, so relative paths work from your current working directory.
