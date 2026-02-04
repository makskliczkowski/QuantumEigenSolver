# HPC Workflow & SLURM Usage

This document describes how to build and run jobs for the **QuantumEigenSolver** project on HPC clusters using SLURM.

## Directory Structure

- `slurm/`: Contains SLURM submission scripts.
  - `run_cpp.sh`: Helper script to submit C++ jobs.
  - `run_py.sh`: Helper script to submit Python jobs.
  - `template_job.sh`: A template for manual SLURM script creation.
  - `utils/slurm_utils.sh`: Library of common functions.

## 1. C++ Jobs

### 1.1. Prerequisites

Ensure you have the following modules available (names may vary by cluster):
- `intel` (OneAPI compilers and MKL)
- `hdf5`
- `cmake` (version 3.23+)

### 1.2. Building the Executable

1.  **Load modules**:
    ```bash
    module load intel hdf5 cmake
    ```

2.  **Create a build directory**:
    ```bash
    mkdir -p build
    cd build
    ```

3.  **Run CMake and Make**:
    ```bash
    cmake ../cpp/library/
    make -j$(nproc)
    ```
    This should produce the `qsolver` executable in the `build` directory.

### 1.3. Running Jobs

You can use the helper script `slurm/run_cpp.sh` to submit jobs easily.

**Usage:**
```bash
./slurm/run_cpp.sh --time=<TIME> --exe=<PATH_TO_EXE> -- "<EXECUTABLE_ARGS>"
```

**Example 1: Small ED Run**
```bash
./slurm/run_cpp.sh \
    --time=01:00:00 \
    --cpu=4 \
    --mem=8 \
    --exe=./build/qsolver \
    -- "mod=1 d=1 lx=10"
```

**Example 2: Larger Run**
```bash
./slurm/run_cpp.sh \
    --time=12:00:00 \
    --cpu=16 \
    --mem=64 \
    --exe=./build/qsolver \
    -- "mod=1 d=2 lx=6 ly=6 J1=1.0"
```

## 2. Python Jobs

### 2.1. Setup

For Python jobs (e.g., using `pyqusolver`), you typically need a virtual environment.

1.  **Prepare a `requirements.txt` file**.
2.  **Create/Update the environment** (handled automatically by `run_py.sh` if you provide requirements, or do it manually):
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

### 2.2. Running Jobs

Use `slurm/run_py.sh` to submit Python scripts.

**Usage:**
```bash
./slurm/run_py.sh --time=<TIME> --script=<SCRIPT_PATH> --venv=<VENV_PATH> -- "<SCRIPT_ARGS>"
```

**Example:**
```bash
./slurm/run_py.sh \
    --time=02:00:00 \
    --script=pyqusolver/examples/example_session.py \
    --venv=./venv \
    --requirements=./pyqusolver/Python/requirements/requirements.txt \
    -- "--option value"
```

## 3. Manual Submission

If you prefer to write your own SLURM scripts or need advanced configurations (e.g., job arrays, MPI), start with the template:

1.  Copy the template:
    ```bash
    cp slurm/template_job.sh my_job.sh
    ```
2.  Edit `my_job.sh` and replace the placeholders (`<PARTITION>`, `<TIME>`, `<COMMAND>`).
3.  Submit:
    ```bash
    sbatch my_job.sh
    ```

## 4. Troubleshooting

-   **"sbatch: command not found"**: You are likely on a login node that restricts submission, or SLURM is not installed. Ensure you are on a cluster submission node.
-   **"Module not found"**: Check available modules with `module avail`. You may need to edit `slurm/utils/slurm_utils.sh` or the generated scripts to match your cluster's module names.
-   **Permissions**: Ensure `run_cpp.sh` and `run_py.sh` are executable (`chmod +x ...`).
