# CI/CD and Validation

This document describes the Continuous Integration (CI) process and local validation tools for the QuantumEigenSolver umbrella repository.

## Overview

The CI system is designed to be minimal and robust, verifying:
1.  **Submodule Integrity**: Ensures submodules are correctly initialized.
2.  **Basic Smoke Tests**: Verifies that the Python package (`QES`) can be imported.
3.  **C++ Build**: Verifies that the C++ library compiles with required dependencies.

## CI Workflows

The CI pipeline is defined in `.github/workflows/ci.yml` and consists of two main jobs:

### 1. Sanity Check (Required)
*   **Runs on**: `ubuntu-latest`
*   **Description**: A lightweight check that verifies submodules are present and runs the Python smoke test (`test_import.py`).
*   **Requirements**: Python 3.10, `numpy`, `scipy`, `numba`.

### 2. Build C++ (Non-Blocking)
*   **Runs on**: `ubuntu-latest`
*   **Description**: Configures and compiles the C++ library using CMake.
*   **Status**: **Non-blocking**. This job is allowed to fail without blocking the PR merge, primarily to handle cases where heavy dependencies (like Intel MKL) might not be available or installable on the runner.
*   **Dependencies**:
    *   `libhdf5-dev` (HDF5) - *Required*
    *   `libarmadillo-dev` (Armadillo) - *Required*
    *   `libmkl-dev` (Intel MKL) - *Optional/Heavy*. The CI attempts to install this. If missing, the build might fail or fall back, but the job is configured to not block the pipeline.
    *   `cmake`
    *   `g++` (C++20 support)

## Local Validation

You can run the same validation steps locally using the provided script `scripts/validate.sh`.

### Prerequisites
*   Bash
*   Python 3
*   (Optional for build) CMake, G++, HDF5, Armadillo, MKL

### Usage

**1. Sanity Check (Submodules + Python Smoke)**
```bash
./scripts/validate.sh
```
This will:
* Check if `cpp/library/source` and `pyqusolver` submodules are initialized.
* Run `python3 test_import.py` to verify `QES` import.

**2. Full Build Verification**
```bash
./scripts/validate.sh --build
```
This will:
* Perform the sanity check.
* Create a `build/` directory (if not present).
* Configure the project using CMake.
* Compile the C++ code.

**Note on C++ Build:**
The C++ build requires HDF5, Armadillo, and preferably Intel MKL. If these are missing on your local machine, the `--build` step will fail.
