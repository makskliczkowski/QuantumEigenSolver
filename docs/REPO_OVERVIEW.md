# Repository Overview

This document provides a map of the `QuantumEigenSolver` repository, explaining the purpose of each top-level directory and subrepository.

## Top-Level Directories

### `cpp/`

Contains the C++ implementation of the quantum solver. This directory includes:

- **`library/`**: The core C++ library code.
- **`library/source/`**: A Git submodule pointing to `general_cpp`, which provides utility functions.
- **`library/inputs/`**: Standard location for input configuration files.

### `pyqusolver/`

This is a **Git submodule** containing the Python framework for simulating quantum systems. It remains the broadest feature surface and the reference implementation for NQS and DQMC workflows.

- It includes its own internal dependencies, such as `general_python`.

### `docs/`

Contains documentation for the project, including:

- `REPO_OVERVIEW.md` (this file): Structural map of the repository.
- `SUBREPOS_AND_WORKFLOWS.md`: Detailed guides on how to use the different components (Python, C++, HPC).
- `FEATURE_GAP_ANALYSIS.md`: Python-vs-Julia maintained feature comparison.
- `DEVELOPMENT_ROADMAP.md`: staged implementation plan for the remaining work.

### `slurm/`

Contains scripts for submitting jobs to HPC clusters using the Slurm workload manager.

- **`run_py.sh`**: Submits Python jobs.
- **`run_cpp.sh`**: Submits C++ jobs.
- **`utils/`**: Helper functions used by the submission scripts.

### `test_import.py`

A root-level utility script used to verify that the Python environment is correctly set up. It attempts to import key modules from `pyqusolver` and reports success or failure.

## Subrepositories

The repository uses Git submodules to manage dependencies and separate components.

### `pyqusolver`

- **Path:** `pyqusolver/`

- **Purpose:** The mature Python framework written around `QES`.
- **Status:** **Reference implementation**. It still contains the broadest feature surface.
- **Dependencies:** It depends on `general_python` (located within `pyqusolver/Python/QES/general_python`).

### `juqusolver`

- **Path:** `juqusolver/`

- **Purpose:** The Julia package `QuantumEigenSolver`.
- **Status:** **Computation-focused implementation**. It already covers the maintained algebra, lattice, and physics core and is intended to become independent.
- **Dependencies:** Local Julia package environment.

### `general_cpp`

- **Path:** `cpp/library/source`

- **Purpose:** A utility library providing common functionality for the C++ solver.
- **Status:** **Utility**. It is automatically used by the CMake build system in `cpp/`.

For detailed instructions on how to work with these components, see [Workflows](SUBREPOS_AND_WORKFLOWS.md).
