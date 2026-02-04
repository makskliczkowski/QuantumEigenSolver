# Repository Overview

This repository acts as a top-level umbrella for the QuantumEigenSolver project, containing both the legacy C++ implementation and the modern Python framework.

## Top-Level Directories

*   **`pyqusolver/`**: The primary Python framework (Git submodule). This is the recommended starting point for new users.
*   **`cpp/`**: The C++ implementation of the solver.
    *   `library/`: Core C++ library source code.
    *   `library/source/`: Git submodule for `general_cpp` utilities.
*   **`docs/`**: Documentation for the repository structure and workflows.
*   **`slurm/`**: HPC job submission scripts and utilities.
*   **`test_import.py`**: A root-level script to test imports (requires specific path configuration).

## Subrepositories and Dependencies

The project is modularized using Git submodules:

1.  **`pyqusolver`** (`pyqusolver/`):
    *   The main Python package for Quantum EigenSolver.
    *   **Dependency**: It internally includes `general_python` as a submodule (located at `pyqusolver/Python/QES/general_python`), which provides core scientific and numerical utilities.

2.  **`general_cpp`** (`cpp/library/source`):
    *   A C++ utility library providing linear algebra wrappers, random number generators, and other tools used by the C++ solver.
