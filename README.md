# QuantumEigenSolver

This is the top-level umbrella repository for the Quantum EigenSolver project. It integrates the Python simulation framework and the high-performance C++ implementation.

**Note:** This repository uses Git submodules. Please clone recursively:
```bash
git clone --recursive https://github.com/makskliczkowski/QuantumEigenSolver.git
```

## Repository Map

- `pyqusolver/`: main Python framework.
- `juqusolver/`: Julia port package (`QuantumEigenSolver`) with source, tests, examples, and docs.
- `cpp/`: C++ implementation and build system.
- `docs/`: top-level repository and workflow docs.
- `porting/`: Julia port contracts and gate reports (`specs/`, `reports/port/`, `reports/review/`).

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

For detailed repository structure and workflows:

- [Repository Overview](docs/REPO_OVERVIEW.md)
- [Workflows and Subrepos](docs/SUBREPOS_AND_WORKFLOWS.md)
- [HPC and Slurm notes](docs/HPC_SLURM.md)

For Julia package organization and module docs:

- [juqusolver README](juqusolver/README.md)
- [Julia docs index](juqusolver/docs/src/index.md)
- [Honeycomb Kitaev ED workflow](juqusolver/docs/src/honeycomb_kitaev_diagonalization.md)

Julia module pages:

- [common](juqusolver/docs/src/common.md)
- [general_maths](juqusolver/docs/src/general_maths.md)
- [general_lattices](juqusolver/docs/src/general_lattices.md)
- [general_physics](juqusolver/docs/src/general_physics.md)
- [general_algebra](juqusolver/docs/src/general_algebra.md)
- [general_python_init](juqusolver/docs/src/general_python_init.md)
- [qes_core](juqusolver/docs/src/qes_core.md)
- [qes_algebra](juqusolver/docs/src/qes_algebra.md)
- [qes_operator](juqusolver/docs/src/qes_operator.md)
- [qes_models](juqusolver/docs/src/qes_models.md)
- [qes_solvers](juqusolver/docs/src/qes_solvers.md)

## Changelog (Julia Port)

- 2026-03-05:
  - Consolidated Phase 1 work in the existing `juqusolver/` package tree.
  - Refreshed Phase 1 module specs and reports for `juqusolver/src/*`.
  - Implemented lightweight SVG plotting backend in `Common.Plotter` and lattice-to-plotter integration.
  - Added top-level capability/listing helpers to mimic `general_python/__init__.py`.
  - Completed `general_physics` expansion with `eigenlevels`, `spectral`, `response`, and `sp` modules in a single Julia-first API, plus tests/docs/example updates.
  - Completed `general_algebra` umbrella with solver/preconditioner/random/ode/eigen/utilities modules, plus tests/docs/example updates.
  - Completed `qes_core` parity for global backend manager, deterministic key helpers, module registry, and session lifecycle (`run` do-block support).
  - Added `qes_algebra` compatibility layer for top-level `QES.Algebra` API (`HilbertSpace/Hamiltonian` configs, registry semantics, backend linalg helper reexports, and module aliases).
  - Upgraded `qes_algebra` Hilbert/symmetry internals with deterministic compact sector mappings and reduced/full basis transforms for one-generator and multi-generator sectors (`Translation_x`, `Reflection`, `Inversion`, `Parity`).
  - Expanded `qes_models` parity in `PhysicsModels`: Python-style alias/class-name dispatch, lattice-aware spin/fermion bond builders, richer Heisenberg-Kitaev (`K/J/Gamma/dlt`) support, `QSM` and `UltrametricModel` parameter parity, and `PLRB`/`RPM` `many_body` behavior.
  - Added `qes_solvers` module in `juqusolver/src/Solvers`: `QES/Solver` parity with `SolverState`, Monte Carlo samplers (`VMCSampler`), initialization/update-rule factories, diagnostics, and `ParallelTempering`.
  - Added `qes_operator` module in `juqusolver/src/Algebra/Operator`: spin-1/2, spin-1, fermion, hardcore, anyon factories, lazy operator loader, catalog registry, and sparse-first `operator_matrix` builder.
  - Removed backend-split naming from exported algebra APIs and consolidated spectral naming to one Julia backend path.
  - Optimized `qes_solvers` VMC inner loop with reusable proposal/swap buffers and in-place built-in update kernels (`LOCAL`, `MULTI_FLIP`), reducing benchmark allocations from `32270` to `14` for the reference sampling case.
  - Added exact-diagonalization honeycomb Kitaev example and regression checks for sorted real spectrum and trace consistency.
  - Polished memory behavior in Monte Carlo update kernels and Aubry-Andre fallback lattice hopping path.
