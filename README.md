# QuantumEigenSolver

Top-level repository for the QuantumEigenSolver project, containing Python, Julia, and C++ implementations.

This repository uses git submodules. Clone recursively:
```bash
git clone --recursive https://github.com/makskliczkowski/QuantumEigenSolver.git
```

## Repository Structure

- `pyqusolver/` Python framework.
- `juqusolver/` Julia package (`QuantumEigenSolver`) with tests/examples/benchmarks/docs.
- `cpp/` C++ implementation.
- `docs/` top-level repository and workflow notes.
- `porting/` Julia port specs and review reports.

## Quick Start

### Python
```bash
cd pyqusolver
pip install -e "Python/[all,dev]"
pytest Python/test/
```

### Julia
```bash
cd juqusolver
~/.juliaup/bin/julia --project -e 'include("test/runtests_all.jl")'
~/.juliaup/bin/julia --project -e 'include("examples/run_all_examples.jl")'
~/.juliaup/bin/julia --project -e 'include("benchmark/run_all_benchmarks.jl")'
```

### C++
```bash
cd cpp/library
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

## Documentation

- Repository and workflow docs:
  - [Repository Overview](docs/REPO_OVERVIEW.md)
  - [Workflows and Subrepos](docs/SUBREPOS_AND_WORKFLOWS.md)
  - [HPC and Slurm notes](docs/HPC_SLURM.md)
- Julia package docs:
  - [juqusolver README](juqusolver/README.md)
  - Custom Hamiltonian construction guide: see `juqusolver/README.md` section `Building Custom Hamiltonians`.
  - [Julia docs index](juqusolver/docs/src/index.md)
  - [Honeycomb Kitaev ED workflow](juqusolver/docs/src/honeycomb_kitaev_diagonalization.md)
