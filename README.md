# QuantumEigenSolver

Main repository containing Python (`pyqusolver`), Julia (`juqusolver`), and C++ components.

Clone with submodules:

```bash
git clone --recursive https://github.com/makskliczkowski/QuantumEigenSolver.git
```

## Implementations

- `pyqusolver/`: Python framework (`QES`) for many-body and variational workflows.
- `juqusolver/`: Julia package (`QuantumEigenSolver`) focused on high-performance computation paths.
- `cpp/`: C++ implementation components - the oldest but potentially the fastest (legacy).

## Documentation Entry Points

- Julia package guide: `juqusolver/README.md`
- Python package guide: `pyqusolver/README.md`
- Julia quadratic builder notes: `juqusolver/docs/src/quadratic_hamiltonians.md`
- Julia quadratic utility notes: `juqusolver/docs/src/quadratic_utilities.md`
- Julia quadratic transform notes: `juqusolver/docs/src/quadratic_transforms.md`
- Julia physics-model notes: `juqusolver/docs/src/physics_models.md`
- Julia time-evolution notes: `juqusolver/docs/src/time_evolution.md`
- Julia entanglement and MES notes: `juqusolver/docs/src/entanglement_mes.md`
- Feature gap analysis: `docs/FEATURE_GAP_ANALYSIS.md`
- Development roadmap: `docs/DEVELOPMENT_ROADMAP.md`
- Julia testing guide: `juqusolver/docs/src/testing.md`
- Python testing guide: `pyqusolver/Python/docs/testing.md`
- Cross-language parity suite: `cross_language/README.md`
- Top-level workflow docs: `docs/REPO_OVERVIEW.md`, `docs/SUBREPOS_AND_WORKFLOWS.md`

## Model Coverage and Tested Physics Invariants

- Spin models covered in maintained tests:
  - Heisenberg-Kitaev
  - QSM
  - Ultrametric
  - Julia-only maintained interacting spin family coverage also includes TFIM, XXZ, and J1-J2
- Fermionic models covered in maintained tests:
  - ManyBodyFreeFermions
  - HubbardModel
- Noninteracting models covered in maintained tests:
  - FreeFermions
  - AubryAndre
  - SYK2
  - PowerLawRandomBanded
  - RosenzweigPorter
- Physics invariants enforced by the maintained suites:
  - Hermiticity
  - deterministic seeded behavior for random ensembles
  - coupling-update rebuild paths
  - total-particle-number conservation for spinless many-body fermion models
  - analytic free-fermion cosine-band reproduction
  - Aubry-Andre localization increase through inverse participation ratio
  - middle-spectrum gap statistics for random ensembles
  - random-spin middle-spectrum entropy, ETH-style local-observable behavior, and finite spectral diagnostics
- Maintained test locations:
  - Julia: `juqusolver/test/models/physics_models_test.jl`
  - Python: `pyqusolver/Python/tests/models/test_random_spin_models.py`
  - Python: `pyqusolver/Python/tests/models/test_fermionic_and_noninteracting_models.py`

## Python vs Julia: Practical Interface Mapping

This repository keeps module-level capability parity where possible, but APIs follow each language style.

### Entropy and density-matrix calculations

- Python:
  - `pyqusolver/Python/QES/general_python/physics/entropy.py`
  - `pyqusolver/Python/QES/general_python/physics/density_matrix.py`
- Julia:
  - `juqusolver/src/Physics/Entropy.jl`
  - `juqusolver/src/Physics/DensityMatrix.jl`

### Hilbert spaces with and without symmetries

- Python:
  - `pyqusolver/Python/QES/Algebra/hilbert.py`
  - `pyqusolver/Python/QES/Algebra/Hilbert/hilbert_base.py`
- Julia:
  - `juqusolver/src/Algebra/HilbertCore.jl`

### Hamiltonian and operator construction

- Python:
  - `pyqusolver/Python/QES/Algebra/Operator/operator.py`
  - `pyqusolver/Python/QES/Algebra/Operator/impl/operators_spin.py`
  - `pyqusolver/Python/QES/Algebra/hamil_quadratic.py`
  - `pyqusolver/Python/QES/Algebra/Quadratic/hamil_quadratic_utils.py`
- Julia:
  - `juqusolver/src/Algebra/OperatorsCore.jl`
  - `juqusolver/src/Algebra/HamiltonianAction.jl`
  - `juqusolver/src/Algebra/HamiltonianCore.jl`
  - `juqusolver/src/Physics/Operators.jl`

### Lattice geometry and neighbor tables

- Python:
  - `pyqusolver/Python/QES/general_python/lattices/lattice.py`
  - `pyqusolver/Python/QES/general_python/lattices/square.py`
  - `pyqusolver/Python/QES/general_python/lattices/honeycomb.py`
- Julia:
  - `juqusolver/src/Lattices.jl`
  - `juqusolver/src/Lattices/Core/Square.jl`
  - `juqusolver/src/Lattices/Core/Honeycomb.jl`

### Time evolution, spectral, and statistics

- Python:
  - `pyqusolver/Python/QES/Algebra/Properties/time_evo.py`
  - `pyqusolver/Python/QES/general_python/common/plotters/spectral_utils.py`
  - `pyqusolver/Python/QES/Algebra/Properties/statistical.py`
- Julia:
  - `juqusolver/src/Algebra/HamiltonianAction.jl`
  - `juqusolver/src/Physics/Spectral.jl`
  - `juqusolver/src/Physics/Statistical.jl`

## Cross-language Validation and Benchmarks

Root-level parity tests and benchmarks are in `cross_language/`.

```bash
# parity tests
PYTHONPATH=pyqusolver/Python pytest cross_language/tests/test_python_julia_parity.py -q

# benchmark snapshots
python cross_language/benchmarks/run_python_julia_benchmarks.py

# full cross-language pipeline
python cross_language/run_all.py
```

## Quick Run Commands

### Python

```bash
cd pyqusolver
pip install -e "Python/[all,dev]"
PYTHONPATH=Python pytest Python/tests -q
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
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```
