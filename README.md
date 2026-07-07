# QuantumEigenSolver

Main repository containing Python (`pyqusolver`), and C++ components.

Clone with submodules:

```bash
git clone --recursive https://github.com/makskliczkowski/QuantumEigenSolver.git
```

## Implementations

- `pyqusolver/`: Python framework (`QES`) for many-body and variational workflows.
- `juqusolver/`: Julia package (`QuantumEigenSolver`) focused on high-performance computation paths (currently developed)
- `cpp/`: C++ implementation components - the oldest but potentially the fastest (legacy).

## Model Coverage
- Spin models covered:
  - Heisenberg-Kitaev
  - QSM
  - Ultrametric
  - ...and others
- Fermionic models covered:
  - ManyBodyFreeFermions
  - HubbardModel
- Noninteracting models covered:
  - FreeFermions
  - AubryAndre
  - SYK2
  - PowerLawRandomBanded
  - RosenzweigPorter
  - ...user-built quadratic systems

### Entropy and density-matrix calculations

- Python:
  - `pyqusolver/Python/QES/general_python/physics/entropy.py`
  - `pyqusolver/Python/QES/general_python/physics/density_matrix.py`

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
  - `pyqusolver/Python/QES/Algebra/Properties/statistical.py`
- Julia:
  - `juqusolver/src/Algebra/HamiltonianAction.jl`
  - `juqusolver/src/Physics/Spectral.jl`
  - `juqusolver/src/Physics/Statistical.jl`

## Quick Run Commands

### Python

```bash
cd pyqusolver
pip install -e "Python/[all,dev]"
PYTHONPATH=Python pytest Python/tests -q
```

### C++

```bash
cd cpp/library
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```
This work was partially supported by the National Science Centre, Poland (Narodowe Centrum Nauki, NCN), grant no. 2024/53/B/ST3/02756.

