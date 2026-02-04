# C++ Alignment to Python Semantics

This document outlines the alignment of the C++ implementation (`cpp/`) with the Python reference implementation (`pyqusolver/`). The Python implementation is considered the source of truth for semantics, naming, and behavior.

## Python Reference Semantics

### Transverse Field Ising Model (TFIM)

**Class:** `TransverseFieldIsing` (in `pyqusolver/Python/QES/Algebra/Model/Interacting/Spin/transverse_ising.py`)

**Hamiltonian:**
$$ H = - \sum_{\langle i,j \rangle} J_{ij} \sigma^z_i \sigma^z_j - \sum_i h_{x,i} \sigma^x_i - \sum_i h_{z,i} \sigma^z_i $$

**Parameters:**
- `j` (Union[List[float], float]): Ising coupling strength. Default: 1.0. (Positive J implies Ferromagnetic interaction due to minus sign in Hamiltonian).
- `hx` (Union[List[float], float]): Transverse field strength. Default: 1.0.
- `hz` (Union[List[float], float]): Perpendicular field strength. Default: 1.0.
- `lattice`: Defines the geometry and neighbors.

**Behavior:**
- Accepts lists for site-dependent parameters (disorder).
- Uses negative signs for all terms in the Hamiltonian construction.

## Mapping Table: Python ↔ C++

| Concept | Python (`TransverseFieldIsing`) | C++ (`IsingModel`) |
| :--- | :--- | :--- |
| **Interaction** | `j` | `J` (was `J1`) |
| **Transverse Field** | `hx` | `hx` (was `g`) |
| **Perpendicular Field** | `hz` | `hz` (was `h`) |
| **Disorder** | List of values | Base value + Random vector (`J0`, `hx0`, `hz0` define width/range) |
| **Hamiltonian Signs** | Negative terms (`-J`, `-hx`, `-hz`) | **Previously:** Positive terms (`+J`, `+g`, `+h`). **Now:** Aligned to Negative terms. |

## Divergences and Alignment Strategy

### 1. Naming
- **Divergence:** C++ used `g` for transverse field and `h` for perpendicular field. CLI used `J1`.
- **Resolution:**
  - Renamed `g` → `hx`, `h` → `hz` in `IsingModel` class.
  - Renamed `J1` → `J` in `ModP` structure.
  - Updated CLI to accept `J`, `hx`, `hz`.
  - Added compatibility shim for `J1` in CLI to map to `J`.

### 2. Hamiltonian Definition (Signs)
- **Divergence:** Python implementation applies a minus sign to all terms: $H = -J ... -hx ...$. C++ implementation added terms with positive signs.
- **Resolution:** Modified `IsingModel::locEnergy` to apply negative signs to `J`, `hx`, `hz` terms, ensuring that passing positive parameter values results in the same physical model (e.g., Ferromagnetic for positive J).

### 3. Disorder Handling
- **Divergence:** Python accepts explicit lists for disordered parameters. C++ generates disorder internally based on a base value and a disorder strength/width.
- **Resolution:** Kept C++ internal generation for now to avoid major rewrite of input parsing, but ensured the naming (`J0`, `hx0`, `hz0`) is consistent. C++ CLI allows seeding and disorder strength configuration.
