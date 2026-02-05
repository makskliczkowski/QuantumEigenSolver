# C++ Alignment to Python Reference Semantics

This document outlines the alignment between the C++ implementation (`cpp/`) and the Python reference (`pyqusolver/`), detailing semantic mappings, divergences, and compatibility shims.

## Python Reference Semantics

The Python implementation (`pyqusolver`) is the maintained reference for:
*   **Hamiltonian Configuration**: Uses `HamiltonianConfig` or `Hamiltonian` subclasses with specific keyword arguments (`is_manybody`, `ns`, `lattice`).
*   **Lattice Definitions**:
    *   **SquareLattice**: Defined by `lx`, `ly`, `lz` (dimensions), and `dim` (1D/2D/3D). `ly` and `lz` default to 1.
    *   **Boundary Conditions (BC)**: Enum `LatticeBC` with values `PBC`, `OBC`, `MBC`, `SBC`. The Enum uses `auto()`, implying 1-based indexing (PBC=1, OBC=2, ...), but typically initialized via string aliases ("pbc", "obc").
*   **Symmetries**: Defined via dictionaries/lists (e.g., `{'translation': 0}`, `{'parity': 1}`).

## Mapping Table: Python Concepts ↔ C++ Components

| Concept | Python (Reference) | C++ (Implementation) | CLI Flag (New) | CLI Flag (Shim) |
| :--- | :--- | :--- | :--- | :--- |
| **Lattice Type** | `SquareLattice`, etc. | `SquareLattice` (class), `LatticeTypes` (enum) | `-lattice` | `-l` |
| **Lattice Dimensions** | `lx`, `ly`, `lz` | `Lx_`, `Ly_`, `Lz_` | `-lx`, `-ly`, `-lz` | (same) |
| **Dimension** | `dim` (1, 2, 3) | `dim` | `-dim` | `-d` |
| **Boundary Conditions** | `LatticeBC` ("pbc", "obc") | `BoundaryConditions` (enum) | `-bc` | (same) |
| **Model Type** | `Hamiltonian` subclass / config | `MY_MODELS` (enum), `modTyp` | `-model` | `-mod` |
| **Sites** | `ns` | `Ntot_` / derived | `-ns` (or implied) | `-Ntot` |
| **Coupling** | `J` | `J_` | `-J` | (same) |
| **Fields** | `hx`, `hz` | `hx_`, `hz_` | `-hx`, `-hz` | (same) |
| **Symmetry (Momentum)**| `translation` (k) | `k_` | `-k` | (same) |
| **Symmetry (Parity X)**| `parity_x` | `px_` | `-px` | (same) |

## Divergences and Resolutions

### 1. Lattice Boundary Condition Values
*   **Divergence**: Python `LatticeBC` uses 1-based indexing (PBC=1) or string parsing. C++ `BoundaryConditions` uses 0-based indexing (PBC=0).
*   **Resolution**: Implement string-based parsing in C++ CLI for `-bc`. C++ will accept "pbc", "obc", etc., and map them to internal C++ enum values (0, 1, ...), matching Python's user-facing behavior.

### 2. Lattice Size Logic
*   **Divergence**: C++ `UI` explicitly calculates `Ns = Lx * Ly * Lz` (and doubles it for non-square 2D lattices like Honeycomb). Python `SquareLattice` sets `ns` internally.
*   **Resolution**: This behavior is semantically consistent for `SquareLattice`. For Honeycomb, both should agree on `Ns = 2 * Lx * Ly`. No change needed if Python's `HoneycombLattice` follows this (standard practice).

### 3. CLI Naming
*   **Divergence**: C++ used terse flags (`-l`, `-d`, `-mod`). Python config uses descriptive keys (`lattice`, `dim`).
*   **Resolution**: Renamed C++ CLI flags to match Python (`-lattice`, `-dim`, `-model`). Retained old flags as shims.

### 4. Configuration Structure
*   **Divergence**: Python uses a structured `HamiltonianConfig`. C++ uses flat `UI_PARAMS` structs (`LatP`, `ModP`).
*   **Resolution**: Kept C++ structure but aligned naming conventions. The CLI parser acts as the bridge.
