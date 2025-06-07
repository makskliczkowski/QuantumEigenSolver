API Reference
=============

QES Package Structure
---------------------

The QES (Quantum EigenSolver) package is organized into several main modules:

Core Modules
~~~~~~~~~~~~

**Algebra**
   
   The Algebra module contains mathematical foundations for quantum mechanics calculations.

**Solver**
   
   The Solver module contains different eigensolving algorithms.

**NQS**
   
   The NQS (Neural Quantum States) module contains neural network implementations.

**General Python Utilities**
   
   The general_python module contains shared utilities for physics, mathematics, and machine learning.

Detailed API Documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For detailed API documentation of individual modules and functions, please refer to the source code docstrings. The QES package structure includes:

**QES.Algebra**: Mathematical operators, models, and Hilbert spaces
  
  - **QES.Algebra.Operator**: Quantum operator implementations
  - **QES.Algebra.Model**: Physical model definitions (Interacting and Noninteracting)
  - **QES.Algebra.Hilbert**: Hilbert space representations  
  - **QES.Algebra.Hamil**: Hamiltonian constructions
  - **QES.Algebra.Properties**: Physical property calculations

**QES.Solver**: Eigensolving algorithms

  - **QES.Solver.solver**: Main solver interface
  - **QES.Solver.MonteCarlo**: Monte Carlo methods

**QES.NQS**: Neural Quantum States

  - **QES.NQS.REF**: Reference implementations

**QES.general_python**: Utility modules

  - **QES.general_python.physics**: Physics calculations including entropy measures
  - **QES.general_python.algebra**: Mathematical utilities and backends
  - **QES.general_python.lattices**: Lattice structure definitions
  - **QES.general_python.maths**: Mathematical functions and algorithms
  - **QES.general_python.common**: Common utilities and plotting functions
  - **QES.general_python.ml**: Machine learning utilities and schedulers

Function Reference
~~~~~~~~~~~~~~~~~~

Key Functions by Module:

**Physics Module (QES.general_python.physics.entropy)**

- ``von_neumann_entropy(rho)``: Calculate von Neumann entropy of density matrix
- ``renyi_entropy(rho, alpha)``: Calculate Rényi entropy with parameter alpha
- ``entanglement_entropy(psi, subsystem)``: Calculate entanglement entropy for a subsystem
- ``mutual_information(rho, subsystem_A, subsystem_B)``: Calculate mutual information between subsystems

**Algebra Utilities (QES.general_python.algebra.utils)**

- Backend management functions for NumPy/JAX compatibility
- Random number generation utilities
- Linear algebra helper functions

**Plotting Utilities (QES.general_python.common.plot)**

- Scientific plotting functions with proper formatting
- Publication-ready figure generation utilities

**Machine Learning (QES.general_python.ml)**

- Neural network training schedulers
- Optimization utilities for quantum machine learning

Note
~~~~

Due to complex dependencies and initialization requirements, some modules may not be fully importable in all environments. The package is designed for high-performance quantum computing applications and requires specific hardware and software configurations for optimal performance.
