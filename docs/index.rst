Welcome to Quantum EigenSolver's Documentation
==============================================

**Quantum EigenSolver (QES)** is a comprehensive framework for solving quantum eigenvalue problems and performing quantum many-body simulations. It provides a unified interface for various computational methods including exact diagonalization, Neural Quantum States (NQS), and Monte Carlo techniques.

Key Features
------------

🔬 **Quantum Many-Body Systems**
   - Support for spin systems, fermionic models, and bosonic systems
   - Flexible Hilbert space construction with symmetry handling
   - Efficient operator representations

⚡ **High-Performance Computing**
   - JIT compilation with Numba for CPU acceleration
   - Optional JAX backend for GPU/TPU support
   - Parallel Monte Carlo implementations
   - C++ Core for critical performance paths

🧠 **Neural Quantum States**
   - Variational quantum Monte Carlo with neural networks
   - Time-dependent variational principle (TDVP)
   - Modern machine learning integration

📊 **Comprehensive Analysis Tools**
   - Entanglement entropy calculations
   - Statistical mechanics utilities
   - Visualization and data management

Package Structure
-----------------

The QES package is organized into several main modules:

- **QES.Algebra**: Core algebraic operations, Hilbert spaces, and Hamiltonians
- **QES.Solver**: Eigenvalue solvers and computational methods  
- **QES.NQS**: Neural Quantum State implementations
- **QES.general_python**: Utilities for scientific computing

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   installation
   usage
   api
   cpp_api
   contributing
   license

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
