Introduction
============

**Quantum EigenSolver (QES)** is a comprehensive computational framework designed for solving quantum many-body problems using classical and quantum-inspired algorithms. The project provides a unified platform for researchers and practitioners working with quantum systems, offering both traditional methods and cutting-edge neural quantum state approaches.

Overview
--------

Quantum many-body systems present some of the most challenging computational problems in physics. The exponential scaling of the Hilbert space dimension with system size makes exact solutions intractable for large systems. QES addresses this challenge by providing:

🔬 **Multiple Solution Approaches**
   - Exact diagonalization for small systems
   - Variational methods with neural quantum states
   - Quantum Monte Carlo techniques
   - Matrix product state methods

⚡ **Performance Optimization**
   - JIT compilation for CPU acceleration
   - GPU support through JAX backend
   - Parallel computing capabilities
   - Memory-efficient algorithms

🎯 **Target Applications**
   - Condensed matter physics simulations
   - Quantum chemistry calculations
   - Statistical mechanics studies
   - Quantum information research

Key Features
------------

Flexible Model Construction
~~~~~~~~~~~~~~~~~~~~~~~~~~~

QES provides tools for constructing various quantum many-body models:

- **Spin Systems**: Heisenberg, Ising, XY models with arbitrary interactions
- **Fermionic Systems**: Hubbard, t-J models with hopping and interactions  
- **Bosonic Systems**: Bose-Hubbard and related models
- **Custom Models**: User-defined Hamiltonians with flexible operator algebra

Advanced Solver Methods
~~~~~~~~~~~~~~~~~~~~~~~~

The framework implements multiple solution strategies:

- **Exact Methods**: Full diagonalization for small systems
- **Neural Quantum States**: Variational optimization with neural networks
- **Monte Carlo**: Statistical sampling for finite-temperature properties
- **Time Evolution**: Real and imaginary time dynamics

Comprehensive Analysis Tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Built-in analysis capabilities include:

- **Entanglement Measures**: Von Neumann, Rényi, and participation entropies
- **Correlation Functions**: Spatial and temporal correlations
- **Thermodynamic Properties**: Specific heat, susceptibility, phase transitions
- **Visualization**: Plotting and data export utilities

Scientific Background
---------------------

The methods implemented in QES are based on established theoretical frameworks:

**Neural Quantum States**
   Neural quantum states, introduced by Carleo and Troyer [1]_, represent quantum many-body wavefunctions using artificial neural networks. This approach has shown remarkable success in capturing complex quantum correlations and ground state properties.

**Variational Monte Carlo**
   The variational principle combined with Monte Carlo sampling provides a powerful framework for approximating ground states and computing expectation values of quantum observables.

**Symmetry Integration**
   QES incorporates quantum symmetries (translational, point group, particle number conservation) to reduce computational complexity and improve accuracy.

References
----------

.. [1] G. Carleo and M. Troyer, "Solving the quantum many-body problem with artificial neural networks," Science 355, 602-606 (2017).

Getting Started
---------------

To begin using QES, see the :doc:`installation` guide for setup instructions and the :doc:`usage` section for examples and tutorials.
