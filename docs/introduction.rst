Introduction
============

**Quantum EigenSolver (QES)** is a multi-language toolkit for quantum
many-body physics. It targets exact and near-exact methods for spin,
fermionic, and bosonic lattice models at zero and finite temperature.

Scope
-----

The three packages share a common physics scope and cross-validated
reference values, but have independent builds and APIs:

**cpqusolver** (C++20)
   - Term-list Hamiltonian over arbitrary binary bases (spin-1/2, spinless
     fermions, spinful fermions).
   - Symmetry-reduced bases: translation, reflection, parity, inversion,
     arbitrary abelian groups with fermionic JW signs.
   - Matrix-free ``matvec`` + Krylov (Lanczos, Krylov time evolution
     ``exp(-iHdt)``), dense and sparse materialization via Armadillo.
   - Spectral functions (T=0 Lehmann, finite-T FTLM, Krylov continued
     fraction), entanglement entropies, level statistics, thermodynamics.
   - Free-fermion (quadratic) solver: correlation matrix, BdG, Aubry-Andre,
     SYK2, power-law random banded.
   - Nine ``qes-*`` command-line apps (diagonalize, entropy, evolve,
     measure, disorder, SQD, Kitaev, quadratic, properties).

**pyqusolver** (Python)
   - ``QES`` package: Hamiltonians, exact diagonalization, NQS/TDVP,
     variational Monte Carlo, Numba/JAX backends.
   - Spin models (Ising, XXZ, Heisenberg-Kitaev, XYZ, J1-J2), fermionic
     (Hubbard, t-J), quadratic (free-fermion, Aubry-Andre, SYK2, PLRB,
     Rosenzweig-Porter).
   - Quantities: entropy, RDM, spectral functions, thermodynamics,
     time evolution, DSF (Lehmann + Krylov + FTLM).

**juqusolver** (Julia)
   - ``QuantumEigenSolver`` package: high-throughput ED, quadratic
     Hamiltonians (BdG, correlation matrix, Aubry-Andre localization),
     Krylov time evolution, entanglement / MES, level statistics.
   - Strong cross-validation suite against pyqusolver.

What is not here
----------------

- **Neural Quantum States (NQS)**: Python only (``pyqusolver``); not in C++
  or Julia. The C++ legacy NQS code has been retired.
- **DMRG / MPS**: not implemented in any package.
- **Quantum chemistry integrals**: not in scope.

Cross-language validation
-------------------------

Shared benchmark suites live in ``cross_language/``. Ground-state energies,
symmetry sector spectra, and free-fermion dispersion are validated between
all three packages to at least 1e-10 relative error. See
``cross_language/README.md`` for the test matrix.

References
----------

.. [1] G. Carleo and M. Troyer, *Science* 355, 602-606 (2017). -- NQS
       method used in pyqusolver.
.. [2] A. Jaklic and P. Prelovsek, *Phys. Rev. B* 49, 5065 (1994). -- FTLM
       finite-temperature Lanczos used in cpqusolver and pyqusolver.
