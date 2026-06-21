Welcome to Quantum EigenSolver's Documentation
==============================================

**Quantum EigenSolver (QES)** is a multi-language framework for quantum
many-body physics: exact diagonalization, symmetry-reduced bases,
free-fermion solvers, spectral functions, time evolution, and thermodynamics.

The repository contains three independent, interoperable packages:

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Package
     - Description
   * - ``cpqusolver/``
     - C++20 static library + ``qes-*`` apps. Header-only core, Armadillo
       backend for dense/sparse linalg, matrix-free Lanczos/Krylov.
       Build: ``cmake -S cpqusolver -B cpqusolver/build-release -DCMAKE_BUILD_TYPE=Release``.
   * - ``pyqusolver/``
     - Python package ``QES``. Spin/fermionic/quadratic models, NQS/TDVP,
       Monte Carlo, Numba/JAX backends. Install: ``pip install -e pyqusolver/Python``.
   * - ``juqusolver/``
     - Julia package ``QuantumEigenSolver``. High-performance ED, quadratic
       solvers, time evolution, entanglement. ``] add juqusolver/``.

Capabilities by package
-----------------------

+------------------------------------+----------+-----------+------------+
| Feature                            | C++ (cpq)| Python    | Julia      |
+====================================+==========+===========+============+
| Exact diagonalization              | yes      | yes       | yes        |
+------------------------------------+----------+-----------+------------+
| Symmetry-reduced basis             | yes      | yes       | yes        |
+------------------------------------+----------+-----------+------------+
| Free-fermion / quadratic           | yes      | yes       | yes        |
+------------------------------------+----------+-----------+------------+
| Krylov / Lanczos (matrix-free)     | yes      | partial   | yes        |
+------------------------------------+----------+-----------+------------+
| Spectral functions / DSF           | yes      | yes       | yes        |
+------------------------------------+----------+-----------+------------+
| Time evolution                     | yes      | yes       | yes        |
+------------------------------------+----------+-----------+------------+
| Thermodynamics                     | yes      | yes       | yes        |
+------------------------------------+----------+-----------+------------+
| NQS / variational Monte Carlo      | —        | yes       | —          |
+------------------------------------+----------+-----------+------------+
| GPU (JAX)                          | —        | yes       | —          |
+------------------------------------+----------+-----------+------------+

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
