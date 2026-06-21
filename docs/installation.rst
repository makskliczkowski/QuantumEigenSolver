Installation
============

Clone with all submodules first:

.. code-block:: bash

    git clone --recursive https://github.com/makskliczkowski/QuantumEigenSolver.git
    cd QuantumEigenSolver

C++ library (cpqusolver)
------------------------

**Prerequisites**

- CMake >= 3.23
- C++20 compiler (GCC 12+, Clang 16+, Apple Clang 15+)
- `Armadillo <https://arma.sourceforge.net/>`_ source tree (header-only; no
  installed library needed). On macOS the BLAS/LAPACK backend is
  ``Accelerate``; on Linux link against OpenBLAS or MKL.
- HDF5 (optional; required only for ``qes-kitaev`` HDF5 output and
  ``qes-disorder`` spectrum dumps)

MKL is **not** required. The default backend is Accelerate (macOS) or
OpenBLAS (Linux).

**Build**

.. code-block:: bash

    export ARMADILLO_INCL_DIR=/path/to/armadillo-15.x.x   # source tree

    cd cpqusolver
    cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
    cmake --build build-release -j$(nproc)
    ctest --test-dir build-release                         # 35/35 expected

To build without Armadillo (core only, no dense/sparse linalg):

.. code-block:: bash

    cmake -S . -B build-noarma -DGENUTILS_USE_ARMADILLO=OFF
    cmake --build build-noarma -j$(nproc)

**Environment variables recognised by CMake**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Variable
     - Purpose
   * - ``ARMADILLO_INCL_DIR``
     - Path to Armadillo source tree (``include/`` dir or root)
   * - ``HDF5_INCL_DIR`` / ``HDF5_LIB_DIR``
     - HDF5 headers / libraries (optional)
   * - ``MKL_INCL_DIR`` / ``MKL_LIB_DIR``
     - Intel MKL (optional; overrides Accelerate/OpenBLAS)

Python package (pyqusolver)
---------------------------

.. code-block:: bash

    cd pyqusolver/Python
    pip install -e .

With optional JAX / ML dependencies:

.. code-block:: bash

    pip install -e .[all]

Requires Python 3.10+. Core dependencies: NumPy, SciPy, h5py.
JAX is optional and pulled in only by ``pip install -e .[jax]``.

Julia package (juqusolver)
--------------------------

.. code-block:: bash

    julia --project=juqusolver -e "using Pkg; Pkg.instantiate()"

Or from the Julia REPL:

.. code-block:: julia

    ] add /path/to/QuantumEigenSolver/juqusolver

Requires Julia 1.9+.
