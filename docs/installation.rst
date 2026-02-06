Installation
============

Quantum EigenSolver (QES) consists of a C++ core library and a Python interface.

Prerequisites
-------------

Ensure the following tools and libraries are installed:

- **Build Tools:**
    - CMake (minimum version 3.23.0)
    - A C++20-compliant compiler (Intel icpx, GCC, or MSVC)
    - Python 3.10+

- **Libraries:**
    - Intel MKL
    - Armadillo
    - HDF5

- **Optional:**
    - OpenMP (for parallel processing)

Clone the Repository
--------------------

.. code-block:: bash

    git clone https://github.com/makskliczkowski/QuantumEigenSolver.git
    cd QuantumEigenSolver

C++ Library Installation
------------------------

1. **Set Environment Variables:**

   Set the paths to your installed libraries. For example:

   .. code-block:: bash

       export MKL_INCL_DIR=/path/to/mkl/include
       export MKL_LIB_DIR=/path/to/mkl/lib
       export HDF5_INCL_DIR=/path/to/hdf5/include
       export HDF5_LIB_DIR=/path/to/hdf5/lib
       export ARMADILLO_INCL_DIR=/path/to/armadillo/include

2. **Configure and Build:**

   .. code-block:: bash

       cd cpp/library
       mkdir build
       cd build
       cmake .. -DCMAKE_BUILD_TYPE=Release
       make -j$(nproc)

Python Package Installation
---------------------------

To install the Python interface:

.. code-block:: bash

    cd pyqusolver/Python
    pip install -e .

To install with all optional dependencies (including JAX, ML tools, etc.):

.. code-block:: bash

    pip install -e .[all]
