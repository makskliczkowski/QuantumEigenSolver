

# Quantum Eigensolver

This project provides a comprehensive framework for simulating quantum systems using various models and functionalities. The project leverages advanced mathematical libraries and parallel computing techniques to ensure efficient and accurate simulations.

It is a general solver for physical Hamiltonians. The work is in progress. Currently, the solver includes:
- ED solutions to multiple Hamiltonians on the Lattice or Hilbert space graphs. This includes standard spin systems and random Hamiltonians. The software also enables solutions to quadratic Hamiltonians with multithreaded correlation matrices calculation.
- Implementation of point and global symmetries for the systems.
- the variational Quantum Monte Carlo solver for ansatz ground (and excited) states with RBM and RBM-PP ansatzes.

For detailed documentation and usage instructions, please refer to the [project wiki](https://github.com/makskliczkowski/QuantumEigenSolver/wiki).

---

## Features

- Cross-platform support (Linux and Windows).
- Utilizes Intel MKL for high-performance linear algebra and numerical computations.
- Leverages Armadillo for user-friendly matrix operations.
- Supports HDF5 for efficient and portable data storage.
- Implements `xorshiro256` for fast and high-quality random number generation.
- Parallel processing support using OpenMP.
- Modular design for ease of extension and customization.

---

## Key Components

### Intel MKL (Math Kernel Library)
Intel MKL provides optimized and threaded math routines for linear algebra, fast Fourier transforms, and random number generation. It is essential for high-performance computations in this project.

- **Installation**: Install via the [Intel oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html).
- **Environment Variables**:
  - `MKL_INCL_DIR`: Path to MKL include directory.
  - `MKL_LIB_DIR`: Path to MKL library directory.

### Armadillo
Armadillo is a C++ linear algebra library designed for efficiency and ease of use, providing a high-level syntax similar to MATLAB.

- **Installation**: Install via package managers (e.g., `sudo apt install libarmadillo-dev` on Ubuntu) or build from source.
- **Environment Variable**: 
  - `ARMADILLO_INCL_DIR`: Path to Armadillo include directory.

### HDF5
HDF5 is a file format and library for storing large amounts of data efficiently and is widely used in scientific computing.

- **Installation**: Download from the [HDF Group](https://www.hdfgroup.org/downloads/hdf5/) or install via package managers.
- **Environment Variables**:
  - `HDF5_INCL_DIR`: Path to HDF5 include directory.
  - `HDF5_LIB_DIR`: Path to HDF5 library directory.

### xorshiro256
`xorshiro256` is a family of pseudorandom number generators known for speed and statistical quality. It is implemented in this project to handle random number generation efficiently.

- **Installation**: Bundled as part of the project; no external dependencies.

---

---

## Features

- Cross-platform support (Linux and Windows).
- Utilizes Intel MKL for high-performance linear algebra and numerical computations.
- Leverages Armadillo for user-friendly matrix operations.
- Supports HDF5 for efficient and portable data storage.
- Implements `xorshiro256` for fast and high-quality random number generation.
- Parallel processing support using OpenMP.
- Modular design for ease of extension and customization.

---

## File Structure

```plaintext
QuantumEigenSolver/
├── CMakeLists.txt             # Build configuration file
├── main.cpp                   # Entry point of the application
├── operators.cpp              # Operator implementation
├── src/                       # Source files
│   ├── operator_parser.cpp
│   ├── user_interface/        # User interface-related files
│   │   ├── ui_check_eth.cpp
│   │   ├── ui_check_nqs.cpp
│   │   ├── ui_check_symmetries.cpp
│   │   └── ui_check_quadratic.cpp
│   └── nqs.cpp
├── include/                   # Header files
│   ├── quantities/            # Statistical quantities and calculations
│   │   ├── density_matrix.h
│   │   ├── thermal.h
│   │   ├── entropy.h
│   │   └── measure.h
│   ├── models/                # Physical models
│   │   ├── quadratic/
│   │   │   ├── AubryAndre.h
│   │   │   ├── FreeFermions.h
│   │   │   ├── SYK2.h
│   │   │   └── PowerLawRandomBanded.h
│   │   ├── heisenberg-kitaev.h
│   │   ├── XYZ.h
│   │   ├── ultrametric.h
│   │   ├── quantum_sun.h
│   │   ├── heisenberg_dots.h
│   │   ├── ising.h
│   │   ├── rosenzweig-porter.h
│   │   └── heisenberg.h
│   ├── hamil.h
│   ├── hamilQ.h
│   ├── hamilF.h
│   ├── NQS/                   # Neural Quantum States
│   │   ├── NQS_base/
│   │   │   ├── nqs.h
│   │   │   └── nqs_definitions_base.h
│   │   ├── RBM_PP_base/rbm_pp.h
│   │   ├── RBM_base/rbm.h
│   │   └── nqs_operator.h
│   ├── hilbert_sym.h
│   ├── user_interface/user_interface.h
│   ├── algebra/               # Algebraic tools
│   │   ├── quantities/
│   │   │   ├── density_matrix.h
│   │   │   ├── thermal.h
│   │   │   ├── entropy.h
│   │   │   └── measure.h
│   │   ├── operators.h
│   │   ├── operator_algebra.h
│   │   ├── hilbert.h
│   │   ├── general_operator.h
│   │   └── global_symmetries.h
│   └── rbm.h
├── source/                    # External and utility source files
│   ├── src/                   # Source files from the external library
│   │   ├── binary.h
│   │   ├── lin_alg.h
│   │   ├── plotter.h
│   │   ├── lattices.h
│   │   ├── xoshiro_pp.h
│   │   ├── Include/
│   │   │   ├── random.h
│   │   │   ├── str.h
│   │   │   ├── directories.h
│   │   │   ├── linalg/
│   │   │   │   ├── generalized_matrix.h
│   │   │   │   ├── diagonalizers.h
│   │   │   └── exceptions.h
│   │   ├── flog.h
│   │   ├── Lattices/
│   │   │   ├── square.h
│   │   │   └── hexagonal.h
│   │   ├── UserInterface/ui.h
│   │   └── common.h
│   └── cpp/
│       ├── time.cpp
│       ├── signatures.cpp
│       ├── exceptions.cpp
│       ├── str.cpp
│       ├── ui.cpp
│       ├── LinearAlgebra/
│       │   ├── preconditioners.cpp
│       │   ├── pfaffian.cpp
│       │   ├── Solvers/
│       │   │   ├── solvers_pseudo.cpp
│       │   │   ├── solvers_direct.cpp
│       │   │   ├── solvers_minres.cpp
│       │   │   ├── solvers_arma.cpp
│       │   │   ├── solvers_minresqlp.cpp
│       │   │   ├── solvers_arnoldi.cpp
│       │   │   └── solvers_cg.cpp
│       │   └── solvers.cpp
│       ├── directories.cpp
│       ├── Lattices/
│       │   ├── square.cpp
│       │   └── hexagonal.cpp
│       ├── common.cpp
│       └── maths.cpp
└── README.md                 # Project overview
```

## Installation Guide
### Step 1: Prerequisites

Ensure the following tools and libraries are installed:

```plaintext
    Build Tools:
        CMake (minimum version 3.23.0)
        A C++20-compliant compiler (Intel icpx, GCC, or MSVC)

    Libraries:
        Intel MKL
        Armadillo
        HDF5

    Optional:
        OpenMP (for parallel processing)
```
Notes on source/

The source/ directory contains files sourced from the external library GenUtils C++ Project[https://github.com/makskliczkowski/general_cpp] available as is. These files include utility functions, random number generators, and linear algebra utilities that are integrated into the project.

### Step 2: Clone the Repository

```bash
git clone https://github.com/makskliczkowski/QuantumEigenSolver.git
cd QuantumEigenSolver
```
### Step 3: Set Environment Variables
```bash
export MKL_INCL_DIR=/path/to/mkl/include
export MKL_LIB_DIR=/path/to/mkl/lib
export HDF5_INCL_DIR=/path/to/hdf5/include
export HDF5_LIB_DIR=/path/to/hdf5/lib
export ARMADILLO_INCL_DIR=/path/to/armadillo/include
```

### Step 4: Configure build
```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
```

## Example imputs 
Example input parameters can be found in cpp/library/inputs.

## Acknowledgements
We would like to thank:
- Intel for the Math Kernel Library.
- HDF Group for the HDF5 library.
- Armadillo developers for their high-level matrix operations library.
- David Blackman and Sebastiano Vigna and [Ryo Suzuki](https://github.com/Reputeless/Xoshiro-cpp) for the xorshiro256 RNG family.

## License

This project is licensed under the [MIT License](LICENSE.md).

---

## Copyright

Copyright 2024
Maksymilian Kliczkowski
PhD candidate
Wroclaw University of Science and Technology
maksymilian.kliczkowski.at.pwr.edu.pl 

