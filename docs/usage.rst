Usage Guide
===========

Quick examples for each package. See each subpackage's own ``README.md``
for full API documentation.

C++ (cpqusolver)
----------------

Build and run an app:

.. code-block:: bash

    cd cpqusolver
    export ARMADILLO_INCL_DIR=/path/to/armadillo-15.x.x
    cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
    cmake --build build-release -j$(nproc)

    # Diagonalize Heisenberg L=10 PBC, Sz=0 sector
    ./build-release/qes-diagonalize --model heisenberg --L 10 --J 1 --pbc 1

    # Entanglement entropy half-cut, Ising L=12
    ./build-release/qes-entropy --model ising --L 12 --J 1 --hx 0.5

    # Krylov time evolution
    ./build-release/qes-evolve --model xxz --L 14 --Jz 1 --dt 0.05 --steps 200

    # Disorder-averaged gap ratio (SLURM array ready)
    ./build-release/qes-disorder --model quantum_sun --seed 42 --L 8

Use ``qes-<app> --help`` for the full flag list. See
``cpqusolver/docs/scripts.md`` for SLURM workflow.

Library API (header-only, C++20):

.. code-block:: cpp

    #include "cpqusolver/cpqusolver.hpp"
    using namespace cpqusolver;

    // Full basis, L=10 spin-1/2
    auto basis = std::make_shared<basis::FullBasis<uint64_t>>(10);

    // Heisenberg Hamiltonian
    auto H = models::heisenberg<std::complex<double>>(10, /*J=*/1.0);

    // Matrix-free matvec
    arma::cx_vec v = arma::randn<arma::cx_vec>(basis->size());
    auto Hv = H.matvec(v);

    // Lanczos lowest eigenvalue
    auto eigs = evolution::lanczos_eigsh(
        [&](auto x, auto y){ H.matvec(x, y); }, basis->size(), /*roots=*/3);

Python (pyqusolver)
-------------------

Install and use the ``QES`` package:

.. code-block:: bash

    pip install -e pyqusolver/Python

.. code-block:: python

    import numpy as np
    from QES.Algebra.hilbert import HilbertSpace
    from QES.Algebra.hamil import Hamiltonian

    # Spin-1/2 Heisenberg chain, L=10
    hilbert = HilbertSpace(L=10, local_hilbert="spin")
    H = Hamiltonian(hilbert, model="heisenberg", J=1.0, pbc=True)
    H.build()
    H.diagonalize()

    # Ground-state entanglement
    from QES.general_python.physics.entropy import vn_entropy
    from QES.general_python.physics.density_matrix import reduced_density_matrix

    gs = H.eigenvectors[:, 0]
    rho = reduced_density_matrix(gs, L_A=5, L=10)
    S = vn_entropy(np.linalg.eigvalsh(rho))
    print(f"S = {S:.6f}")

    # NQS (requires pip install -e .[jax] or .[all])
    from QES.NQS import quick_start
    quick_start(mode="ground")

Julia (juqusolver)
------------------

.. code-block:: julia

    using QuantumEigenSolver
    using LinearAlgebra

    # Heisenberg chain L=10
    basis = FullBasis(10)
    H = heisenberg_hamiltonian(basis; J=1.0, pbc=true)

    # Exact diagonalization
    vals, vecs = eigen(Hermitian(Matrix(H)))
    println("GS energy = ", vals[1])

    # Free-fermion chain
    h = tight_binding_matrix(10; t=1.0, pbc=true)
    qh = QuadraticHamiltonian(h)
    ε, U = diagonalize(qh)

    # Entanglement entropy
    C = correlation_matrix(qh, 5)   # half-chain
    S = entanglement_entropy(C)
    println("S = ", S)

Cross-language validation
-------------------------

Run the shared benchmark suite:

.. code-block:: bash

    python cross_language/run_benchmarks.py

See ``cross_language/README.md`` for the model list and tolerance table.
