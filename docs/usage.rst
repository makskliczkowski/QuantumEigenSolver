Usage Guide
===========

This guide provides examples and tutorials for using the Quantum EigenSolver (QES) framework.

Python Interface
----------------

Quick Start
~~~~~~~~~~~

Import the main QES package and start working with quantum systems:

.. code-block:: python

    import QES
    from QES.Algebra import HilbertSpace
    from QES.general_python.lattices import SquareLattice
    import numpy as np

Basic Hilbert Space Construction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a Hilbert space for a spin-1/2 system:

.. code-block:: python

    # Create a 1D lattice with 4 sites
    lattice = SquareLattice(lx=4, dim=1)
    
    # Create Hilbert space for spin-1/2 particles
    hilbert = HilbertSpace(lattice=lattice, local_space="spin-1/2")
    
    print(f"Hilbert space dimension: {hilbert.dim}")

Hamiltonian Construction
~~~~~~~~~~~~~~~~~~~~~~~~

Build a transverse-field Ising model Hamiltonian:

.. code-block:: python

    from QES.Algebra.Model.Interacting.Spin.transverse_ising import TransverseFieldIsing
    
    # Create model parameters
    H = TransverseFieldIsing(
        lattice=lattice,
        hilbert_space=hilbert,
        j=1.0,
        hx=0.3,
        hz=0.0,
    )
    H.build()

Exact Diagonalization
~~~~~~~~~~~~~~~~~~~~~

Solve small systems exactly:

.. code-block:: python

    # Diagonalize the Hamiltonian
    H.diagonalize(method="exact")
    
    # Access ground state
    eigenvalues = H.eigenvalues
    eigenvectors = H.eigenvectors
    ground_energy = eigenvalues[0]
    ground_state = eigenvectors[:, 0]
    
    print(f"Ground state energy: {ground_energy}")

Neural Quantum States
~~~~~~~~~~~~~~~~~~~~~

Use neural networks for larger systems:

.. code-block:: python

    from QES.NQS import quick_start
    
    # Print a runnable boilerplate script for NQS workflows
    quick_start(mode="ground")

Entanglement Analysis
~~~~~~~~~~~~~~~~~~~~~

Calculate entanglement entropy:

.. code-block:: python

    from QES.general_python.physics.eigenlevels import reduced_density_matrix
    from QES.general_python.physics.entropy import vn_entropy
    
    # Reduced density matrix for subsystem A
    L = lattice.ns
    La = L // 2
    rho_A = reduced_density_matrix(ground_state, La, L)
    
    # Calculate von Neumann entropy
    eigenvals = np.linalg.eigvals(rho_A)
    entropy = vn_entropy(eigenvals)
    
    print(f"Entanglement entropy: {entropy}")

Time Evolution
~~~~~~~~~~~~~~

Evolve quantum states in time:

.. code-block:: python

    from QES.Algebra.Properties.time_evo import TimeEvolutionModule
    
    # Define time evolution parameters
    dt = 0.1
    t_max = 10.0
    times = np.arange(0, t_max, dt)
    
    # Evolve the ground state
    H.diagonalize(method="exact")
    evolver = TimeEvolutionModule(H)
    evolved_states = evolver.evolve_batch(ground_state, times)
    
    # Example: check normalization over time
    norms = np.sum(np.abs(evolved_states) ** 2, axis=0)

Monte Carlo Sampling
~~~~~~~~~~~~~~~~~~~~

Perform statistical sampling:

.. code-block:: python

    from QES.NQS import NetworkFactory
    from QES.Solver.MonteCarlo.vmc import VMCSampler
    
    # JAX-based VMC sampling (requires JAX/Flax extras)
    net = NetworkFactory.create("rbm", input_shape=(lattice.ns,), alpha=1.0)
    sampler = VMCSampler(
        net=net,
        shape=(lattice.ns,),
        numsamples=1000,
        numchains=4,
        therm_steps=100,
        sweep_steps=10,
        backend="jax",
    )
    
    # Generate samples
    final_state, samples, probs = sampler.sample()

Visualization
~~~~~~~~~~~~~

Plot results using built-in utilities:

.. code-block:: python

    from QES.general_python.common.plot import Plotter
    import matplotlib.pyplot as plt
    
    # Plot entanglement entropy scaling
    system_sizes = [4, 6, 8, 10, 12]
    def calculate_entropy(L):
        return 0.1 * L
    entropies = [calculate_entropy(L) for L in system_sizes]
    
    fig, ax = plt.subplots()
    Plotter.plot(ax, system_sizes, entropies, marker='o')
    ax.set_xlabel('System Size')
    ax.set_ylabel('Entanglement Entropy')
    ax.set_title('Entropy Scaling')
    plt.show()

Advanced Features
~~~~~~~~~~~~~~~~~

GPU Acceleration with JAX
^^^^^^^^^^^^^^^^^^^^^^^^^

Enable GPU support for large-scale calculations:

.. code-block:: python

    # Set JAX backend for GPU acceleration (if available)
    from QES.general_python.algebra.utils import backend_mgr
    
    if backend_mgr.is_jax_available:
        backend_mgr.set_active_backend("jax")
    
    # Now all computations will use JAX/GPU when available

Symmetry-Adapted Calculations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use quantum symmetries to reduce computational cost:

.. code-block:: python

    # Add translation symmetry (1D momentum sector k=0)
    sym_gen = {"translation": 0}
    hilbert_sym = HilbertSpace(lattice=lattice, local_space="spin-1/2", sym_gen=sym_gen)
    
    print(f"Reduced Hilbert space dimension: {hilbert_sym.dim}")

Custom Models
^^^^^^^^^^^^^

Define your own quantum models:

.. code-block:: python

    class CustomModel:
        def __init__(self, hilbert, J1, J2):
            self.hilbert = hilbert
            self.J1 = J1  # Nearest neighbor coupling
            self.J2 = J2  # Next-nearest neighbor coupling
            
        def build_hamiltonian(self):
            # Implement custom Hamiltonian construction
            pass

For more detailed examples and advanced usage, see the :doc:`api` reference.

C++ Library Usage
-----------------

The C++ library provides high-performance implementations of key algorithms.
To use the C++ executable after building:

.. code-block:: bash

    ./build/qsolver

Input files can be configured in `cpp/library/INPUTS`.

See the :doc:`cpp_api` for detailed C++ class documentation.
