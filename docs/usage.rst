Usage Guide
===========

This guide provides examples and tutorials for using the Quantum EigenSolver (QES) framework.

Quick Start
-----------

Import the main QES package and start working with quantum systems:

.. code-block:: python

    import QES
    from QES.Algebra import HilbertSpace, Hamiltonian
    from QES.general_python.lattices import square
    import numpy as np

Basic Hilbert Space Construction
--------------------------------

Create a Hilbert space for a spin-1/2 system:

.. code-block:: python

    # Create a 4x4 square lattice
    lattice = square.SquareLattice(4, 4)
    
    # Create Hilbert space for spin-1/2 particles
    hilbert = HilbertSpace(lattice, particle_type='spin', local_dim=2)
    
    print(f"Hilbert space dimension: {hilbert.get_dim()}")

Hamiltonian Construction
------------------------

Build a Heisenberg model Hamiltonian:

.. code-block:: python

    from QES.Algebra.Model.Interacting.Spin import heisenberg_kitaev as hk
    
    # Create Heisenberg model parameters
    params = {
        'J': 1.0,      # Exchange coupling
        'h': 0.1,      # Magnetic field
        'bc': 'pbc'    # Boundary conditions
    }
    
    # Build the Hamiltonian
    H = hk.HeisenbergModel(hilbert, **params)
    hamiltonian = H.build()

Exact Diagonalization
----------------------

Solve small systems exactly:

.. code-block:: python

    from QES.Solver import Solver
    
    # Create exact diagonalization solver
    solver = Solver.ExactSolver(hamiltonian)
    
    # Find ground state
    eigenvalues, eigenvectors = solver.solve(k=1)  # k=1 for ground state only
    
    ground_energy = eigenvalues[0]
    ground_state = eigenvectors[:, 0]
    
    print(f"Ground state energy: {ground_energy}")

Neural Quantum States
---------------------

Use neural networks for larger systems:

.. code-block:: python

    from QES.NQS import NeuralQuantumState
    from QES.Solver import VariationalMonteCarlo
    
    # Create neural quantum state ansatz
    nqs = NeuralQuantumState(
        hilbert=hilbert,
        hidden_units=32,
        activation='tanh'
    )
    
    # Set up variational Monte Carlo solver
    vmc = VariationalMonteCarlo(
        hamiltonian=hamiltonian,
        ansatz=nqs,
        optimizer='adam',
        learning_rate=0.001
    )
    
    # Optimize the neural network
    energies = vmc.optimize(n_steps=1000, n_samples=1000)
    
    print(f"Final energy: {energies[-1]}")

Entanglement Analysis
---------------------

Calculate entanglement entropy:

.. code-block:: python

    from QES.general_python.physics.entropy import EntropyPredictions
    
    # Reduced density matrix for subsystem A
    rho_A = hilbert.reduced_density_matrix(ground_state, subsystem='A')
    
    # Calculate von Neumann entropy
    eigenvals = np.linalg.eigvals(rho_A)
    entropy = EntropyPredictions.vn_entropy(eigenvals)
    
    print(f"Entanglement entropy: {entropy}")

Time Evolution
--------------

Evolve quantum states in time:

.. code-block:: python

    from QES.Algebra.Properties import time_evo
    
    # Define time evolution parameters
    dt = 0.1
    t_max = 10.0
    times = np.arange(0, t_max, dt)
    
    # Evolve the ground state
    evolved_states = time_evo.evolve_state(
        initial_state=ground_state,
        hamiltonian=hamiltonian,
        times=times
    )
    
    # Calculate expectation values during evolution
    magnetization = []
    for state in evolved_states:
        mag = hilbert.expectation_value(state, 'Sz_total')
        magnetization.append(mag)

Monte Carlo Sampling
--------------------

Perform statistical sampling:

.. code-block:: python

    from QES.Solver.MonteCarlo import MetropolisSampler
    
    # Set up Monte Carlo sampler
    sampler = MetropolisSampler(
        hilbert=hilbert,
        hamiltonian=hamiltonian,
        beta=1.0,  # Inverse temperature
        n_samples=10000
    )
    
    # Generate samples
    samples = sampler.sample()
    
    # Calculate thermal averages
    energy_avg = sampler.estimate_energy(samples)
    specific_heat = sampler.estimate_specific_heat(samples)
    
    print(f"Thermal energy: {energy_avg}")
    print(f"Specific heat: {specific_heat}")

Visualization
-------------

Plot results using built-in utilities:

.. code-block:: python

    from QES.general_python.common.plot import Plot
    import matplotlib.pyplot as plt
    
    # Plot energy convergence
    Plot.energy_convergence(energies)
    
    # Plot entanglement entropy scaling
    system_sizes = [4, 6, 8, 10, 12]
    entropies = [calculate_entropy(L) for L in system_sizes]
    
    plt.figure()
    plt.plot(system_sizes, entropies, 'o-')
    plt.xlabel('System Size')
    plt.ylabel('Entanglement Entropy')
    plt.title('Entropy Scaling')
    plt.show()

Advanced Features
-----------------

GPU Acceleration with JAX
~~~~~~~~~~~~~~~~~~~~~~~~~~

Enable GPU support for large-scale calculations:

.. code-block:: python

    # Set JAX backend for GPU acceleration
    from QES.general_python.algebra.utils import set_global_backend
    
    set_global_backend('jax')
    
    # Now all computations will use JAX/GPU when available

Symmetry-Adapted Calculations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use quantum symmetries to reduce computational cost:

.. code-block:: python

    from QES.Algebra.symmetries import TranslationSymmetry
    
    # Add translation symmetry
    symmetry = TranslationSymmetry(lattice)
    hilbert_sym = HilbertSpace(lattice, symmetries=[symmetry])
    
    print(f"Reduced Hilbert space dimension: {hilbert_sym.get_dim()}")

Custom Models
~~~~~~~~~~~~~

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

    python main.py --N 3 --L 4 --U 2.0 --layers 5
