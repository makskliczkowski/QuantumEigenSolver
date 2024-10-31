# Hamiltonian Solver

This is the general solver for lattice Hamiltonians. The works is in progress but currently it includes:
- ED
- ED with symmetries
- the variational Quantum Monte Carlo solver for ansatz ground state Jastrov functions

The plan for the future is to extend it to either more than 1D Hamitlonians in the case of symmetries or add symmetries to the variational ansatz potentially 
in higher dimensions and non-spin Hamiltonians.

Copyright 2022 <br> 
Maksymilian Kliczkowski <br> 
PhD candidate <br> 
Wroclaw University of Science and Technology <br> 
maksymilian.kliczkowski.at.pwr.edu.pl <br> 

# Quantum State Simulation: INI Configuration

This document outlines the configuration parameters used in the quantum state simulation software. The parameters are organized into different sections, categorizing models and functionalities based on their features and characteristics.

## Table of Contents
- [Functionality Commands](#functionality-commands)
- [Interacting Models](#interacting-models)
- [Quadratic Models](#quadratic-models)
- [Model Parameters](#model-parameters)
- [Lattice Choice](#lattice-choice)
- [Symmetry Parameters](#symmetry-parameters)
- [NQS Parameters](#nqs-parameters)
- [Directory and Weight Loading](#directory-and-weight-loading)

## Functionality Commands

### General Functionality
- `10 - NQS`: Implements functionalities related to Quantum Neural States, including training, evaluating, and measuring quantum states.
- `20 - Symmetry Testing`: Test the symmetries in interacting models.
  - `21`: Run a single interacting Hamiltonian with or without symmetries.
  - `22`: Sweep all the symmetries for a specified Hamiltonian.
  - `23`: Save the Hilbert space sizes for a specific Hamiltonian and its symmetries.
  - `24`: Save the eigenstates of a Hamiltonian in a specific symmetry sector or without the resolved symmetries.
  - `25`: Save the degeneracies of a specific Hamiltonian.
  - `26`: Save and create the degeneracies of a specific Hamiltonian (with or without symmetries).

### Quadratic Models Functionality
- `30`: Run the simulation for entropies of a quadratic Hamiltonian.
- `31`: Run the simulation for the spectral functions of the quadratic Hamiltonians.

### QSM ETH Simulations
- `40`: Run the QSM ETH simulation - Ljubljana 2024.
- `41`: Run the QSM ETH simulation - sweep parameters - Ljubljana 2024.
- `42`: Run the QSM ETH simulation - statistics - Ljubljana 2024.
- `43`: Run the QSM ETH simulation - statistics - sweep parameters - Ljubljana 2024.
- `45`: Run the QSM ETH simulation - statistics - sweep time evolution - Ljubljana 2024.
- `46`: Run the QSM ETH simulation - statistics - time evolution - Ljubljana 2024.

## Interacting Models

Interacting models describe systems where particles or spins interact with each other, leading to complex quantum phenomena. These models are essential for studying phase transitions, quantum entanglement, and other fundamental aspects of quantum mechanics.

- **1 - Transverse Ising Model**: A fundamental model in quantum magnetism characterized by transverse magnetic fields and interactions between neighboring spins. This model exhibits a quantum phase transition driven by the strength of the transverse field.
  
- **2 - XYZ Model**: A generalization of the Heisenberg model that includes anisotropic interactions between spins in three dimensions (x, y, z directions). This model is crucial for understanding various magnetic phases and transitions.

- **3 - Heisenberg-Kitaev Model**: A hybrid model that combines the Heisenberg interaction with Kitaev's fractionalization, leading to exotic quantum phases such as spin liquids. It is particularly relevant in the context of topological order and quantum computing.

- **4 - Quantum Statistical Mechanics (QSM)**: A framework that studies the statistical properties of quantum systems, focusing on understanding thermalization, ergodicity, and the emergence of classical behavior in quantum systems.

- **5 - Rosenzweig-Porter Model**: A model that examines the effects of randomness on quantum states, often associated with the study of quantum chaos. It investigates the transition from localized to delocalized states in disordered systems.

- **6 - Ultrametric Model**: A model characterized by an ultrametric distance structure, which influences the distribution of eigenstates. This model is useful for studying complex systems with hierarchical structures.

## Quadratic Models

Quadratic models represent systems where interactions are captured by quadratic terms in the Hamiltonian. They often yield analytical solutions and provide insight into non-interacting and weakly interacting particles.

- **100 - FF Model**: A free fermion model where particles do not interact, serving as a baseline for more complex systems. It is instrumental in understanding fermionic statistics and quantum transport.

- **101 - Aubry-André Model**: A model that exhibits localization phenomena due to aperiodic potentials. This model is significant in studying the interplay between disorder and quantum interference.

- **102 - SYK2 Model**: The Sachdev-Ye-Kitaev model, a prototype of a quantum chaotic system, demonstrating maximal chaos and a connection to holography. It provides insights into black hole thermodynamics and quantum gravity.

- **103 - Anderson Model**: A model that describes the localization of electrons in a disordered medium. It is crucial for understanding electron transport in materials and the metal-insulator transition.

- **104 - Power Law Random Bandwidth Model**: A model that examines the effects of power-law distributions on quantum states. It is particularly relevant for systems with long-range interactions and critical behavior.

## Model Parameters

This section specifies parameters relevant to each model, such as interaction strengths, randomness, and other model-specific variables.

### General Parameters
- `q_R`: Number of realizations of the mixed (unmixed) states.
- `q_CN`: Number of combinations to choose random states from (applicable when the number of sites > 22).
- `q_gamma`: Number of states to combine together.
- `q_man`: Use degenerate manifold? (Currently applies to FF model).
- `q_mb`: Calculate many-body states (currently only for Slater determinants).

### Example Model Parameters
- **Transverse Ising Model**:
  - `hx`: Transverse field strength.
  - `hz`: Longitudinal field strength.
  - `J1`, `J2`: Coupling constants.

- **Heisenberg-Kitaev Model**:
  - `heiJ`: Coupling strength.
  - `heiHz`: Longitudinal field strength.
  - `Kx`, `Ky`, `Kz`: Kitaev interaction parameters.

- **QSM Parameters**:
  - `qsm_alpha`: Parameter controlling the spectral density.
  - `qsm_xi`: Localization length.
  - `qsm_h`: External field values.

## Lattice Choice

Lattice parameters define the geometry of the system under consideration, influencing the interaction and propagation of quantum states.

- `-bc 0`: Boundary conditions (0 = periodic).
- `-Lx`, `-Ly`, `-Lz`: Dimensions of the lattice.
- `-l`: Lattice types.
  - `0`: Square lattice.
  - `1`: Hexagonal lattice.

## Symmetry Parameters

Symmetry parameters define the symmetry operations applied to the models, which can simplify calculations and reveal conserved quantities.

- `-S`: Define the symmetry type (e.g., U(1), Z_2).
- `-U1`: Define the U(1) symmetry parameter, associated with global phase invariance.

### Symmetry Descriptions
- **U(1) Symmetry**: This symmetry involves invariance under a global phase shift, commonly found in systems with conserved particle number.
  
- **Z_2 Symmetry**: A discrete symmetry that corresponds to flipping certain degrees of freedom, often relevant in models with order-disorder transitions.

## Neural Quantum States (NQS)

## Neural Quantum States (NQS)

Neural Quantum States (NQS) represent a powerful framework for modeling quantum many-body systems using neural networks. They allow for the variational approximation of quantum states, making them particularly suitable for ground state and thermal state calculations.

- **Types of NQS**:
  - `0 - RBM`: Restricted Boltzmann Machine, a type of neural network suited for capturing complex correlations in quantum states.
  - `1 - RBM+PP`: An extension of RBM incorporating Projected Phase (PP) techniques to enhance the expressiveness of the model.

### Training Parameters
- `nqs_tr_epo`: Number of epochs for training the NQS.
- `nqs_tr_mc`: Number of Monte Carlo steps per epoch.
- `nqs_tr_bs`: Block size for training; defines the number of sweeps within a block.
- `nqs_tr_th`: Threshold for optimization.
- `nqs_tr_pinv`: Threshold for the pseudo-inverse (if negative, no pseudo-inverse is applied).
- `nqs_tr_reg`: Regularization parameter to prevent overfitting.
- `nqs_tr_pc`: Percentage of data to be used during training.

### Excited State Parameters
- `nqs_ex_mc`: Number of Monte Carlo steps for excited state calculations.
- `nqs_ex_bn`: Number of blocks used for excited state sampling.
- `nqs_ex_th`: Threshold for excited state optimization.
- `nqs_ex_bs`: Block size for excited state calculations.
- `nqs_ex_beta`: List of beta values for the excited states.

### Collection Parameters
- `nqs_col_mc`: Number of Monte Carlo steps for collecting data.
- `nqs_col_bn`: Number of blocks for collecting observables.
- `nqs_col_th`: Threshold for data collection.
- `nqs_col_bs`: Block size for collection processes.

### Learning Rate Scheduler
- `nqs_lr`: Initial learning rate for the optimizer.
- `nqs_lrd`: Learning rate decay factor.
- `nqs_lr_pat`: Patience for learning rate reduction.
- `nqs_sch`: Type of learning rate scheduler:
  - `0`: No momentum (constant learning rate)
  - `1`: Exponential decay
  - `2`: Step decay
  - `3`: Cosine decay
  - `4`: Adaptive learning rate

### Early Stopping Parameters
- `nqs_es_pat`: Patience for early stopping; how many epochs to wait before stopping if there is no improvement.
- `nqs_es_del`: Minimum change required to consider an improvement.

### Hidden Units Configuration
- `nqs_nh`: Number of hidden units in the NQS (if negative, it acts as a multiplier of the visible units).

### Load Weights
Load pre-trained weights for the NQS using the following format:
-lNQS [path_to_weights]

## Directory and Weight Loading

### Directory Configuration
- `-dir`: Specifies the directory for data storage and retrieval during simulations. Example:
  - `-dir ./RANDOM_MODELS_STRUCTURE`: Custom path for random model structures.
  - `-dir /media/klimak/ObiTwo/DATA_NQS`: Path for NQS data storage.
  - `-dir ./`: Current directory for output files.


