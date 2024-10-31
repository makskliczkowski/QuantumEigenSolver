# Hamiltonian Solver

This is the general solver for lattice Hamiltonians. The works is in progress but currently it includes:
- ED
- ED with symmetries
- the variational Quantum Monte Carlo solver for ansatz ground state Jastrov functions

The plan for the future is to extend it to either more than 1D Hamitlonians in the case of symmetries or add symmetries to the variational ansatz potentially 
in higher dimensions and non-spin Hamiltonians.

# Configuration Parameters for Quantum Simulation Models

This document outlines the configuration parameters for running simulations of various quantum models using a specified command structure. The commands allow for flexibility in exploring different models, settings, and properties relevant to quantum simulations. Below is a comprehensive description of the command structure and their associated parameters.

## Table of Contents

1. [Functionality Commands](#functionality-commands)
2. [Model Choice](#model-choice)
3. [Lattice Choice](#lattice-choice)
4. [Interacting Model Parameters](#interacting-model-parameters)
5. [Symmetry Parameters](#symmetry-parameters)
6. [Quadratic Parameters](#quadratic-parameters)
7. [Neural Quantum States (NQS) Configuration](#neural-quantum-states-nqs-configuration)
8. [Early Stopping Configuration](#early-stopping-configuration)
9. [Directory Management](#directory-management)
10. [Command Usage Examples](#command-usage-examples)

## Functionality Commands

The following commands specify the type of simulation and the operations to perform:

- **10 - NQS**: Configure settings related to Neural Quantum States (NQS).
- **20 - Test Symmetries**: Evaluate symmetries in interacting models.
- **21 - Single Hamiltonian**: Run a single interacting Hamiltonian with or without symmetries.
- **22 - Sweep Symmetries**: Explore all symmetries for a specified Hamiltonian.
- **23 - Hilbert Space Sizes**: Save the Hilbert space sizes for a specific Hamiltonian and its symmetries.
- **24 - Save Eigenstates**: Store the eigenstates of a Hamiltonian in a specific symmetry sector or without resolved symmetries.
- **25 - Save Degeneracies**: Save the degeneracies of a specific Hamiltonian.
- **26 - Create Degeneracies**: Generate and save degeneracies of a specific Hamiltonian, with or without symmetries.
- **30 - Entropy Simulation**: Run a simulation to calculate entropies of a quadratic Hamiltonian.
- **31 - Spectral Functions Simulation**: Execute simulations for spectral functions of quadratic Hamiltonians.
- **40-46 - QSM ETH Simulation**: Execute various simulations related to Quantum Statistical Mechanics (QSM) and Eigenstate Thermalization Hypothesis (ETH) with parameter sweeps and statistics. 

## Model Choice

Specify the model type for the simulation:

### Interacting Models
1. **Transverse Ising**: Model index 1.
2. **XYZ**: Model index 2.
3. **Heisenberg-Kitaev**: Model index 3.
4. **Quantum Statistical Mechanics (QSM)**: Model index 4.
5. **Rosenzweig-Porter**: Model index 5.
6. **Ultrametric**: Model index 6.

### Quadratic Models
100. **FF**: Model index 100.
101. **Aubry-André**: Model index 101.
102. **SYK2**: Model index 102.
103. **Anderson**: Model index 103.
104. **Power Law Random Bandwidth**: Model index 104.

**Randomness Configuration**:
- **q_R**: Number of realizations of mixed/unmixed states.
- **q_CN**: Number of combinations from which to choose random states (only applies if the number of sites > 22).
- **q_gamma**: Number of states to combine together.
- **q_man**: Use degenerate manifold (currently applies to FF).
- **q_mb**: Calculate many-body state (currently Slater determinant only).

## Lattice Choice

Specify the lattice configuration:

- **Boundary Conditions (-bc)**: 
  - `0`: Open boundary conditions.
  - (Other options can be specified as required.)

- **Lattice Dimensions**:
  - **Lx**: Length in the x-dimension (e.g., `10`).
  - **Ly**: Length in the y-dimension (e.g., `1`).
  - **Lz**: Length in the z-dimension (e.g., `1`).

- **Lattice Types (-l)**:
  - `0`: Square lattice.
  - `1`: Hexagonal lattice.

- **Total Sites (-Ntot)**: The total number of sites in the system. Various configurations can be defined.

## Interacting Model Parameters

This section defines parameters specific to the chosen interacting model:

- **hx**: Field strength in the x-direction (e.g., `-0.5`).
- **hz**: Field strength in the z-direction (e.g., `-1.0`).
- **J1**, **J2**: Coupling constants.
- Additional parameters specific to the Heisenberg-Kitaev model, QSM, and other interacting models as required.

## Symmetry Parameters

Define parameters related to the symmetries of the model:

- **S**: Specific symmetry configurations.
- **U1**: Additional symmetry parameters.

## Quadratic Parameters

Parameters that apply to quadratic models:

- **q_R**: Number of realizations.
- **q_CN**: Number of combinations.
- **q_gamma**: States to combine.
- **q_man**: Use degenerate manifold.
- **q_mb**: Many-body state calculation.
- **aa_lambda**: Aubry-André model parameters.

## Neural Quantum States (NQS) Configuration

Parameters related to the configuration of Neural Quantum States:

- **nqst**: Type of NQS (e.g., RBM, RBM+PP).
- **Training Parameters**:
  - **nqs_tr_epo**: Number of epochs.
  - **nqs_tr_mc**: Monte Carlo steps.
  - **nqs_tr_bs**: Block size.
  - **nqs_tr_th**: Threshold for training.
  - **nqs_tr_reg**: Regularization parameter.

- **Excited State Parameters**:
  - **nqs_ex_mc**: Number of Monte Carlo steps for excited states.
  - **nqs_ex_bn**: Number of blocks.
  - **nqs_ex_beta**: Temperature for excited states.

- **Collecting Data**:
  - **nqs_col_mc**: Monte Carlo steps for data collection.
  - **nqs_col_bn**: Number of blocks for data collection.

## Early Stopping Configuration

To prevent overfitting during training:

- **nqs_es_pat**: Patience for early stopping.
- **nqs_es_del**: Minimum delta for improvement.

## Directory Management

Configuration for managing input and output directories:

- **Directory Paths**:
  - **-dir**: Set the output directory for results.
  
## Command Usage Examples

Here are a few examples of how to use the commands effectively:

1. **Run a single interacting Hamiltonian without symmetries**:
   ```bash
   -fun 21
   -mod 2
   -th 1

Copyright 2022 <br> 
Maksymilian Kliczkowski <br> 
PhD candidate <br> 
Wroclaw University of Science and Technology <br> 
maksymilian.kliczkowski.at.pwr.edu.pl <br> 
