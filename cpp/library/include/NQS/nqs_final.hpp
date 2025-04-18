/***********************************************
* @file nqs_final.hpp
* @brief Final header file for the NQS class, defining the generic NQS Solver and its supporting methods for solving neural quantum states.
* 
* ## Overview:
* 
* The NQS (Neural Quantum States) class provides a general framework for implementing neural network-based quantum eigensolvers.
* This class is designed to be highly flexible, allowing for specific NQS models to inherit from it and customize various operations
* related to training, energy estimation, and state evolution. The class incorporates methods for handling state flips, weight updates,
* and the calculation of transition probabilities in the context of quantum Monte Carlo sampling.
*
* This implementation relies heavily on Armadillo for matrix operations and can be extended for use with multiple spin modes and Hamiltonian
* representations. Additionally, the class supports multi-threading for more efficient computations during training. This approach
* is well-suited for variational quantum algorithms and provides a modular framework for future extensions.
*
* ## Key Features:
* - Supports multi-spin models (e.g., hardcore bosons, fermions) - Work in progress (WIP).
* - Provides a consistent interface for computing probabilities, variational derivatives, and energy estimations.
* - Can be extended to incorporate stochastic reconfiguration (SR) techniques for optimizing neural network weights.
* - Includes methods for both random and deterministic flips of quantum states.
* - Implements threading for parallel computations - leveraging OpenMP and a thread-safe pool.
*
* ## Usage:
* This class serves as the foundation for developing specific NQS models. A derived class should implement the abstract methods that
* are model-specific, such as probability ratios (`pRatio`) and state transitions. The class also provides high-level functions for
* training and energy estimation.
*
* ### `train` function:
* The `train` function is used to perform a single training iteration of the Neural Quantum State (NQS) with parameters controlling the Monte Carlo steps, blocks, and sampling specifics.
*
* #### Parameters:
* - `mcSteps`   : Total number of Monte Carlo steps (iterations) to be executed during training. It serves as an outer loop that determines the number of weight updates.
* - `nThrm`     : Number of Monte Carlo steps dedicated to thermalizing the system (burn-in period).
* - `nBlck`     : Number of blocks used to compute the covariance matrix. It affects the gradient calculation for each update.
* - `bSize`     : Size of a block, which determines the number of sampling steps within each block.
* - `nFlip`     : Number of state flips per update. By default, it's set to 1.
* - `quiet`     : Boolean flag to suppress output. When set to `false`, the training process displays progress.
* - `_t`        : Timestamp for logging purposes.
* - `progPrc`   : Progress percentage for displaying a progress bar.
*
* #### Return:
* The function returns an `arma::Col<_T>` containing the mean energies obtained throughout the Monte Carlo steps.
*
* #### Example:
* ```cpp
* MonteCarlo::MCS_train_t params = {1000, 100, 10, 50, "training_data", 1}; // example parameters
* auto meanEnergies = nqs.train(params, false, std::chrono::system_clock::now(), 25);
* ```
*
* ### Setup:
* One sets up and trains the NQS for both the ground and excited states, initializes the system with 
* a Hamiltonian (e.g., Ising model) and trains multiple NQS instances.
*
* #### Key Operations:
* - **Hamiltonian Setup**: (not necessary) The Hilbert space is initialized, and the Hamiltonian is constructed, such as an Ising model.
* - **Exact Diagonalization (ED)**: The Hamiltonian is diagonalized to compare energies against NQS results.
* - **Ground State Training**: The ground state NQS is trained using the `train` function.
* - **Excited State Training**: A secondary NQS is trained for the excited state by initializing with the trained ground state weights.

* ## References:
* - **Armadillo C++ Library**: A high-performance linear algebra library utilized for matrix operations throughout the codebase. For more details, visit [Armadillo website](http://arma.sourceforge.net).
* - **Neural Network Quantum States**: A modern approach to quantum many-body problems using neural networks. This concept is introduced in the paper by Carleo and Troyer (2017): 
*   *Giuseppe Carleo, Matthias Troyer, “Solving the quantum many-body problem with artificial neural networks,” Science, 355(6325), 2017.* 
*   Available at [Science](https://www.science.org/doi/10.1126/science.aag2302).
* - **Quantum Monte Carlo Methods**: Used in the training of NQS to sample quantum states. For background on Monte Carlo techniques in physics, refer to:
*   *D. Ceperley and M. Kalos, “Quantum Monte Carlo Methods,” in Monte Carlo Methods in Statistical Physics, Topics in Current Physics, Springer, 1986.*
*
* December 2023. Under continuous development.
* Author: Maksymilian Kliczkowski, WUST, Poland.
************************************************/

#include "./NQS_base/nqs_training.tpp"

// ##########################################################################################################################################