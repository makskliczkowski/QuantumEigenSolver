Usage
=====

After installation, you can run the QuantumEigenSolver as follows:

.. code-block:: bash

    python main.py --options

Replace `--options` with the appropriate command-line arguments as described below.

**Command-Line Arguments:**

- `--N` (int): Number of SU(N) components of the fermions. (default = 3)
- `--Np` (list of ints): Number of spins per color. (default = 1 per color)
- `--L` (int): Number of sites of the instance. (default = 3)
- `--t` (float or list of floats): Constants for the hopping terms. (default = [1.0])
- `--U` (float): On-site interaction. (default = 1.0)
- `--V` (float or list of floats): Interactions between fermions at distance > 0. (default = [0.0])
- `--mu` (float): Chemical potential.
- `--layers` (int): Number of maximum layers of the ansatz for the VQE implementation. (default = 3)
- `--phi_max` (float): Maximum flux value. (default = 0.5)
- `--phi_num` (int): Number of flux values to consider from 0 to phi_max. (default = 25)
- `--backend` (str): Qibo backend to use for the computation. (default = "qibotf")
- `--exact`: Include this flag to get values using exact diagonalization.
- `--perturb`: Include this flag to apply a small perturbation to the last best parameters of the VQE before the next execution.
- `--open_chain`: Include this flag to switch to an open chain rather than a closed one.

**Example:**

.. code-block:: bash

    python main.py --N 3 --L 4 --U 2.0 --layers 5
