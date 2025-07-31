import os, sys
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
for i in range(1, 5): # Check up to 4 levels up
    dir_up = os.path.abspath(os.path.join(script_dir, *(['..'] * i)))
    if dir_up not in sys.path:
        sys.path.append(dir_up)
    sys.path.append(os.path.join(dir_up, "QES", "general_python"))

try:
    import jax
    import jax.numpy as jnp
    import gymnasium as gym
    # for Flax
    import optax
    import flax.linen as nn
    from flax.training.train_state import TrainState
    from flax.linen.initializers import orthogonal
except ImportError as e:
    print("Please install the required packages: jax, jaxlib, gymnasium, optax, flax")
    sys.exit(1)
    
# ------------------------------------------------------------------------------------

try:
    from QES.general_python.physics.density_matrix import rho_numpy, schmidt_numpy
    from QES.general_python.physics.entropy import vn_entropy
except ImportError as e:
    print("Please ensure the QES package is correctly installed and accessible.")
    sys.exit(1)

# ------------------------------------------------------------------------------------

def normalize_state(c):
    return c / np.linalg.norm(c)

def loss_entanglement_unitary(unitary: np.array, prepared_states: np.array, dim_a: int, dim_b: int) -> np.array:
    new_states          = prepared_states @ unitary
    dim_a               = dim_a if isinstance(dim_a, int) else dim_a[0]
    dim_b               = dim_b if isinstance(dim_b, int) else dim_b[0]
    schmidt_values      = np.array([schmidt_numpy(state, dim_a, dim_b, use_eig=False)[0] for state in new_states.T])
    entropies           = [vn_entropy(s) for s in schmidt_values]
    return np.array(entropies)

def loss_entanglement_single(coefficients: np.array, prepared_states: np.array, dim_a: int, dim_b: int) -> np.array:
    """
    Computes the entanglement entropy of a mixed state defined by coefficients and prepared states.
    
    Parameters:
        coefficients     : Coefficients for the prepared states.
        prepared_states  : Prepared states as a matrix.
        dim_a            : Dimension of subsystem A.
        dim_b            : Dimension of subsystem B.
    Returns:
        np.array         : Entanglement entropy of the mixed state.
    """
    # compute mixed state vector consisting of \sum _i c_i |psi_i>
    mixed_state         = prepared_states @ coefficients
    # normalize the mixed state
    normalized_state    = normalize_state(mixed_state)
    # compute the density matrix
    rho                  = rho_numpy(normalized_state, dim_a, dim_b)
    # compute the entanglement entropy
    schmidt_values      = schmidt_numpy(rho, dim_a, dim_b, use_eig=False)[0]
    return vn_entropy(schmidt_values)

# ------------------------------------------------------------------------------------