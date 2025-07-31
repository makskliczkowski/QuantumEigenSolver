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

# ------------------------------------------------------------------------------------
#! SCHMIDT RANK LOSS FUNCTIONS
# ------------------------------------------------------------------------------------

def loss_schmidt_rank_unitary(c: np.array, prepared_states: np.array, dim_a: int, dim_b: int) -> float:
    """
    Computes the Schmidt rank of a mixed state defined by coefficients and prepared states.
    
    Parameters:
        c               : Coefficients for the prepared states.
        prepared_states : Prepared states as a matrix.
        dim_a           : Dimension of subsystem A.
        dim_b           : Dimension of subsystem B.
    Returns:
        float           : Schmidt rank of the mixed state.
    """
    # if c is a matrix, compute the Schmidt rank for each column
    ranks       = []
    mixed_state = prepared_states @ c
    for state in mixed_state.T:
        normalized_state    = normalize_state(state)
        schmidt_values      = schmidt_numpy(normalized_state, dim_a, dim_b, eig=False)[0]
        ranks.append(np.count_nonzero(schmidt_values > 1e-10))
    return np.sum(ranks)

def loss_schmidt_rank_single(c: np.array, prepared_states: np.array, dim_a: int, dim_b: int) -> float:
    """
    Computes the Schmidt rank of a mixed state defined by coefficients and prepared states.
    
    Parameters:
        c               : Coefficients for the prepared states.
        prepared_states : Prepared states as a matrix.
        dim_a           : Dimension of subsystem A.
        dim_b           : Dimension of subsystem B.
    Returns:
        float           : Schmidt rank of the mixed state.
    """
    # compute mixed state vector consisting of \sum _i c_i |psi_i>
    mixed_state         = prepared_states @ c
    # normalize the mixed state
    normalized_state    = normalize_state(mixed_state)
    # compute the Schmidt decomposition
    schmidt_values      = schmidt_numpy(normalized_state, dim_a, dim_b, eig=False)[0]
    return np.count_nonzero(schmidt_values > 1e-10)  # Count non-zero singular values

def loss_schmidt_rank_states(states: np.array, dim_a: int, dim_b: int) -> float:
    """
    Computes the Schmidt rank for a set of states.
    
    Parameters:
        states  : Array of states (shape: [n_states, state_dim]).
        dim_a   : Dimension of subsystem A.
        dim_b   : Dimension of subsystem B.
    Returns:
        float   : Total Schmidt rank of the states.
    """
    if states.ndim == 1:
        return np.count_nonzero(schmidt_numpy(states, dim_a, dim_b, eig=False)[0] > 1e-10)
    
    ranks = []
    for state in states.T:
        schmidt_values = schmidt_numpy(state, dim_a, dim_b, eig=False)[0]
        ranks.append(np.count_nonzero(schmidt_values > 1e-10))
    return np.sum(ranks)

# ------------------------------------------------------------------------------------
#! PURITY LOSS FUNCTIONS
# ------------------------------------------------------------------------------------

def loss_purity_subsystem_unitary(c: np.array, prepared_states: np.array, dim_a: int, dim_b: int) -> float:
    # if c is a matrix, compute the purity for each column
    purities            = []
    mixed_state         = prepared_states @ c
    for state in mixed_state.T:
        normalized_state    = normalize_state(state)
        rho_a               = rho_numpy(normalized_state, dim_a, dim_b, eig=False)[0]
        rho_b               = rho_numpy(normalized_state, dim_b, dim_a, eig=False)[0]
        purity_a            = np.trace(rho_a @ rho_a)
        purity_b            = np.trace(rho_b @ rho_b)
        purities.append(1 - (purity_a + purity_b) / 2)
    return np.sum(purities)

def loss_purity_subsystem_single(c: np.array, prepared_states: np.array, dim_a: int, dim_b: int) -> float:
    # if c is a vector, compute the mixed state
    mixed_state         = prepared_states @ c
    normalized_state    = normalize_state(mixed_state)
    rho_a               = rho_numpy(normalized_state, dim_a, dim_b, eig=False)[0]
    rho_b               = rho_numpy(normalized_state, dim_b, dim_a, eig=False)[0]
    purity_a            = np.trace(rho_a @ rho_a)
    purity_b            = np.trace(rho_b @ rho_b)
    return 1 - (purity_a + purity_b) / 2

def loss_purity_states(states: np.array, dim_a: int, dim_b: int) -> float:
    """
    Computes the purity of a set of states.
    
    Parameters:
        states  : Array of states (shape: [n_states, state_dim]).
        dim_a   : Dimension of subsystem A.
        dim_b   : Dimension of subsystem B.
    Returns:
        float   : Purity of the states.
    """
    
    if states.ndim == 1:
        rho_a = rho_numpy(states, dim_a, dim_b, eig=False)[0]
        rho_b = rho_numpy(states, dim_b, dim_a, eig=False)[0]
        purity_a = np.trace(rho_a @ rho_a)
        purity_b = np.trace(rho_b @ rho_b)
        return 1 - (purity_a + purity_b) / 2
    
    purities = []
    for state in states.T:
        rho_a = rho_numpy(state, dim_a, dim_b, eig=False)[0]
        rho_b = rho_numpy(state, dim_b, dim_a, eig=False)[0]
        purity_a = np.trace(rho_a @ rho_a)
        purity_b = np.trace(rho_b @ rho_b)
        purities.append(1 - (purity_a + purity_b) / 2)
    return np.sum(purities)

# ------------------------------------------------------------------------------------
#! ENTANGLEMENT LOSS FUNCTIONS
# ------------------------------------------------------------------------------------

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
    # compute the entanglement entropy
    schmidt_values      = schmidt_numpy(normalized_state, dim_a, dim_b, eig=False)[0]
    return vn_entropy(schmidt_values)

def loss_entanglement_states(states: np.array, dim_a: int, dim_b: int) -> np.array:
    """
    Computes the entanglement entropy for a set of states.
    
    Parameters:
        states  : Array of states (shape: [n_states, state_dim]).
        dim_a   : Dimension of subsystem A.
        dim_b   : Dimension of subsystem B.
    Returns:
        np.array: Entanglement entropy for each state.
    """
    if states.ndim == 1:
        return np.array([vn_entropy(schmidt_numpy(states, dim_a, dim_b, eig=False)[0])])
    
    entropies = []
    for state in states.T:
        schmidt_values = schmidt_numpy(state, dim_a, dim_b, eig=False)[0]
        entropies.append(vn_entropy(schmidt_values))
    return np.array(entropies)

# ------------------------------------------------------------------------------------
#! SIMPLE FERMIONIC OPERATORS
# ------------------------------------------------------------------------------------

def apply_annihilation(psi: np.ndarray, l: int, Ns: int) -> np.ndarray:
    """
    Apply fermionic annihilation operator c_l to state vector psi (in Fock basis) - binary representation.
    
    Parameters:
        psi : np.ndarray
            State vector in Fock basis.
        l   : int
            Mode index to apply the annihilation operator on, counted from left (0-indexed) to right.
    For example, if l=2, Ns=4, the state vector is 0b'1100' (binary representation of 12),
    the l'th bit is 1 (particle present), and the result will be 0b'1000' (binary representation of 8) times the sign and scaled by the amplitude.
    """
    assert 0 <= l < Ns, "Mode index l must be in range [0, Ns-1]"
    
    result  = np.zeros_like(psi)
    mask    = 1 << (Ns - 1 - l)
    shift   = Ns - l
    
    for idx in np.flatnonzero(psi):
        if idx & mask:                          # n_l == 1 ?
            j           = idx ^ mask            # flip that bit off
            parity      = (idx >> shift).bit_count() & 1
            sign        = -1 if parity else 1
            result[j]  += sign * psi[idx]
    return result

def apply_creation(psi: np.ndarray, l: int, Ns: int) -> np.ndarray:
    """
    Apply fermionic creation operator c_l^† to state vector psi (in Fock basis) - binary representation.

    Parameters:
        psi : np.ndarray
            State vector in Fock basis.
        l   : int
            Mode index to apply the creation operator on, counted from left (0-indexed) to right.
    """
    assert 0 <= l < Ns, "Mode index l must be in range [0, Ns-1]"
    result  = np.zeros_like(psi)
    mask    = 1 << (Ns - 1 - l)
    shift   = Ns - l
    for idx in np.flatnonzero(psi):
        if not (idx & mask):                     # n_l == 0 ?
            j           = idx ^ mask              # flip that bit on
            parity      = (idx >> shift).bit_count() & 1
            sign        = -1 if parity else 1
            result[j]  += sign * psi[idx]
    return result

def correlation_matrix(psi: np.ndarray, Ns: int) -> np.ndarray:
    """
    Compute the single-particle correlation matrix C_{ll'} = <psi| c^\u202Fdagger_l c_l' |psi>
    for a fermionic state vector psi of length 2^n_modes.
    """
    C = np.zeros((Ns, Ns), dtype=psi.dtype)
    for lp in range(Ns):
        psi1 = apply_annihilation(psi, lp, Ns)
        for l in range(Ns):
            psi2 = apply_creation(psi1, l, Ns)
            C[l, lp] = np.vdot(psi, psi2)
    return C

def loss_nongaussianity_unitary(unitary: np.ndarray, prepared_states: np.ndarray, Ns: int) -> float:
    r"""
    Computes the non-Gaussianity of a mixed state defined by coefficients and prepared states.
    
    Parameters:
        unitary         : Unitary operator applied to the prepared states.
        prepared_states : Prepared states as a matrix.
        Ns              : Number of modes (subsystems).
    Returns:
        float           : Non-Gaussianity of the mixed state.
    """
    new_states          = prepared_states @ unitary
    nongaussianities    = []
    for i in range(new_states.shape[1]):
        new_states[:, i] = normalize_state(new_states[:, i])
        # Compute the correlation matrix
        C = correlation_matrix(new_states, Ns)
        # Measure non-Gaussianity as deviation from identity matrix
        # diagonalize it and calculate the function \sum (1+lambda_i)/2 * log(1+lambda_i/2) + (1-lambda_i)/2 * log(1-lambda_i/2)
        # where lambda_i are the eigenvalues of C
        eigenvalues = np.linalg.eigvalsh(C)
        # get the nonzero eigenvalues
        nonzero_eigenvalues = eigenvalues[eigenvalues > 1e-10]
        if nonzero_eigenvalues.size == 0:
            return 0.0
        # calculate the non-Gaussianity
        for lambda_i in nonzero_eigenvalues:
            if lambda_i > 0:
                nongaussianities.append(np.sum((1 + lambda_i) / 2 * np.log(1 + lambda_i / 2) + (1 - lambda_i) / 2 * np.log(1 - lambda_i / 2)))
    return -np.mean(nongaussianities) # Average over all states

def loss_nongaussianity_single(coefficients: np.ndarray, prepared_states: np.ndarray, Ns: int) -> float:
    r"""
    Computes the non-Gaussianity of a mixed state defined by coefficients and prepared states.
    
    Parameters:
        coefficients    : Coefficients for the prepared states.
        prepared_states : Prepared states as a matrix.
        Ns              : Number of modes (subsystems).
    Returns:
        float           : Non-Gaussianity of the mixed state.
    """
    # compute mixed state vector consisting of \sum _i c_i |psi_i>
    mixed_state         = prepared_states @ coefficients
    # normalize the mixed state
    normalized_state    = normalize_state(mixed_state)
    # Compute the correlation matrix
    C = correlation_matrix(normalized_state, Ns)
    # Measure non-Gaussianity as deviation from identity matrix
    eigenvalues = np.linalg.eigvalsh(C)
    # get the nonzero eigenvalues
    nonzero_eigenvalues = eigenvalues[eigenvalues > 1e-10]
    if nonzero_eigenvalues.size == 0:
        return 0.0
    # calculate the non-Gaussianity
    nongaussianities = []
    for lambda_i in nonzero_eigenvalues:
        if lambda_i > 0:
            nongaussianities.append(np.sum((1 + lambda_i) / 2 * np.log(1 + lambda_i / 2) + (1 - lambda_i) / 2 * np.log(1 - lambda_i / 2)))
    return -np.mean(nongaussianities) if nongaussianities else 0.0

def loss_nongaussianity_states(states: np.ndarray, Ns: int) -> float:
    r"""
    Computes the non-Gaussianity for a set of states.
    We define non-Gaussianity as the average of the non-Gaussianity of each state.

    NG_i(psi) = -\sum (1+lambda_i)/2 * log(1+lambda_i/2) - (1-lambda_i)/2 * log(1-lambda_i/2)
    where lambda_i are the eigenvalues of the correlation matrix C_{ll'} = <psi| c^\dag_l c_l' |psi>
    
    Parameters:
        states  : Array of states (shape: [n_states, state_dim]).
        Ns      : Number of modes (subsystems).
    Parameters:
        states  : Array of states (shape: [n_states, state_dim]).
        Ns      : Number of modes (subsystems).
    Returns:
        float   : Non-Gaussianity of the states.
    """
    if states.ndim == 1:
        C                   = correlation_matrix(states, Ns)
        eigenvalues         = np.linalg.eigvalsh(C)
        nonzero_eigenvalues = eigenvalues[eigenvalues > 1e-10]
        if nonzero_eigenvalues.size == 0:
            return 0.0
        nongaussianity      = 0.0
        for lambda_i in nonzero_eigenvalues:
            if lambda_i > 0:
                nongaussianity += np.sum((1 + lambda_i) / 2 * np.log(1 + lambda_i / 2) + (1 - lambda_i) / 2 * np.log(1 - lambda_i / 2))
        return -nongaussianity / nonzero_eigenvalues.size if nonzero_eigenvalues.size > 0 else 0.0
    nongaussianities = []
    for state in states.T:
        C = correlation_matrix(state, Ns)
        eigenvalues = np.linalg.eigvalsh(C)
        nonzero_eigenvalues = eigenvalues[eigenvalues > 1e-10]
        if nonzero_eigenvalues.size == 0:
            continue
        nongaussianity = 0.0
        for lambda_i in nonzero_eigenvalues:
            if lambda_i > 0:
                nongaussianity += np.sum((1 + lambda_i) / 2 * np.log(1 + lambda_i / 2) + (1 - lambda_i) / 2 * np.log(1 - lambda_i / 2))
        nongaussianities.append(nongaussianity / nonzero_eigenvalues.size if nonzero_eigenvalues.size > 0 else 0.0)
    return -np.mean(nongaussianities) if nongaussianities else 0.0

# ------------------------------------------------------------------------------------
#! EOF
# ------------------------------------------------------------------------------------