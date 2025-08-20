'''
Necessary bit of code to extract entanglement entropy from a single Slater determinant and 
many-body quantum states.
'''

import numpy as np
import numba
import math
from enum import Enum, unique

from typing import Sequence, Tuple, Optional
from functools import partial

_EPS                = 1e-10

try:
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax import jit, vmap
    JAX_AVAILABLE   = True
except ImportError:
    jax             = None
    jnp             = None
    lax             = None
    jit             = None
    vmap            = None
    JAX_AVAILABLE   = False

#############################################
#! Single particle correlation matrix
#############################################

# a) Numpy version

def corr_single(
    W_A                 : np.ndarray,   # shape (Ls, La)
    occ                 : np.ndarray,   # boolean / 0-1 vector, shape (Ls,)
    subtract_identity   : bool = True,
    W_A_CT              : Optional[np.ndarray] = None,
    raw                 : bool = True) -> np.ndarray:
    """
    Correlation matrix C_A of a single Slater determinant.

    Parameters
    ----------
    W_A  : ndarray (Ls, La)
        Rows = orbitals q, columns = real-space sites i\inA.
        Matrix of coefficients from the single particle Hamiltonian diagonalization.
    W_A_CT : ndarray (La, Ls), optional
        Conjugate transpose of W_A, if not provided it will be computed.
    raw  : bool, default True
        If True, return the raw correlation matrix C_A.
        If False, return the entanglement Hamiltonian correlation matrix C_A.
    occ  : ndarray (Ls,) of {0,1}
        Occupation numbers n_q of the determinant.
    subtract_identity : bool, default True
        If True, return `C_A - I` (frequently needed for ent-H calcs).

    Returns
    -------
    C    : ndarray (La, La)
    """
    # Ensure W_A_CT is provided or compute it as the conjugate transpose of W_A
    if W_A_CT is None:
        W_A_CT = W_A.conj().T  # (La, Ls)

    # Use occupation vector directly for raw mode, otherwise map to ±1 for entanglement Hamiltonian
    if raw:
        indices     =       occ
        Wp          =       W_A[indices, :] 
        W_prime     =       W_A_CT[:, indices]
        # C           =       2.0 * np.matmul(W_prime, Wp)
        C           =       2 * (W_prime @ Wp)  # (La, La)
    else:
        prefactors  =       2 * occ - 1 # maps 0→-1, 1→+1
        # Efficiently compute C = W_A\dag · diag(prefactors) · W_A without explicit diag
        # (La, Ls) * (Ls,) → (La, Ls), then @ (Ls, La) → (La, La)
        C           =       (W_A_CT * prefactors) @ W_A

    if subtract_identity:
        np.fill_diagonal(C, C.diagonal() - 1.0)
    return C

    from functools import partial
    import jax
    import jax.numpy as jnp

# b) JAX version

if JAX_AVAILABLE:
    
    @partial(jax.jit, static_argnums=(2,))
    def corr_single_jax(
        W_A                 : jnp.ndarray,  # (Ls, La)
        occ                 : jnp.ndarray,  # (Ls,)
        subtract_identity   : bool = True) -> jnp.ndarray:
        """
        Correlation matrix C_A of a single Slater determinant using JAX.
        """
        occ_f   = occ.astype(W_A.dtype)
        C       = (jnp.conjugate(W_A).T * occ_f) @ W_A
        if subtract_identity:
            C   = C - jnp.eye(W_A.shape[1], dtype=W_A.dtype)
        return C
else:
    # If JAX is not available, we will use the Numpy version of the correlation matrix function
    corr_single_jax = corr_single
    
#! Usage example for the correlation matrix function
# use as:
#   eigv, W_A               = np.linalg.eigh(H_A)       # where H_A is the single particle Hamiltonian
#   orb_4_mb_state_1_hot    = np.array([1, 0, 0, 0])    # example occupation vector
#   W_A_CT                  = W_A.conj().T              # conjugate transpose of W_A
#   correlation_matrix      = Corr.corr_single(W_A, orb_4_mb_state_1_hot, W_A_CT=W_A_CT, raw=True, subtract_identity=True)
#   corr_val, _             = np.linalg.eigh(correlation_matrix)
#   entropy_correlation     = Entropy.entropy(corr_val, typek=Entropy.Entanglement.SINGLE)

#############################################
#! Entanglement Entropy
#############################################

@numba.njit(cache=True)
def _clean_probs(p: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """Clip negatives & zeros from round-off, renormalise."""
    q   = np.where(p < eps, 0.0, p)
    s   = q.sum()
    return q / s if s != 1.0 else q

# ----------------------------------

@numba.njit(cache=True)
def vn_entropy(lam: np.ndarray, base: float = np.e) -> float:
    """
    Calculates the von Neumann entropy for a given probability distribution.

    Parameters
    ----------
    lam : np.ndarray
        np.ndarray of eigenvalues or probabilities. Should sum to 1 and contain non-negative values.
    base : float, optional
        The logarithm base to use for entropy calculation. Default is the natural logarithm (np.e).

    Returns
    -------
    float
        The von Neumann entropy of the probability distribution.

    Notes
    -----
    The function internally cleans the probability array to ensure numerical stability.
    A small constant (1e-30) is added to probabilities before taking the logarithm to avoid log(0).
    """
    lam     = _clean_probs(lam)
    log     = np.log(lam + 1e-30)
    if base != np.e:
        log /= np.log(base)
    return -np.dot(lam, log)

@numba.njit(cache=True)
def renyi_entropy(lam: np.ndarray, q: float, base: float = np.e, threshold: float = 1e-12) -> float:
    """
    Calculates the Rényi entropy of a probability distribution.

    The Rényi entropy is a generalization of the Shannon (von Neumann) entropy, parameterized by q.
    For q = 1, it reduces to the von Neumann entropy.

    Args:
        lam (np.ndarray):
            Probability distribution (array of probabilities or eigenvalues).
        q (float):
            Order of the Rényi entropy. Must be a positive real number.
        base (float, optional):
            Logarithm base to use. Default is the natural logarithm (np.e).

    Returns:
        float:
            The Rényi entropy of the distribution.

    Raises:
        ValueError: If q is not a positive real number.

    Notes:
        - If q == 1, the function returns the von Neumann entropy.
        - The input probabilities are cleaned to remove zeros and ensure normalization.
    """
    if q == 1.0:
        return vn_entropy(lam, base)
    lam     = _clean_probs(lam)
    s       = (lam ** q).sum()
    log     = np.log(s)
    
    if base != np.e:
        log /= np.log(base)
    return log / (1.0 - q)

@numba.njit(cache=True)
def sp_correlation_entropy(lam: np.ndarray, q: float, base: float = np.e):    
    """
    Compute the single-particle correlation entropy for a set of eigenvalues.
    This function calculates either the von Neumann entropy (for q=1) or the Rényi entropy (for generic q)
    of a set of eigenvalues `lam` (typically from a correlation matrix), after mapping each eigenvalue
    from the interval [-1, 1] to a probability in [0, 1].
    Parameters
    ----------
    lam : np.ndarray
        np.ndarray of eigenvalues (λ), each in the interval [-1, 1].
    q : float
        Entropy order parameter. If q == 1, computes the von Neumann entropy; otherwise, computes the Rényi entropy.
    base : float, optional
        The logarithm base to use (default is the natural logarithm, base e).
    Returns
    -------
    float
        The computed entropy value.
    Notes
    -----
    - For q == 1, the function computes the von Neumann entropy:
          S = -Σ [p * log(p) + (1-p) * log(1-p)]
      where p = 0.5 * (1 + λ).
    - For q ≠ 1, the function computes the Rényi entropy:
          S_q = (1 / (1-q)) * Σ log(p^q + (1-p)^q)
    - The entropy is normalized by the logarithm of the specified base.
    """
    
    log_base = np.log(base)
    
    #! von‑Neumann entropy  (q == 1)
    if np.abs(q - 1.0) < 1e-12:
        s = 0.0
        LOG_TWO = np.log(2.0)
        for l in lam:
            if l > -1.0:
                s += (1.0 + l) * (np.log1p(l) - LOG_TWO)
            if l < 1.0:
                s += (1.0 - l) * (np.log1p(-l) - LOG_TWO)
        return -0.5 * s

    #! Rényi entropy  (generic q)
    inv_1mq = 1.0 / (1.0 - q)
    s = 0.0
    for l in lam:
        p  = 0.5 * (1.0 + l)
        pm = 1.0 - p
        s += np.log(p ** q + pm ** q)
    return inv_1mq * s / log_base

# b) JAX version

if JAX_AVAILABLE:

    @jax.jit
    def _clean_probs_jax(p: jnp.ndarray, eps: float = 1e-15):
        q  = jnp.where(p < eps, 0.0, p)
        s  = jnp.sum(q)
        return jnp.where(jnp.abs(s - 1.0) > 1e-12, q / s, q)

    # --------------------------------------------------------------------

    @partial(jax.jit, static_argnames=("base",))
    def vn_entropy_jax(lam: jnp.ndarray, base: float = jnp.e) -> float:
        r"""
        Von Neumann (Shannon) entropy

        \[
            S = -\sum_i p_i \log_b p_i ,
            \qquad  p_i \ge 0,\; \sum_i p_i = 1 .
        \]

        Parameters
        ----------
        lam  : jnp.ndarray
            Probability vector.
        base : float, optional
            Logarithm base \(b\).  Default = \(e\).

        Returns
        -------
        float
            \(S\).
        """
        lam      = _clean_probs_jax(lam)
        logp     = jnp.log(lam + _EPS)
        if base != jnp.e:
            logp /= jnp.log(base)
        return -jnp.vdot(lam, logp)

    @partial(jax.jit, static_argnames=("base",))
    def renyi_entropy_jax(lam: jnp.ndarray, q: float, base: float = jnp.e) -> float:
        r"""
        Rényi entropy of order \(q\neq 1\)

        \[
            S_q = \frac{\log_b \!\bigl(\sum_i p_i^{\,q}\bigr)}{1-q},
            \qquad  q>0 .
        \]

        For \(q\to1\) it reduces to the von Neumann entropy.

        Parameters
        ----------
        lam  : jnp.ndarray
        q    : float
        base : float, optional

        Returns
        -------
        float
        """
        lam = _clean_probs_jax(lam)

        def _vn(_: None) -> float:
            return vn_entropy(lam, base)

        def _generic(_: None) -> float:
            s       = jnp.sum(lam ** q)
            log_s   = jnp.log(s)
            if base != jnp.e:
                log_s   /= jnp.log(base)
            return log_s / (1.0 - q)

        return jax.lax.cond(jnp.isclose(q, 1.0), _vn, _generic, operand=None)
    
    @partial(jax.jit, static_argnames=("base",))
    def sp_correlation_entropy_jax(lam: jnp.ndarray, q: float, base: float = jnp.e) -> float:
        r"""
        Compute the single-particle correlation entropy for a set of eigenvalues.

        This function calculates either the von Neumann entropy (for q=1) or the Rényi entropy (for q≠1)
        associated with the eigenvalues of a correlation matrix. The eigenvalues λ are assumed to be in [-1, 1].
        The probabilities are defined as \( p = \frac{1}{2}(1 + \lambda) \).

        Formulas:
            - For q = 1 (von Neumann entropy):
                \( S = -\sum_i \left[ p_i \log_b p_i + (1 - p_i) \log_b (1 - p_i) \right] \)
            - For q ≠ 1 (Rényi entropy):
                \( S_q = \frac{1}{1 - q} \sum_i \log_b \left( p_i^q + (1 - p_i)^q \right) \)

        lam : jnp.ndarray
            np.ndarray of correlation-matrix eigenvalues (\(\lambda\)), each in the interval [-1, 1].
        q : float
            Entropy order parameter. Use q=1 for von Neumann entropy, q≠1 for Rényi entropy.
            Logarithm base for entropy calculation (default: natural logarithm, e).

            The computed entropy value.

        Notes
        -----
        - The function is numerically stable for probabilities close to 0 or 1.
        - For q=1, the result is the standard von Neumann entropy.
        - For q≠1, the result is the Rényi entropy of order q.
        """
        log_base = jnp.log(base)
        p        = 0.5 * (1.0 + lam)
        pm       = 1.0 - p

        def _vn(_: None) -> float:
            ent  = -jnp.sum(p  * jnp.log(p  + _EPS) +
                            pm * jnp.log(pm + _EPS))
            if base != jnp.e:
                ent /= log_base
            return ent

        def _renyi(_: None) -> float:
            s = jnp.sum(jnp.log(p ** q + pm ** q))
            return s / ((1.0 - q) * log_base)

        return jax.lax.cond(jnp.isclose(q, 1.0), _vn, _renyi, operand=None)
else:
    # If JAX is not available, we will use the Numpy version of the entropy functions
    vn_entropy              = vn_entropy
    renyi_entropy           = renyi_entropy
    sp_correlation_entropy  = sp_correlation_entropy

############################################
#! Entropy
#############################################

@unique
class Entanglement(Enum):
    VN          = 1
    RENYI       = 2
    TSALLIS     = 3
    SINGLE      = 4
    PARTIC      = 5

def entropy(lam: np.ndarray, q: float = 1.0, base: float = np.e, *,
        typek: Entanglement = Entanglement.VN, backend: str = "numpy", **kwargs) -> float:
    """
    Calculates the entropy of a probability distribution using the specified entanglement entropy type.

    Parameters:
        lam (np.ndarray):
            The probability distribution (eigenvalues) for which to compute the entropy.
        q (float, optional):
            The order parameter for Rényi and Tsallis entropies. Default is 1.0.
        base (float, optional):
            The logarithm base to use in entropy calculations. Default is the natural logarithm (np.e).
        typek (Entanglement, optional):
            The type of entanglement entropy to compute. Must be one of:
                - Entanglement.VN:
                    Von Neumann entropy
                - Entanglement.RENYI:
                    Rényi entropy
                - Entanglement.TSALLIS:
                    Tsallis entropy
                - Entanglement.SINGLE:
                    Single-particle correlation entropy

    Returns:
        float: The computed entropy value.

    Raises:
        ValueError: If an unsupported entanglement type is provided.
    """
    if backend.lower() == 'numpy':
        lam = np.asarray(lam)
        if typek == Entanglement.VN:
            return vn_entropy(lam, base)
        elif typek == Entanglement.RENYI:
            return renyi_entropy(lam, q, base)
        elif typek == Entanglement.TSALLIS:
            pass
        elif typek == Entanglement.SINGLE:
            return sp_correlation_entropy(lam, q, base)
        elif typek == Entanglement.PARTIC:
            pass
    elif backend.lower() == 'jax' and JAX_AVAILABLE:
        if typek == Entanglement.VN:
            return jnp.vn_entropy(lam, base)
        elif typek == Entanglement.RENYI:
            return jnp.renyi_entropy(lam, q, base)
        elif typek == Entanglement.TSALLIS:
            pass
        elif typek == Entanglement.SINGLE:
            return jnp.sp_correlation_entropy(lam, q, base)
        elif typek == Entanglement.PARTIC:
            pass
    else:
        raise ValueError(f"Unsupported backend: {backend}. Use 'numpy' or 'jax'.")

#############################################
#! Density matrix
#############################################

# @numba.njit
def rho_numba(state: np.ndarray, dimA: int, dimB: int) -> np.ndarray:
    psi = state.reshape(dimA, dimB, order="F")
    return psi @ psi.conj().T

# @numba.njit(cache=True)
def rho_numba_mask(state    : np.ndarray,
                order       : tuple,
                size_a      : int) -> np.ndarray:
    """
    Reshapes and reorders a quantum state vector for subsystem partitioning.
    This function takes a 1D quantum state vector and reshapes it into a multi-dimensional array,
    then transposes and flattens it to produce a 2D array suitable for partial trace or reduced
    density matrix calculations. The partitioning is determined by the `order` and `size_a` arguments.
    Args:
        state (np.ndarray):
            The input quantum state vector as a 1D complex-valued array of length 2**N, where N is the number of qubits.
        order (tuple):
            A tuple specifying the new order of qubits after partitioning. The first `size_a` elements correspond to subsystem A.
        size_a (int):
            The number of qubits in subsystem A.
    Returns:
        np.ndarray:
            A 2D array of shape (2**size_a, 2**(N - size_a)), where N = len(order), representing the reshaped and reordered wavefunction.
    """
    psi_nd      = state.reshape((2, ) * len(order), order='F') # no copy, reshape to (2, 2, ..., 2)
    dA          = 1 << size_a
    return psi_nd.transpose(order).reshape(dA, -1)
    
                # extract_a   : Callable[[int], int],
                # extract_b   : Callable[[int], int],
                # size_a      : int,
                # size        : int,
                # tol         : float = 1e-14):
    # """
    # Constructs a reshaped wavefunction (psi) from a given quantum state vector by mapping indices using provided extraction functions.

    # This function is typically used to prepare a state for partial trace or reduced density matrix calculations by reshaping the state vector into a 2D array according to subsystem partitions.

    # Args:
    #     state (np.ndarray):
    #         The input quantum state vector (1D complex array).
    #     extract_a (Callable[[int], int]):
    #         Function to extract the subsystem-A index from a basis index.
    #     extract_b (Callable[[int], int]):
    #         Function to extract the subsystem-B index from a basis index.
    #     size_a (int):
    #         Number of qubits (or bits) in subsystem A.
    #     size (int):
    #         Total number of qubits (or bits) in the system.
    #     tol (float, optional):
    #         Amplitude threshold below which values are ignored (default: 1e-12).

    # Returns:
    #     np.ndarray:
    #         A 2D array of shape (2**size_a, 2**(size - size_a)) representing the reshaped wavefunction.
    # """
    
    # psi_nd  = state.reshape((2,) * size) # no copy, reshape to (2, 2, ..., 2)
    
    
    
    # size_b  = size - size_a
    # dA      = 1 << size_a
    # dB      = 1 << size_b
    # psi     = np.zeros((dA, dB), dtype=state.dtype)
    # for i in numba.prange(state.size):
    #     i               = np.int64(i)
    #     amp             = state[i]
    #     if np.abs(amp) <= tol:        
    #         continue
    #     iA              = extract_a(i)
    #     iB              = extract_b(i)
    #     psi[iA, iB]    += amp
    # return psi

# @numba.njit(cache=True, fastmath=True)
def schmidt_numba(  psi     : np.ndarray,
                    dimA    : int,
                    dimB    : int,
                    eig     : bool) -> Tuple[np.ndarray, np.ndarray]:
    if eig:
        if dimA <= dimB:
            rho         = psi @ psi.conj().T
            vals, vecs  = np.linalg.eigh(rho)
        else:
            sigma       = psi.conj().T @ psi
            vals, V     = np.linalg.eigh(sigma)
            vecs        = psi @ V / np.sqrt(np.maximum(vals, 1e-30))
    else:
        vecs, s, _      = np.linalg.svd(psi, full_matrices=False)
        vals            = s * s
    return vals[::-1], vecs[:, ::-1]

@numba.njit(cache=True, fastmath=True)
def schmidt_numba_mask(psi      : np.ndarray,
                    order       : tuple,
                    size_a      : int,
                    eig         : bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the Schmidt decomposition of a bipartite quantum state using Numba.
    Parameters
    ----------
    state : np.ndarray
        The input state vector representing the bipartite quantum system.
    order : tuple
        A tuple specifying the new order of qubits after partitioning. The first `size_a` elements correspond to subsystem A.
    size_a : int
        Dimension of subsystem A.
    eig : bool
        If True, use eigenvalue decomposition of the reduced density matrix; 
        if False, use singular value decomposition (SVD).
    Returns
    -------
    vals : np.ndarray
        The Schmidt coefficients (squared singular values or eigenvalues), sorted in descending order.
    vecs : np.ndarray
        The corresponding Schmidt vectors, columns ordered according to descending Schmidt coefficients.
    """
    
    if eig:
        if size_a <= len(order) - size_a:
            rho         = psi @ psi.conj().T
            vals, vecs  = np.linalg.eigh(rho)
        else:
            sigma       = psi.conj().T @ psi
            vals, V     = np.linalg.eigh(sigma)
            vecs        = psi @ V / np.sqrt(np.maximum(vals, 1e-30))
    else:
        vecs, s, _      = np.linalg.svd(psi, full_matrices=False)
        vals            = np.square(s)
    return vals[::-1], vecs[:, ::-1]

if JAX_AVAILABLE:
    
    @partial(jax.jit, static_argnums=(1, 2))
    def rho_jax(state, dimA: int, dimB: int):
        psi = jnp.reshape(state, (dimA, dimB), order="F")
        return psi @ jnp.conj(psi).T

    @partial(jax.jit, static_argnums=(1, 2))
    def rho_mask_jax(state  : jnp.ndarray,
                order   : tuple[int, ...],
                size_a  : int) -> jnp.ndarray:
        r"""
        Reshape-and-permute a pure \(N\)-qubit state
        \(|\psi\rangle\in\mathbb C^{2^N}\)
        into a matrix  
        \(\psi_{A,B}\in\mathbb C^{2^{|A|}\times 2^{|B|}}\)
        that is ready for a partial-trace / reduced-density-matrix
        calculation with respect to subsystem *A* (first ``size_a`` qubits in
        ``order``).

        Parameters
        ----------
        state : jnp.ndarray
            Flat state vector, length \(2^N\).
        order : tuple[int]
            Permutation of qubit indices; the *first* ``size_a`` entries
            define subsystem *A*.
        size_a : int
            \(|A|\) - number of qubits in subsystem *A*.

        Returns
        -------
        jnp.ndarray
            Array of shape \((2^{|A|},\,2^{N-|A|})\).
        """
        N = len(order)                          # total number of qubits

        # reshape: 1-D → N-D tensor (row-major)
        psi_nd = jnp.reshape(state, (2,) * N)   # JAX supports only row-major

        # mimic Fortran-order semantics
        #   For a (2,2,…,2) tensor, Fortran layout is equivalent to
        #   row-major layout with *reversed* axis numbering.
        perm = tuple(N - 1 - o for o in order)  # map Fortran axes → C axes

        # reorder qubits and flatten back to matrix
        psi_perm = jnp.transpose(psi_nd, perm)
        dA       = 1 << size_a                  # 2**size_a
        return jnp.reshape(psi_perm, (dA, -1))

    @partial(jax.jit, static_argnums=(1, 2, 3))
    def schmidt_jax(state, dimA: int, dimB: int, use_eig: bool):
        psi = jnp.reshape(state, (dimA, dimB), order="F")
        if use_eig:
            if dimA <= dimB:
                rho         = psi @ jnp.conj(psi).T
                vals, vecs  = jnp.linalg.eigh(rho)
            else:
                sigma   = jnp.conj(psi).T @ psi
                vals, V = jnp.linalg.eigh(sigma)
                vecs    = psi @ V / jnp.sqrt(jnp.maximum(vals, 1e-30))
        else:
            vecs, s, _  = jnp.linalg.svd(psi, full_matrices=False)
            vals        = s * s
        # flip for descending order
        return vals[::-1], vecs[:, ::-1]
    
    @partial(jax.jit, static_argnums=(1, 2, 3))
    def schmidt_mask_jax(state: jnp.array, order: tuple, size_a: int, eig: bool = False):
        """
        Computes the Schmidt decomposition for a given state and mask.
        
        Args:
            state (Array): The input state vector.
            order (int): The order of the mask.
            size_a (int): The size of the first subsystem.
        
        Returns:
            Array: The Schmidt decomposition.
        """
        # Reshape the state vector into a matrix
        psi_nd  = jnp.reshape(state, (2,) * len(order), order="F")
        dA      = 1 << size_a
        psi     = psi_nd.transpose(order).reshape((dA, -1))
        if eig:
            if size_a <= len(order) - size_a:
                rho         = psi @ jnp.conj(psi).T
                vals, vecs  = jnp.linalg.eigh(rho)
            else:
                sigma       = jnp.conj(psi).T @ psi
                vals, V     = jnp.linalg.eigh(sigma)
                vecs        = psi @ V / jnp.sqrt(jnp.maximum(vals, 1e-30))
        else:
            vecs, s, _      = jnp.linalg.svd(psi, full_matrices=False)
            vals            = s * s
        return vals[::-1], vecs[:, ::-1]
    
#! usage

if __name__ == "__main__":
    # Example usage of the rho_numba_mask function
    nh      = 2**4
    state   = np.random.rand(nh) + 1j * np.random.rand(nh)  # Random complex state vector
    state  /= np.linalg.norm(state)
    la      = 2
    lb      = 2
    dima    = 2**la
    dimb    = 2**lb
    
    #! in numpy
    rho             = rho_numba(state, dimA=dima, dimB=dimb)
    schmidt_val, _  = schmidt_numba(rho, dima, dimb, eig = False)
    vn_entropy_np   = vn_entropy(schmidt_val)
    print("Von Neumann Entropy (NumPy):", vn_entropy_np)

    #! in jax
    if JAX_AVAILABLE:
        state_jax           = jnp.array(state)
        rho                 = rho_jax(state_jax, dima, dimb)
        schmidt_val_jax, _  = schmidt_jax(rho, dima, dimb, use_eig=False)
        vn_entropy_jnp      = vn_entropy_jax(schmidt_val_jax)
        print("Von Neumann Entropy (JAX):", vn_entropy_jnp)
    else:
        print("JAX is not available.")
    