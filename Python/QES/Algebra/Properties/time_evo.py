"""
time_evo.py

This module provides functions for simulating the time evolution of quantum systems using their Hamiltonian eigenstates and eigenvalues. It supports both NumPy and JAX backends for efficient computation and automatic differentiation.

Key Features:
-------------
- Time evolution of quantum states via matrix exponentiation.
- Batch evolution for multiple time points.
- Expectation value calculation for observables after time evolution.
- Construction of initial states for various quantum quench protocols.
- Calculation of diagonal ensemble averages and mean energies.

Author  : Maksymilian Kliczkowski
Date    : 2025-02-01
"""

import os
import numpy as np
from enum import Enum

try:
    from general_python.common import binary as BinaryMod
    from general_python.algebra.utils import Array
except ImportError as e:
    raise ImportError("Error importing modules in time_evo.py: " + str(e))

_JAX_AVAILABLE = os.environ.get("PY_JAX_AVAILABLE", 0) == "1"
if _JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
    from jax import lax
else:
    jax = None
    jnp = np
    lax = None

# -----------------------------------------------------------------------------
#! Constants
# -----------------------------------------------------------------------------

SYSTEM_PROPERTIES_MIN_SPACING       = 1e-15
SYSTEM_PROPERTIES_THROW_DEGENERATE  = 1
SYSTEM_PROPERTIES_COEFF_THRESHOLD   = 1e-11
SYSTEM_PROPERTIES_USE_OPENMP        = 0

# -----------------------------------------------------------------------------
#! Time Evolution Functions
# -----------------------------------------------------------------------------

if _JAX_AVAILABLE:
    @jax.jit
    def time_evo_jax(eigenstates    : Array,
                    eigvals         : Array,
                    overlaps        : Array,
                    time            : float) -> Array:
        """
        Evolves a quantum state in time using the eigenstates and eigenvalues of the system Hamiltonian.

        Parameters
        ----------
        eigenstates : Array
            A matrix whose columns are the eigenstates of the Hamiltonian.
        eigvals : Array
            A 1D array of eigenvalues corresponding to the eigenstates.
        overlaps : Array
            A 1D array of overlaps (projections) of the initial state onto each eigenstate.
        time : float
            The time at which to evaluate the evolved state.

        Returns
        -------
        Array
            The quantum state at the specified time, as a linear combination of eigenstates with time-dependent phases.
        """

        phases = jnp.exp(-1j * eigvals * time)
        return eigenstates @ (overlaps * phases)

    @jax.jit
    def time_evo_block_jax(eigenstates  : Array,
                        eigvals         : Array,
                        overlaps        : Array,
                        times           : Array) -> Array:
        """
        Evolves a quantum state in time using a block of eigenstates, eigenvalues, and overlaps.
        Args:
            eigenstates (Array):
                Array of eigenstates, shape (M, N), where M is the Hilbert space dimension and N is the number of eigenstates.
            eigvals (Array):
                Array of eigenvalues, shape (N,).
            overlaps (Array):
                Array of overlaps between the initial state and eigenstates, shape (N,).
            times (Array):
                Array of time points at which to evaluate the evolved state, shape (T,).
        Returns:
            Array: The time-evolved state at each time in `times`, shape (M, T).
        """
        t_arr               = jnp.asarray(times)
        # Compute all evolved states at once
        evolved_states      = jnp.exp(-1j * jnp.outer(eigvals, t_arr)) * overlaps[:, jnp.newaxis]
        # Project back to basis
        quenched_states_t   = eigenstates @ evolved_states # shape: (dim, len(t_arr))
        # Compute expectation values for all times
        return quenched_states_t

    @jax.jit
    def time_evo_evaluate_jax(quenched_states_t : Array,
                            quench_operator_m   : Array) -> Array:
        """
        Evaluates the expectation value of a quench operator for a set of time-evolved quantum states.
        Parameters
        ----------
        quenched_states_t : Array
            A 2D array where each column (or row, depending on convention) represents a quantum state at a specific time.
        quench_operator_m : Array
            The operator (as a matrix) whose expectation value is to be computed for each time-evolved state.
        Returns
        -------
        Array
            A 1D array of expectation values of the quench operator for each time-evolved state.
        Notes
        -----
        The function computes ⟨ψ(t)|O|ψ(t)⟩ for each time-evolved state |ψ(t)⟩, where O is the quench operator.
        """
        quenched_values_t = jnp.einsum('ij,ji->i', jnp.conj(quenched_states_t.T), quench_operator_m @ quenched_states_t)
        return quenched_values_t    
else:
    time_evo_jax = None
    time_evo_block_jax = None
    time_evo_evaluate_jax = None

# numpy version

def time_evolution(eigenstates    : Array,
            eigvals         : Array,
            overlaps        : Array,
            time            : float) -> Array:
    phases = np.exp(-1j * eigvals * time)
    return eigenstates @ (overlaps * phases)

def time_evo_block(eigenstates          : Array,
                    eigvals             : Array,
                    quench_overlaps     : Array,
                    times               : Array) -> Array:
    """
    Evolves a quantum state in time using a given set of eigenstates, eigenvalues, and initial overlaps.
    Parameters
    ----------
    eigenstates : Array
        Array of eigenstates of the Hamiltonian, shape (dim, N), where dim is the Hilbert space dimension and N is the number of eigenstates.
    eigvals : Array
        Array of eigenvalues corresponding to the eigenstates, shape (N,).
    quench_overlaps : Array
        Overlaps of the initial state with each eigenstate, shape (N,).
    times : Array
        Array of time points at which to compute the evolved state, shape (T,).
    Returns
    -------
    Array
        Array of evolved states at each time, shape (dim, T), where each column corresponds to the state at a given time.
    """
    
    # build (N,T) phase matrix and multiply by overlaps
    # Vectorized time evolution for all times (excluding t=0)
    t_arr               = times
    # Compute all evolved states at once
    evolved_states      = np.exp(-1j * np.outer(eigvals, t_arr)) * quench_overlaps[:, np.newaxis]
    # Project back to basis
    quenched_states_t   = eigenstates @ evolved_states # shape: (dim, len(t_arr))
    # Compute expectation values for all times
    return quenched_states_t

def time_evo_block_optimized(
    eig_vec     : np.ndarray,           # shape (M, N), eigenvectors as columns
    eig_val     : np.ndarray,           # shape (N,)
    overlaps    : np.ndarray,           # shape (N,)
    time_steps  : np.ndarray,           # shape (T,)
    *,
    out         : np.ndarray | None = None,    # optional preallocated (M, T) complex
    dtype       = np.complex128,               # output/compute dtype
    block_cols  : int | None = None,           # if None, auto; else number of time columns per block
    max_temp_gb : float | None = 1.0           # soft cap for temporary phases (in GiB) when block_cols is None
    ) -> np.ndarray:
    """
    Compute quench states for arbitrary times:
        Q(:, t_k) = eig_vec @ (exp(-i * eig_val * t_k) * overlaps)

    Memory/perf notes:
    - Uses a single NtimesT temporary "phases" when it fits under `max_temp_gb`,
        then one GEMM. Otherwise processes in blocks of `block_cols`.
    - Ensures Fortran order for the right operand of GEMM and for the output.

    Shapes:
        eig_vec     : (M, N)
        eig_val     : (N,)
        overlaps    : (N,)
        time_steps  : (T,)
        returns out : (M, T) with dtype `dtype`
    """
    
    # ---- sanitize & dtypes ----
    E   = np.ascontiguousarray(eig_val, dtype=np.float64)           # real energies
    ovl = np.ascontiguousarray(overlaps, dtype=dtype)               # complex overlaps
    t   = np.ascontiguousarray(time_steps, dtype=np.float64)        # real times

    # BLAS likes column-major on the right operand; left matrix can be either, but
    # giving it Fortran also helps. Cast once to target dtype.
    V   = np.asfortranarray(eig_vec, dtype=dtype)

    M, N    = V.shape
    T       = t.size
    if E.size != N or ovl.size != N:
        raise ValueError("Shape mismatch: eig_vec (M,N), eig_val (N,), overlaps (N,) must agree.")

    # prepare output
    if out is None:
        out = np.empty((M, T), dtype=dtype, order='F')
    else:
        if out.shape != (M, T):
            raise ValueError(f"'out' must be shape {(M, T)}, got {out.shape}.")
        if out.dtype != dtype:
            raise ValueError(f"'out' dtype must be {dtype}, got {out.dtype}.")
        # ensure Fortran for better slice assignment into columns
        if not out.flags.f_contiguous:
            out = np.asfortranarray(out)

    #! choose blocking - this is important for large T
    if block_cols is None:
        if max_temp_gb is None:
            max_temp_gb = 1.0 # sensible default
        bytes_per_c128  = np.dtype(dtype).itemsize
        # NtimesB complex buffer; try to keep under cap (leave ~20% headroom for intermediates)
        max_bytes       = int(max_temp_gb * (1024**3) * 0.8)
        # ensure at least 1 column
        block_cols      = max(1, min(T, max_bytes // max(1, (N * bytes_per_c128))))
        # if it all fits comfortably, do everything at once
        if block_cols >= T:
            block_cols = T

    #! full GEMM path (fits in memory)
    if block_cols == T:
        # Build phases once: phases = exp(-i * E[:,None] * t[None,:]) * overlaps[:,None]
        # Make it Fortran for BLAS (right operand).
        phases = np.asfortranarray(np.exp(-1j * (E[:, None] * t[None, :])), dtype=dtype)
        # Row-scale in-place by overlaps (avoid extra temp)
        np.multiply(phases, ovl[:, None], out=phases)
        # One GEMM
        out[:, :] = V @ phases
        return out

    #! blocked path (arbitrary times, large T)
    # Precompute E[:,None] once to save tiny broadcast overhead inside the loop
    Ecol = E.reshape(N, 1)
    for p in range(0, T, block_cols):
        q           = min(p + block_cols, T)
        t_blk       = t[p:q].reshape(1, q - p)  # (1, B)
        # phases_blk = exp(-i * (E[:,None] * t_blk))  -> (N, B), Fortran order
        phases_blk  = np.asfortranarray(np.exp(-1j * (Ecol * t_blk)), dtype=dtype)
        # scale rows by overlaps in-place
        np.multiply(phases_blk, ovl[:, None], out=phases_blk)
        # GEMM for the block
        out[:, p:q] = V @ phases_blk
    return out

def time_evo_evaluate(quenched_states_t : Array,
                    quench_operator_m   : Array) -> Array:
    """
    Evaluates the expectation value of a quench operator for a set of time-evolved quantum states.

    Parameters
    ----------
    quenched_states_t : Array
        A 2D array where each column (or row, depending on convention) represents a quantum state at a specific time.
    quench_operator_m : Array
        The operator (as a matrix) whose expectation value is to be computed for each time-evolved state.

    Returns
    -------
    Array
        A 1D array of expectation values of the quench operator for each time-evolved state.

    Notes
    -----
    The function computes ⟨ψ(t)|O|ψ(t)⟩ for each time-evolved state |ψ(t)⟩, where O is the quench operator.
    """
    quenched_values_t = np.einsum('ij,ji->i', np.conj(quenched_states_t.T), quench_operator_m @ quenched_states_t)
    return quenched_values_t
    # quenched_values_t = np.conj(quenched_states_t.T) @ quench_operator_m @ quenched_states_t
    # return quenched_values_t.diagonal()

# -----------------------------------------------------------------------------
#! Diagonal Ensemble
# -----------------------------------------------------------------------------

def diagonal_ensemble_jax(  soverlaps    : Array,
                            diag_mat      : Array) -> Array:
    """
    Computes the diagonal ensemble of a given matrix using the overlaps.
    Args:
        overlaps (Array): A 1D array of overlaps (projections) of the initial state onto each eigenstate.
        matrix (Array): A 2D array representing the matrix for which to compute the diagonal ensemble.
    Returns:
        Array: The diagonal ensemble, computed as the sum of the product of overlaps and the diagonal elements of the matrix.
    """
    # \sum _n a_nn |<ψ|n>|²
    return jnp.dot(soverlaps, diag_mat)
    return jnp.sum(overlaps * diag_mat)    

def diagonal_ensemble(soverlaps  : Array,
                    diag_mat    : Array):
    # \sum _n a_nn |<ψ|n>|²
    return np.dot(soverlaps, diag_mat)
    return np.sum(overlaps * diag_mat)

# -----------------------------------------------------------------------------
#! Quench Types Enum
# -----------------------------------------------------------------------------

class QuenchTypes(Enum):
    '''
    Defines the types of quenches that can be performed on the system. The quench
    types are used to specify the type of perturbation that is applied to the system.
    '''
    RANDP      = 0  # Random product state
    RANDN      = 1  # Random normal state
    RANDU      = 2  # Random uniform
    AF_UP      = 3  # Antiferromagnetic up - first site down
    AF_DN      = 4  # Antiferromagnetic down - first site up
    F_UP       = 5  # Ferromagnetic up
    F_DN       = 6  # Ferromagnetic down
    DW_HALF_UP = 7  # Domain wall - half up
    DW_HALF_DN = 8  # Domain wall - half down
    DW_THIRD_UP= 9  # Domain wall - third up
    DW_THIRD_DN= 10 # Domain wall - third down
    MEAN       = 11 # Mean state 
    SEEK       = 12 # Seek state with a given energy

# -----------------------------------------------------------------------------

def create_initial_quench_state(quench_type : QuenchTypes, 
                                Nh          : int,
                                Ns          : int,
                                Eseek       : float = 0.0, 
                                energies            = None,
                                backend             = 'default', 
                                key                 = None,
                                state               = None):
    """
    Creates the initial state vector after a quench.
    
    Args:
        quench_type: A member of the QuenchTypes enum.
        Nh: Dimension of the Hilbert space (length of the state vector).
        Ns: Number of spins.
        Eseek: (For SEEK type) the energy to seek.
        energies: (For SEEK/MEAN types) 1D array of energies.
        backend: jnp (if using JAX) or np.
        key: A PRNGKey for JAX random number generation (if needed).
        
    Returns:
        A 1D array of length Nh representing the initial state (one-hot vector).
    """
    # Initialize state to zeros.
    if state is None:
        state = backend.zeros(Nh, dtype=backend.float64)
    is_jax = backend == jax

    if quench_type == QuenchTypes.RANDP:
        # For RANDP, generate a random bit pattern over Ns bits.
        if is_jax and key is not None:
            bits = jax.random.bernoulli(key, p=0.5, shape=(Ns,))
            # Convert boolean bits to integer index.
            idx = 0
            # Here we convert bits to a NumPy array for the Python loop.
            for i, b in enumerate(np.array(bits)):
                if b:
                    idx |= 1 << i
        else:
            bits = np.random.rand(Ns) < 0.5
            idx = 0
            for i, b in enumerate(bits):
                if b:
                    idx |= 1 << i
        # Update state: for JAX, use functional update; for NumPy, in-place.
        if _JAX_AVAILABLE:
            state = state.at[idx].set(1.0)
        else:
            state[idx] = 1.0

    elif quench_type == QuenchTypes.RANDN:
        # Create a normally distributed state, then normalize.
        if _JAX_AVAILABLE:
            if key is None:
                raise ValueError("A PRNG key must be provided for JAX random generation.")
            state = jax.random.normal(key, (Nh,))
        else:
            state = np.random.randn(Nh)
        norm = backend.linalg.norm(state)
        state = state / norm

    elif quench_type == QuenchTypes.RANDU:
        # Create a uniformly distributed state, then normalize.
        if _JAX_AVAILABLE:
            if key is None:
                raise ValueError("A PRNG key must be provided for JAX random generation.")
            state = jax.random.uniform(key, (Nh,), minval=0.0, maxval=1.0)
        else:
            state = np.random.rand(Nh)
        norm = backend.linalg.norm(state)
        state = state / norm

    elif quench_type == QuenchTypes.AF_UP:
        idx = int(Nh / 3) if (Ns % 2 == 0) else int((Nh + 1) / 3)
        if _JAX_AVAILABLE:
            state = state.at[idx].set(1.0)
        else:
            state[idx] = 1.0

    elif quench_type == QuenchTypes.AF_DN:
        idx = int(Nh / 3) if (Ns % 2 == 0) else int((Nh + 1) / 3)
        idx = flip_all(idx, Ns)
        if _JAX_AVAILABLE:
            state = state.at[idx].set(1.0)
        else:
            state[idx] = 1.0

    elif quench_type == QuenchTypes.F_UP:
        idx = Nh - 1
        if _JAX_AVAILABLE:
            state = state.at[idx].set(1.0)
        else:
            state[idx] = 1.0

    elif quench_type == QuenchTypes.F_DN:
        idx = 0
        if _JAX_AVAILABLE:
            state = state.at[idx].set(1.0)
        else:
            state[0] = 1.0

    elif quench_type == QuenchTypes.DW_HALF_UP:
        idx = ULLPOW(max(int(Ns / 2) - 1, 0))
        if _JAX_AVAILABLE:
            state = state.at[idx].set(1.0)
        else:
            state[idx] = 1.0

    elif quench_type == QuenchTypes.DW_HALF_DN:
        idx = ULLPOW(max(int(Ns / 2) - 1, 0))
        idx = flip_all(idx, Ns)
        if _JAX_AVAILABLE:
            state = state.at[idx].set(1.0)
        else:
            state[idx] = 1.0

    elif quench_type == QuenchTypes.DW_THIRD_UP:
        idx = BinaryMod.ULLPOW(max(int(Ns / 3) - 1, 0))
        if _JAX_AVAILABLE:
            state = state.at[idx].set(1.0)
        else:
            state[idx] = 1.0

    elif quench_type == QuenchTypes.DW_THIRD_DN:
        idx = BinaryMod.ULLPOW(max(int(Ns / 3) - 1, 0))
        idx = BinaryMod.flip_all(idx, Ns)
        if _JAX_AVAILABLE:
            state = state.at[idx].set(1.0)
        else:
            state[idx] = 1.0

    elif quench_type in (QuenchTypes.MEAN, QuenchTypes.SEEK):
        if energies is None:
            raise ValueError("Energies must be provided for MEAN or SEEK quench types.")
        if quench_type == QuenchTypes.SEEK:
            if Eseek is None:
                raise ValueError("Eseek must be provided for SEEK quench type.")
            diff    = backend.abs(energies - Eseek)
            idx     = int(backend.argmin(diff))
            if is_jax:
                state = state.at[idx].set(1.0)
            else:
                state[idx] = 1.0
        else:  # MEAN
            mean_val = backend.mean(energies)
            diff = backend.abs(energies - mean_val)
            idx = int(backend.argmin(diff))
            print(f"MEAN energy: {mean_val}, chosen index: {idx}")
            if is_jax:
                state = state.at[idx].set(1.0)
            else:
                state[idx] = 1.0

    else:
        # Default: set the first element.
        if is_jax:
            state = state.at[0].set(1.0)
        else:
            state[0] = 1.0

    return state

# -----------------------------------------------------------------------------
# Mean Energy Calculation After Quench
# -----------------------------------------------------------------------------
def calc_mean_energy_quench(H, state, backend='default'):
    """
    Calculates the mean energy after the quench, defined as <state| H |state>.
    
    Args:
        H: The Hamiltonian matrix (dense or sparse) as a 2D array.
        state: The state vector.
        backend: jnp if JAX is available, otherwise np.
    
    Returns:
        A scalar representing the mean energy.
    """
    # For complex inner products, use vdot which conjugates the first argument.
    return backend.vdot(state, backend.dot(H, state))
