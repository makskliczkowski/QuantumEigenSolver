'''
This module contains the time evolution of the system. The time evolution is computed
using the Hamiltonian matrix and the initial state vector. The time evolution is
computed using the matrix exponential method.

@file time_evo.py
@author Maksymilian Kliczkowski
@date 2025-02-01
'''

import numpy as np
from enum import Enum

import sys, os 

# Add the parent directory to the sys.path list.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '...')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from .. import BinaryMod

# Try to import JAX. If unavailable, fall back to NumPy.
try:
    import jax
    import jax.numpy as jnp
    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False

# Use JAX’s numpy if available, otherwise NumPy.
Backend = jnp if _JAX_AVAILABLE else np

# -----------------------------------------------------------------
# Constants
# -----------------------------------------------------------------

SYSTEM_PROPERTIES_MIN_SPACING    = 1e-15
SYSTEM_PROPERTIES_THROW_DEGENERATE = 1
SYSTEM_PROPERTIES_COEFF_THRESHOLD  = 1e-11
SYSTEM_PROPERTIES_USE_OPENMP       = 0

# -----------------------------------------------------------------------------
# Time Evolution Function
# -----------------------------------------------------------------------------

def time_evo(eigenstates, eigvals, overlaps, time, backend=Backend):
    """
    Computes the time evolution of a quantum state.
    
    Args:
        eigenstates : 2D array whose columns are eigenstates.
        eigvals     : 1D array of eigenvalues.
        overlaps    : 1D array of coefficients (overlaps of the initial state with the eigenstates).
        time        : The time at which to evaluate the evolved state.
        backend     : jnp (if JAX is available) or np.
        
    Returns:
        1D complex array representing the evolved state.
    """
    # Compute the exponential factor exp(-i * time * eigenvals)
    exp_factors     = backend.exp(-1j * time * eigvals)
    coeffs          = exp_factors * overlaps
    # The C++ loop (in the original code: cpp/library/include/quantities/statistics.h):
    #   for (i = 0; i < n_cols; ++i)
    #       ret += (exp(i)*overlap(i)) * eigenstates.col(i);
    # is equivalent to a matrix–vector product.
    return backend.dot(eigenstates, coeffs)

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Quench Types Enum
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
                                backend             = Backend, 
                                key                 = None):
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
    state = backend.zeros(Nh, dtype=backend.float64)
    
    if quench_type == QuenchTypes.RANDP:
        # For RANDP, generate a random bit pattern over Ns bits.
        if _JAX_AVAILABLE and key is not None:
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
            diff = backend.abs(energies - Eseek)
            idx = int(backend.argmin(diff))
            if _JAX_AVAILABLE:
                state = state.at[idx].set(1.0)
            else:
                state[idx] = 1.0
        else:  # MEAN
            mean_val = backend.mean(energies)
            diff = backend.abs(energies - mean_val)
            idx = int(backend.argmin(diff))
            print(f"MEAN energy: {mean_val}, chosen index: {idx}")
            if _JAX_AVAILABLE:
                state = state.at[idx].set(1.0)
            else:
                state[idx] = 1.0

    else:
        # Default: set the first element.
        if _JAX_AVAILABLE:
            state = state.at[0].set(1.0)
        else:
            state[0] = 1.0

    return state

# -----------------------------------------------------------------------------
# Mean Energy Calculation After Quench
# -----------------------------------------------------------------------------
def calc_mean_energy_quench(H, state, backend=Backend):
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
