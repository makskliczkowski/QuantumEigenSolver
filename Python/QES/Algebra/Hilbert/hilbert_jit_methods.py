"""
file        : QES/Algebra/hilbert_jit_methods.py
description : JIT methods for Hilbert space operations.
author      : Maksymilian Kliczkowski
email       : maksymilian.kliczkowski@pwr.edu.pl

version     : 1.0.1
changes     : 
"""

import numpy as np
import numba

#! private
_INT_BINARY_REPR       = 2**6
_SYM_NORM_THRESHOLD    = 1e-12
from general_python.common.binary import binary_search, __BAD_BINARY_SEARCH_STATE, binary_search_numpy
from general_python.algebra.utils import get_backend, JAX_AVAILABLE, ACTIVE_INT_TYPE


####################################################################################################
#! NUMBA METHODS
####################################################################################################

if True:
    
    # a) mapping
    
    @numba.njit
    def get_mapping(mapping, state):
        """
        Get the mapping of the state.
        
        Args:
            mapping (list):
                The mapping of the states.
            state (int):
                The state to get the mapping for.
        
        Returns:
            int:
                The mapping of the state.
        """
        return mapping[state] if len(mapping) > state else state

    # b) find the representative of a given state
    
    @numba.jit(nopython=False)
    def find_repr_int(state,
                    _sym_group,
                    _reprmap    : np.ndarray = None):
        """
        Find the representative of a given state. This corresponds 
        to finding the lowest state in the symmetry sector.
        as the representative is the state with the lowest integer
        value in the symmetry sector.
        
        Args:
            state (int):
                The state to find the representative for.
            _sym_group (list):
                The symmetry group of the system.
            _reprmap (np.ndarray):
                The mapping of the states to their representatives.
                If None, the representative is calculated.
        Returns:
            int:
                The representative of the state.
        """
                
        # If mapping exists, return saved representative.
        if _reprmap is not None and len(_reprmap) > 0:
            idx     = _reprmap[state, 0]
            sym_eig = _reprmap[state, 1]
            return idx, sym_eig
        
        # If no mapping exists, find the representative.
        if _sym_group is None or len(_sym_group) == 0:
            return state, 1.0
        
        _sec = (1 << _INT_BINARY_REPR) - 1 # by default, using 64-bit integers
        _val = 1.0
        
        # loop over all states in symmetry sectors
        for g in _sym_group:
            _st, _retval = g(state)
            if _st < _sec:
                _sec = _st
                _val = _retval
        return _sec, _val

    # c) find the representative of a given state in a given symmetry sector

    @numba.jit(nopython=False)
    def find_representative_int(
                            _state                  : int,
                            _mapping                : np.ndarray,
                            _normalization          : np.ndarray,
                            _normalization_beta     : float,
                            _sym_group,
                            _reprmap                : np.ndarray = None
            ):
        """
        Find the representative of a given state.
        """
        if _mapping is None or len(_mapping) == 0:
            return (_state, 1.0)
        
        # if the map exists, use it!
        if _reprmap is not None and len(_reprmap) > 0:
            idx, sym_eig    = _reprmap[_state, 0], _reprmap[_state, 1]
            sym_eigc        = sym_eig.conjugate() if hasattr(sym_eig, "conjugate") else sym_eig
            return (idx, _normalization[idx] / _normalization_beta * sym_eigc)
        
        mapping_size = len(_mapping)
        
        # find the representative already in the mapping (can be that the matrix element already 
        # represents the representative state)
        idx = binary_search_numpy(_mapping, 0, mapping_size - 1, _state)
        
        if idx != __BAD_BINARY_SEARCH_STATE:
            return (idx, _normalization[idx] / _normalization_beta)
        
        # otherwise, we need to find the representative by acting on the state with the symmetry operators
        # and finding the one that gives the smallest value - standard procedure
        idx, sym_eig = find_repr_int(_state, _sym_group, _reprmap)
        if idx != __BAD_BINARY_SEARCH_STATE:
            sym_eigc = sym_eig.conjugate() if hasattr(sym_eig, "conjugate") else sym_eig
            return (idx, _normalization[idx] / _normalization_beta * sym_eigc)
        
        # didn't find the representative - this may be different sector
        return (_state, 0.0)

    # d) get the matrix element of a given state

    @numba.jit(nopython=False)
    def get_matrix_element(
            k               : int,
            new_k           : int,
            kmap            = None,
            h_conj          = False,
            _mapping        : np.ndarray = None,
            _norm           : np.ndarray = None,
            _sym_group                   = None,
            _reprmap        : np.ndarray = None
        ):
        '''
        Get the matrix element of a given state using information provided from the symmetry group and 
        a given Hilbert space.
        Args:
            k (int):
                The state to get the matrix element for.
            new_k (int):
                The new state to get the matrix element for.
            kmap (int):
                The mapping of the states.
            h_conj (bool):
                A flag to indicate if the Hamiltonian is conjugated.
            _mapping (list):
                The mapping of the states.
            _norm (list):
                The normalization of the states.
            _sym_group:
                The symmetry group.
            _reprmap:
                The mapping of the representatives.
        '''
        
        # check the mapping, if it is None, we need to get the mapping
        if kmap is None:
            kmap = get_mapping(_mapping, k)
        
        # try to process the elements
        if kmap == new_k:
            # the element k is already the same as new_k and obviously we 
            # and we add this at k (not kmap as it only checks the representative)
            return (new_k, k), 1
        
        # otherwise we need to check the representative of the new k
        # get the norm of the k'th element of the Hilbert space - how to return to the representative
        # of the new k
        norm = _norm[k] if _norm is not None else 1.0
        # find the representative of the new k
        idx, symeig = find_representative_int(new_k, _mapping, _norm, norm, _sym_group, _reprmap)
        return ((idx, k), symeig) if not h_conj else ((k, idx), symeig)

    ###############################

    @numba.jit(nopython=False)
    def jitted_find_repr_int(state, _sym_group, _reprmap = None):
        return find_repr_int(state, _sym_group, _reprmap)

    @numba.jit(nopython=False)
    def jitted_find_representative_int(_state, _mapping, _normalization, _normalization_beta, _sym_group, _reprmap = None):
        return find_representative_int(_state, _mapping, _normalization, _normalization_beta, _sym_group, _reprmap)
        
    @numba.njit
    def jitted_get_mapping(mapping, state):
        return get_mapping(mapping, state)

    @numba.jit(nopython=False)
    def jitted_get_matrix_element(k, new_k, kmap = None, h_conj = False, _mapping = None, _norm = None, _sym_group = None, _reprmap = None):
        return get_matrix_element(k, new_k, kmap, h_conj, _mapping, _norm, _sym_group, _reprmap)
    


# JAX version needs jax imports
try:
    import jax
    import jax.numpy as jnp
    from jax.lax import fori_loop
    from functools import partial

    @partial(jax.jit, static_argnames=('ns',))
    def _calculate_slater_det_jax(sp_eigvecs: jnp.ndarray, # U matrix (Ns x Norb)
                                  occupied_orbitals: jnp.ndarray,
                                  basis_state_int: int,
                                  ns: int) -> jnp.ndarray:
        """Calculates the Slater determinant using JAX."""
        n_particles = occupied_orbitals.shape[0]

        # JAX way to count bits and get indices (more functional)
        site_indices = jnp.arange(ns)
        occupied_sites_mask = ((basis_state_int >> site_indices) & 1) == 1
        n_particles_fock = jnp.sum(occupied_sites_mask)

        # Conditional execution in JAX
        def calculate_det(sp_eigvecs, occupied_orbitals, occupied_sites_mask):
            # Get the actual site indices (might be slower than mask indexing)
            occupied_sites = jnp.where(occupied_sites_mask, site_indices, -1)
            occupied_sites = occupied_sites[occupied_sites != -1] # Filter padding

            # Construct Slater matrix using advanced indexing
            # JAX prefers indices to be JAX arrays
            # Ensure occupied_orbitals is also a JAX array if passed as list/numpy
            row_indices = occupied_sites[:, None] # Shape (N, 1)
            col_indices = occupied_orbitals[None, :] # Shape (1, N)
            slater_matrix = sp_eigvecs[row_indices, col_indices] # Direct N x N construction

            return jnp.linalg.det(slater_matrix.astype(jnp.complex128)) # Ensure complex

        # Return 0 if particle numbers don't match, otherwise calculate det
        # Need to handle N=0 case: det is 1
        det_val = jax.lax.cond(
            n_particles == 0,
            lambda: jnp.array(1.0 + 0.0j, dtype=jnp.complex128), # N=0 case
            lambda: jax.lax.cond(
                n_particles_fock != n_particles,
                lambda: jnp.array(0.0 + 0.0j, dtype=jnp.complex128), # Mismatch case
                lambda: calculate_det(sp_eigvecs, occupied_orbitals, occupied_sites_mask) # Match case
            )
        )
        return det_val

except ImportError:
    warnings.warn("JAX not found. JAX functions for Slater/Permanent will not be available.", ImportWarning)
    _calculate_slater_det_jax = None

# --- Permanent (Bosons) ---

# Ryser's Formula implementation for NumPy/Numba
@njit(cache=True)
def _calculate_permanent_np(M: np.ndarray) -> complex:
    """
    Calculates the permanent of a square matrix M using Ryser's formula.
    Optimized for Numba.

    Args:
        M (np.ndarray): Input square matrix (N x N). Must be complex or float.

    Returns:
        complex: The permanent of the matrix.
    """
    n = M.shape[0]
    if n == 0:
        return 1.0 + 0.0j
    if M.shape[1] != n:
        raise ValueError("Input matrix must be square.")

    total_sum = 0.0 + 0.0j
    one_complex = 1.0 + 0.0j # Precompute complex one

    # Iterate through all 2^n subsets S of columns {0, ..., n-1}
    # Represent subset S by integer k (bitmask) from 1 to 2^n - 1
    for k in range(1, 1 << n):
        sum_prod = one_complex # Product Π_{i=1..N} (sum_{j∈S} A_{ij})
        popcount_k = 0 # Size of subset S, |S|

        # Calculate sum over selected columns for each row
        for i in range(n): # Iterate rows
            row_sum = 0.0 + 0.0j
            col_idx = 0
            temp_k = k
            while temp_k > 0:
                if temp_k & 1: # If j-th column is in subset S
                    row_sum += M[i, col_idx]
                temp_k >>= 1
                col_idx += 1
            sum_prod *= row_sum # Accumulate product

        # Calculate popcount (size of subset |S|) for sign
        temp_k = k
        while temp_k > 0:
            temp_k &= (temp_k - 1)
            popcount_k += 1

        # Calculate sign (-1)^(n - |S|)
        sign = -one_complex if (n - popcount_k) % 2 else one_complex

        # Add to total sum
        total_sum += sign * sum_prod

    # The formula uses (-1)^n * sum. Let's adjust the sign definition.
    # Ryser: perm(A) = sum_{S subset {1..n}} (-1)^{n-|S|} prod_{i=1..n} sum_{j in S} A_ij
    # Our loop calculates sum (-1)^{n-popcount} * prod(...) which matches.

    return total_sum

@njit(cache=True)
def _calculate_permanent_ryser_np_wrapper(sp_eigvecs: np.ndarray, # U matrix (Ns x Norb)
                                          occupied_orbitals: np.ndarray,
                                          basis_state_int: int,
                                          ns: int) -> complex:
    """
    Calculates the permanent using NumPy/Numba via Ryser's formula.
    Represents the amplitude <Fock(R)|State_B(O)>.

    Args:
        sp_eigvecs (np.ndarray): Eigenvector matrix U (Ns x Norb), where
                                 sp_eigvecs[i, α] = <i|ψ_α>. Must be complex or float.
        occupied_orbitals (np.ndarray): 1D array of integer indices of the
                                        N occupied single-particle orbitals {α_k}.
        basis_state_int (int): Integer representation of the Fock state in the
                               site basis. Bit `i` is 1 if site `i` is occupied.
        ns (int): Total number of sites/modes.

    Returns:
        complex: The value of the permanent perm(M), where M_{jk} = U_{i_j, α_k}.
                 Normalization factors are omitted.
    """
    n_particles = occupied_orbitals.shape[0]

    # Efficiently count set bits
    n_particles_fock = 0
    temp_basis = basis_state_int
    while temp_basis > 0:
        temp_basis &= (temp_basis - 1)
        n_particles_fock += 1

    if n_particles_fock != n_particles:
        return 0.0 + 0.0j

    if n_particles == 0:
        return 1.0 + 0.0j

    # Find indices of occupied sites
    occupied_sites = np.empty(n_particles, dtype=np.int64)
    count = 0
    for i in range(ns):
        if (basis_state_int >> i) & 1:
            occupied_sites[count] = i
            count += 1
            if count == n_particles:
                break

    # Construct the relevant N x N matrix M
    # M_{j,k} = U_{occupied_sites[j], occupied_orbitals[k]}
    M = np.empty((n_particles, n_particles), dtype=sp_eigvecs.dtype)
    for j in range(n_particles):
        site_j = occupied_sites[j]
        for k in range(n_particles):
            orbital_k = occupied_orbitals[k]
            M[j, k] = sp_eigvecs[site_j, orbital_k]

    # Calculate permanent using Ryser's formula
    perm = _calculate_permanent_np(M)
    return perm


# JAX version of Permanent using Ryser's Formula
try:
    import jax
    import jax.numpy as jnp
    from jax.lax import fori_loop, scan, psum, popcount
    from functools import partial

    # JAX implementation of Ryser's formula
    @partial(jax.jit, static_argnames=('n',))
    def _calculate_permanent_ryser_jax_core(M: jnp.ndarray, n: int) -> jnp.ndarray:
        """Core JAX Ryser's formula calculation."""
        if n == 0:
            return jnp.array(1.0 + 0.0j, dtype=jnp.complex128)

        # Constants
        one_complex = jnp.array(1.0 + 0.0j, dtype=jnp.complex128)
        minus_one_complex = jnp.array(-1.0 + 0.0j, dtype=jnp.complex128)
        zero_complex = jnp.array(0.0 + 0.0j, dtype=jnp.complex128)
        indices = jnp.arange(n)

        def loop_body(k, current_sum):
            # k represents the subset S (from 1 to 2^n - 1)
            # Create boolean mask for columns in subset S
            col_mask = ((k >> indices) & 1) == 1

            # Calculate sum over selected columns for each row
            # Use mask to select columns: M[:, col_mask] doesn't work directly this way in JAX?
            # Alternative: zero out non-selected columns
            masked_M = jnp.where(col_mask[None, :], M, zero_complex)
            row_sums = jnp.sum(masked_M, axis=1) # Sum over columns for each row

            # Product over rows: Π_{i=1..N} (sum_{j∈S} A_{ij})
            prod_row_sums = jnp.prod(row_sums)

            # Calculate popcount |S|
            # Use jax.lax.popcount if available, otherwise implement
            popcount_k = jnp.sum(col_mask) # Simple sum of boolean mask works
            # popcount_k = popcount(jnp.array(k, dtype=jnp.uint32)) # If using lax.popcount

            # Calculate sign (-1)^(n - |S|)
            exponent = n - popcount_k
            # sign = jnp.where(exponent % 2 == 1, minus_one_complex, one_complex)
            # Using powers might be simpler for JIT:
            sign = (minus_one_complex)**exponent

            # Add to total sum
            term = sign * prod_row_sums
            return current_sum + term

        # Loop from k = 1 to 2^n - 1
        initial_sum = zero_complex
        # Use fori_loop for better JIT compilation than python range
        final_sum = fori_loop(1, 1 << n, loop_body, initial_sum)

        return final_sum

    @partial(jax.jit, static_argnames=('ns',))
    def _calculate_permanent_jax(sp_eigvecs: jnp.ndarray, # U matrix (Ns x Norb)
                                 occupied_orbitals: jnp.ndarray,
                                 basis_state_int: int,
                                 ns: int) -> jnp.ndarray:
        """Calculates the permanent using JAX via Ryser's formula."""
        n_particles = occupied_orbitals.shape[0]

        # JAX way to count bits and get indices
        site_indices = jnp.arange(ns)
        occupied_sites_mask = ((basis_state_int >> site_indices) & 1) == 1
        n_particles_fock = jnp.sum(occupied_sites_mask)

        # Conditional execution in JAX
        def compute_permanent(sp_eigvecs, occupied_orbitals, occupied_sites_mask):
            occupied_sites = jnp.where(occupied_sites_mask, site_indices, -1)
            occupied_sites = occupied_sites[occupied_sites != -1] # Filter padding

            # Construct the N x N matrix M
            row_indices = occupied_sites[:, None]
            col_indices = occupied_orbitals[None, :]
            M = sp_eigvecs[row_indices, col_indices]

            # Calculate permanent using Ryser's core function
            perm = _calculate_permanent_ryser_jax_core(M.astype(jnp.complex128), n_particles)
            return perm

        # Return 0 if particle numbers don't match, otherwise calculate permanent
        perm_val = jax.lax.cond(
            n_particles == 0,
            lambda: jnp.array(1.0 + 0.0j, dtype=jnp.complex128), # N=0 case
            lambda: jax.lax.cond(
                n_particles_fock != n_particles,
                lambda: jnp.array(0.0 + 0.0j, dtype=jnp.complex128), # Mismatch case
                lambda: compute_permanent(sp_eigvecs, occupied_orbitals, occupied_sites_mask) # Match case
            )
        )
        return perm_val

except ImportError:
    _calculate_permanent_jax = None


# --- Wrapper Function (Example Usage) ---

def calculate_many_body_amplitude(particle_type: str, # 'fermion' or 'boson'
                                  backend: str,       # 'numpy' or 'jax'
                                  sp_eigvecs: np.ndarray, # U matrix (Ns x Norb)
                                  occupied_orbitals: list or np.ndarray,
                                  basis_state_int: int,
                                  ns: int) -> complex or np.ndarray:
    """
    Dispatcher function to calculate Slater determinant or permanent.

    Args:
        particle_type (str): 'fermion' or 'boson'.
        backend (str): 'numpy' or 'jax'.
        sp_eigvecs (np.ndarray): Eigenvector matrix U (Ns x Norb),
                                 <site|orbital>. Assumed NumPy for input, converted if JAX.
        occupied_orbitals (list/array): Indices of occupied orbitals.
        basis_state_int (int): Integer representation of the Fock state.
        ns (int): Number of sites.

    Returns:
        complex or jnp.ndarray: The calculated amplitude.
    """
    if backend == 'numpy':
        # Ensure inputs are numpy arrays for Numba function
        occ_orb_np = np.asarray(occupied_orbitals, dtype=np.int64)
        sp_eigvecs_np = np.asarray(sp_eigvecs) # Numba handles dtype

        if particle_type == 'fermion':
            # Numba version assumes sp_eigvecs_T input (<alpha|i>), need to adjust
            sp_eigvecs_T_np = sp_eigvecs_np.T.conj()
            return _calculate_slater_det_np(sp_eigvecs_T_np, occ_orb_np, basis_state_int, ns)
        elif particle_type == 'boson':
            return _calculate_permanent_ryser_np_wrapper(sp_eigvecs_np, occ_orb_np, basis_state_int, ns)
        else:
            raise ValueError("particle_type must be 'fermion' or 'boson'")

    elif backend == 'jax':
        if _calculate_slater_det_jax is None or _calculate_permanent_jax is None:
             raise ImportError("JAX not found or JAX functions failed to import.")
        # Ensure inputs are JAX arrays
        occ_orb_jnp = jnp.asarray(occupied_orbitals, dtype=jnp.int32) # JAX often uses int32
        sp_eigvecs_jnp = jnp.asarray(sp_eigvecs)

        if particle_type == 'fermion':
            return _calculate_slater_det_jax(sp_eigvecs_jnp, occ_orb_jnp, basis_state_int, ns)
        elif particle_type == 'boson':
            return _calculate_permanent_jax(sp_eigvecs_jnp, occ_orb_jnp, basis_state_int, ns)
        else:
            raise ValueError("particle_type must be 'fermion' or 'boson'")
    else:
        raise ValueError("backend must be 'numpy' or 'jax'")
