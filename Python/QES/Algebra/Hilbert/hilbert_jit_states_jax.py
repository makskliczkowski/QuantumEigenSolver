import numpy as np
from typing import Union, Optional, Callable
from functools import partial

from general_python.algebra.utils import JAX_AVAILABLE

if JAX_AVAILABLE:
    import jax.numpy as jnp
    import jax
else:
    jax = None
    jnp = np

#######################################################################

if JAX_AVAILABLE:
    
    @partial(jax.jit)
    def calculate_slater_det_jax(sp_eigvecs         : jnp.ndarray,      # U matrix (Ns x Norb)
                                occupied_orbitals   : jnp.ndarray,      # Indices {α_k}
                                org_basis_state     : Union[int, jnp.ndarray],
                                ns                  : int
                                ) -> jnp.ndarray: # Returns JAX array (scalar)
        """
        Calculates the Slater determinant using JAX.

        Represents the amplitude <Fock(R)|State_F(O)>. See NumPy version docstring
        for mathematical background.

        Args:
            sp_eigvecs (jnp.ndarray):
                Eigenvector matrix U (shape: Ns x Norb). Assumes columns are eigenvectors,
                so sp_eigvecs[i, α] = <i|ψ_α>.
            occupied_orbitals (jnp.ndarray):
                1D array (length N) of integer indices of the occupied orbitals {α_k}.
            org_basis_state (Union[int, jnp.ndarray]):
                Represents the Fock state R.
                If int: 
                    Bitmask where bit i is set if site i is occupied.
                If jnp.ndarray:
                    Boolean or integer array (length Ns) where entry i is
                    1 or True if site i is occupied.
        Returns:
            jnp.ndarray: Scalar JAX array containing the complex value of the
                        Slater determinant det(M).
        """
        ns              = sp_eigvecs.shape[0]
        n_particles     = occupied_orbitals.shape[0]


        #! Determine occupied sites mask and check particle number
        site_indices = jnp.arange(ns, dtype=jnp.int32)
        if isinstance(org_basis_state, (int, np.integer)):
            basis_state_mask = ((int(org_basis_state) >> site_indices) & 1) == 1
        elif isinstance(org_basis_state, jnp.ndarray):
            basis_state_mask = org_basis_state.astype(jnp.bool_)
        else:
            # JAX requires traceable types. Raise error outside JIT context if needed.
            # Inside JIT, return 0 or handle based on expected usage.
            # For safety within JIT, return 0 if type is unexpected.
            return jnp.zeros(1)

        n_particles_fock = jnp.sum(basis_state_mask)

        #! Conditional Calculation
        def compute_det(sp_eigvecs, occupied_orbitals, basis_state_mask):
            # Get occupied site indices using nonzero with fixed size.
            # This requires n_particles_fock == n_particles, ensured by outer cond.
            occupied_sites  = jnp.nonzero(basis_state_mask, size=n_particles, fill_value=-1)[0]
            # Ensure no fill_value used (safety check, shouldn't happen if counts match)
            valid_indices   = jnp.all(occupied_sites != -1)

            #! Construct Slater Matrix M using vectorized indexing
            # M_{jk} = U_{i_j, α_k} = sp_eigvecs[occupied_sites[j], occupied_orbitals[k]]
            row_idx         = occupied_sites[:, None]       # Shape (N, 1)
            col_idx         = occupied_orbitals[None, :]    # Shape (1, N)
            slater_matrix   = sp_eigvecs[row_idx, col_idx]  # Shape (N, N)

            #! Calculate Determinant
            sign, logdet    = jnp.linalg.slogdet(slater_matrix)

            # Combine sign and logdet, handle sign=0 (singular matrix)
            det_val = jnp.where(sign == 0,
                                jnp.zeros(1, dtype=logdet.dtype),
                                sign * jnp.exp(logdet))

            # Return 0 if indices were invalid (e.g., wrong particle count detected subtly)
            return jnp.where(valid_indices, det_val, jnp.zeros(1, dtype=det_val.dtype))

        # Use jax.lax.cond for conditional execution based on particle numbers
        det_result = jax.lax.cond(
            n_particles == 0,
            lambda: jnp.ones(1),        # N=0 case
            lambda: compute_det(sp_eigvecs, occupied_orbitals, basis_state_mask)
        )
        return det_result
    
    #############################################################################
    #! Bogolubov - de'Gennes - a BCS-like state amplitudes
    #############################################################################
    
    def calculate_bcs_amp_jax(f_mat                 : jnp.ndarray,   # (ns x ns) pairing matrix f = v u^{-1}
                        occupied_sites              : jnp.ndarray,   # length 2N
                        pfaffian_function           : Callable
                        ):
        pass
    
    #############################################################################
    #! Permanents
    #############################################################################
    
    @jax.jit
    def calculate_permament_jax(sp_eigvecs          : jnp.ndarray,   # (Ns x Norb) matrix of eigenvectors
                            occupied_orbitals       : jnp.ndarray,   # 1D array of integer indices of the occupied single-particle orbitals {α_k}.
                            org_basis_state         : Union[int, jnp.ndarray]):
        return 1.0
else:
    def calculate_slater_det_jax(sp_eigvecs         : np.ndarray,
                                occupied_orbitals   : np.ndarray,
                                org_basis_state     : Union[int, np.ndarray],
                                ns                  : int
                            ) -> np.ndarray:
        """
        JAX is not available. This function is a placeholder.
        """
        raise ImportError("JAX is not available. Cannot compute Slater determinant.")
    
    def calculate_bcs_amp_jax(f_mat                 : np.ndarray,   # (ns x ns) pairing matrix f = v u^{-1}
                        occupied_sites              : np.ndarray,   # length 2N
                        pfaffian_function           : Callable
                        ):
        pass
    
    def calculate_permament_jax(sp_eigvecs          : np.ndarray,   # (Ns x Norb) matrix of eigenvectors
                            occupied_orbitals       : np.ndarray,   # 1D array of integer indices of the occupied single-particle orbitals {α_k}.
                            org_basis_state         : Union[int, np.ndarray]):
        return 1.0
        
#######################################################################