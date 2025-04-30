'''
file    : Algebra/Hilbert/hilbert_jit_states.py
author  : Maksymilian Kliczkowski
email   : maksymilian.kliczkowski@pwr.edu.pl
'''


from Algebra.Hilbert.hilbert_jit_methods import *
import numpy as np
from numba import njit, prange
from numba.typed import List

####################################################################################################
#! NUMBA METHODS
####################################################################################################

if True:
    @njit(cache=True)
    def calculate_slater_det_np(sp_eigvecs_T        : np.ndarray,   # Transposed: (Orbital x Site)
                                occupied_orbitals   : np.ndarray,   # 1D array of integer indices of the occupied single-particle orbitals {α_k}.
                                basis_state         : np.ndarray,   # 1D array of 0s and 1s
                                ns                  : int):
        """
        Calculates the Slater determinant using NumPy/Numba.

        Represents the amplitude <Fock(R)|State_F(O)>
        where R are occupied sites
        derived from basis_state, and O are the occupied_orbitals.

        Args:
            sp_eigvecs_T (np.ndarray): 
                Transposed eigenvector matrix (Norb x Ns),
                where sp_eigvecs_T[α, i] = <ψ_α|i>.
                Must be complex or float.
            occupied_orbitals (np.ndarray):
                1D array of integer indices of the
                N occupied single-particle orbitals {α_k}.
            basis_state (np.ndarray):
                1D array of 0s and 1s representing the Fock state.
                The number of particles in the Fock state
                is equal to the number of 1s in the array.
            ns (int):
                Total number of sites/modes.

        Returns:
            complex or float:
                The value of the Slater determinant det(M),
                where
                    M_{jk} = <i_j|ψ_{α_k}> = U_{i_j, α_k} = sp_eigvecs_T[α_k, i_j]^*
                    (Note the conjugate if using <ψ|i> format).
                Assuming U_{iα} = <i|ψ_α>, then M_{jk} = U_{i_j, α_k}.
                If sp_eigvecs_T[α, i] = <ψ_α|i>, then U_{iα}^* = sp_eigvecs_T[α, i],
                so U_{iα} = sp_eigvecs_T[α, i]^*.
                M_{jk} = U_{i_j, α_k} = sp_eigvecs_T[α_k, i_j]^*.
                Let's assume sp_eigvecs_T[α, i] = U_{iα}^* = ⟨ψ_α|i⟩ directly.
                Then M_{jk} = ⟨i_j|ψ_{α_k}⟩ = U_{i_j, α_k}.
                Requires U matrix = sp_eigvecs_T.T.conj()
                If we define M_{jk} = sp_eigvecs_T[α_k, i_j] for simplicity:
                This corresponds to <ψ_{α_k}|i_j> if T matrix is <ψ|i>.
        """
        
        # get the number of particles from the occupied orbitals
        n_particles         = occupied_orbitals.shape[0]

        # Efficiently count set bits (number of particles in Fock state)
        n_particles_fock    = np.sum(basis_state > 0)

        # Check if the number of particles in the Fock state matches the number of occupied orbitals
        # This is a necessary condition for the determinant to be non-zero.
        # If they don't match, return 0.0 + 0.0j
        if n_particles_fock != n_particles:
            return 0.0 + 0.0j

        # Check if the number of particles is zero
        if n_particles == 0:
            return 1.0 + 0.0j

        # Find indices of occupied sites
        occupied_sites = np.flatnonzero(basis_state)

        # Construct the Slater Matrix M (N x N)
        # M_{jk} = <site j | orbital k> = U_{ij, αk}
        # If sp_eigvecs_T[α, i] = <ψ_α|i>, then U = sp_eigvecs_T.T.conj()
        # M_{jk} = U[occupied_sites[j], occupied_orbitals[k]]
        #      = sp_eigvecs_T.T.conj()[occupied_sites[j], occupied_orbitals[k]]
        #      = sp_eigvecs_T[occupied_orbitals[k], occupied_sites[j]].conj()

        # Let's assume sp_eigvecs_T is already U^dagger, so sp_eigvecs_T[alpha, i] = <alpha | i>
        # Then U = sp_eigvecs_T.T, and M_jk = U[occupied_sites[j], occupied_orbitals[k]]
        # M_jk = sp_eigvecs_T.T[occupied_sites[j], occupied_orbitals[k]]
        # M_jk = sp_eigvecs_T[occupied_orbitals[k], occupied_sites[j]] # Check indices again!
        # If M_jk = <site_j | orbital_k>, then M should use eigenvectors directly U.
        # Let U = sp_eigvecs (Ns x Norb). M_jk = U[occupied_sites[j], occupied_orbitals[k]]

        # Let's redefine: Assume sp_eigvecs is Ns x Norb, where sp_eigvecs[:, alpha] = |psi_alpha> in site basis.
        # Then sp_eigvecs[i, alpha] = <i | psi_alpha> = U_{i, alpha}
        # M_{j,k} = U_{occupied_sites[j], occupied_orbitals[k]}

        # Reverting to original code's assumption for consistency:
        # M_{jk} = sp_eigvecs[occupied_sites[j], occupied_orbitals[k]] seems correct if sp_eigvecs is U.
        # Let's rename sp_eigvecs_T to sp_eigvecs (Ns x Norb) for clarity.

        # Renaming input for clarity
        sp_eigvecs = sp_eigvecs_T.T.conj() # U matrix (Ns x Norb) from <alpha|i> input

        slater_matrix = np.empty((n_particles, n_particles), dtype=sp_eigvecs.dtype)
        for j in range(n_particles):
            site_j = occupied_sites[j]
            for k in range(n_particles):
                orbital_k = occupied_orbitals[k]
                slater_matrix[j, k] = sp_eigvecs[site_j, orbital_k]

        # Calculate determinant
        # Use np.linalg.slogdet for potentially better numerical stability?
        # sign, logdet = np.linalg.slogdet(slater_matrix)
        # if sign == 0:
        #     return 0.0 + 0.0j
        # det = sign * np.exp(logdet)
        # Using standard det for simplicity now:
        det = np.linalg.det(slater_matrix)

        return det