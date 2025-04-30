'''
file    : Algebra/Hilbert/hilbert_jit_states.py
author  : Maksymilian Kliczkowski
email   : maksymilian.kliczkowski@pwr.edu.pl
'''


from Algebra.Hilbert.hilbert_jit_methods import *
import numpy as np
from numba import njit, prange
from numba.typed import List
from typing import Union, Optional, Callable

#! jax
import Algebra.Hilbert.hilbert_jit_states_jax as jnp

####################################################################################################
#! NUMBA METHODS
####################################################################################################

if True:
    
    @njit(cache=True)
    def calculate_slater_det(sp_eigvecs         : np.ndarray,   # (Ns x Norb) matrix of eigenvectors
                            occupied_orbitals   : np.ndarray,   # 1D array of integer indices of the occupied single-particle orbitals {α_k}.
                            org_basis_state     : Union[int, np.ndarray],
                            ):
        """
        Calculates the Slater determinant using NumPy/Numba.
        
        It takes the eigenvectors of single particle Hamiltonian that represents 
        a vector of coefficients that return to the original basis state.
        
        After the diagonalization:
        
        $$ \hat{H} = \sum _n E_n b_n^\dagger b_n, $$
        where b_n^\dagger and b_n are the creation and annihilation operators for the n-th single-particle state.
        
        One constructs a single particle state by occupying the single particle orbitals as
        a product state:
        
        $$|ψ_α> = \Pi _ i b_i^\dagger |0>$$
        
        Now, to return to the original basis state, one needs to transform the coefficients according to
        
        $$
        b^\dagger_i = \sum _x U_{xi} a_x^\dagger,
        $$
        where U is the unitary matrix of eigenvectors
        and now inverting the transformation gives (where $a^\dagger_x$
        is the creation operator in the original basis - like occupied sites):
        $$
        a^\dagger_x = \sum _i U_{xi} b_i^\dagger.    
        $$ 
        
        The matrix of overlaps $\psi _{n_i} = {<i|ψ_n>}$
        is given by the eigenvectors of the single particle Hamiltonian.
        
        Therefore, one needs to check the overlaps of the occupied orbitals
        with the corresponding sites (occupation) or other basis to obtain
        the $\psi (x)$. One constructs the matrix from the definition:
        
        $$ M _{jk} = <i_j|q_k> = <0|a_j b_k^\dagger |0> = <0|a_j \sum _{x} U_{xk} a_x^\dagger |0> = U_{jk}$$
        
        and therefore the matrix row is simply given by Row[j] = U[occupied_sites[j], occupied_orbitals],
        where occupied_sites are the sites that are occupied in the Fock state (\vec{x}) and 
        occupied_orbitals are the orbitals that are occupied in the Fock state original product state
        with given energy eigenvalues $\Pi _i b_i^\dagger |0>$.
        
        Afterwards, this represents the amplitude connected to the slater determinant
        $$\psi(x) = \frac{1}{\sqrt{N!}} \det M

        Args:
            sp_eigvecs (np.ndarray):
                2D array of eigenvectors of the single-particle Hamiltonian.
                The shape is (ns, norb), where ns is the number of sites/modes
                and norb is the number of single-particle orbitals.
            occupied_orbitals (np.ndarray):
                1D array of integer indices of the
                N occupied single-particle orbitals {α_k}.
            org_basis_state (Union[int, np.ndarray]):
                The original basis state represented as a bit string or
                a 1D array of integers. The length of the array should be equal
                to the number of sites/modes (ns).

        Returns:
            complex or float:
                The value of the Slater determinant.
        """
        
        # number of sites
        ns                  = sp_eigvecs.shape[0]  
        
        # get the number of particles from the occupied orbitals
        n_particles         = occupied_orbitals.shape[0]

        # Efficiently count set bits (number of particles in Fock state)
        if isinstance(org_basis_state, (int, np.integer)):
            # Integer bitmask case
            occupied_sites      = np.empty(n_particles, dtype=np.int64)
            n_particles_fock    = 0
            idx_count           = 0
            temp_basis          = int(org_basis_state)
            for i in range(ns):
                if (temp_basis >> i) & 1:
                    n_particles_fock += 1
                    # Avoid buffer overflow if more bits are set than n_particles
                    if idx_count < n_particles:
                        occupied_sites[idx_count] = i
                    idx_count += 1
        elif isinstance(org_basis_state, np.ndarray):
            # Array case (assume boolean or 0/1 int)
            basis_state         = org_basis_state.astype(np.bool_) if org_basis_state.dtype != np.bool_ else org_basis_state
            n_particles_fock    = np.sum(basis_state)
            if n_particles_fock == n_particles:
                occupied_sites  = np.flatnonzero(basis_state)
                
        # Check if the number of particles in the Fock state matches the number of occupied orbitals
        # This is a necessary condition for the determinant to be non-zero.
        # If they don't match, return 0.0 + 0.0j
        if n_particles_fock != n_particles:
            return 0.0

        # Check if the number of particles is zero
        if n_particles == 0:
            return 1.0

        # Find indices of occupied sites
        occupied_sites = np.flatnonzero(basis_state)

        # Construct the Slater Matrix M (N x N)
        slater_matrix = np.empty((n_particles, n_particles), dtype=sp_eigvecs.dtype)
        for j in range(n_particles):
            # Get the index of the occupied site
            site_j = occupied_sites[j]
            
            # Fill the j-th row of the Slater matrix
            slater_matrix[j, :] = sp_eigvecs[site_j, 1]
            for k in range(n_particles):
                orbital_k           = occupied_orbitals[k]
                slater_matrix[j, k] = sp_eigvecs[site_j, orbital_k]

        # Calculate determinant
        # Use np.linalg.slogdet for potentially better numerical stability?
        sign, logdet = np.linalg.slogdet(slater_matrix)
        if sign == 0:
            return 0.0 + 0.0j
        det = sign * np.exp(logdet)
        # det = np.linalg.det(slater_matrix)
        return det
    
    #############################################################################
    
    @njit(cache=True)
    def calculate_bcs_amp(f_mat                     : np.ndarray,   # (ns x ns) pairing matrix f = v u^{-1}
                        occupied_sites              : np.ndarray,   # length 2N
                        pfaffian_function           : Callable
                        ):
        """
        Returns Psi(x) = Pf[f_{ij}] for occupied_sites i,j.
        
        The Pfaffian is computed using the antisymmetrized matrix f_{ij} = v_{ij} u^{-1}_{ij}.
        The function takes the pairing matrix f and the occupied sites as input.
        
        Parameters:
            f_mat (np.ndarray):
                2D array of the pairing matrix f_{ij} = v_{ij} u^{-1}_{ij}.
                The shape is (ns, ns), where ns is the number of sites/modes.
            occupied_sites (np.ndarray):
                1D array of integer indices of the occupied sites.
                The length should be 2N, where N is the number of particles.
            pfaffian_function (Callable):
                Function to compute the Pfaffian of a matrix.
        Returns:
            complex or float:
                The value of the coefficient Psi(x) = Pf[f_{ij}].     
        """
        N2      = occupied_sites.size
        #!TODO: Test this and implement further...
        
        # extract 2N×2N antisymmetric submatrix
        F_sub   = np.empty((N2, N2), dtype=f_mat.dtype)
        for p in range(N2):
            ip = occupied_sites[p]
            for q in range(N2):
                F_sub[p, q] = f_mat[ip, occupied_sites[q]]
        # compute Pfaffian via skew‐symmetric elimination
        return pfaffian_function(F_sub)
    
    #############################################################################
    #! Permanents
    #############################################################################
    
    