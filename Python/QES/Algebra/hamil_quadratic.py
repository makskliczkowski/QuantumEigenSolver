'''
file    : QES/Algebra/hamil_quadratic.py
author  : Maksymilian Kliczkowski
email   : maksymilian.kliczkowski@pwr.edu.pl

'''

import numpy as np
import scipy as sp

from typing import List, Tuple, Union, Optional, Callable
from enum import Enum, unique
from abc import ABC
from functools import partial

##############################################################################

@unique
class QuadraticTerm(Enum):
    '''
    Types of terms to be added to the quadratic Hamiltonian
    '''
    Onsite  =   0
    Hopping =   1
    Pairing =   2
    
    @property
    def mode_num(self):
        return 1 if self == QuadraticTerm.Onsite else 2
    
##############################################################################

from Algebra.hamil import Hamiltonian, HilbertSpace, Lattice, JAX_AVAILABLE
from Algebra.Hilbert.hilbert_jit_states import (
    calculate_slater_det,
    calculate_bcs_amp,
    calculate_permament
)
if JAX_AVAILABLE:
    import jax 
    import jax.numpy as jnp
    from jax.experimental.sparse import BCOO
    from Algebra.Hilbert.hilbert_jit_states_jax import (
    calculate_slater_det_jax,
    calculate_bcs_amp_jax,
    calculate_permament_jax
    )
else:
    jax                         = None
    jnp                         = None
    BCOO                        = None
    calculate_slater_det_jax    = None
    calculate_bcs_amp_jax       = None
    calculate_permament_jax     = None

##############################################################################

class QuadraticHamiltonian(Hamiltonian):
    """
    Specialized Hamiltonian for non-interacting (quadratic) systems.
    Builds and diagonalizes a smaller Ns x Ns (or 2Ns x 2Ns) matrix.
    Provides methods related to single-particle spectrum and transformations.
    
    The standard Ns x Ns matrix is used for particle-conserving systems, while
    the 2Ns x 2Ns matrix is used for non-particle-conserving systems (BdG).
    
    This class will provide the following functionalities:
    - Add quadratic terms (hopping, pairing, onsite).
    - Build the Hamiltonian matrix from the added terms.
    - Diagonalize the Hamiltonian matrix.
    - Calculate many-body energy from single-particle eigenvalues.
    - Provide methods to retrieve eigenvalues and eigenvectors of the Hamiltonian.
    """
    
    def __init__(self,
                ns                      : int, # Ns is mandatory
                particle_conserving     : bool                  = True,
                dtype                   : Optional[np.dtype]    = None,
                backend                 : str                   = 'default',
                is_sparse               : bool                  = False,
                constant_offset         : float                 = 0.0,
                particles               : str                   = 'fermions',
                # Allow passing lattice/logger
                hilbert_space           : Optional[HilbertSpace]= None,
                lattice                 : Optional[Lattice]     = None,
                logger                  : Optional['Logger']    = None,
                **kwargs):
        """
        Initialize a Quadratic Hamiltonian.

        Args:
            ns (int):
                Number of single-particle modes/sites.
            particle_conserving (bool):
                If True, uses Ns x Ns matrix. If False,
                implies pairing, potentially uses 2Ns x 2Ns
                Bogoliubov-de Gennes structure (requires specific build logic).
                (Default: True)
            dtype (Optional[data-type]):
                Matrix data type.
            backend (str):
                Computation backend ('np' or 'jax').
            particles (str):
                Whether the particles are fermions or bosons
                for the construction of a many-body state.
            is_sparse (bool):
                Use sparse matrices.
            constant_offset (float):
                A constant energy offset added *after* diagonalization.
            **kwargs:
                Passed to base class (e.g., logger, lattice).
        """
        
        # Call base class init, explicitly setting is_many_body=False
        super().__init__(is_many_body   =   False,
                        ns              =   ns,
                        lattice         =   lattice,
                        hilbert_space   =   hilbert_space,
                        is_sparse       =   is_sparse,
                        dtype           =   dtype,
                        backend         =   backend,
                        logger          =   logger
                        **kwargs)

        # setup the arguments first
        self._particle_conserving   = particle_conserving
        self._constant_offset       = constant_offset
        self._isfermions            = particles.lower() == 'fermions'
        self._isbosons              = not self._isfermions
        if self._hilbert_space.particle_conserving != particle_conserving:
            raise self._ERR_MODE_MISMATCH
        
        # Determine shape based on conservation (simple case)
        # BdG case (not particle conserving) needs 2Ns x 2Ns and different term handling.
        if not particle_conserving:
            self._log("Warning: particle_conserving=False implies BdG Hamiltonian structure. Ensure _build_quadratic handles 2Nsx2Ns matrix and pairing terms correctly.",
                    log='warning')
            self._hamil_sp_size     = 2 * ns
        else:
            self._hamil_sp_size     = ns
        
        # Set matrix shape
        self._hamil_sp_shape        = (self._hamil_sp_size, self._hamil_sp_size)
        self._dtypeint              = self._backend.int32 if self.ns < 2**32 - 1 else self._backend.int64

        # Store quadratic terms (hopping, pairing, onsite)
        self._quadratic_terms       = [] # List to store term info [(type, sites, value), ...]
        self._name                  = f"QuadraticHamiltonian(Ns={self._ns})"
        self._mb_calculator         = self._many_body_state_calculator()
        
    #!TODO: Add term adding such as in many-body?

    ##########################################################################
    #! Build the Hamiltonian
    ##########################################################################

    def _add_term(self,
                term_type   : QuadraticTerm,
                sites       : Union[Tuple, int],
                value       : Union[float, complex]):
        """
        Note:
            Is only for testing purposes probably
        
        Adds a quadratic term definition to be used during build.
        Hamiltonian is generally of this type:
            $$
            H = \sum _i h_i c_i+c_i + \sum _{i != j} t_{ij} c_i^+ c_j 
                + \sum _{i,j} \Delta _{ij} [c_i^+] c_i^+] + \Delta _{ij}^* c_jc_i]
            $$
        Args:
            term_type (str):
                Type of term ('onsite', 'hopping', 'pairing').
            sites (List[int]):
                List of site indices involved (1 for onsite, 2 for hopping/pairing).
                Indices must be in the range [0, ns-1].
            value (complex):
                Strength/value of the term (e.g., onsite energy, hopping amplitude, pairing gap).
        """
                
        # Validate term type and number of sites
        expected_sites = term_type.mode_num
        
        if len(sites) != expected_sites:
            raise ValueError(f"Term type '{term_type}' requires {expected_sites} site(s), got {len(sites)}: {sites}.")
        
        # Validate site indices
        if any(s < 0 or s >= self._ns for s in sites):
            raise IndexError(f"Site index out of bounds [0, {self._ns-1}] in term: {term_type}, {sites}")
        
        # Validate pairing term based on conservation flag
        if term_type == QuadraticTerm.Pairing and self._particle_conserving:
            self._log(f"Warning: Adding 'pairing' term {sites} while particle_conserving=True. Term will be ignored during build.", log='warning')
        if term_type != QuadraticTerm.Pairing and not self._particle_conserving:
            # Hopping/onsite terms are handled differently in BdG matrix construction
            pass # Logic is handled in _build_quadratic

        # Store term definition
        self._quadratic_terms.append({'type': term_type, 'sites': sites, 'value': value}) # Store value as complex
        self._log(f"Added term: {term_type}, sites={sites}, value={value:.4g}", lvl=3, log='debug')

        # Invalidate existing matrix if terms are added after build
        if self.hamil is not None:
            self.clear()

    #!TODO: Optimize this BS
    def _hamiltonian_quadratic(self, use_numpy: bool = False):
        """ 
        Builds the quadratic Hamiltonian matrix from the stored terms. 
        This is a general purpose method for convenience, otherwise, 
        we will allow to set the matrices differently - much faster
        """
        self._log(f"Building quadratic matrix ({self._hamil_sp_shape}, Sparse={self.sparse})...", lvl=1, color="blue")
        backend             = np if use_numpy or self._is_numpy else self._backend

        # empty
        if not self._quadratic_terms:
            self._log("No quadratic terms added. Resulting Hamiltonian matrix will be zero.", log='warning')
            self.init(use_numpy=use_numpy)
            return

        rows, cols, data    = [], [], []

        #! Populate COO lists based on terms
        for term in self._quadratic_terms:
            term_type       = term['type']
            sites           = term['sites']
            value           = term['value']

            try:
                if self._particle_conserving:
                    #! Particle Conserving Case (Ns x Ns Matrix)
                    if term_type == 'onsite':
                        i = sites[0]
                        rows.append(i)
                        cols.append(i)
                        data.append(value)
                    elif term_type == 'hopping': 
                        # t_ij c_i^dagger c_j + h.c.
                        i, j = sites[0], sites[1]
                        rows.append(i)
                        cols.append(j)
                        data.append(value) # H_ij = t_ij
                        if i != j:
                            rows.append(j) 
                            cols.append(i)
                            # H_ji = t_ij^*
                            data.append(value.conjugate())
                else:
                    #! Non-Particle Conserving Case (BdG: 2Ns x 2Ns Matrix)
                    Ns = self._ns
                    if term_type == QuadraticTerm.Onsite:
                        # mu_i c_i^dagger c_i
                        i = sites[0]
                        # H_hop block: H[i, i] = mu_i
                        rows.append(i)
                        cols.append(i)
                        data.append(value)
                        
                        # H_hop^T block: H[N+i, N+i] = -mu_i (diagonal is real)
                        rows.append(i + Ns)
                        cols.append(i + Ns)
                        data.append(-value)
                        
                    elif term_type == QuadraticTerm.Hopping:
                        # t_ij c_i^dagger c_j + h.c.
                        i, j = sites[0], sites[1]
                        # H_hop block: H[i, j] = t_ij, H[j, i] = t_ij^*
                        rows.append(i)
                        cols.append(j)
                        data.append(value)
                        if i != j: 
                            rows.append(j)
                            cols.append(i)
                            data.append(value.conjugate())
                        # -H_hop^T block:   H[N+i, N+j] = -H_hop[j, i] = -t_ij^*
                        #                   H[N+j, N+i] = -H_hop[i, j] = -t_ij
                        rows.append(i + Ns)
                        cols.append(j + Ns)
                        data.append(-value.conjugate())
                        if i != j: 
                            rows.append(j + Ns)
                            cols.append(i + Ns)
                            data.append(-value)
                    elif term_type == QuadraticTerm.Pairing:
                        # Delta_ij c_i c_j + Delta_ij^* c_j^dagger c_i^dagger
                        # Assume input 'value' is Delta_ij. We need antisymmetry Delta_ji = -Delta_ij.
                        i, j = sites[0], sites[1]
                        # Delta block (top right): H[i, N+j] = Delta_ij
                        rows.append(i)
                        cols.append(j + Ns)
                        data.append(value)
                        if i != j: 
                            rows.append(j)
                            cols.append(i + Ns)
                            # Delta_ji = -Delta_ij
                            data.append(-value)
                        # Delta^dagger block (bottom left): H[N+j, i] = Delta_ij^* (from h.c. of Delta_ij c_i c_j)
                        # Note the index swap for dagger block element definition H[N+row, col]
                        rows.append(j + Ns)
                        cols.append(i)
                        # Delta_ji^* = (-Delta_ij)^*
                        data.append(value.conjugate()) 
                        if i != j:
                            rows.append(i + Ns)
                            cols.append(j)
                            # Delta_ij^* = (-Delta_ji)^*
                            data.append(-value.conjugate())
            except IndexError:
                # This should ideally not happen due to initial checks in add_term
                self._log(f"Internal Error: Index out of bounds processing term {term}", log='error')
                raise

        #! Create Matrix from COO data
        try:
            coo_data        = backend.asarray(data, dtype=self._dtype)
            idx_dtype       = np.int32 if self._hamil_sp_size < 2**31 else np.int64
            jax_idx_dtype   = jnp.int32 if JAX_AVAILABLE and self._hamil_sp_size < 2**31 else jnp.int64

            if self.sparse:
                if backend == np:
                    coo_rows_np     = np.array(rows, dtype=idx_dtype)
                    coo_cols_np     = np.array(cols, dtype=idx_dtype)
                    self._hamil_sp  = sp.sparse.csr_matrix((coo_data, (coo_rows_np, coo_cols_np)), shape=self._hamil_sp_shape)
                    self._log(f"Built sparse quadratic matrix (CSR) with {self._hamil_sp.nnz} non-zeros.", lvl=2, log='debug')
                else: 
                    # JAX BCOO
                    coo_rows_jax    = backend.asarray(rows, dtype=jax_idx_dtype)
                    coo_cols_jax    = backend.asarray(cols, dtype=jax_idx_dtype)
                    indices         = backend.stack([coo_rows_jax, coo_cols_jax], axis=1)
                    # --- Build dense and convert for simplicity/robustness in JAX sparse ---
                    self._log("Building JAX sparse matrix via dense intermediate to handle potential duplicate COO entries.", log='debug')
                    H_dense         = backend.zeros(self._hamil_sp_shape, dtype=self._dtype)
                    H_dense         = H_dense.at[coo_rows_jax, coo_cols_jax].add(coo_data)
                    self._hamil_sp  = BCOO.fromdense(H_dense)
                    self._log(f"Built sparse quadratic matrix (BCOO) via dense intermediate with {self._hamil_sp.nse} non-zeros.", lvl=2, log='debug')
            else: # Dense
                H               = backend.zeros(self._hamil_sp_shape, dtype=self._dtype)
                coo_rows_be     = backend.asarray(rows, dtype=jax_idx_dtype if backend == jnp else idx_dtype)
                coo_cols_be     = backend.asarray(cols, dtype=jax_idx_dtype if backend == jnp else idx_dtype)
                H               = H.at[coo_rows_be, coo_cols_be].add(coo_data)
                self._hamil_sp  = H
                self._log("Built dense quadratic matrix.", lvl=2, log='debug')

        except Exception as e:
            self._log(f"Error during quadratic matrix final construction: {e}", log='error', color='red')
            self._hamil_sp      = None
            raise RuntimeError(f"Failed to construct quadratic matrix: {e}") from e

    ###########################################################################
    
    def diagonalize(self, verbose: bool = False, **kwargs):
        """
        Diagonalizes the quadratic matrix and applies constant offset
        for a constant term that can be included in the quadratic system. 
        """
        
        # Calls base diagonalize on self.hamil (which is _hamil_sp)
        super().diagonalize(verbose=verbose, **kwargs)
        
        # Apply constant offset after diagonalization
        if self._eig_val is not None and self._constant_offset != 0.0:
            if verbose:
                self._log(f"Adding constant offset {self._constant_offset} to eigenvalues.", lvl=2, log='debug')
            self._eig_val += self._constant_offset
            # Recalculate energy stats if offset was applied
            self._calculate_av_en()

    ###########################################################################

    def _set_local_energy_operators(self):
        """ Not applicable for standard quadratic Hamiltonians built from terms. """
        if self._is_quadratic:
            self._log("Method _set_local_energy_operators is not used for standard QuadraticHamiltonian build.", log='debug')
        else:
            raise NotImplementedError("ManyBody Hamiltonian subclass must implement _set_local_energy_operators.")

    ###########################################################################

    def many_body_energy(self, occupied_orbitals: Union[List[int], np.ndarray]) -> float:
        """
        Calculates the total energy of a many-body state defined by occupying
        single-particle orbitals (or quasiparticle orbitals for BdG).

        Args:
            occupied_orbitals (list/array):
                Indices of the occupied single-particle
                eigenstates (orbitals α or quasiparticles γ).

        Returns:
            The total energy E = Σ_{α∈occupied} ε_α (or E = Σ_{γ∈occupied} E_γ for BdG).
            Result includes the constant_offset.
        """
        
        if self.eig_val is None:
            raise ValueError("Single-particle eigenvalues not calculated. Call diagonalize() first.")

        occ_indices_arr     = self._backend.asarray(occupied_orbitals, dtype=self._dtypeint)
        num_eigvals         = self.eig_val.shape[0]

        if self._backend.any(occ_indices_arr < 0) or self._backend.any(occ_indices_arr >= num_eigvals):
            raise IndexError(f"Occupied orbital index out of bounds [0, {num_eigvals-1}].")

        if self._particle_conserving:
            return self._backend.sum(self.eig_val[occ_indices_arr])
        else:
            # BdG case: Eigenvalues come in pairs (+E_gamma, -E_gamma) sorted around 0.
            # Energy is sum of occupied *positive* quasiparticle energies E_gamma.
            # occupied_orbitals refer to indices 0..Ns-1 (positive energies)
            #   occupied_orbitals are indices k=0..Ns-1 referring to the
            #   positive energy quasiparticle states E_k (which are typically eig_val[Ns+k]).
            #   Ns = self._ns
            # positive_energy_indices = Ns + occ_indices_arr # Indices from Ns to 2Ns-1
            # if self._backend.any(positive_energy_indices >= num_eigvals):
            #      raise IndexError("Occupied BdG quasiparticle index implies index >= 2Ns.")
            # mb_energy = self._backend.sum(self.eig_val[positive_energy_indices])

            Ns              = self._ns
            middle_idx      = Ns - 1            # Index corresponding to "zero energy" or boundary
            mb_energy       = 0.0
            for i in occ_indices_arr:
                idx_plus    = middle_idx + i + 1
                idx_minus   = middle_idx - i

                # Check bounds
                if not (0 <= idx_plus < num_eigvals and 0 <= idx_minus < num_eigvals):
                    raise IndexError(f"BdG occupied index {i} leads to out-of-bounds eigenvalue access.")
                
                # Sum E_i + E_{-i}
                mb_energy   += self._eig_val[idx_plus] - self._eig_val[idx_minus]
        return mb_energy

    ###########################################################################

    def _many_body_state_calculator(self):
        '''
        Returns the function to calculate the many-body coefficients from
        '''
        #! Select Calculation Function
        calculator      = None
        if self._isfermions:
            if self._particle_conserving:
                if self._is_numpy:
                    calculator = calculate_slater_det
                else:
                    calculator = calculate_slater_det_jax
            else:
                if self._is_numpy:
                    calculator = calculate_bcs_amp
                else:
                    calculator = calculate_bcs_amp_jax
        elif self._isbosons:
            if self._is_numpy:
                calculator = calculate_permament
            else:
                calculator = calculate_permament_jax
        else: # Should not happen
            raise TypeError("Unknown particle type setting.")
        return calculator

    def many_body_state(self,
                            occupied_orbitals       : Union[List[int], np.ndarray],
                            target_basis            : str                       = 'sites',
                            many_body_hs            : Optional[HilbertSpace]    = None,
                            batch_size              = 1
                            ):
        """
        Constructs the Many-Body state vector in a specified basis using Slater
        determinants (fermions) or permanents (bosons).

        Args:
            occupied_orbitals (list/array):
                Indices (0 to Ns-1 or 0 to 2Ns-1 for BdG)
                of the occupied single-particle orbitals or quasiparticles.
            target_basis (str):
                The basis for the output vector ('sites' for Fock). (Default: 'sites')
            many_body_hs (Optional[HilbertSpace]):
                Target Many-Body HilbertSpace object. Required if using a symmetry-reduced basis.
                If None, assumes the full Fock space over Ns sites.

        Returns:
            np.ndarray:
                state vector in the target MB basis (NumPy or JAX array).
        """
        if target_basis.lower() != 'sites':
            raise NotImplementedError("Currently only supports transformation to site ('Fock') basis.")
        
        if self.eig_vec is None:
            raise ValueError("Single-particle eigenvectors not calculated. Call diagonalize() first.")
        
        if not self._particle_conserving and self._isbosons:
            raise NotImplementedError("Bosonic transformation for non-particle conserving (BdG) case not implemented.")

        #! Check the calculator function
        if self._calculator is None:
            raise RuntimeError(f"Calculation function for {'fermions' if self._isfermions else 'bosons'}.")

        #! Prepare Inputs
        # Use Ns (number of sites) for basis state representation size
        ns_sites        = self._ns
        # Eigenvectors: use sp_eig_vec property which points to self._eig_vec
        sp_eigvecs_arr  = self._backend.asarray(self.eig_vec)
        # Occupied orbitals: ensure correct backend array type
        occ_orb_arr     = self._backend.asarray(occupied_orbitals, dtype=self._dtypeint)
        #!TODO: Finish!
        
        #! Determine Target Basis
        # (Logic remains the same as previous refinement using many_body_hilbert_space)
        use_representatives = False
        target_hs           = many_body_hs

        if target_hs is not None:
             if not target_hs._is_many_body: raise ValueError("Target HilbertSpace must be many-body.")
             if target_hs.get_Ns() != ns_sites: raise ValueError(f"Ns mismatch: QuadH Ns={ns_sites}, Target MB HS Ns={target_hs.get_Ns()}.")
             if target_hs.mapping is not None: # Use reduced basis
                  use_representatives = True
                  target_basis_states = target_hs.mapping # These are representative states (int)
                  target_size = target_hs.get_Nh()
                  self._log(f"Constructing MB state in reduced basis (Nh={target_size}).", log='debug')
             else: # Use full basis filtered by global symmetries
                   full_map_list = target_hs.get_full_map_int() # Returns list or None
                   if full_map_list is None: # Assume full space if no global syms
                       target_size = target_hs.get_Nh_full()
                       target_basis_states = self._backend.arange(target_size, dtype=self._dtypeint)
                   else:
                       target_basis_states = self._backend.asarray(full_map_list, dtype=self._dtypeint)
                       target_size = len(target_basis_states)
                   self._log(f"Constructing MB state in full basis satisfying global symmetries (Size={target_size}).", log='debug')
        else: # Assume full Fock space over ns_sites
             target_size = 2**ns_sites
             if target_size > 1e7: # Warning for large spaces
                  self._log(f"Warning: Constructing state in full Fock space (Ns={ns_sites}, NhFull={target_size}). This may be slow/memory intensive.", log='warning')
             target_basis_states = self._backend.arange(target_size, dtype=self._dtypeint) # Iterate all ints
             self._log(f"Constructing MB state in full Fock basis (NhFull={target_size}).", log='debug')

        # --- Perform Calculation (Vectorized or Loop) ---
        mb_state_vector_unnorm : np.ndarray # Type hint
        if self._is_jax:
             if not _HAS_JAX_HELPERS: raise RuntimeError("JAX helpers not available.")
             self._log("Using JAX vmap for state construction.", log='debug')
             # JAX calculator expects ns as static arg if jitted that way
             vmap_calculator = jax.vmap(calculator, in_axes=(None, None, 0, None))
             mb_state_vector_unnorm = vmap_calculator(sp_eigvecs_arr, occ_orb_arr, target_basis_states, ns_sites)
        else: # NumPy
             if not _HAS_NUMBA_HELPERS: raise RuntimeError("NumPy/Numba helpers not available.")
             self._log("Using NumPy loop for state construction.", log='debug')
             mb_state_vector_unnorm = np.zeros(target_size, dtype=np.complex128)
             # Convert target_basis_states to NumPy for Numba loop if it's not already
             target_basis_states_np = np.asarray(target_basis_states)
             sp_eigvecs_np = np.asarray(sp_eigvecs_arr) # Ensure NumPy array for Numba
             occ_orb_np = np.asarray(occ_orb_arr)       # Ensure NumPy array for Numba

             for i in range(target_size):
                 basis_int_scalar = target_basis_states_np[i].item() # Use item() for scalar
                 amplitude = calculator(sp_eigvecs_np, occ_orb_np, basis_int_scalar, ns_sites)
                 mb_state_vector_unnorm[i] = amplitude
             # Convert back to backend array if needed (though usually stays np here)
             # mb_state_vector_unnorm = self._backend.asarray(mb_state_vector_unnorm)


        # --- Normalization & Symmetry Adjustment ---
        self._log("Normalizing generated many-body state...", log='debug')
        norm_sq = self._backend.sum(self._backend.abs(mb_state_vector_unnorm)**2)
        norm_factor = self._backend.sqrt(norm_sq)
        # Avoid division by zero/NaN
        mb_state_vector = self._backend.where(norm_factor > 1e-14,
                                              mb_state_vector_unnorm / norm_factor,
                                              self._backend.zeros_like(mb_state_vector_unnorm))

        if use_representatives and target_hs.normalization is not None:
             self._log("Applying symmetry normalization factor...", log='debug')
             norms_hs = target_hs.normalization # Should match target_size
             # Divide by sqrt(orbit_size) = norm_hs
             safe_norms = self._backend.where(self._backend.abs(norms_hs) > 1e-14, norms_hs, 1.0)
             mb_state_vector = mb_state_vector / safe_norms
             # Re-normalize after applying symmetry factors (optional but recommended)
             norm_sq_final = self._backend.sum(self._backend.abs(mb_state_vector)**2)
             norm_factor_final = self._backend.sqrt(norm_sq_final)
             mb_state_vector = self._backend.where(norm_factor_final > 1e-14,
                                                   mb_state_vector / norm_factor_final,
                                                   self._backend.zeros_like(mb_state_vector))

        self._log("Many-body state construction complete.", log='debug')
        return mb_state_vector