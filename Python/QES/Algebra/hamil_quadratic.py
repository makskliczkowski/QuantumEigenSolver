'''
file    : QES/Algebra/hamil_quadratic.py
author  : Maksymilian Kliczkowski
email   : maksymilian.kliczkowski@pwr.edu.pl

'''

import numpy as np
import scipy as sp
from typing import List, Tuple, Union, Optional, Callable
from abc import ABC
from functools import partial
import time

from QES.Algebra.hamil import Hamiltonian, HilbertSpace

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
                hilbert_space           : Optional[HilbertSpace]= None,
                particle_conserving     : bool                  = True,
                dtype                   : Optional[np.dtype]    = None,
                backend                 : str                   = 'default',
                is_sparse               : bool                  = False,
                constant_offset         : float                 = 0.0,
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
                        hilbert_space   =   None, # No MB HilbertSpace needed directly
                        is_sparse       =   is_sparse,
                        dtype           =   dtype,
                        backend         =   backend,
                        **kwargs)

        self._particle_conserving   = particle_conserving
        self._constant_offset       = constant_offset

        # Determine shape based on conservation (simple case)
        # BdG case (not particle conserving) needs 2Ns x 2Ns and different term handling.
        if not particle_conserving:
            self._log("Warning: particle_conserving=False implies BdG Hamiltonian structure. Ensure _build_quadratic handles 2Nsx2Ns matrix and pairing terms correctly.", log='warning')
            self._hamil_sp_size     = 2 * ns
        else:
            self._hamil_sp_size     = ns
        
        # Set matrix shape
        self._hamil_sp_shape        = (self._hamil_sp_size, self._hamil_sp_size)

        # Store quadratic terms (hopping, pairing, onsite)
        self._quadratic_terms       = [] # List to store term info [(type, sites, value), ...]
        self._name                  = f"QuadraticHamiltonian(Ns={self._ns})"

    # Term Management
    def add_term(self, term_type: str, sites: List[int], value: Union[float, complex]):
        """ Add a quadratic term definition (e.g., hopping, pairing, onsite). """
        # Perform basic validation
        term_type = term_type.lower()
        expected_sites = 0
        if term_type == 'onsite': expected_sites = 1
        elif term_type in ['hopping', 'pairing']: expected_sites = 2
        else: raise ValueError(f"Unknown quadratic term type: {term_type}")

        if len(sites) != expected_sites:
            raise ValueError(f"Term type '{term_type}' requires {expected_sites} site(s), got {len(sites)}.")
        if any(s < 0 or s >= self._ns for s in sites): # Basic bounds check on original Ns sites
             raise IndexError(f"Site index out of bounds [0, {self._ns-1}] in term: {term_type}, {sites}")

        # Specific checks for BdG terms if not particle conserving
        if not self._particle_conserving and term_type == 'pairing':
             # Pairing Δ_{ij} c_i c_j involves indices up to 2Ns in BdG representation
             pass # Add specific BdG matrix index logic if needed here or in build

        self._quadratic_terms.append({'type': term_type, 'sites': sites, 'value': value})
        self._log(f"Added quadratic term: {term_type} on {sites} with value {value}", lvl=3, log='debug')
        # Reset Hamiltonian if terms are added after build
        if self.hamil is not None: self.clear()

    # --- Build Implementation ---
    def _build_quadratic(self, use_numpy: bool = False):
        """ Overrides base method. Builds the Ns x Ns (or 2Ns x 2Ns) matrix. """
        self._log(f"Building quadratic matrix ({self._hamil_sp_shape}, Sparse={self.sparse})...", lvl=1, color="blue")
        backend = np if use_numpy or self._is_numpy else self._backend # np or jnp

        rows, cols, data = [], [], [] # COO format lists

        for term in self._quadratic_terms:
            term_type = term['type']
            sites = term['sites']
            value = term['value']

            # --- Logic to get matrix elements for each term ---
            if self._particle_conserving:
                # --- Particle Conserving Case (Ns x Ns Matrix) ---
                if term_type == 'onsite':
                    i = sites[0]
                    rows.append(i); cols.append(i); data.append(value)
                elif term_type == 'hopping': # t_ij c_i^dagger c_j + h.c.
                    i, j = sites[0], sites[1]
                    rows.append(i); cols.append(j); data.append(value)      # H_ij = t_ij
                    # Add hermitian conjugate only if i != j to avoid double counting diagonal
                    if i != j:
                         rows.append(j); cols.append(i); data.append(self._backend.conjugate(value)) # H_ji = t_ij^*
                elif term_type == 'pairing':
                     self._log("Warning: 'pairing' term added but particle_conserving=True. Ignoring term.", log='warning')

            else:
                # --- Non-Particle Conserving Case (BdG: 2Ns x 2Ns Matrix) ---
                # Assumes basis (c_0, ..., c_{N-1}, c_0^\dagger, ..., c_{N-1}^\dagger)
                # H_BdG = [  H_hopping     Delta       ]
                #         [ Delta^dagger  -H_hopping^T ]
                Ns = self._ns
                if term_type == 'onsite': # mu_i c_i^dagger c_i -> mu_i in H_hop[i,i], -mu_i in -H_hop^T[N+i, N+i]
                    i = sites[0]
                    rows.append(i); cols.append(i); data.append(value)          # H_hop block
                    rows.append(i + Ns); cols.append(i + Ns); data.append(-value) # -H_hop^T block (diagonal is real)
                elif term_type == 'hopping': # t_ij c_i^dagger c_j + h.c.
                    i, j = sites[0], sites[1]
                    # H_hop block
                    rows.append(i); cols.append(j); data.append(value)
                    if i != j: rows.append(j); cols.append(i); data.append(self._backend.conjugate(value))
                    # -H_hop^T block (careful with conjugate and sign)
                    # Element (N+i, N+j) corresponds to -H_hop^T[i, j] = -H_hop[j, i]
                    rows.append(i + Ns); cols.append(j + Ns); data.append(-self._backend.conjugate(value)) # -H_ji^* = -t_ij
                    if i != j: rows.append(j + Ns); cols.append(i + Ns); data.append(-value) # -H_ij^* = -t_ij^*
                elif term_type == 'pairing': # Delta_ij c_i c_j + h.c. (Assume Delta_ij = -Delta_ji for fermions)
                    # This term contributes to off-diagonal blocks Delta and Delta^dagger
                    i, j = sites[0], sites[1]
                    # Delta block (top right): Element (i, N+j) is Delta_ij
                    rows.append(i); cols.append(j + Ns); data.append(value)
                    # Delta_ji = -Delta_ij if fermionic pairing assumed
                    if i != j: rows.append(j); cols.append(i + Ns); data.append(-value)
                    # Delta^dagger block (bottom left): Element (N+i, j) is Delta_ji^* = (-Delta_ij)^*
                    rows.append(i + Ns); cols.append(j); data.append(self._backend.conjugate(value)) # Delta_ij^*
                    if i != j: rows.append(j + Ns); cols.append(i); data.append(-self._backend.conjugate(value)) # (-Delta_ij)^*


        # --- Create Matrix from COO data ---
        try:
            # Convert data list to backend array
            coo_data = backend.asarray(data, dtype=self._dtype)

            if not rows: # Handle case with no terms added
                 self._log("No quadratic terms added, creating empty/zero matrix.", log='warning')
                 if self.sparse:
                     if backend == np: self._hamil_sp = sp.sparse.csr_matrix(self._hamil_sp_shape, dtype=self._dtype)
                     else: # JAX BCOO
                          indices = backend.zeros((0, 2), dtype=ACTIVE_INT_TYPE)
                          data_empty = backend.zeros((0,), dtype=self._dtype)
                          self._hamil_sp = BCOO((data_empty, indices), shape=self._hamil_sp_shape)
                 else: # Dense
                      self._hamil_sp = backend.zeros(self._hamil_sp_shape, dtype=self._dtype)
                 return # Exit build

            # Ensure indices are correct integer type for backend
            idx_dtype = np.int32 if max(self._hamil_sp_size, len(rows)) < 2**31 else np.int64 # Choose appropriate numpy int type
            jax_idx_dtype = jnp.int32 if max(self._hamil_sp_size, len(rows)) < 2**31 else jnp.int64 # JAX int type

            if self.sparse:
                if backend == np:
                    coo_rows_np = np.array(rows, dtype=idx_dtype)
                    coo_cols_np = np.array(cols, dtype=idx_dtype)
                    # Sum duplicates implicitly during conversion
                    self._hamil_sp = sp.sparse.csr_matrix((coo_data, (coo_rows_np, coo_cols_np)), shape=self._hamil_sp_shape)
                    self._log(f"Built sparse quadratic matrix (CSR) with {self._hamil_sp.nnz} non-zeros.", lvl=2, log='debug')
                else: # JAX BCOO
                    coo_rows_jax = backend.asarray(rows, dtype=jax_idx_dtype)
                    coo_cols_jax = backend.asarray(cols, dtype=jax_idx_dtype)
                    indices = backend.stack([coo_rows_jax, coo_cols_jax], axis=1)
                    # BCOO requires summing duplicates explicitly *before* creation if COO data has them.
                    # This is complex. Simplest JAX approach is often dense build then sparsify if needed,
                    # or use external libraries, or ensure COO data has unique indices.
                    # For now, demonstrate the naive BCOO build (assumes unique indices or requires pre-processing)
                    num_entries = len(coo_data)
                    # Check for duplicate indices (slow for large data)
                    unique_indices_flat = np.unique(np.array(rows)*self._hamil_sp_shape[1] + np.array(cols))
                    if len(unique_indices_flat) < num_entries:
                         self._log("Warning: Duplicate indices found in COO data for JAX sparse build. Summing required but not implemented here. Result may be incorrect or slow.", log='warning')
                         # Fallback: build dense then convert (inefficient)
                         H_dense = backend.zeros(self._hamil_sp_shape, dtype=self._dtype)
                         H_dense = H_dense.at[coo_rows_jax, coo_cols_jax].add(coo_data) # Sums duplicates
                         self._hamil_sp = BCOO.fromdense(H_dense)
                         self._log(f"Built sparse quadratic matrix (BCOO) via dense intermediate with {self._hamil_sp.nse} non-zeros.", lvl=2, log='debug')
                    else:
                         # Assume unique indices for direct BCOO creation
                         self._hamil_sp = BCOO((coo_data, indices), shape=self._hamil_sp_shape)
                         self._log(f"Built sparse quadratic matrix (BCOO) directly with {self._hamil_sp.nse} non-zeros.", lvl=2, log='debug')

            else: # Dense
                H = backend.zeros(self._hamil_sp_shape, dtype=self._dtype)
                coo_rows_be = backend.asarray(rows, dtype=jax_idx_dtype if backend == jnp else idx_dtype)
                coo_cols_be = backend.asarray(cols, dtype=jax_idx_dtype if backend == jnp else idx_dtype)
                # Use .at[].add() for efficient summation of duplicates
                H = H.at[coo_rows_be, coo_cols_be].add(coo_data)
                self._hamil_sp = H
                self._log("Built dense quadratic matrix.", lvl=2, log='debug')

        except Exception as e:
            self._log(f"Error during quadratic matrix construction: {e}", log='error', color='red')
            raise

    # --- Overridden Methods ---
    def diagonalize(self, verbose: bool = False, **kwargs):
        """ Diagonalizes the quadratic matrix and applies constant offset. """
        super().diagonalize(verbose=verbose, **kwargs) # Calls base diagonalize on self.hamil (which is _hamil_sp)
        # Apply constant offset after diagonalization
        if self._eig_val is not None and self._constant_offset != 0.0:
            if verbose: self._log(f"Adding constant offset {self._constant_offset} to eigenvalues.", lvl=2)
            self._eig_val += self._constant_offset
            # Recalculate energy stats if offset was applied
            self._calculate_av_en()

    def _set_local_energy_operators(self):
        """ Not applicable for standard quadratic Hamiltonians built from terms. """
        if self._is_quadratic:
             self._log("Method _set_local_energy_operators is not used for standard QuadraticHamiltonian build.", log='debug')
        else: # Should be implemented by ManyBody subclasses
             raise NotImplementedError("ManyBody Hamiltonian subclass must implement _set_local_energy_operators.")


    # --- Quadratic Specific Methods ---
    def get_many_body_energy_from_sp(self, occupied_sp_indices: Union[List[int], np.ndarray]) -> float:
        """ Calculates many-body energy from occupied single-particle eigenvalues. """
        if not self._particle_conserving:
             # BdG case: Energy is sum E_alpha for occupied quasiparticles alpha > 0
             raise NotImplementedError("Energy calculation for BdG quasiparticles not implemented here.")
        if self.sp_eig_val is None:
            raise ValueError("Single-particle eigenvalues not calculated. Diagonalize first.")

        occupied_sp_indices_arr = self._backend.asarray(occupied_sp_indices)
        # Ensure indices are valid
        if self._backend.any(occupied_sp_indices_arr < 0) or self._backend.any(occupied_sp_indices_arr >= len(self.sp_eig_val)):
             raise IndexError("Occupied orbital index out of bounds.")

        # Sum eigenvalues corresponding to occupied orbitals (+ offset already handled by diagonalize)
        mb_energy = self._backend.sum(self.sp_eig_val[occupied_sp_indices_arr])
        return mb_energy

    def get_many_body_state(self,
                            occupied_orbitals: Union[List[int], np.ndarray],
                            particle_type: str = 'fermion',
                            target_basis: str = 'sites', # 'sites' or 'orbitals' (trivial)
                            many_body_hilbert_space: Optional[HilbertSpace] = None
                            ) -> np.ndarray:
        """
        Constructs the Many-Body state vector in a specified basis.

        Args:
            occupied_orbitals (list/array): Indices (0 to Ns-1) of the occupied SP orbitals.
            particle_type (str): 'fermion' or 'boson'. (Default: 'fermion')
            target_basis (str): The basis for the output state vector ('sites' for Fock state).
                                (Default: 'sites')
            many_body_hilbert_space (Optional[HilbertSpace]):
                A Many-Body HilbertSpace object defining the target basis (required if
                using symmetry-reduced basis). If None, assumes full Fock space over Ns sites.

        Returns:
            np.ndarray: The complex-valued state vector in the target many-body basis.
                        Backend matches self.backend.
        """
        if target_basis.lower() != 'sites':
             raise NotImplementedError("Currently only supports transformation to site ('Fock') basis.")
        if self.sp_eig_vec is None:
             raise ValueError("Single-particle eigenvectors not calculated. Diagonalize first.")
        if not self._particle_conserving and particle_type=='boson':
             raise NotImplementedError("Bosonic transformation for BdG not implemented.")

        # Use helper functions defined earlier
        calculator = None
        if particle_type.lower() == 'fermion':
            calculator = calculate_slater_det_np if self._is_numpy else calculate_slater_det_jax
        elif particle_type.lower() == 'boson':
            calculator = calculate_permanent_np if self._is_numpy else calculate_permanent_jax
        else:
            raise ValueError("particle_type must be 'fermion' or 'boson'")

        if calculator is None:
             raise RuntimeError(f"Calculation function for {particle_type} on backend {self.backend} not available.")

        # Prepare inputs
        occ_orb_arr = self._backend.asarray(occupied_orbitals, dtype=self._dtypeint)
        sp_eigvecs_arr = self._backend.asarray(self.sp_eig_vec) # Ensure backend array

        # Determine target basis states and size
        use_representatives = False
        target_hs = many_body_hilbert_space
        if target_hs is not None:
             if not target_hs._is_many_body: raise ValueError("Target HilbertSpace must be many-body.")
             if target_hs.get_Ns() != self.ns: raise ValueError("Ns mismatch between quadratic H and target HilbertSpace.")
             if target_hs.mapping is not None: # Use reduced basis
                  use_representatives = True
                  target_basis_states = target_hs.mapping # These are representative states (int)
                  target_size = target_hs.get_Nh()
                  self._log(f"Constructing MB state in reduced basis (Nh={target_size}).", log='debug')
             else: # Use full basis filtered by global symmetries
                   target_basis_states = self._backend.asarray(target_hs.get_full_map_int())
                   target_size = len(target_basis_states)
                   self._log(f"Constructing MB state in full basis satisfying global symmetries (Size={target_size}).", log='debug')
        else: # Assume full Fock space
             if self._ns > 20: # Add warning for large full spaces
                  self._log(f"Warning: Constructing state in full Fock space (Ns={self._ns}, NhFull={2**self._ns}). This can be very large.", log='warning')
             target_size = 2**self._ns
             target_basis_states = self._backend.arange(target_size, dtype=self._dtypeint) # Iterate all ints
             self._log(f"Constructing MB state in full Fock basis (NhFull={target_size}).", log='debug')


        # --- Vectorized calculation if possible (especially JAX) ---
        if self._is_jax:
             # Use jax.vmap
             vmap_calculator = jax.vmap(calculator, in_axes=(None, None, 0, None))
             mb_state_vector_unnorm = vmap_calculator(sp_eigvecs_arr, occ_orb_arr, target_basis_states, self.ns)
        else: # NumPy: Loop for now, consider vectorization if feasible/faster
             mb_state_vector_unnorm = self._backend.zeros(target_size, dtype=np.complex128)
             for i in range(target_size):
                 basis_state_int = target_basis_states[i]
                 # Need item() for Numba compatibility if target_basis_states is array
                 basis_int_scalar = basis_state_int.item() if hasattr(basis_state_int, 'item') else int(basis_state_int)
                 amplitude = calculator(sp_eigvecs_arr, occ_orb_arr, basis_int_scalar, self.ns)
                 mb_state_vector_unnorm[i] = amplitude

        # --- Normalization ---
        norm_sq = self._backend.sum(self._backend.abs(mb_state_vector_unnorm)**2)
        norm_factor = self._backend.sqrt(norm_sq)
        mb_state_vector = self._backend.where(norm_factor > 1e-12,
                                              mb_state_vector_unnorm / norm_factor,
                                              mb_state_vector_unnorm) # Avoid division by zero

        # Apply symmetry normalization if using reduced basis
        if use_representatives and target_hs.normalization is not None:
             norms_hs = target_hs.normalization # Should match target_size
             # Ensure norms are broadcastable and handle potential zeros safely
             valid_norms = self._backend.where(self._backend.abs(norms_hs) > HilbertSpace.__SYM_NORM_THRESHOLD, norms_hs, 1.0)
             mb_state_vector = mb_state_vector / valid_norms # Divide by sqrt(orbit size)

        return mb_state_vector
