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

from Algebra.hamil import Hamiltonian, HilbertSpace, Lattice, JAX_AVAILABLE, Logger, Array
from Algebra.Hilbert.hilbert_jit_states import (
    calculate_slater_det,
    bogolubov_decompose,
    pairing_matrix,
    calculate_bogoliubov_amp,
    calculate_bogoliubov_amp_exc,
    calculate_bosonic_gaussian_amp,
    calculate_permanent,
    many_body_state_full,
    many_body_state_mapping,
    nrg_particle_conserving,
    nrg_bdg,
    
)
if JAX_AVAILABLE:
    import jax 
    import jax.numpy as jnp
    from jax.experimental.sparse import BCOO
    from Algebra.Hilbert.hilbert_jit_states_jax import (
    calculate_slater_det_jax,                               # for calculating fermionic states
    calculate_bcs_amp_jax,                                  # for calculating BCS-like states
    calculate_permament_jax                                 # for calculating permanent states
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
    r"""
    QuadraticHamiltonian: Specialized Hamiltonian for non-interacting (quadratic) quantum systems.

    This class represents Hamiltonians of the general quadratic form:
        .. math::

            H = \sum_{i,j} \left[ h_{ij} c_i^\dagger c_j + \Delta_{ij} c_i^\dagger c_j^\dagger + \Delta_{ij}^* c_j c_i \right]

    where:
        - :math:`h_{ij}` encodes onsite energies and hopping amplitudes,
        - :math:`\Delta_{ij}` encodes pairing (superconducting) terms,
        - :math:`c_i^\dagger`, :math:`c_j` are creation/annihilation operators (fermionic or bosonic).

    For particle-conserving systems (:math:`\Delta_{ij}=0`), the Hamiltonian reduces to:
        .. math::

            H = \sum_{i,j} h_{ij} c_i^\dagger c_j

    and is represented by an :math:`N_s \times N_s` matrix.

    For non-particle-conserving systems (e.g., Bogoliubov–de Gennes, BdG), the Hamiltonian is:
        .. math::

            H = \frac{1}{2} \Psi^\dagger H_\mathrm{BdG} \Psi

    with

        .. math::

            \Psi = (c_1, \ldots, c_{N_s}, c_1^\dagger, \ldots, c_{N_s}^\dagger)^T,

            H_\mathrm{BdG} = \begin{bmatrix} h & \Delta \\ \Delta^\dagger & -h^T \end{bmatrix}

    and is represented by a :math:`2N_s \times 2N_s` matrix.

    Features:
    - Add quadratic terms (onsite, hopping, pairing) using a flexible interface.
    - Build the Hamiltonian matrix from the stored terms, supporting both dense and sparse representations.
    - Diagonalize the Hamiltonian to obtain single-particle eigenvalues and eigenvectors.
    - Compute many-body energies and wavefunctions for Slater determinants (fermions), permanents (bosons), and Bogoliubov vacua (superconductors).
    - Support for both NumPy and JAX backends for high-performance and differentiable computations.

    Mathematical background:
    - For fermions, the ground state of a quadratic Hamiltonian is a Slater determinant (particle-conserving) or a Bogoliubov vacuum (BdG).
    - For bosons, the ground state is a permanent (particle-conserving) or a Gaussian state (BdG).
    - Diagonalization yields the single-particle spectrum, which determines the many-body ground state and excitations.

    References:
    - Altland, A., & Simons, B. (2010). "Condensed Matter Field Theory" (2nd ed.), Cambridge University Press.
    - Blaizot, J.-P., & Ripka, G. (1986). "Quantum Theory of Finite Systems", MIT Press.
    - Peschel, I., & Eisler, V. (2009). "Reduced density matrices and entanglement entropy in free lattice models", J. Phys. A: Math. Theor. 42, 504003.
    - See also: https://en.wikipedia.org/wiki/Bogoliubov–de_Gennes_equations

    --------------------------------------------------------------------
    Key properties used in this class:
    - self._isfermions            : bool        # True for fermions, False for bosons
    - self._isbosons              : bool        # True for bosons, False for fermions
    - self._particle_conserving   : bool        # True if particle number is conserved
    - self._is_numpy              : bool        # True for NumPy backend, False for JAX
    - self._ns                    : int         # Number of sites/modes
    - self._dtype                 : np.dtype    # Matrix/vector precision
    - self._U                     : ndarray     # (ns × n_orb) eigenvectors (fermions, N-conserving)
    - self._F                     : ndarray     # (ns × ns) pairing matrix (fermions, BdG)
    - self._Ub                    : ndarray     # (ns × N_qp) columns of u (for excitations)
    - self._G                     : ndarray     # (ns × ns) pairing matrix (bosons, BdG)
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
        
        if self._is_sparse:
            raise NotImplementedError("Sparse matrix support not implemented yet. TODO: implement sparse matrix handling.")
        
        # Determine shape based on conservation (simple case)
        # BdG case (not particle conserving) needs 2Ns x 2Ns and different term handling.
        if not particle_conserving:
            self._log("Warning: particle_conserving=False implies BdG Hamiltonian structure. Ensure _build_quadratic handles 2Nsx2Ns matrix and pairing terms correctly.",
                    log='warning')
            self._hamil_sp_size         = 2 * ns
        else:
            self._hamil_sp_size         = ns
        
        # Set matrix shape
        self._hamil_sp_shape            = (self._hamil_sp_size, self._hamil_sp_size)
        self._dtypeint                  = self._backend.int32 if self.ns < 2**32 - 1 else self._backend.int64

        # Store quadratic terms (hopping, pairing, onsite)
        self._name                      = f"QuadraticHamiltonian(Ns={self._ns},{'BdG' if not self._particle_conserving else 'N-conserving'})"
        self._mb_calculator             = self._many_body_state_calculator()
        
        # for storing the pairing terms (Bogoliubov-de Gennes terms when not conserving particles)
        self._F                         = None
        self._G                         = None
        self._U                         = None
        self._V                         = None
        self._occupied_orbitals_cached  = None
        self._mb_calculator             = self._many_body_state_calculator()

    ##########################################################################
    #! Build the Hamiltonian
    ##########################################################################

    def _invalidate_cache(self):
        """Wipe eigenvalues, eigenvectors, cached many-body calculator."""
        self._eig_val           = None
        self._eig_vec           = None
        self._mb_calculator     = self._many_body_state_calculator()

    def add_term(self,
                term_type   : QuadraticTerm,
                sites       : tuple[int, ...] | list[int] | int,
                value       : complex,
                remove      : bool = False):
        """
        Adds a quadratic term to the Hamiltonian or pairing matrix.

        Parameters
        ----------
        term_type : QuadraticTerm
            The type of quadratic term to add. Must be one of QuadraticTerm.Onsite,
            QuadraticTerm.Hopping, or QuadraticTerm.Pairing.
        sites : tuple[int, ...] | list[int] | int
            The site indices involved in the term. For Onsite, a single index is required.
            For Hopping and Pairing, two indices are required.
        value : complex
            The coefficient of the term to be added.

        Raises
        ------
        ValueError
            If the number of site indices does not match the requirements for the term type.
        TypeError
            If the term_type is not recognized.

        Notes
        -----
        - For Onsite terms, adds `value` to the diagonal element of the Hamiltonian matrix.
        - For Hopping terms, adds `value` to the off-diagonal elements and ensures Hermiticity.
        - For Pairing terms, modifies the pairing matrix according to particle statistics.
            If the system is particle-conserving, pairing terms are ignored.
        - Logs the operation and invalidates cached eigensystem and many-body states.
        """
        xp      = self._backend
        if isinstance(sites, int):
            sites = (sites,)
        val     = -value if remove else value
        valc    = val.conjugate() if isinstance(val, (complex, np.complex)) else val

        if term_type is QuadraticTerm.Onsite:
            if len(sites) != 1:
                raise ValueError("Onsite term needs one index")
            i                       = sites[0]
            self._hamil_sp[i, i]   += val
        elif term_type is QuadraticTerm.Hopping:
            if len(sites) != 2:
                raise ValueError("Hopping term needs two indices")
            i, j                    = sites
            self._hamil_sp[i, j]   += val
            self._hamil_sp[j, i]   += valc
        elif term_type is QuadraticTerm.Pairing:
            if self._particle_conserving:
                self._log("Pairing ignored: particle_conserving=True", lvl=2, log='warning')
                return
            if len(sites) != 2:
                raise ValueError("Pairing term needs two indices")
            i, j                    = sites
            if self._isfermions: # antisymmetric
                self._delta_sp[i, j]   +=  value
                self._delta_sp[j, i]   += -value
            else: # bosons: symmetric
                self._delta_sp[i, j]   +=  value
                self._delta_sp[j, i]   +=  value
        else:
            raise TypeError(term_type)
        self._log(f"add_term: {term_type.name} {sites} {value:+.4g}", lvl=3, log='debug')

    def set_single_particle_matrix(self, H: Array):
        if not self._particle_conserving:
            raise RuntimeError("Use set_bdg_matrices for non-conserving case")
        if H.shape != (self._ns, self._ns):
            raise ValueError(f"shape mismatch, expected {(self._ns, self._ns)}")
        self._hamil_sp = H
        self._invalidate_cache()
        self._log(f"set_single_particle_matrix: {H.shape}", lvl=3, log='debug')

    def set_bdg_matrices(self, K: np.ndarray, Delta: np.ndarray):
        """
        Set the Bogoliubov-de Gennes (BdG) matrices for the Hamiltonian.

        Parameters
        ----------
        K : np.ndarray
            The single-particle (kinetic) Hamiltonian matrix. Must be a square matrix of shape (self._ns, self._ns).
        Delta : np.ndarray
            The pairing (superconducting gap) matrix. Must be a square matrix of shape (self._ns, self._ns).

        Raises
        ------
        RuntimeError
            If the Hamiltonian is particle-conserving (i.e., self._particle_conserving is True).
        ValueError
            If the shapes of K or Delta do not match (self._ns, self._ns).

        Notes
        -----
        This method is only applicable for non-particle-conserving Hamiltonians (i.e., when self._particle_conserving is False).
        After setting the matrices, the internal cache is invalidated.
        """

        if self._particle_conserving:
            raise RuntimeError("BdG matrices only for particle_conserving=False")
        
        if K.shape != (self._ns, self._ns) or Delta.shape != (self._ns, self._ns):
            raise ValueError("shape mismatch")
        
        self._hamil_sp[:]   = K
        self._delta_sp[:]   = Delta
        self._invalidate_cache()
        self._log(f"set_bdg_matrices: {K.shape}, {Delta.shape}", lvl=3, log='debug')
        
    def _hamiltonian_quadratic(self, use_numpy: bool = False):
        '''
        Generates the Hamiltonian matrix whenever the Hamiltonian is single-particle. 
        This method needs to be implemented by the subclasses.
        '''
        #!TODO: To be overriden by others
        pass

    ###########################################################################
    #! Diagonalization
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
    #! UNUSED METHODS
    ###########################################################################
    
    def _set_local_energy_operators(self):
        """ Not applicable for standard quadratic Hamiltonians built from terms. """
        if self._is_quadratic:
            self._log("Method _set_local_energy_operators is not used for standard QuadraticHamiltonian build.", log='debug')
        else:
            raise NotImplementedError("ManyBody Hamiltonian subclass must implement _set_local_energy_operators.")

    ###########################################################################
    #! Many-Body Energy Calculation
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

        occ = np.asarray(occupied_orbitals, dtype=self._dtypeint)
        if occ.ndim != 1:
            raise ValueError("occupied_orbitals must be 1-D")
        e   = 0.0
        
        if self._is_jax:
            occ     = jnp.asarray(occ, dtype=self._dtypeint)
            vmax    = self._eigvals.shape[0]

            def _check_bounds(x):
                if int(jnp.min(x)) < 0 or int(jnp.max(x)) >= vmax:
                    raise IndexError("orbital index out of bounds")
                return x
            occ = _check_bounds(occ)

            if self._particle_conserving:
                e = jnp.sum(self._eigvals[occ])
            else:
                if int(jnp.max(occ)) >= self._ns:
                    raise IndexError("BdG index must be in 0…Ns-1")
                mid = self._ns - 1
                e   = jnp.sum(self._eigvals[mid + occ + 1] -
                            self._eigvals[mid - occ])
        else:
            vmax = self._eigvals.shape[0]
            if occ.min() < 0 or occ.max() >= vmax:
                raise IndexError("orbital index out of bounds")

            if self._particle_conserving:
                e = nrg_particle_conserving(self._eigvals, occ)
            else:
                if occ.max() >= self._ns:
                    raise IndexError("BdG index must be in 0…Ns-1 (positive branch)")
                e = nrg_bdg(self._eigvals, self._ns, occ)
        return float(e) + self._constant_offset

    ###########################################################################
    #! Many-Body State Calculation
    ###########################################################################
    
    def _many_body_state_calculator(self):
        """
        Return a function object that implements

            ψ = calc(matrix_arg, basis_state_int, ns)
        together with the constant `matrix_arg` it needs.

        The closure is JIT-compatible with Numba (`nopython=True`) when
        `self._is_numpy` is True; otherwise the returned function calls the
        JAX variant of the same kernel.
        
        The function is used to calculate the many-body state vector
        in the Fock basis (or other specified basis) from the single-particle
        eigenvectors and the occupied orbitals.
        """

        #! Fermions
        if self._isfermions:
            if self._particle_conserving:
                #! Slater determinant needs (U, α_k) - U is the matrix of eigenvectors
                if not hasattr(self, "_occupied_orbitals_cached"):
                    raise RuntimeError( "call many_body_state(...) with "
                                        "`occupied_orbitals` first.")
                occ = self._occupied_orbitals_cached

                if self._is_numpy:
                    calc = lambda U, st, _ns: calculate_slater_det(U, occ, st, _ns)
                else:
                    calc = lambda U, st, _ns: calculate_slater_det_jax(U, occ, st, _ns)
                return calc, self._eig_vec
            else:
                #! Bogoliubov vacuum / Pfaffian
                if self._is_numpy:
                    if self._F is None:
                        if self._U is None or self._V is None:
                            self._U, self._V, _ = bogolubov_decompose(self._eig_val, self._eig_vec)
                        self._F = pairing_matrix(self._U, self._V)
                    calc                    = lambda F, st, _ns: calculate_bogoliubov_amp(F, st, _ns)
                else:
                    raise NotImplementedError("JAX Bogoliubov vacuum calculation not implemented.")
                    # calc = lambda F, st, _ns: calculate_bogoliubov_amp_jax(F, st, _ns)
                return calc, self._F

        #! Bosons 
        if self._isbosons:
            if self._particle_conserving:
                #! Permanent / Gaussian state
                if self._is_numpy:
                    calc = lambda G, st, _ns: calculate_permanent(G, st, _ns)
                else:
                    calc = lambda G, st, _ns: calculate_permament_jax(G, st, _ns)
                return calc, self._eig_vec
            else:
                #! Gaussian squeezed vacuum / Hafnian
                if self._is_numpy:
                    if self._G is None:
                        if self._U is None or self._V is None:
                            self._U, self._V, _ = bogolubov_decompose(self._eig_val, self._eig_vec)
                        self._G = pairing_matrix(self._eig_val, self._eig_vec)
                        
                    calc = lambda G, st, _ns: calculate_bosonic_gaussian_amp(G, st, _ns)
                else:
                    raise NotImplementedError("JAX Bosonic Gaussian state calculation not implemented.")
                return calc, self._G
    
    def many_body_state(self,
                        occupied_orbitals : Union[list[int], np.ndarray] | None = None,
                        target_basis      : str                                 = "sites",
                        many_body_hs      : Optional[HilbertSpace]              = None):
        """
        Return the coefficient vector `|Ψ〉` in the *computational* basis.

        Parameters
        ----------
        occupied_orbitals
            For **particle-conserving fermions/bosons**: list/array of α_k.
            Ignored otherwise.
        target_basis
            Currently only `"sites"` is supported.
        many_body_hs
            If provided, must expose mapping → 1-D np.ndarray`.
            The output vector is ordered according to that mapping.
            If `None`, a full vector of length `2**ns` is produced.
        batch_size
            If >0, the Fock space is processed in slices of that length
            to keep peak memory low.  `0` (default) disables batching.

        Returns
        -------
        np.ndarray
            Coefficient vector `ψ(x)`.
        """
        if target_basis != "sites":
            raise NotImplementedError("Only the site/bitstring basis "
                                    "is implemented for now.")

        #! cache α_k for later reuse (Slater path)
        if occupied_orbitals is not None:
            self._occupied_orbitals_cached = np.ascontiguousarray(occupied_orbitals,dtype=self._dtypeint)

        #! obtain (calculator, matrix_arg)
        calculator, matrix_arg = self._many_body_state_calculator()

        #! choose mapping / dimensions
        ns           = self._ns
        dtype        = getattr(self, "_dtype", np.complex128)

        if many_body_hs is None or not many_body_hs.modifies:
            return many_body_state_full(matrix_arg, calculator, ns, dtype)
        else:            
            mapping = many_body_hs.mapping
            return many_body_state_mapping(matrix_arg,
                                        calculator,
                                        mapping,
                                        ns,
                                        dtype)
        return None # should not be reached

    ##########################################################################