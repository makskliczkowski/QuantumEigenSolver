'''
file    : QES/Algebra/hamil_quadratic.py
author  : Maksymilian Kliczkowski
email   : maksymilian.kliczkowski@pwr.edu.pl

'''

import numpy as np
import scipy as sp

from typing import List, Tuple, Union, Optional, Sequence
from enum import Enum, unique
from abc import ABC
from functools import partial
from scipy.special import comb
from itertools import combinations
from collections import defaultdict
from scipy.stats import unitary_group
from dataclasses import dataclass, field

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
    many_body_state_closure,
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
    jnp                         = np
    BCOO                        = None
    calculate_slater_det_jax    = None
    calculate_bcs_amp_jax       = None
    calculate_permament_jax     = None

from general_python.common.binary import int2base, base2int, extract as Extractor
from general_python.common import indices_from_mask, complement_indices

##############################################################################

class QuadraticSelection:
    '''
    A utility class for generating and manipulating selections of orbitals, commonly used in quantum chemistry and physics applications. Provides methods to compute the number of possible orbital selections, generate all possible combinations, and randomly select subsets of orbitals.
    
    Methods
        all_orbitals_size(n, k)
            Calculate the number of ways to choose k orbitals from n available orbitals (binomial coefficient).
        all_orbitals(n, k)
            Generate all possible combinations of k orbitals from a set of n orbitals or a given array-like collection.
        ran_orbitals(n, k)
            Randomly select k orbitals from a set of n orbitals or a given array-like collection.
    
    - If `n` is an integer, orbitals are indexed from 0 to n-1.
    - If `n` is array-like, its elements are treated as the set of available orbitals.
    '''
    def all_orbitals_size(n, k):
        """
        Calculate the number of ways to choose k orbitals from n available orbitals.

        This function returns the binomial coefficient "n choose k", representing the number
        of possible selections of k orbitals from a total of n orbitals.

        Args:
            n (int): The total number of available orbitals.
            k (int): The number of orbitals to select.

        Returns:
            int: The number of possible combinations (binomial coefficient).

        Example:
            >>> all_orbitals(4, 2)
            6
        """
        return comb(n, k, exact=True)

    def all_orbitals(n, k):
        """
        Generate all possible combinations of orbitals.
        Parameters
        ----------
        n : int or array-like
            If int, the number of orbitals (orbitals are indexed from 0 to n-1).
            If array-like, the collection of orbitals to choose from.
        k : int
            The number of orbitals to select in each combination.
        Returns
        -------
        iterator
            An iterator over tuples, each containing a unique combination of k orbitals.
        Notes
        -----
        If `n` is an integer, combinations are generated from the range [0, n).
        If `n` is array-like, combinations are generated from the elements of `n`.
        """
        
        if isinstance(n, (int, np.integer)):
            arange = np.arange(0, n, dtype = np.int64)
            return arange, combinations(arange, k)
        else:
            return n, combinations(n, k)

    def ran_orbitals(n, k, rng=None):
        """
        Generate a set of orbital indices and randomly select a subset.

        Parameters
        ----------
        n : int or array-like
            If int, defines the range of orbital indices from 0 to n-1.
            If array-like, treated as the set of available orbital indices.
        k : int
            Number of orbitals to randomly select.

        Returns
        -------
        arange : numpy.ndarray
            Array of orbital indices.
        selected : numpy.ndarray
            Array of randomly selected orbital indices of length k.

        Notes
        -----
        If `n` is an integer, the function creates an array of indices from 0 to n-1.
        If `n` is array-like, it is used directly as the set of indices.
        """
        if isinstance(n, (int, np.integer)):
            arange = np.arange(0, n, dtype=np.int64)
        else:
            arange = np.array(n, dtype=np.int64)
        selected = rng.choice(arange, k, replace=False) if rng else np.random.choice(arange, k, replace=False)
        return arange, selected

    def mask_orbitals(ns        : int, 
                    n_occupation: Union[int, float] = 0.5,
                    *,
                    ordered     : bool = True, 
                    rng         : Optional[np.random.Generator] = None,
                    return_b    : bool = False) -> dict:
        """
        Generate a mask of occupied orbitals based on the number of orbitals and the occupation fraction.
        Parameters
        ----------
        n : int
            Total number of orbitals.
        n_occupation : int or float
            Number of occupied orbitals or fraction of occupied orbitals.
            If `n_occupation` is a float, it should be in the range [0, 1].
        ordered : bool
            If True, the mask will be ordered (0s followed by 1s).
            If False, the mask will be randomly shuffled.
        rng : numpy.random.Generator or None
            Random number generator for shuffling the mask.
            If None, uses the default random generator.
        Returns
            dict
                Dictionary with keys 'mask_a' (occupied orbitals in subsystem a)
                and 'mask_b' (occupied orbitals in subsystem b - if applicable).
        """
        if n_occupation < 0:
            raise ValueError("`n_occupation` must be non-negative.")
        elif n_occupation > ns:
            raise ValueError("`n_occupation` must be less than or equal to `ns`.")
        elif 0 < n_occupation < 1:
            n_occupation = int(n_occupation * ns)
            ordered                 = True

        out_dict = {}
        if ordered:
            mask_a  = np.arange(n_occupation)
        else:    
            mask_a  = np.sort(rng.choice(np.arange(ns), n_occupation, replace=False))
            
        out_dict['mask_a']      = mask_a
        out_dict['mask_a_1h']   = Extractor.to_one_hot(mask_a, ns)
        out_dict['mask_a_int']  = base2int(out_dict['mask_a_1h'], spin=False, spin_value=1)
        if return_b:
            mask_b                  = np.setdiff1d(np.arange(ns), mask_a)
            out_dict['mask_b']      = mask_b
            out_dict['mask_b_1h']   = Extractor.to_one_hot(mask_b, ns)
            out_dict['mask_b_int']  = base2int(out_dict['mask_b_1h'], spin=False, spin_value=1)
            out_dict['order']       = tuple(mask_a) + tuple(mask_b)
        return out_dict
    
    # ------------------------------------------------------------------------
    #! Haar random coefficients
    # ------------------------------------------------------------------------

    def haar_random_coeff(gamma : int,
                        *,
                        rng     : np.random.Generator | None = None,
                        dtype   = np.complex128) -> np.ndarray:
        r"""
        Return a length-gamma complex vector distributed with the **Haar
        measure** on the unit sphere (i.e. what you get from the first
        column of a Haar-random unitary).

        Parameters
        ----------
        gamma : int
            Dimension of states to mix.
        rng : numpy.random.Generator, optional
            Random-number generator to use. `np.random.default_rng()` is the
            default; pass your own for reproducibility.
        dtype : np.dtype, default=np.complex128
            Precision of the returned coefficients.

        Notes
        -----
        * **Mathematical equivalence** - Drawing
        $$\psi_i = x_i + i y_i \\ (x_i,y_i\\sim 𝒩(0,1))$$
        and normalising,  
        $$\\psi/\\lVert\\psi\\rVert$$  
        gives exactly the same distribution as the first column of a Haar
        unitary (see, e.g., Mezzadri 2006).
        * If SciPy ≥ 1.4 is available we use `scipy.stats.unitary_group.rvs`
        (QR‑based) instead, but the Gaussian trick is used as a fallback and
        is typically faster.

        Examples
        --------
        >>> gen_random_state_coefficients(4) # doctest: +ELLIPSIS
        array([ 0.44...+0.05...j, -0.40...-0.11...j,  0.63...+0.48...j,
                0.16...+0.29...j])
        """
        if gamma < 1:
            raise ValueError("`gamma` must be at least 1.")
        if gamma == 1:
            return np.ones(1, dtype=dtype)

        rng = np.random.default_rng() if rng is None else rng

        #! fast path: SciPy's true Haar unitary, if available
        try:
            from scipy.stats import unitary_group
            return unitary_group.rvs(gamma, random_state=rng).astype(dtype)[:, 0]
        except Exception:
            # fall back to Gaussian normalise trick  (still Haar-correct)
            z   = rng.normal(size=gamma) + 1j * rng.normal(size=gamma)
            z   = z.astype(dtype, copy=False)
            z  /= np.linalg.norm(z)
            return z

    def haar_random_unitary(gamma   : int,
                            *,
                            rng     : np.random.Generator | None = None,
                            dtype   = np.complex128) -> np.ndarray:
        r"""
        Generate a Haar-random unitary matrix of shape (gamma, gamma).

        Parameters
        ----------
        gamma : int
            Dimension of the unitary matrix.
        rng : numpy.random.Generator, optional
            Random number generator (default: np.random.default_rng()).
        dtype : np.dtype, default=np.complex128
            Desired complex dtype.

        Returns
        -------
        np.ndarray
            Haar-distributed unitary matrix of shape (gamma, gamma).

        Notes
        -----
        If SciPy ≥ 1.4 is available, uses `scipy.stats.unitary_group.rvs`,
        which samples unitaries via QR decomposition with Haar measure
        (Mezzadri 2006). Otherwise, performs the QR-based method manually.

        Reference
        ---------
        Mezzadri, F. (2006). How to generate random matrices from the classical groups.
        Notices of the AMS, 54(5), 592–604.

        Examples
        --------
        >>> U = haar_random_unitary(4)
        >>> np.allclose(U.conj().T @ U, np.eye(4))
        True
        """
        
        if gamma < 1:
            raise ValueError("`gamma` must be at least 1.")

        rng = np.random.default_rng() if rng is None else rng

        try:
            return unitary_group.rvs(gamma, random_state=rng).astype(dtype)
        except Exception:
            # Fallback: generate complex Ginibre matrix and QR-decompose
            z      = rng.normal(size=(gamma, gamma)) + 1j * rng.normal(size=(gamma, gamma))
            z      = z.astype(dtype, copy=False)
            q, r   = np.linalg.qr(z)
            # Normalize phases to ensure uniqueness
            d      = np.diag(r)
            ph     = d / np.abs(d)
            q     *= ph[np.newaxis, :]
            return q

    def haar_random_coeff_real(gamma    : int,
                                *,
                                rng     : np.random.Generator | None = None,
                                dtype   = np.float64) -> np.ndarray:
        r"""
        Return a length-gamma real vector distributed with the Haar measure
        on the unit sphere (i.e. what you get from the first column of a Haar
        random orthogonal matrix).

        Parameters
        ----------
        gamma : int
            Dimension of states to mix.
        rng : numpy.random.Generator, optional
            Random-number generator to use. `np.random.default_rng()` is the
            default; pass your own for reproducibility.
        dtype : np.dtype, default=np.float64
            Precision of the returned coefficients.

        Returns
        -------
        np.ndarray
            A real vector of length `gamma` sampled from the Haar measure.

        Notes
        -----
        Uses Gaussian normalisation trick to sample from the Haar measure.
        
        Examples
        --------
        >>> haar_random_coeff_real(4) # doctest: +ELLIPSIS
        array([ 0.44..., -0.40...,  0.63...,  0.16...])
        """
        
        if gamma < 1:
            raise ValueError("`gamma` must be at least 1.")
        if gamma == 1:
            return np.ones(1, dtype=dtype)

        rng = np.random.default_rng() if rng is None else rng
        # Generate normally distributed real numbers - normalize to unit sphere
        z   = rng.normal(size=gamma)
        z   = z.astype(dtype, copy=False)
        z  /= np.linalg.norm(z)
        return z

    def haar_random_unitary_real(gamma   : int,
                                *,
                                rng     : np.random.Generator | None = None,
                                dtype   = np.float64) -> np.ndarray:
        r"""
        Generate a Haar-random real orthogonal matrix of shape (gamma, gamma).
        Parameters
        ----------
        gamma : int
            Dimension of the orthogonal matrix.
        rng : numpy.random.Generator, optional
            Random number generator (default: np.random.default_rng()).
        dtype : np.dtype, default=np.float64
            Desired real dtype.

        Returns
        -------
        np.ndarray
            Haar-distributed orthogonal matrix of shape (gamma, gamma).

        Notes
        -----
        Uses QR decomposition of a Gaussian matrix to sample from the Haar measure.

        Examples
        --------
        >>> U = haar_random_unitary_real(4)
        >>> np.allclose(U.T @ U, np.eye(4))
        True
        """
        
        if gamma < 1:
            raise ValueError("`gamma` must be at least 1.")

        rng     = np.random.default_rng() if rng is None else rng

        # Generate a random real orthogonal matrix using QR decomposition
        A       = rng.normal(size=(gamma, gamma))
        Q, R    = np.linalg.qr(A)
        d       = np.diag(R)
        Q      *= np.sign(d)
        
        # Return the first column as the Haar-random real vector
        return Q.astype(dtype, copy=False)

    # ------------------------------------------------------------------------
    #! Energy concerned
    # ------------------------------------------------------------------------
    
    def bin_energies(sorted_energies, digits: int = 10):
        """
        Groups indices of sorted energies by their rounded values.
        
        Args:
            sorted_energies (list[tuple[int, float]]):
                A list of (index, energy) pairs, sorted by energy.
            digits (int):
                Number of decimal digits for rounding - used for 
                binning the energies to avoid floating point errors.
        
        Returns:
            dict[float, list[int]]:
                Dictionary mapping rounded energy to list of indices.
        """
        binned = defaultdict(list)
        for idx, energy in sorted_energies:
            key = round(energy, digits)
            binned[key].append(idx)
        return dict(binned)
    
    def man_energies(binned_energies, dtype: np.dtype = np.int32):
        """
        Calculates the number of elements in each energy manifold from a dictionary of binned energies.
        Args:
            binned_energies (dict):
                A dictionary where keys represent energy bins and values are lists of items in each bin.
            dtype (np.dtype, optional):
                The desired data type for the output array of manifold sizes. Defaults to np.int32.
        Returns:
            tuple:
                - energy_manifolds (dict): A dictionary mapping each energy bin to the number of items in that bin.
                - energy_manifold_values (np.ndarray): An array containing the number of items in each energy bin, in the order of iteration.
        """
        
        energy_manifolds        = {}
        energy_manifold_values  = []
        for k, v in binned_energies.items():
            n                   = len(v)
            energy_manifolds[k] = n
            energy_manifold_values.append(n)
        energy_manifold_values = np.array(energy_manifold_values, dtype=dtype)
        return energy_manifolds, energy_manifold_values
    
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
    - self._U                     : ndarray     # (ns  \times  n_orb) eigenvectors (fermions, N-conserving)
    - self._F                     : ndarray     # (ns  \times  ns) pairing matrix (fermions, BdG)
    - self._Ub                    : ndarray     # (ns  \times  N_qp) columns of u (for excitations)
    - self._G                     : ndarray     # (ns  \times  ns) pairing matrix (bosons, BdG)
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
                seed                    : Optional[int]         = None,
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
        super().__init__(is_manybody    =   False,
                        ns              =   ns,
                        lattice         =   lattice,
                        hilbert_space   =   hilbert_space,
                        is_sparse       =   is_sparse,
                        dtype           =   dtype,
                        backend         =   backend,
                        seed            =   seed,
                        logger          =   logger,
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
        self._occupied_orbitals_cached  = None
        self._mb_calculator             = self._many_body_state_calculator()
        
        # for storing the pairing terms (Bogoliubov-de Gennes terms when not conserving particles)
        self._F                         = None
        self._G                         = None
        self._U                         = None
        self._V                         = None
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
    #! Transformation Preparation
    ###########################################################################


    @dataclass(frozen=True)
    class PCTransform:
        """Particle-conserving transformation handle."""
        W           : np.ndarray          # (Ns, Ns) single-particle eigenvectors (unitary)
        occ_idx     : np.ndarray          # (Na,)
        unocc_idx   : np.ndarray          # (Ns-Na,)

        # on-demand helpers (allocate only when used)

        def W_A(self) -> np.ndarray:
            """Form columns W[:, occ_idx]. NOTE: column gather copies."""
            return np.take(self.W, self.occ_idx, axis=1)

        def W_A_CT(self) -> np.ndarray:
            """Conjugate transpose of W_A (allocates as above)."""
            WA = np.take(self.W, self.occ_idx, axis=1)
            return WA.conj().T

        def order_occ_then_unocc(self) -> np.ndarray:
            """Permutation indices [occ, unocc]."""
            return np.concatenate((self.occ_idx, self.unocc_idx), dtype=np.int64)

    @dataclass(frozen=True)
    class BdGTransform:
        """BdG/Nambu transform handle, avoids big temporaries."""
        W           : np.ndarray            # (2N, 2N) quasiparticle eigenvectors in Nambu basis
        N           : int                   # single-particle dimension
        occ_idx     : np.ndarray            # subset in physical modes (0..N-1)
        unocc_idx   : np.ndarray            # complement in physical modes

        # Block views (no copies): U= W[:N,:N], V= W[:N,N:]
        @property
        def U(self) -> np.ndarray:
            return self.W[:self.N, :self.N]

        @property
        def V(self) -> np.ndarray:
            return self.W[:self.N, self.N:]

        # ---- Prefer row slicing (views) when projecting to a *mode subset* ----

        def U_rows_A(self) -> np.ndarray:
            """Rows of U restricted to A (view). Shape (|A|, N)."""
            return self.U[self.occ_idx, :]

        def V_rows_A(self) -> np.ndarray:
            """Rows of V restricted to A (view). Shape (|A|, N)."""
            return self.V[self.occ_idx, :]

        # Columns-for-A (copies; only form when needed)
        def U_cols_A(self) -> np.ndarray:
            """Columns of U for A (copy; unavoidable in NumPy). Shape (N, |A|)."""
            return np.take(self.U, self.occ_idx, axis=1)

        def V_cols_A(self) -> np.ndarray:
            """Columns of V for A (copy; unavoidable). Shape (N, |A|)."""
            return np.take(self.V, self.occ_idx, axis=1)

    def prepare_transformation(self, occ, *, bdg: bool | None = None):
        """
        Memory-conscious preparation of subspace transforms.

        Parameters
        ----------
        occ : IndexLike
            - int k             : take first k orbitals/modes
            - 1D bool mask      : occupied mask over Ns (PC) or N (BdG physical sector)
            - 1D int indices    : occupied indices in 0..Ns-1 (PC) or 0..N-1 (BdG)
        bdg : bool | None
            - If None: infer from self._particle_conserving (bdg = not particle_conserving).
            - If True : treat eigenvector matrix as (2N x 2N) BdG/Nambu.
            - If False: treat as particle-conserving (Ns x Ns).

        Returns
        -------
        QuadraticHamiltonian.PCTransform   if bdg == False
        QuadraticHamiltonian.BdGTransform  if bdg == True

        Notes
        -----
        - Does not eagerly form large submatrices.
        - Column gathers (W_A, U_cols_A, V_cols_A) allocate only when invoked.
        - Row slices (U_rows_A, V_rows_A) are views.
        """
        if bdg is None:
            bdg = not self._particle_conserving

        W = self._eig_vec
        if W is None:
            raise RuntimeError("Eigenvectors not available. Call diagonalize() first.")

        if bdg:
            # BdG / Nambu path
            if W.ndim != 2 or W.shape[0] != W.shape[1] or (W.shape[0] % 2 != 0):
                raise ValueError(f"Expect square (2N,2N) BdG eigenvector matrix; got {W.shape}")
            
            twoN        = W.shape[0]
            N           = twoN // 2

            occ_idx     = indices_from_mask(occ, N)
            unocc_idx   = complement_indices(N, occ_idx)
            return self.BdGTransform(W=W, N=N, occ_idx=occ_idx, unocc_idx=unocc_idx)

        # Particle-conserving path
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValueError(f"Expect square (Ns,Ns) eigenvector matrix; got {W.shape}")
        Ns          = W.shape[1]
        occ_idx     = indices_from_mask(occ, Ns)
        unocc_idx   = complement_indices(Ns, occ_idx)
        return self.PCTransform(W=W, occ_idx=occ_idx, unocc_idx=unocc_idx)
    
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
    
    def many_body_energy(self, occupied_orbitals: Union[int, List[int], np.ndarray]) -> float:
        """
        Calculates the total energy of a many-body state defined by occupying
        single-particle orbitals (or quasiparticle orbitals for BdG).

        Args:
            occupied_orbitals (list/array):
                Indices of the occupied single-particle
                eigenstates (orbitals α or quasiparticles γ).

        Returns:
            The total energy E = Σ_{α\inoccupied} ε_α (or E = Σ_{γ\inoccupied} E_γ for BdG).
            Result includes the constant_offset.
        """
        
        if self.eig_val is None:
            raise ValueError("Single-particle eigenvalues not calculated. Call diagonalize() first.")
        if isinstance(occupied_orbitals, int):
            occupied_orbitals = int2base(occupied_orbitals, self._ns, spin=False, spin_value=1, backend=self._backend).astype(self._dtypeint)
            
        occ = np.asarray(occupied_orbitals, dtype=self._dtypeint)
        if occ.shape[0] == 0:
            return 0.0
        
        if occ.ndim != 1:
            raise ValueError("occupied_orbitals must be 1-D")
        e   = 0.0
        
        if self._is_jax:
            occ     = jnp.asarray(occ, dtype=self._dtypeint)
            vmax    = self._eig_val.shape[0]

            def _check_bounds(x):
                if int(jnp.min(x)) < 0 or int(jnp.max(x)) >= vmax:
                    raise IndexError("orbital index out of bounds")
                return x
            occ = _check_bounds(occ)

            if self._particle_conserving:
                e = jnp.sum(self._eig_val[occ])
            else:
                if int(jnp.max(occ)) >= self._ns:
                    raise IndexError("BdG index must be in 0…Ns-1")
                mid = self._ns - 1
                e   = jnp.sum(self._eig_val[mid + occ + 1] -
                            self._eig_val[mid - occ])
        else:
            vmax = self._eig_val.shape[0]
            if occ.min() < 0 or occ.max() >= vmax:
                raise IndexError("orbital index out of bounds")

            if self._particle_conserving:
                e = nrg_particle_conserving(self._eig_val, occ)
            else:
                if occ.max() >= self._ns:
                    raise IndexError("BdG index must be in 0…Ns-1 (positive branch)")
                e = nrg_bdg(self._eig_val, self._ns, occ)
        return self._backend.real(e) + self._constant_offset

    def many_body_energies(self, n_occupation: Union[float, int] = 0.5, 
                    nh: Optional[int] = None, use_combinations: bool = False) -> dict[int, float]:
        '''
        Returns a dictionary of many-body energies for all possible
        configurations of occupied orbitals.
        
        The keys are integers representing the configuration of occupied orbitals,
        and the values are the corresponding many-body energies.
        The function iterates over all possible configurations of occupied orbitals
        and calculates the many-body energy for each configuration.
        
        Parameters
        ----------
        n_occupation : float or int
            The number of occupied orbitals. If a float, it is interpreted as a fraction
            of the total number of sites (ns). If an int, it is the exact number of
            occupied orbitals.
        nh : int, optional
            The number of configurations to consider. If None, it defaults to 2^ns.    
        use_combinations : bool
            If True, use combinations to generate occupied orbitals.
            If False, use a loop over all possible configurations.
            This is useful for large systems where the number of configurations
            is too large to handle with combinations.
        Returns
        -------
        dict[int, float]
            A dictionary where the keys are integers representing the configuration
            of occupied orbitals and the values are the corresponding many-body energies.
        Notes
        -----
        - The function uses the `int2base` function to convert integers to binary
        representations of occupied orbitals.
        - The function uses the `many_body_energy` method to calculate the energy         
        '''
        
        if 0 < n_occupation < 1:
            n_occupation = int(self._ns * n_occupation)
        elif n_occupation > self._ns:
            raise ValueError("n_occupation must be less than or equal to the number of sites.")
        elif n_occupation < 0:
            raise ValueError("n_occupation must be greater than or equal to 0.")
        
        if nh is None:
            nh = 2**self._ns
        
        # what is faster?
        many_body_energies = {}
        #! 1. Loop over all possible configurations of occupied orbitals
        if not use_combinations:
            for i in range(nh):
                occupied_orbitals       = int2base(i, self._ns, spin=False, spin_value=1).astype(self._dtypeint)
                occupied_orbitals       = np.nonzero(occupied_orbitals)[0]
                if len(occupied_orbitals) != n_occupation:
                    continue
                many_body_energies[i]   = self.many_body_energy(occupied_orbitals)
            return many_body_energies
        #! 2. Use combinations
        else:
            all_combinations        = QuadraticSelection.all_orbitals(self._ns, n_occupation)
            for i, occupied_orbitals in enumerate(all_combinations[1]):
                occupied_orbitals       = np.array(occupied_orbitals, dtype=self._dtypeint)
                many_body_energies[i]   = self.many_body_energy(occupied_orbitals)        
            return many_body_energies

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
                    calc = many_body_state_closure (
                            calculator_func = calculate_slater_det,
                            matrix_arg      = occ)
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
                    calc = lambda F, st, _ns: calculate_bogoliubov_amp(F, st, _ns)
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
                        many_body_hs      : Optional[HilbertSpace]              = None,
                        resulting_state   : Optional[np.ndarray]                = None):
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
            Coefficient vector `psi(x)`.
        """
        if target_basis != "sites":
            raise NotImplementedError("Only the site/bitstring basis is implemented for now.")

        # If new occupied_orbitals are provided, or the cached state is missing (e.g., after cache invalidation)
        if occupied_orbitals is not None or self._occupied_orbitals_cached is None:
            if isinstance(occupied_orbitals, (list, np.integer)):
                # transform to array
                self._occupied_orbitals_cached = int2base(occupied_orbitals, self._ns, spin=False, backend=self._backend).astype(self._dtypeint)
            else:
                self._occupied_orbitals_cached = np.ascontiguousarray(occupied_orbitals, dtype=self._dtypeint)
        
        #! obtain (calculator, matrix_arg)
        calculator, matrix_arg = self._many_body_state_calculator()

        #! choose mapping / dimensions
        ns           = self._ns
        dtype        = getattr(self, "_dtype", np.result_type(matrix_arg))
        if resulting_state is not None:
            dtype = np.result_type(resulting_state, dtype)
        
        if many_body_hs is None or not many_body_hs.modifies:
            return many_body_state_full(matrix_arg, calculator, ns, resulting_state, dtype=dtype)
        else:            
            mapping = many_body_hs.mapping
            return many_body_state_mapping(matrix_arg,
                                        calculator,
                                        mapping,
                                        ns,
                                        dtype)
        return None # should not be reached

    ##########################################################################
    
##############################################################################