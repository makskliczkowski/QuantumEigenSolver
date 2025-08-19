"""
Analytic translational-invariant free-fermion model
file    : QES/Algebra/Model/Noninteracting/Conserving/free_fermions.py
author  : Maksymilian Kliczkowski
email   : maksymilian.kliczkowski@pwr.edu.pl
date    : 2025-05-01
"""

import numpy as np
import numba
from typing import Optional, Union

# import the quadratic base
from Algebra.hamil_quadratic import QuadraticHamiltonian, JAX_AVAILABLE, Array

# ---------------------------------------------------------------------
#! Spectrum
# ---------------------------------------------------------------------

@numba.njit
def _free_fermions_spectrum(ns: int, t) -> tuple[np.ndarray, np.ndarray]:
    """
    Analytic spectrum of the free fermion model.

    Parameters
    ----------
    ns : int
        Number of sites.
    t : float
        Hopping amplitude.

    Returns
    -------
    tuple
        Eigenvalues and eigenvectors.
    """
    k               = np.arange(ns)
    twopi_over_L    = 2.0 * np.pi / ns
    eig_val         = -2.0 * t * np.cos(twopi_over_L * k)
    
    #! plane waves
    j               = np.arange(ns)[:, None]      # (ns,1)
    phase           = np.exp(1j * twopi_over_L * j * k) / np.sqrt(ns)
    return eig_val, phase.astype(np.complex128)

if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp

    @jax.jit
    def _free_fermions_spectrum_jax(ns: int, t: float) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Analytic spectrum of the free fermion model.

        Parameters
        ----------
        ns : int
            Number of sites.
        t : float
            Hopping amplitude.

        Returns
        -------
        tuple
            Eigenvalues and eigenvectors.
        """
        k               = jnp.arange(ns)
        twopi_over_L    = 2.0 * jnp.pi / ns
        eig_val         = -2.0 * t * jnp.cos(twopi_over_L * k)
        
        #! plane waves
        j               = jnp.arange(ns)[:, None]
        phase           = jnp.exp(1j * twopi_over_L * j * k) / jnp.sqrt(ns)
        return eig_val, phase.astype(jnp.complex128)
else:
    jax = None
    jnp = np
    def _free_fermions_spectrum_jax(ns: int, t: float) -> tuple[np.ndarray, np.ndarray]:
        raise ImportError("JAX is not available. Please install JAX to use this function.")

# ---------------------------------------------------------------------

class FreeFermions(QuadraticHamiltonian):
    r"""
    1D translationally invariant tight-binding chain of free fermions with periodic boundary conditions.

    The Hamiltonian is given by:

    .. math::

        H = -t \sum_{\langle i, j \rangle} \left( c_i^\dagger c_j + c_j^\dagger c_i \right) + \text{constant\_offset}

    where:
        - :math:`t > 0` is the nearest-neighbor hopping (coupling) amplitude,
        - :math:`c_i^\dagger` and :math:`c_j` are fermionic creation and annihilation operators,
        - the sum :math:`\langle i, j \rangle` runs over nearest-neighbor pairs with periodic boundary conditions.

    The exact single-particle energy spectrum for periodic boundary conditions is:

    .. math::

        \varepsilon_k = -2t \cos\left( \frac{2\pi k}{N_s} \right), \quad k = 0, 1, \ldots, N_s-1


    .. math::

        U_{j, k} = \frac{1}{\sqrt{N_s}} \exp\left( \frac{2\pi i j k}{N_s} \right)

    where:
        - :math:`N_s` is the number of sites,
        - :math:`j` is the site index,
        - :math:`k` is the momentum index.
    """

    def __init__(self,
                ns                  : int,
                t                   : Union[Array, float]   = 1.0,
                constant_offset     : float                 = 0.0,
                dtype               : Optional[np.dtype]    = None,
                backend             : str                   = "default",
                logger              = None,
                **kwargs
                ):
        super().__init__(ns                     = ns,
                        particle_conserving     = True,
                        dtype                   = dtype,
                        backend                 = backend,
                        constant_offset         = constant_offset,
                        particles               = "fermions",
                        is_sparse               = False,
                        lattice                 = None,
                        hilbert_space           = None,
                        logger                  = logger,
                        **kwargs
        )
        self._t = self._set_some_coupling(t).astype(self._dtype)
        # # Allocate dummy single-particle matrix so that parent methods that
        # # expect one (e.g. set_single_particle_matrix) still work.
        # xp = self._backend
        # self._hamil_sp = xp.zeros((ns, ns), dtype=self._dtype)
        # Pre-compute spectrum immediately
        self._set_free_spectrum()

    # -----------------------------------------------------------------
    #! analytic spectrum
    # -----------------------------------------------------------------
    
    def _set_free_spectrum(self):
        t = self._backend.asarray(self._t, dtype=self._dtype)
        if self._is_jax:
            self._eig_val, self._eig_vec = _free_fermions_spectrum_jax(self._ns, jnp.asarray(t))
        else:
            self._eig_val, self._eig_vec = _free_fermions_spectrum(self._ns, t)

        # If target dtype is non-complex -> keep only the real part (cast in-place, no extra copy)
        if np.issubdtype(np.dtype(self._dtype), np.complexfloating) or self._is_jax and jnp.issubdtype(self._dtype, jnp.complexfloating):
            self._eig_vec  = self._eig_vec.astype(self._dtype, copy=False)
        else:
            # use backend.real for NumPy/JAX symmetry; cast to requested float dtype
            self._eig_vec  = self._backend.real(self._eig_vec).astype(self._dtype, copy=False)

    # -----------------------------------------------------------------
    #! override parent diagonalisation (nothing to diagonalise)
    # -----------------------------------------------------------------
    
    def diagonalize(self, verbose: bool = False, **kwargs):
        if verbose:
            self._log("FreeFermions: spectrum set analytically.", lvl=2, log='debug')
            
        # eigenvalues/vectors already cached
        # still apply constant offset if needed
        if self._constant_offset != 0.0 and not getattr(self, "_offset_applied", False):
            self._eig_val           = self._eig_val + self._constant_offset
            self._offset_applied    = True
            
        # parent bookkeeping
        self._calculate_av_en()

    # -----------------------------------------------------------------
    #! the quadratic builder is a no-op
    # -----------------------------------------------------------------
    
    def _hamiltonian_quadratic(self, use_numpy: bool = False):
        """
        Returns the Hamiltonian in quadratic form.

        Parameters
        ----------
        use_numpy : bool, optional
            If True, use numpy instead of the backend for the Hamiltonian matrix.

        Returns
        -------
        np.ndarray
            The Hamiltonian matrix in quadratic form.
        """
        if use_numpy:
            self._hamil_sp = np.zeros((self._ns, self._ns), dtype=self._dtype)
        else:
            self._hamil_sp = self._backend.zeros((self._ns, self._ns), dtype=self._dtype)
        
        for i in range(self._ns):
            self._hamil_sp[i, (i + 1) % self._ns] = -self._t[i]
            self._hamil_sp[i, (i - 1) % self._ns] = -self._t[i]

    # -----------------------------------------------------------------
    #! adding terms not allowed (would spoil analyticity)
    # -----------------------------------------------------------------
    
    def add_term(self, *_, **__):
        raise NotImplementedError("FreeFermions is fully analytic - "
                                "use QuadraticHamiltonian directly for "
                                "arbitrary hopping matrices.")

    # -----------------------------------------------------------------
    
    def __repr__(self):
        return f"FreeFermions(ns={self._ns},t={self._t[0]},c={self._constant_offset})"
    
    def __str__(self):
        return self.__repr__()
    
# ---------------------------------------------------------------------
