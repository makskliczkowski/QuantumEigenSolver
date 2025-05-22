


import numpy as np
import numba
from enum import Enum
from typing import Tuple, Union, Optional
from functools import partial

from general_python.common import binary as BinaryMod
from general_python.algebra.utils import JAX_AVAILABLE, get_backend, Array

if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
    from jax import lax
else:
    jax = None
    jnp = np
    lax = None
    
# -----------------------------------------------------------------------------
#! LDOS
# -----------------------------------------------------------------------------

if JAX_AVAILABLE:
    
    @partial(jax.jit, static_argnames=["degenerate", "tol"])
    def ldos_jax(
            energies    : Array,
            overlaps    : Array,
            degenerate  : bool = False,
            tol         : float = 1e-8) -> Array:
            """
            JAX version of LDOS/strength function.
            """
            if not degenerate:
                return jnp.abs(overlaps) ** 2

            # for each E_i sum |overlaps[j]|^2 over j with |E_j - E_i| < tol
            def _ldos_i(E_i):
                mask = jnp.abs(energies - E_i) < tol
                return jnp.sum(jnp.abs(overlaps) ** 2 * mask)

            return jax.vmap(_ldos_i)(energies)

    @partial(jax.jit, static_argnames=["nbins"])
    def dos_jax(energies: Array, nbins: int = 100, **kwargs) -> Array:
        """
        JAX version of DOS via histogram binning.
        """
        counts, _ = jnp.histogram(energies, bins=nbins, **kwargs)
        return counts
else:
    ldos_jax = None
    dos_jax = None

def ldos(energies: Array, overlaps: Array, degenerate: bool = False, tol: float = 1e-8) -> Array:
    r"""
    Local density of states (LDOS) or strength function.

    If non-degenerate:
    .. math::
        \mathrm{LDOS}_i = |\,\langle i\,|\,\psi\rangle|^2.

    If degenerate, energies within `tol` are grouped:
    .. math::
        \mathrm{LDOS}_i = \sum_{j:|E_j - E_i|<\mathrm{tol}}
                        |\langle j|\psi\rangle|^2.

    Parameters
    ----------
    energies
        Eigenenergies \(E_n\), shape (N,).
    overlaps
        Overlap amplitudes \(\langle n|\psi\rangle\), shape (N,).
    degenerate
        Whether to sum over (nearly) degenerate levels.
    tol
        Tolerance for degeneracy grouping.

    Returns
    -------
    Array1D
        LDOS for each energy index.
    """
    if not degenerate:
        return np.abs(overlaps)**2

    N = energies.size
    ldos = np.empty(N, dtype=float)
    for i in range(N):
        mask      = np.abs(energies - energies[i]) < tol
        ldos[i]   = np.sum(np.abs(overlaps[mask])**2)
    return ldos

def dos(energies: Array, nbins: int = 100, **kwargs) -> Array:
    r"""
    Density of states via histogram binning.

    Parameters
    ----------
    energies
        Eigenenergies array, shape (N,).
    nbins
        Number of bins.

    Returns
    -------
    Array1D
        Counts per energy bin.
    """
    counts, _ = np.histogram(energies, bins=nbins, **kwargs)
    return counts

# -----------------------------------------------------------------------------
#! Matrix elements
# -----------------------------------------------------------------------------

@numba.njit(fastmath=True)
def f_function( start               : int,
                stop                : int,
                overlaps            : np.ndarray,
                eigvals             : np.ndarray,
                omegas_allocated    : np.ndarray,
                vals_allocated      : np.ndarray,
                energy_target       : float = 0.0,
                bw                  : float = 1.0,
                energy_diff_cut     : float = 0.015):
    """
    Computes and allocates omega values and corresponding squared overlaps for pairs of eigenvalues within a specified energy window.

    Parameters:
        start (int):
            Starting index for the outer loop over eigenvalues.
        stop (int):
            Stopping index (exclusive) for the outer loop over eigenvalues.
        overlaps (np.ndarray):
            2D array of overlap values between eigenstates, shape (N, N).
        eigvals (np.ndarray):
            1D array of eigenvalues, shape (N,).
        omegas_allocated (np.ndarray):
            1D array to store computed omega values.
        vals_allocated (np.ndarray):
            1D array to store squared overlap values.
        energy_target (float, optional):
            Target energy value for filtering pairs. Default is 0.0.
        bw (float, optional):
            Bandwidth parameter for energy window. Default is 1.0.
        energy_diff_cut (float, optional):
            Energy difference cutoff factor. Default is 0.015.

    Returns:
        int: The number of (omega, squared overlap) pairs allocated in the output arrays.
    """
    tol    = bw * energy_diff_cut
    cnt    = 0
    nh     = eigvals.shape[0]

    for i in range(start, stop):
        e_i    = eigvals[i]
        # derive allowed e_j range:
        low    = 2.0*(energy_target - tol) - e_i
        high   = 2.0*(energy_target + tol) - e_i

        # find j‐window via binary search
        j0     = np.searchsorted(eigvals, low)
        if j0 <= i:
            j0 = i + 1
        j1     = np.searchsorted(eigvals, high)
        if j1 > nh:
            j1 = nh

        # only loop inside the small window
        for j in range(j0, j1):
            diff                    = e_i - eigvals[j]
            omega                   = diff if diff >= 0.0 else -diff
            m2                      = abs(overlaps[i, j] * overlaps[j, i])
            omegas_allocated[cnt]   = omega
            vals_allocated[cnt]     =  m2
            cnt                    += 1
    return cnt

# -----------------------------------------------------------------------------
#! Fidelity susceptibility
# -----------------------------------------------------------------------------

if JAX_AVAILABLE:
    @partial(jax.jit, static_argnames=["idx"])
    def fidelity_susceptibility_jax(
        energies    : Array,
        V           : Array,
        mu          : float,
        idx         : Optional[int] = None) -> Array:
        """
        JAX version of fidelity susceptibility. If idx is given (and in-range),
        returns a scalar χ_idx; otherwise returns an Array of shape (N,) with all χ_i.
        """
        mu2 = mu * mu

        if idx is not None and 0 <= idx < energies.shape[0]:
            E      = energies[idx]
            dE     = energies - E               # shape (N,)
            omm    = dE**2                      # (E_j - E_i)^2
            denom2 = (omm + mu2) ** 2
            V2_row = jnp.abs(V[idx, :]) ** 2
            return jnp.sum(V2_row * omm / denom2)

        # full-vector version
        dE      = energies[:, None] - energies[None, :]  # shape (N,N)
        omm     = dE**2
        denom2  = (omm + mu2) ** 2
        V2      = jnp.abs(V) ** 2
        return jnp.sum(V2 * omm / denom2, axis=1)
else:
    fidelity_susceptibility_jax = None

def fidelity_susceptibility(energies: Array, V: Array, mu: float, idx: Optional[int] = None) -> float:
    r"""
    Compute fidelity susceptibility for state `idx`:
    
    .. math::
        \chi_i = \sum_{j\neq i}
        \frac{|V_{ij}|^2\,(E_j - E_i)^2}
            {\bigl[(E_j - E_i)^2 + \mu^2\bigr]^2}\,.

    Parameters
    ----------
    idx
        Index of the reference eigenstate \(i\).
    energies
        1D array of eigenenergies \(E_n\).
    V
        2D overlap (or perturbation) matrix \(V_{nm}\).
    mu
        Broadening/cutoff parameter \(\mu\).

    Returns
    -------
    float
        Fidelity susceptibility \(\chi_i\).
    """
    if idx is not None and 0 <= idx < len(energies):
        E       = energies[idx]
        dE      = energies - E
        omm     = dE**2
        mu2     = mu*mu
        # overlap squared - take a given row
        V2_row  = np.abs(V[idx, :])**2
        # denominator squared
        denom2  = (omm + mu2)**2
        # vectorized sum; diag term is zero since omm[idx]=0
        return np.sum(V2_row * omm / denom2)
    else:
        dE      = energies[:, None] - energies[None, :]
        omm     = dE**2
        mu2     = mu*mu
        denom2  = (omm + mu2)**2
        V2      = np.abs(V)**2
        # sum over columns j for each row i
        return np.sum(V2 * omm / denom2, axis=1)

# -----------------------------------------------------------------------------
#! State information
# -----------------------------------------------------------------------------

if JAX_AVAILABLE:
    @partial(jax.jit, static_argnames=["q", "new_basis"])
    def inverse_participation_ratio_jax(state: Array, q: float = 1.0, new_basis: Optional[Array] = None) -> float:
        r"""
        Compute the inverse participation ratio (IPR) of a quantum state.

        The IPR is defined as:

        .. math::
            \mathrm{IPR} = \sum_{i=1}^{N} |\psi_i|^{2q}\,.

        Parameters
        ----------
        state
            Quantum state, either a 1D array or a NumPy array.
        q
            Exponent for the IPR calculation.

        Returns
        -------
        float
            Inverse participation ratio.
        """
        if new_basis is not None:
            return jnp.sum(jnp.abs(new_basis.T @ state)**(2*q))
        return jnp.sum(jnp.abs(state)**(2*q))
else:
    inverse_participation_ratio_jax = None

@numba.njit(parallel=True, fastmath=True)
def inverse_participation_ratio(states: np.ndarray, q: float = 1.0, new_basis: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute IPR_j = ∑_i |ψ_{i j}|^{2q} for each column j of `states`.
    If `new_basis` is provided (shape n \times n), then ψ → B^T·ψ is used
    before raising to the 2q power.  Works on 1D or 2D `states`.

    Parameters
    ----------
    states : np.ndarray
        Complex array, shape (n,) or (n, m).
    q : float
        Exponent in the IPR definition (default 1.0).
    new_basis : np.ndarray, optional
        Change-of-basis matrix (n \times n).  If not None, each state ψ_j is
        transformed via B^T·ψ_j before computing |·|^(2q).

    Returns
    -------
    np.ndarray
        If input was 1D, returns a scalar in a 0-d array; if 2D, returns
        a length-m array of IPR values.
    """
    
    # reshape 1D->2D so we can always write m-parallel loops
    single = False
    if states.ndim == 1:
        states = states.reshape(states.shape[0], 1)
        single = True

    n, m    = states.shape
    out     = np.zeros(m, dtype=np.float64)
    two_q   = 2.0 * q

    if new_basis is None:
        # no transform 
        for j in numba.prange(m):
            acc = 0.0
            for i in range(n):
                c       = states[i, j]
                p       = c.real*c.real + c.imag*c.imag
                acc    += p**q
            out[j] = acc
    else:
        # on-the-fly transform: φ_i = ∑_k B[k,i]*ψ_k
        # then acc += |φ_i|^(2q)
        B = new_basis
        for j in numba.prange(m):
            acc = 0.0
            for i in range(n):
                re = 0.0
                im = 0.0
                # compute (B^T·ψ)_i = ∑_k B[k,i] * ψ[k,j]
                for k in range(n):
                    b   = B[k, i]
                    s   = states[k, j]
                    # complex multiply: (b_r + i b_i)*(s_r + i s_i)
                    re += b.real*s.real - b.imag*s.imag
                    im += b.real*s.imag + b.imag*s.real
                p       = re*re + im*im
                acc    += p**q
            out[j] = acc

    return out

# -----------------------------------------------------------------------------

#! EOF 



