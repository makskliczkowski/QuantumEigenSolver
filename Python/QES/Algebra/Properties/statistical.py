


import numpy as np
import numba
import math
from enum import Enum
from typing import Tuple, Union, Optional
from functools import partial

from tenpy import eig

from general_python.common import binary as BinaryMod
from general_python.algebra.utils import JAX_AVAILABLE, get_backend, Array

if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
else:
    jax = None
    jnp = np
    
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

@numba.njit(fastmath=True, cache=True)
def extract_indices_window(
        start               : int,
        stop                : int,
        eigvals             : np.ndarray,
        energy_target       : float = 0.0,
        bw                  : float = 1.0,
        energy_diff_cut     : float = 0.015,
        whole_spectrum      : bool  = False):
    '''
    Extract indices of eigenvalues within a specified energy window.
    '''

    if whole_spectrum:
        return np.empty((0, 3), dtype=np.int64), 0

    #! allocate -> idx_i, idx_j_start, idx_j_end
    if stop < start:
        tmp     = start
        start   = stop
        stop    = tmp
    if stop > eigvals.shape[0]:
        stop = eigvals.shape[0]
    if start < 0:
        start = 0
    indices_alloc = np.zeros(((stop - start), 3), dtype=np.int64)

    tol     = bw * energy_diff_cut
    j_lo    = stop - 1
    j_hi    = stop - 1

    # iterate i descending so j_lo/j_hi move forward only
    cnt     = 0
    for i in range(start, stop):
        e_i         = eigvals[i]
        # [|(E_i + E_j)/2 - e_target| < eps] -> [E_j < 2*e_target + eps - E_i] & [E_j > 2*e_target - eps - E_i]
        low         = 2.0 * (energy_target - tol) - e_i
        high        = 2.0 * (energy_target + tol) - e_i
        
        # advance j_hi to first eigvals[j] > high
        j_hi        = stop - 1
        while eigvals[j_hi] >= high:
            j_hi   -= 1        
        
        # advance j_lo to first eigvals[j] >= low
        # we can start from j_hi!
        j_lo        = j_hi
        while eigvals[j_lo] > low and j_lo > i:
            j_lo   -= 1 # decrement in the upper right triangle

        if j_hi <= j_lo:
            break # we finished the upper triangle
        
        indices_alloc[cnt, 0] = i
        indices_alloc[cnt, 1] = j_lo
        indices_alloc[cnt, 2] = j_hi + 1 # exclusive end
        cnt                  += 1
    return indices_alloc, cnt

@numba.njit(fastmath=True, cache=True)
def _m2_hermitian(v):
    # Works for real or complex
    a = abs(v)
    return a * a

@numba.njit(fastmath=True, cache=True)
def _m2_generic(x, y):
    # |x*y| = |x|*|y|
    return abs(x) * abs(y)

@numba.njit(fastmath=True, cache=True)
def _omega_abs(ei, ej):
    d = ei - ej
    if d < 0:
        return -d
    return d

@numba.njit(cache=True, fastmath=True, inline='always')
def _bin_index( omega, 
                bins,
                bin0,
                inv_binw,
                uniform_bins        = False,
                uniform_log_bins    = False):
    # returns (-1) if out-of-range for uniform path; for non-uniform caller must still range-check
    if uniform_bins:
        idx = int((omega - bin0) * inv_binw)
        return idx
    elif uniform_log_bins:
        # Handle uniform log bins
        if omega <= 0.0:
            return -1

        # bin0 should be the log of the first bin edge
        # inv_binw should be the inverse width of the log bins
        
        t = math.log(omega) - bin0
        b = int(t * inv_binw)  # floor
        # caller should still range-check (0 <= b < nbins)
        return b
    #! non-uniform log bins
    return np.searchsorted(bins, omega, side='right') - 1

@numba.njit(fastmath=True, cache=True)
def _alloc_values_or_bins(nh: int, bins: Optional[np.ndarray] = None, indices_alloc: Optional[np.ndarray] = None) -> Tuple[np.ndarray, int]:
    if bins is not None and bins.shape[0] >= 2:
        nbins           = bins.shape[0]
        counts          = np.zeros(nbins, dtype=np.uint64)
        sums            = np.zeros(nbins, dtype=np.float64)
        empty_values    = np.empty((0, 2), dtype=np.float64)
        return (counts, sums, nbins), empty_values
    else:
        if indices_alloc is not None and indices_alloc.shape[0] > 0 and indices_alloc.shape[1] == 3:
            cap = indices_alloc.shape[0]
        else:
            cap = nh * (nh - 1) // 2
    
        values  = np.empty((cap, 2), dtype=np.float64)
        counts  = np.empty(0, dtype=np.uint64)
        sums    = np.empty(0, dtype=np.float64)
        return (counts, sums, 0), values

@numba.njit(fastmath=True, cache=True)
def _alloc_bin_info(uniform_bins: bool, uniform_log_bins: bool, bins: Optional[np.ndarray]) -> Tuple[float, float, int]:
    '''
    Allocate bin information for histogramming.
    '''
    if (not uniform_bins and not uniform_log_bins) or (bins is None) or (bins.shape[0] < 2):
        return 0.0, 0.0, (False, False)

    if uniform_bins:
        bin0                = bins[0]
        binw                = bins[1] - bins[0]
        inv_binw            = 1.0 / binw if binw > 0.0 else 0.0
        uniform_log_bins    = False
        return bin0, inv_binw, (True, False)
    elif uniform_log_bins:
        log_bin0            = math.log(bins[0]) if bins[0] > 0.0 else -np.inf
        log_binw            = math.log(bins[1]) - log_bin0
        uniform_bins        = False
        
        bin0                = log_bin0
        inv_binw            = 1.0 / log_binw if log_binw > 0.0 else 0.0
        return bin0, inv_binw, (False, True)
    else:
        #! Non-uniform bins
        bin0                = 0.0
        inv_binw            = 0.0
        uniform_bins        = False
        uniform_log_bins    = False
        return bin0, inv_binw, (False, False)

# -----------------------------------------------------------------------------

@numba.njit(fastmath=True, cache=True)
def f_function( overlaps        : np.ndarray,
                eigvals         : np.ndarray,
                indices_alloc   : Optional[np.ndarray] = None,
                bins            : Optional[np.ndarray] = None,
                # other info about the bins
                typical         : bool = False,
                uniform_bins    : bool = False,
                uniform_log_bins: bool = False,
                log_eps         : float = 1e-24,
                ):
    """
    Compute the f-function for a given set of overlaps and eigenvalues.
    """
    
    nh                              = eigvals.shape[0]
    use_hist                        = (bins is not None) and (bins.shape[0] >= 2)
    #! allocation
    (counts, sums, nbins), values   = _alloc_values_or_bins(nh, bins)
    bin0, inv_binw, (uniform_bins, uniform_log_bins) = _alloc_bin_info(uniform_bins, uniform_log_bins, bins)

    if indices_alloc is not None and indices_alloc.shape[0] > 0 and indices_alloc.shape[1] == 3:
        #! Fast path over precomputed pairs
        if use_hist:
            for k in range(indices_alloc.shape[0]):
                i       = indices_alloc[k, 0]
                j_start = indices_alloc[k, 1]
                j_end   = indices_alloc[k, 2]
                ei      = eigvals[i]
                for j in range(j_start, j_end):
                    omega   = _omega_abs(ei, eigvals[j])
                    b       = _bin_index(omega, bins, bin0, inv_binw, uniform_bins, uniform_log_bins)
                    if 0 <= b < nbins:
                        m2         = _m2_generic(overlaps[i, j], overlaps[j, i])
                        sums[b]   += (np.log(m2 + log_eps) if typical else m2)
                        counts[b] += 1
            return values, counts, sums
        else:
            cnt = 0
            for k in range(indices_alloc.shape[0]):
                i       = indices_alloc[k, 0]
                j_start = indices_alloc[k, 1]
                j_end   = indices_alloc[k, 2]

                for j in range(j_start, j_end):
                    omega           = eigvals[i] - eigvals[j]
                    if omega < 0:
                        omega = -omega

                    m2              = _m2_generic(overlaps[i, j], overlaps[j, i])
                    values[cnt, 0]  = omega
                    values[cnt, 1]  = m2 if not typical else math.log(m2 + 1e-30)
                    cnt            += 1
            
            return values[:cnt], counts, sums

    # ----------
    
    #! No indices provided: generate the pair set on the fly
    if use_hist:
        for i in range(nh):
            e_i     = eigvals[i]
            for j in range(i + 1, nh):
                omega   = abs(e_i - eigvals[j])
                b       = _bin_index(omega, bins, bin0, inv_binw, uniform_bins, uniform_log_bins)
                if 0 <= b < nbins:
                    m2          = _m2_generic(overlaps[i, j], overlaps[j, i])
                    counts[b]  += 1
                    sums[b]    += m2 if not typical else math.log(m2 + log_eps)
        return values, counts, sums
    else:
        cnt = 0
        for i in range(nh):
            e_i = eigvals[i]
            for j in range(i + 1, nh):
                omega           = abs(e_i - eigvals[j])
                m2              = _m2_generic(overlaps[i, j], overlaps[j, i])
                values[cnt, 0]  = omega
                values[cnt, 1]  = m2 if not typical else math.log(m2 + 1e-30)
                cnt            += 1
        return values[:cnt], counts, sums

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
    mu2 = mu * mu

    if idx is not None:
        E       = energies[idx]
        dE      = energies - E
        omm     = dE**2
        V_row   = np.abs(V[idx])**2

        mask    = np.ones(len(energies), dtype=bool)
        mask[idx] = False

        denom   = omm[mask] + mu2
        return np.sum(V_row[mask] * omm[mask] / denom**2)
    else:
        dE      = energies[:, None] - energies[None, :]
        omm     = dE**2
        denom2  = (omm + mu2)**2
        np.fill_diagonal(denom2, 1.0)  # avoid div-by-zero (will get zero in numerator anyway)
        V2      = np.abs(V)**2
        np.fill_diagonal(V2, 0.0)      # eliminate diagonal contribution explicitly
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
def inverse_participation_ratio(states: np.ndarray, q: float = 1.0, new_basis: Optional[np.ndarray] = None, square: bool = True) -> np.ndarray:
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
    two_q   = 2.0 * q if square else q

    if new_basis is None:
        # no transform 
        for j in numba.prange(m):
            acc = 0.0
            for i in range(n):
                c       = states[i, j]
                p       = np.abs(c)**two_q
                acc    += p
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
#! K - function
# -----------------------------------------------------------------------------

@numba.njit(fastmath=True, cache=True)
def k_function( ldos            : np.ndarray,
                eigvals         : np.ndarray,
                indices_alloc   : Optional[np.ndarray] = None,
                bins            : Optional[np.ndarray] = None,
                # additional parameters
                typical         : bool = False,
                uniform_bins    : bool = False,
                uniform_log_bins: bool = False,
                log_eps         : float = 1e-24) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the f-function for a given set of overlaps and eigenvalues.
    """
    nh                              = eigvals.shape[0]
    use_hist                        = (bins is not None) and (bins.shape[0] >= 2)
    #! allocation
    (counts, sums, nbins), values   = _alloc_values_or_bins(nh, bins)
    bin0, inv_binw, (uniform_bins, uniform_log_bins) = _alloc_bin_info(uniform_bins, uniform_log_bins, bins)
    
    if indices_alloc is not None and indices_alloc.shape[0] > 0 and indices_alloc.shape[1] == 3:
        #! Fast path over precomputed pairs
        if use_hist:
            for k in range(indices_alloc.shape[0]):
                i       = indices_alloc[k, 0]
                j_start = indices_alloc[k, 1]
                j_end   = indices_alloc[k, 2]
                ei      = eigvals[i]
                for j in range(j_start, j_end):
                    omega   = _omega_abs(ei, eigvals[j])
                    b       = _bin_index(omega, bins, bin0, inv_binw, uniform_bins, uniform_log_bins)
                    if 0 <= b < nbins:
                        val        = ldos[i] * ldos[j]
                        sums[b]   += (np.log(val + log_eps) if typical else val)
                        counts[b] += 1
            return values, counts, sums
        else:
            cnt = 0
            for k in range(indices_alloc.shape[0]):
                i       = indices_alloc[k, 0]
                j_start = indices_alloc[k, 1]
                j_end   = indices_alloc[k, 2]

                for j in range(j_start, j_end):
                    omega           = eigvals[i] - eigvals[j]
                    if omega < 0:
                        omega = -omega

                    m2              = ldos[i] * ldos[j]
                    values[cnt, 0]  = omega
                    values[cnt, 1]  = m2 if not typical else math.log(m2 + 1e-30)
                    cnt            += 1
            
            return values[:cnt], counts, sums

    # ----------
    
    #! No indices provided: generate the pair set on the fly
    if use_hist:
        for i in range(nh):
            e_i     = eigvals[i]
            for j in range(i + 1, nh):
                omega   = abs(e_i - eigvals[j])
                b       = _bin_index(omega, bins, bin0, inv_binw, uniform_bins, uniform_log_bins)
                if 0 <= b < nbins:
                    m2          = ldos[i] * ldos[j]
                    counts[b]  += 1
                    sums[b]    += m2 if not typical else math.log(m2 + log_eps)
        return values, counts, sums
    else:
        cnt = 0
        for i in range(nh):
            e_i = eigvals[i]
            for j in range(i + 1, nh):
                omega           = abs(e_i - eigvals[j])
                m2              = ldos[i] * ldos[j]
                values[cnt, 0]  = omega
                values[cnt, 1]  = m2 if not typical else math.log(m2 + 1e-30)
                cnt            += 1
        return values[:cnt], counts, sums

# -----------------------------------------------------------------------------
#! Fourier spectrum function - S(omega) = \sum _{n \neq m} |c_n|^2 |c_m|^2 |O_mn|^2 \delta (omega - |E_m - E_n|)
# -----------------------------------------------------------------------------

@numba.njit(fastmath=True, cache=True)
def s_function( ldos            : np.ndarray,
                overlaps        : np.ndarray,
                eigvals         : np.ndarray,
                indices_alloc   : Optional[np.ndarray] = None,
                bins            : Optional[np.ndarray] = None,
                # additional parameters
                typical         : bool = False,
                uniform_bins    : bool = False,
                uniform_log_bins: bool = False,
                log_eps         : float = 1e-24
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Compute the power spectrum S(omega) for a given set of overlaps and eigenvalues.
    Coarse-grained representation in energy space.

    Power spectrum S(omega) is defined as:
    S(omega) = \sum _{n \neq m} |c_n|^2 |c_m|^2 |O_mn|^2 \delta (omega - |E_m - E_n|)

    Parameters
    ----------
    ldos : np.ndarray
        Local density of states.
    overlaps : np.ndarray
        Overlap matrix.
    eigvals : np.ndarray
        Eigenvalues.
    indices_alloc : Optional[np.ndarray], optional
        Precomputed indices for allocation.
    bins : Optional[np.ndarray], optional
        Bins for histogram.
    typical : bool, optional
        If True, use typical values.
    uniform_bins : bool, optional
        If True, use uniform bins.
    uniform_log_bins : bool, optional
        If True, use uniform log bins.

    Returns
    -------
    np.ndarray
        Power spectrum S(omega).
    """
    
    nh                              = eigvals.shape[0]
    use_hist                        = (bins is not None) and (bins.shape[0] >= 2)
    #! allocation
    (counts, sums, nbins), values   = _alloc_values_or_bins(nh, bins)
    bin0, inv_binw, (uniform_bins, uniform_log_bins) = _alloc_bin_info(uniform_bins, uniform_log_bins, bins)
    
    if indices_alloc is not None and indices_alloc.shape[0] > 0 and indices_alloc.shape[1] == 3:
        #! Fast path over precomputed pairs
        if use_hist:
            for k in range(indices_alloc.shape[0]):
                i       = indices_alloc[k, 0]
                j_start = indices_alloc[k, 1]
                j_end   = indices_alloc[k, 2]
                ei      = eigvals[i]
                for j in range(j_start, j_end):
                    omega   = _omega_abs(ei, eigvals[j])
                    b       = _bin_index(omega, bins, bin0, inv_binw, uniform_bins, uniform_log_bins)
                    if 0 <= b < nbins:
                        val        = ldos[i] * ldos[j]
                        val       *= _m2_generic(overlaps[i, j], overlaps[j, i])
                        sums[b]   += (np.log(val + log_eps) if typical else val)
                        counts[b] += 1
            return values, counts, sums
        else:
            cnt = 0
            for k in range(indices_alloc.shape[0]):
                i       = indices_alloc[k, 0]
                j_start = indices_alloc[k, 1]
                j_end   = indices_alloc[k, 2]

                for j in range(j_start, j_end):
                    omega           = eigvals[i] - eigvals[j]
                    if omega < 0:
                        omega = -omega

                    m2              = ldos[i] * ldos[j]
                    m2             *= _m2_generic(overlaps[i, j], overlaps[j, i])
                    values[cnt, 0]  = omega
                    values[cnt, 1]  = m2 if not typical else math.log(m2 + 1e-30)
                    cnt            += 1
            return values[:cnt], counts, sums

    # ----------
    
    #! No indices provided: generate the pair set on the fly
    if use_hist:
        for i in range(nh):
            e_i     = eigvals[i]
            for j in range(i + 1, nh):
                omega   = abs(e_i - eigvals[j])
                b       = _bin_index(omega, bins, bin0, inv_binw, uniform_bins, uniform_log_bins)
                if 0 <= b < nbins:
                    m2          = ldos[i] * ldos[j]
                    m2         *= _m2_generic(overlaps[i, j], overlaps[j, i])
                    counts[b]  += 1
                    sums[b]   += m2 if not typical else math.log(m2 + log_eps)
        return values, counts, sums
    else:
        cnt = 0
        for i in range(nh):
            e_i = eigvals[i]
            for j in range(i + 1, nh):
                omega           = abs(e_i - eigvals[j])
                m2              = ldos[i] * ldos[j]
                m2             *= _m2_generic(overlaps[i, j], overlaps[j, i])
                values[cnt, 0]  = omega
                values[cnt, 1]  = m2 if not typical else math.log(m2 + 1e-30)
                cnt            += 1
        return values[:cnt], counts, sums

# -----------------------------------------------------------------------------
#! Spectral CDF
# -----------------------------------------------------------------------------

@staticmethod
def spectral_cdf(x, y, gammaval = 0.5, BINVAL = 21):
    """
    Calculate the cumulative distribution function (CDF) and find the gamma value.

    Parameters:
    x (array-like): The independent variable values.
    y (array-like): The dependent variable values, which may contain NaNs.
    gammaval (float, optional): The target CDF value to find the corresponding gamma value. Default is 0.5.

    Returns:
    tuple: A tuple containing:
        - x (array-like): The input independent variable values.
        - y (array-like): The input dependent variable values with NaNs removed.
        - cdf (array-like): The cumulative distribution function values.
        - gammaf (float): The value of the independent variable corresponding to the target CDF value.
    """
    # Apply the moving average to smooth y
    y_smoothed  = np.convolve(y, np.ones(BINVAL)/BINVAL, mode='same')
    cdf         = np.cumsum(y_smoothed * np.diff(np.insert(x, 0, 0)))
    cdf         /= cdf[-1]
    y_smoothed  /= cdf[-1]
    gammaf      = x[np.argmin(np.abs(cdf - gammaval))]
    return x, y_smoothed, cdf, gammaf

# -----------------------------------------------------------------------------
#! Survival probability
# -----------------------------------------------------------------------------

@numba.njit(fastmath=True, cache=True)
def survival_prob(psi0  : np.ndarray,
                psi_t   : np.ndarray,
                axis    : int = 0,
                out     : np.ndarray | None = None) -> np.ndarray:
    """
    P_k = |<psi(0) | psi(t_k)>|^2

    psi0 : (H,) complex
    psi_t : (H, N) complex
        - axis=0 -> (H, N)  columns are states at times t_k
        - axis=1 -> (N, H)  rows    are states at times t_k
    """
    if axis == 0:
        H, N = psi_t.shape[0], psi_t.shape[1]
        if psi0.shape[0] != H:
            raise ValueError("psi0 length mismatch with psi_t (axis=0).")
        # allocate output if needed
        if out is None or out.shape[0] != N:
            P = np.empty(N, dtype=psi_t.real.dtype)
        else:
            P = out
        # P_k = | sum_h conj(psi0[h]) * psi_t[h,k] |^2
        for k in range(N):
            re = 0.0
            im = 0.0
            for h in range(H):
                ar = psi_t[h, k].real
                ai = psi_t[h, k].imag
                br = psi0[h].real
                bi = psi0[h].imag
                # a * conj(b) = (ar+iai)*(br-ibi)
                re += ar*br + ai*bi
                im += -ar*bi + ai*br
            P[k] = re*re + im*im
        return P

    elif axis == 1:
        N, H = psi_t.shape[0], psi_t.shape[1]
        if psi0.shape[0] != H:
            raise ValueError("psi0 length mismatch with psi_t (axis=1).")
        if out is None or out.shape[0] != N:
            P = np.empty(N, dtype=psi_t.real.dtype)
        else:
            P = out
        # P_k = | sum_h conj(psi0[h]) * psi_t[k,h] |^2
        for k in range(N):
            re = 0.0
            im = 0.0
            for h in range(H):
                ar = psi_t[k, h].real
                ai = psi_t[k, h].imag
                br = psi0[h].real
                bi = psi0[h].imag
                re += ar*br + ai*bi
                im += -ar*bi + ai*br
            P[k] = re*re + im*im
        return P

    else:
        raise ValueError("axis must be 0 (psi_t shape (H,N)) or 1 (psi_t shape (N,H)).")

# -----------------------------------------------------------------------------
#! Structures
# -----------------------------------------------------------------------------

def spectral_structure(data: np.ndarray, window: int) -> np.ndarray:
    """
    Compute the residuals of a moving average (spectral structure) for each row in the input data.

    For each row, the function subtracts a moving average from the data:
    - For the first `window` points, the moving average is computed with a growing denominator (1, 2, ..., window).
    - For the remaining points, a fixed-size window is used.

    Parameters
    ----------
    data : np.ndarray
        Input 2D array of shape (N, T), where N is the number of rows (e.g., signals or samples)
        and T is the number of time points.
    window : int
        Size of the moving average window.

    Returns
    -------
    np.ndarray
        Array of the same shape as `data`, containing the residuals after subtracting the moving average.
    """

    N, T      = data.shape
    cumsum    = np.cumsum(data, axis=1)      # shape (N, T)
    residual  = np.empty_like(data, dtype=float)

    # first `window` points use growing denominator (1,2,…,window)
    t0        = min(window, T)
    counts    = np.arange(1, t0+1)           # [1, 2, …, t0]
    residual[:, :t0] = data[:, :t0] - cumsum[:, :t0] / counts

    # remaining points use fixed window
    if T > window:
        numer = cumsum[:, window:] - cumsum[:, :-window]
        ma    = numer / window
        residual[:, window:] = data[:, window:] - ma

    return residual

# -----------------------------------------------------------------------------
#! EOF 



