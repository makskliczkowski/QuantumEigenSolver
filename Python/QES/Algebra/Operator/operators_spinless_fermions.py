"""
file        : Algebra/Operator/operators_spinless_fermions.py

This module implements spin operators for quantum systems.
It provides functionality to create and anihilate fermions, as well as to
compute the number of fermions in a given state.

\section{Local Fock basis}

For a chain of $N_s$ spinless fermionic modes we label sites from the \emph{left},
$i=0,1,\dots,N_s-1$.
The on-site occupation number
\[
  n_i = c_i^\dagger c_i \in\{0,1\}
\]
defines the computational Fock basis
\[
  \bigl|\,\mathbf{n}\bigr\rangle
  =
  \bigl|\,n_0\,n_1\ldots n_{N_s-1}\bigr\rangle.
\]

\paragraph{Integer encoding.}
A basis vector is encoded as a non-negative integer
\[
  s(\mathbf{n})
  =
  \sum_{i=0}^{N_s-1} n_i\,2^{N_s-1-i},
\]
so that bit~$N_s-1-i$ of $s$ stores $n_i$.
The \verb|popcount| (Hamming weight) of $s$ equals
$\sum_i n_i$.

%------------------------------------------------------------------------------
\section{Canonical anticommutation relations}

Creation and annihilation operators obey
\begin{equation}
  \{c_i,c_j\}=0,\qquad
  \{c_i^\dagger,c_j^\dagger\}=0,\qquad
  \{c_i,c_j^\dagger\}=\delta_{ij}.
  \label{eq:car}
\end{equation}

%------------------------------------------------------------------------------
\section{Jordan-Wigner phase (fermionic sign)}

Because fermions anticommute, the operator must “pass through’’
all modes to its left before acting.
Define the parity operator up to (but not including) site~$i$
\[
  \mathcal P_i
  =
  (-1)^{\displaystyle\sum_{j=0}^{i-1} n_j}.
\]
With the integer representation this is
\[
  \mathcal P_i(s)
  =
  (-1)^{\mathrm{popcount}\!\bigl(s \,\&\, \bigl[(1\!\ll\! (N_s-i))-1\bigr]\bigr)}.
\]

%------------------------------------------------------------------------------
\section{Action on a basis state}

\begin{align}
  c_i^\dagger\,
  \bigl|\,\mathbf{n}\bigr\rangle
  &=
  \mathcal P_i(\mathbf n)\;
  (1-n_i)\;
  \bigl|\,n_0\!\ldots 1_i\!\ldots n_{N_s-1}\bigr\rangle,
  \label{eq:create}\\[4pt]
  c_i\,
  \bigl|\,\mathbf{n}\bigr\rangle
  &=
  \mathcal P_i(\mathbf n)\;
  n_i\;
  \bigl|\,n_0\!\ldots 0_i\!\ldots n_{N_s-1}\bigr\rangle.
  \label{eq:annih}
\end{align}
If the occupation constraint $(1-n_i)$ or $n_i$ vanishes, the
state is annihilated.

%------------------------------------------------------------------------------
\section{Momentum-space operator}

For any subset $S\subseteq\{0,\dots,N_s-1\}$ we define
\[
  c_k
  =
  \frac{1}{\sqrt{|S|}}
  \sum_{i\in S} e^{-ik i}\,c_i,
  \qquad k\in[0,2\pi).
\]
Acting on a basis state gives a superposition
with amplitudes
\[
  \langle\mathbf n|\,
  c_k\,
  |\mathbf n\rangle
  =
  \frac{1}{\sqrt{|S|}}
  \sum_{i\in S}
        \mathcal P_i(\mathbf n)\;
        n_i\;
        e^{-ik i}.
\]



Author      : Maksymilian Kliczkowski, WUST, Poland
Date        : May 2025
Version     : 1.0
"""

import math
import numpy as np
import numba
from typing import List, Union, Optional, Callable

################################################################################
from Algebra.Operator.operator import Operator, OperatorTypeActing, SymmetryGenerators, create_operator
################################################################################

################################################################################

from general_python.common.tests import GeneralAlgebraicTest
from general_python.lattices.lattice import Lattice
from general_python.algebra.utils import DEFAULT_NP_INT_TYPE, DEFAULT_NP_FLOAT_TYPE, DEFAULT_NP_CPX_TYPE
from general_python.common.binary import BACKEND_REPR as _SPIN, BACKEND_DEF_SPIN, JAX_AVAILABLE
import general_python.common.binary as _binary

if JAX_AVAILABLE:
    from Algebra.Operator.operators_spinless_fermions_jax import c_dag_jnp, c_jnp, c_k_jnp, c_k_dag_jnp


_DEFAULT_INT    = DEFAULT_NP_INT_TYPE
_DEFAULT_FLOAT  = DEFAULT_NP_FLOAT_TYPE
_SPIN           = 1.0
_bit            = _binary.check_int
_flip           = _binary.flip_int

@numba.njit(inline="always")
def _popcount_mask(x: int, mask_bits: int) -> int:
    """Return number of 1-bits in x & mask_bits."""
    return numba.cgutils.int_popcount(x & mask_bits) # LLVM popcnt

###############################################################################
#! Jordan-Wigner sign  (-1)^{#occupied to the *left* of site}
#   * we use the convention that the creation operators are ordered from right to left
#   * $i_1 < i_2 < \ldots < i_n$ and $c_{i_m}^\dagger \cdots c_{i_1}^\dagger |0\rangle$
#   * then, to count how many fermions from site k one needs to pass through, we need to count
#     the number of fermions to the left of $i_1$, $i_2$, and so on until $i_k$.
#   * we use the reverse order of the sites as oposed to the binary representation
#     - the leftmost bit is the first site
#   * the sign is given by the parity of the number of occupied sites to the left of the site
###############################################################################

@numba.njit
def f_parity_int(state: int, ns: int, site: int) -> float:
    
    # mask   = bits for sites < site  ⇒  positions  ns-1-(0 … site-1)
    shift      = ns - site              # first of the sites to the left of site
    mask_bits  = ((1 << shift) - 1)     # faster than per-bit loop - moves shift to the left and fills with 1s to the right.
    # then the mask is shifted to the left by site
    # >>> site = 2
    # >>> ns   = 6
    # >>> mask = 0b010000 -> 0b001111 -> we need to pass through 4 bits
    parity     = _popcount_mask(state, mask_bits) & 1
    return -1.0 if parity else 1.0

@numba.njit
def f_parity_np(state: np.ndarray, site: int) -> float:
    """Return ±1 for NumPy occupation array."""
    parity = 0
    for i in range(site):
        parity ^= int(state[i] > 0) # modulo-2 sum
    return -1.0 if parity else 1.0

###############################################################################
#! Creation / annihilation on *integer* occupation number
###############################################################################

@numba.njit
def c_dag_int_np(state      : int,
                 ns         : int,
                 site       : int,
                 prefactor  : float = 1.0):
    """
    Applies the creation operator (c†) for spinless fermions on a given site of an integer-represented Fock state.
    Parameters
    ----------
    state : int
        The integer representation of the Fock state.
    ns : int
        The total number of sites.
    site : int
        The site index (0-based, leftmost site is 0).
    prefactor : float, optional
        A prefactor to multiply the resulting coefficient (default is 1.0).
    Returns
    -------
    out_state : numpy.ndarray
        Array of length 1 containing the new state(s) as integer(s) after applying the creation operator.
    out_coeff : numpy.ndarray
        Array of length 1 containing the corresponding coefficient(s) after applying the creation operator.
    Notes
    -----
    - If the site is already occupied, the result is zero (fermionic exclusion principle).
    - The sign is determined by the fermionic parity up to the given site.
    - Helper functions `_bit`, `_flip`, and `f_parity_int` are used for bit manipulation and parity calculation.
    
    Example
    -------
    >>> state = 0b000000
    >>> ns = 6
    >>> site = 2
    >>> prefactor = 1.0
    >>> out_state, out_coeff = c_dag_int_np(state, ns, site, prefactor)
    >>> print(out_state)  # Output: [0b000100]
    >>> print(out_coeff)  # Output: [1.0]
    """
    
    # position of the site in the integer representation
    pos   = ns - 1 - site
    
    # check the bit to see whether even bother to flip
    if _bit(state, pos):                         # already occupied → annihilation
        coeff_val = 0.0
        new_state = state
    else:
        sign      = f_parity_int(state, ns, site)
        new_state = _flip(state, pos)
        coeff_val = sign * prefactor

    out_state  = np.empty(1, dtype=_DEFAULT_INT)
    out_coeff  = np.empty(1, dtype=_DEFAULT_FLOAT)
    out_state[0] = new_state
    out_coeff[0] = coeff_val
    return out_state, out_coeff

@numba.njit
def c_int_np(state       : int,
             ns          : int,
             site        : int,
             prefactor   : float = 1.0):
    """
    Applies the annihilation operator (c) for spinless fermions on a given site of a basis state represented as an integer.
    Parameters
    ----------
    state : int
        The integer representation of the basis state.
    ns : int
        The total number of sites in the system.
    site : int
        The site index (0-based, leftmost site is 0) where the operator acts.
    prefactor : float, optional
        A prefactor to multiply the resulting coefficient (default is 1.0).
    Returns
    -------
    out_state : numpy.ndarray
        Array of shape (1,) containing the new basis state(s) as integer(s) after applying the operator.
    out_coeff : numpy.ndarray
        Array of shape (1,) containing the corresponding coefficient(s) after applying the operator.
    Notes
    -----
    - If the site is unoccupied (bit is 0), the output coefficient is 0 and the state is unchanged.
    - The function accounts for the fermionic sign (parity) when applying the operator.
    - Helper functions `_bit`, `_flip`, and `f_parity_int` are assumed to be defined elsewhere in the module.
    """

    # position of the site in the integer representation
    pos = ns - 1 - site
    
    # check the bit to see whether even bother to flip
    if _bit(state, pos) == 0:                    # empty → 0
        coeff_val = 0.0
        new_state = state
    else:
        sign      = f_parity_int(state, ns, site)
        new_state = _flip(state, pos)
        coeff_val = sign * prefactor

    out_state  = np.empty(1, dtype=_DEFAULT_INT)
    out_coeff  = np.empty(1, dtype=_DEFAULT_FLOAT)
    out_state[0] = new_state
    out_coeff[0] = coeff_val
    return out_state, out_coeff

###############################################################################
#!  Creation / annihilation on *NumPy* occupation array
###############################################################################

@numba.njit
def c_dag_np(state      : np.ndarray,
             site       : int,
             prefactor  : float = 1.0):
    """
    Applies the fermionic creation operator (c†) at a given site to a spinless fermion state.

    Parameters
    ----------
    state : np.ndarray
        The occupation number representation of the fermionic state (1D array of 0s and 1s).
    site : int
        The site index at which to apply the creation operator.
    prefactor : float, optional
        A multiplicative prefactor to apply to the resulting amplitude (default is 1.0).

    Returns
    -------
    tuple
        A tuple (new_state, amplitude) where:
            - new_state (np.ndarray): The updated state after applying the creation operator.
            - amplitude (float): The resulting amplitude, including the sign from fermionic parity and the prefactor.
        If the site is already occupied, returns (state, 0.0).

    Notes
    -----
    This function modifies the input state in-place. The sign is determined by the fermionic parity up to the given site.
    """
    if state[site] > 0: # occupied
        return state, 0.0
    sign        = f_parity_np(state, site)
    state[site] = 1
    return state, sign * prefactor

@numba.njit
def c_np(state       : np.ndarray,
         site        : int,
         prefactor   : float = 1.0):
    """
    Applies the annihilation operator (c) to a spinless fermion state at a given site.

    Parameters
    ----------
    state : np.ndarray
        The occupation number representation of the fermionic state (1 for occupied, 0 for unoccupied).
    site : int
        The site index at which to apply the annihilation operator.
    prefactor : float, optional
        A multiplicative prefactor to apply to the result (default is 1.0).

    Returns
    -------
    tuple
        A tuple containing:
            - The updated state as a numpy array.
            - The resulting coefficient (float), which is zero if the site is unoccupied,
                or the product of the sign from the parity function and the prefactor otherwise.

    Notes
    -----
    The function uses the Jordan-Wigner transformation convention, where the sign is determined
    by the parity of occupied sites to the left of the target site.
    """
    if state[site] == 0:
        return state, 0.0
    sign        = f_parity_np(state, site)
    state[site] = 0
    return state, sign * prefactor

###############################################################################
#!  Momentum-space fermionic operator  c_k and c_k†
#      c_k = (1/√N) Σ_i  e^{-ik i} c_i
###############################################################################

@numba.njit
def c_k_int_np(state      : int,
               ns         : int,
               sites      : List[int],
               k          : float,
               prefactor  : float = 1.0):
    """
    Applies the momentum-space annihilation operator c_k to a given spinless fermion state.

    This function constructs the action of the annihilation operator in momentum space
    on a given basis state, for a set of site indices, and returns the resulting states
    and their coefficients.

    Args:
        state (int):
            The integer representation of the input basis state.
        ns (int):
            The total number of sites in the system.
        sites (List[int]):
            List of site indices where the operator acts.
        k (float):
            The momentum value (in radians) for the operator.
        prefactor (float, optional):
            A prefactor to multiply the operator by. Defaults to 1.0.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - out_state: Array of resulting basis states after applying the operator at each site.
            - out_coeff: Array of corresponding coefficients for each resulting state.

    Notes:
        - The coefficients are normalized by sqrt(len(sites)) if sites is not empty, otherwise by 1.0.
        - The function assumes the existence of `c_int_np`, which applies the site-local annihilation operator.
    """
    # count non-zero bits in the state
    non_zero        = _popcount_mask(state, (1 << ns) - 1)
    if non_zero == 0:
        return np.empty(0, dtype=_DEFAULT_INT), np.empty(0, dtype=_DEFAULT_FLOAT)
    out_state       = np.empty(non_zero, dtype=_DEFAULT_INT)
    out_coeff       = np.empty(non_zero, dtype=_DEFAULT_FLOAT)
    index           = 0
    for i in sites:
        res_state, coeff    = c_int_np(state, ns, i, prefactor)
        if coeff == 0.0:
            continue
        out_state[index]    = res_state
        out_coeff[index]    = coeff * np.exp(-1j * k * i)
        index              += 1
    return out_state, out_coeff / np.sqrt(non_zero)

@numba.njit
def c_k_np(state       : np.ndarray,
           sites       : List[int],
           k           : float,
           prefactor   : float = 1.0):
    ns               = state.shape[0]
    # number of occupied sites => upper bound on output length
    non_zero         = 0
    for j in range(ns):
        non_zero    += 1 if state[j] else 0

    if non_zero == 0:
        return (np.empty((0, ns), dtype=state.dtype),
                np.empty(0,        dtype=_DEFAULT_FLOAT))

    out_state        = np.empty((non_zero, ns), dtype=state.dtype)
    out_coeff        = np.empty(non_zero,        dtype=_DEFAULT_FLOAT)

    index            = 0
    for i in sites:
        if state[i] == 0:                         # empty ⇒ no contribution
            continue

        tmp_state     = state.copy()
        tmp_state, c  = c_np(tmp_state, i, prefactor)   # local c_i
        if c == 0.0:
            continue

        out_state[index, :] = tmp_state
        out_coeff[index]    = c * np.exp(-1j * k * i)
        index              += 1

    return (out_state[:index],
            out_coeff[:index] / np.sqrt(max(index, 1)))

@numba.njit
def c_k_dag_int_np(state      : int,
                   ns         : int,
                   sites      : List[int],
                   k          : float,
                   prefactor  : float = 1.0):
    """
    Applies the momentum-space fermionic creation operator c†_k to a given basis state.

    This function constructs the action of the creation operator in momentum space
    on a spinless fermion basis state represented as an integer bitstring. It sums
    over all possible site indices, applying the real-space creation operator and
    weighting each term by the appropriate phase factor exp(1j * k * i).

    Args:
        state (int):
            The input basis state represented as an integer bitstring.
        ns (int):
            The total number of sites in the system.
        sites (List[int]): 
            List of site indices to consider for the creation operator.
        k (float):
            The momentum value for the creation operator.
        prefactor (float, optional):
            An overall prefactor to multiply the coefficients. Defaults to 1.0.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - out_state: Array of resulting basis states (as integers) after applying the operator.
            - out_coeff: Array of corresponding coefficients (complex floats), normalized by sqrt(number of nonzero terms).

    Notes:
        - If the input state is fully occupied, returns empty arrays.
        - The coefficients are normalized by the square root of the number of nonzero terms.
        - Requires the helper functions `_popcount_mask` and `c_dag_int_np`, as well as the constants `_DEFAULT_INT` and `_DEFAULT_FLOAT`.
    """
    # number of empty sites
    occ_bits        = _popcount_mask(state, (1 << ns) - 1)
    non_zero        = ns - occ_bits
    if non_zero == 0:
        return (np.empty(0, dtype=_DEFAULT_INT),
                np.empty(0, dtype=_DEFAULT_FLOAT))

    out_state       = np.empty(non_zero, dtype=_DEFAULT_INT)
    out_coeff       = np.empty(non_zero, dtype=_DEFAULT_FLOAT)

    index           = 0
    for i in sites:
        res_state, coeff = c_dag_int_np(state, ns, i, prefactor)
        if coeff == 0.0:
            continue
        out_state[index]  = res_state
        out_coeff[index]  = coeff * np.exp(1j * k * i)
        index            += 1

    return (out_state[:index],
            out_coeff[:index] / np.sqrt(max(index, 1)))

@numba.njit
def c_k_dag_np(state       : np.ndarray,
               sites       : List[int],
               k           : float,
               prefactor   : float = 1.0):
    """
    Apply momentum‑space creation operator c_k† to a NumPy occupation array.
    """
    ns               = state.shape[0]
    # empty sites count ⇒ upper bound
    non_zero         = ns
    for j in range(ns):
        non_zero    -= 1 if state[j] else 0

    if non_zero == 0:
        return (np.empty((0, ns), dtype=state.dtype),
                np.empty(0,        dtype=_DEFAULT_FLOAT))

    out_state        = np.empty((non_zero, ns), dtype=state.dtype)
    out_coeff        = np.empty(non_zero,        dtype=_DEFAULT_FLOAT)

    index            = 0
    for i in sites:
        if state[i] == 1:                         # already occupied
            continue

        tmp_state     = state.copy()
        tmp_state, c  = c_dag_np(tmp_state, i, prefactor)  # local c_i†
        if c == 0.0:
            continue

        out_state[index, :] = tmp_state
        out_coeff[index]    = c * np.exp(1j * k * i)
        index              += 1

    return (out_state[:index],
            out_coeff[:index] / np.sqrt(max(index, 1)))

###############################################################################
#! Public dispatch helpers  (match your σ-operator API)
###############################################################################

def c_dag(state       : Union[int, np.ndarray],
          ns          : int,
          site        : int,
          prefactor   : float = 1.0):
    """Creation operator dispatcher."""
    if isinstance(state, (int, np.integer)):
        return c_dag_int_np(int(state), ns, site, prefactor)
    if isinstance(state, np.ndarray):
        return c_dag_np(state, site, prefactor)
    return c_dag_jnp(state, ns, site, prefactor)

def c(state          : Union[int, np.ndarray],
      ns             : int,
      site           : int,
      prefactor      : float = 1.0):
    """Annihilation operator dispatcher."""
    if isinstance(state, (int, np.integer)):
        return c_int_np(int(state), ns, site, prefactor)
    if isinstance(state, np.ndarray):
        return c_np(state, site, prefactor)
    return c_jnp(state, ns, site, prefactor)

def c_k(state        : Union[int, np.ndarray],
        ns           : int,
        sites        : Optional[List[int]],
        k            : float,
        prefactor    : float = 1.0):
    """Momentum-space annihilation operator dispatcher."""
    if sites is None:
        sites = list(range(ns))
    if isinstance(state, (int, np.integer)):
        return c_k_int_np(int(state), ns, sites, k, prefactor)
    if isinstance(state, np.ndarray):
        return c_k_np(state, sites, k, prefactor)
    return c_k_jnp(state, ns, sites, k, prefactor)

def c_k_dag(state     : Union[int, np.ndarray],
            ns        : int,
            sites     : Optional[List[int]],
            k         : float,
            prefactor : float = 1.0):
    """Momentum-space creation operator dispatcher."""
    if sites is None:
        sites = list(range(ns))
    if isinstance(state, (int, np.integer)):
        return c_k_dag_int_np(int(state), ns, sites, k, prefactor)
    if isinstance(state, np.ndarray):
        return c_k_dag_np(state, sites, k, prefactor)
    return c_k_dag_jnp(state, ns, sites, k, prefactor)

##############################################################################
#! Factory for the operator
##############################################################################

def c( lattice  : Optional[Lattice]     = None,
    ns          : Optional[int]         = None,
    type_act    : OperatorTypeActing    = OperatorTypeActing.Local,
    sites       : Optional[List[int]]   = None,
    prefactor   : float                 = 1.0) -> Operator:
    """
    Factory for the fermionic annihilation operator c_i.
    """
    return create_operator(
        type_act    = type_act,
        op_func_int = c_int_np,            # integer kernel dispatcher already wraps
        op_func_np  = c_np,
        op_func_jnp = c_jnp if JAX_AVAILABLE else None,
        lattice     = lattice,
        ns          = ns,
        sites       = sites,
        extra_args  = (prefactor,),
        name        = "c",
        modifies    = True
    )

def cdag( lattice   : Optional[Lattice]     = None,
        ns          : Optional[int]         = None,
        type_act    : OperatorTypeActing    = OperatorTypeActing.Local,
        sites       : Optional[List[int]]   = None,
        prefactor   : float                 = 1.0) -> Operator:
    """
    Factory for the fermionic creation operator c_i†.
    """
    return create_operator(
        type_act    = type_act,
        op_func_int = c_dag_int_np,
        op_func_np  = c_dag_np,
        op_func_jnp = c_dag_jnp if JAX_AVAILABLE else None,
        lattice     = lattice,
        ns          = ns,
        sites       = sites,
        extra_args  = (prefactor,),
        name        = "cdag",
        modifies    = True
    )

def ck( k           : float,
        lattice     : Optional[Lattice]     = None,
        ns          : Optional[int]         = None,
        type_act    : OperatorTypeActing    = OperatorTypeActing.Global,
        sites       : Optional[List[int]]   = None,
        prefactor   : float                 = 1.0) -> Operator:
    r"""
    Factory for the momentum‑space annihilation operator
        c_k = \frac{1}{\sqrt{|S|}} \sum_{i\in S} e^{-ik i}\,c_i .
    """
    return create_operator(
        type_act    = type_act,
        op_func_int = c_k_int_np,
        op_func_np  = c_k_np,
        op_func_jnp = c_k_jnp if JAX_AVAILABLE else None,
        lattice     = lattice,
        ns          = ns,
        sites       = sites,
        extra_args  = (k, prefactor),
        name        = f"ck(k={k:.3g})",
        modifies    = True
    )

def ckdag(  k         : float,
            lattice   : Optional[Lattice]     = None,
            ns        : Optional[int]         = None,
            type_act  : OperatorTypeActing    = OperatorTypeActing.Global,
            sites     : Optional[List[int]]   = None,
            prefactor : float                 = 1.0) -> Operator:
    r"""
    Factory for the momentum‑space creation operator
        c_k^{\dagger} = \frac{1}{\sqrt{|S|}} \sum_{i\in S} e^{+ik i}\,c_i^{\dagger}.
    """
    return create_operator(
        type_act    = type_act,
        op_func_int = c_k_dag_int_np,
        op_func_np  = c_k_dag_np,
        op_func_jnp = c_k_dag_jnp if JAX_AVAILABLE else None,
        lattice     = lattice,
        ns          = ns,
        sites       = sites,
        extra_args  = (k, prefactor),
        name        = f"ckdag(k={k:.3g})",
        modifies    = True
    )

##############################################################################
#! End of file
##############################################################################