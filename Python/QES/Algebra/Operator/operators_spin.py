"""
file        : Algebra/Operator/operators_spin.py

This module implements spin operators for quantum systems.
It includes functions for sigma_x, sigma_y, sigma_z, sigma_plus (raising),
sigma_minus (lowering), their products, and a Fourier-transformed sigma_k operator.
The implementation is based on the provided C++ code and uses a general Operator class.

Author      : Maksymilian Kliczkowski, WUST, Poland
Date        : February 2025
Version     : 1.0

Future Work:
- Implement sigma_x, sigma_y, sigma_z operator functions.
- Define functionality for raising and lowering operator functions.
- Implement products of spin operators.
- Add tests for each functionality implemented.
"""

import math
import numpy as np
from typing import List, Tuple, Union, Optional, Callable
from numba import njit

################################################################################
from Algebra.Operator.operator import Operator, OperatorTypeActing, SymmetryGenerators
################################################################################

################################################################################
from general_python.common.tests import GeneralAlgebraicTest
from general_python.lattices.__lattice__ import Lattice
from general_python.algebra.utils import DEFAULT_BACKEND, get_backend as __backend, maybe_jit
from general_python.algebra.utils import DEFAULT_NP_INT_TYPE, DEFAULT_NP_FLOAT_TYPE, DEFAULT_NP_CPX_TYPE
from general_python.common.binary import _BACKEND_REPR as _SPIN, _BACKEND_DEF_SPIN, _JAX_AVAILABLE
from general_python.common.binary import flip, flip_all, check, base2int, int2base, int2binstr
import general_python.common.binary as _binary
################################################################################

_I      = 1j
if _JAX_AVAILABLE:
    from jax import lax
    from jax import numpy as jnp
    from general_python.algebra.utils import DEFAULT_JP_INT_TYPE, DEFAULT_JP_FLOAT_TYPE, DEFAULT_JP_CPX_TYPE

################################################################################
#! Standard Pauli matrices
################################################################################

# Define the Pauli matrices for reference
_SIG_X = np.array([[0, 1],
                [1, 0]], dtype=float)
_SIG_Y = np.array([[0, -1j],
                [1j, 0]], dtype=complex)
_SIG_Z = np.array([[1,  0],
                [0, -1]], dtype=float)
_SIG_P = np.array([[0, 1],
                [0, 0]], dtype=float)
_SIG_M = np.array([[0, 0],
                [1, 0]], dtype=float)
# -----------------------------------------------------------------------------
# Sigma-X (σₓ) operator
# -----------------------------------------------------------------------------

@maybe_jit
def _sigma_x_int_jnp(state, ns, sites, spin_value=_SPIN, backend=DEFAULT_BACKEND):
    """
    Apply the Pauli-X (σₓ) operator on the given sites.
    For each site, flip the bit at position (ns-1-site) using a JAX-compatible flip function.
    
    Args:
        state: A JAX integer (or traced array) representing the state.
        ns (int): Number of sites.
        sites (Union[List[int], None]): A list of site indices.
        spin_value (float): Spin value (default _SPIN).
        backend (str): Backend flag (unused in this JAX version).
    
    Returns:
        A tuple (state, coeff) with the updated state and accumulated coefficient.
    """
    backend = __backend(backend)
    sites   = backend.array(sites)
    
    def body(i, carry):
        curr_state, curr_coeff  = carry
        # sites is static, so extract the site.
        site                    = sites[i]
        pos                     = ns - 1 - site
        # flip is assumed to be a JAX-compatible function that flips the bit at position pos.
        new_state               = flip(curr_state, pos, spin_value=spin_value, backend=backend)
        new_coeff               = curr_coeff * spin_value
        return (new_state, new_coeff)

    num_sites   = len(sites)
    init        = (state, 1.0)
    final_state, final_coeff = lax.fori_loop(0, num_sites, body, init)
    return final_state, final_coeff

@njit
def _sigma_x_integer(state, ns, sites, spin_value=_SPIN):
    """
    Apply the Pauli-X (σₓ) operator on the given sites.
    For each site, flip the bit at position (ns-1-site) using binary.flip.
    """
    coeff = 1.0
    for site in sites:
        pos     = ns - 1 - site
        state   = _binary.flip_int(state, pos)
        coeff   *= spin_value
    
    out_state       = np.empty(1, dtype=np.int64)
    out_state[0]    = state
    out_coeff       = np.empty(1, dtype=np.float64)
    out_coeff[0]    = coeff
    return out_state, out_coeff

def _sigma_x_int(state  : int,
            ns          : int,
            sites       : Union[List[int], None],
            spin_value  : float     = _SPIN,
            backend     : str       = DEFAULT_BACKEND):
    """
    Apply the Pauli-X (σₓ) operator on the given sites.
    For each site, flip the bit at position (ns-1-site) using binary.flip.
    """
    if not isinstance(state, int):
        return _sigma_x_int_jnp(state, ns, sites, spin_value, backend)
    return _sigma_x_integer(state, ns, sites, spin_value)

def _sigma_x_np(state   : np.ndarray,
            ns          : int,
            sites       : Union[List[int], None],
            spin        : bool      = _BACKEND_DEF_SPIN,
            spin_value  : float     = _SPIN):
    """
    Apply the Pauli-X (σₓ) operator on the given sites.
    For each site, flip the bit at position (ns-1-site) using binary.flip.
    """
    coeff = 1.0
    for site in sites:
        state   = flip(state, site, spin=spin, spin_value=spin_value)
        coeff   *= spin_value
    return state, coeff

@maybe_jit
def _sigma_x_jnp(state,
                ns          : int,
                sites       : Union[List[int], None],
                spin        : bool = _BACKEND_DEF_SPIN,
                spin_value  : float = _SPIN,
                backend     : str = DEFAULT_BACKEND):
    """
    Apply the Pauli-X (σₓ) operator on a JAX array state.
    For each site, flip the bit at position (ns-1-site) using binary.flip.
    
    This version is optimized by using a JAX for-loop (lax.fori_loop)
    to avoid Python-level loops and enable JIT compilation.
    """
    if sites is None:
        sites = list(range(ns))
    
    # Pre-calculate the overall coefficient as spin_value raised to the number of flips.
    coeff = spin_value ** len(sites)
    
    # Define the loop body function.
    def body_fun(i, state_val):
        # Get the current site from the (static) list.
        site            = sites[i]
        pos             = ns - 1 - site
        # Flip the bit at the computed position.
        new_state       = flip(state_val, pos, spin=spin, spin_value=spin_value)
        return new_state
    
    # Use lax.fori_loop to iterate over sites.
    new_state           = lax.fori_loop(0, len(sites), body_fun, state)
    return new_state, coeff

def sigma_x(state,
            ns          : int, 
            sites       : Union[List[int], None],
            spin        : bool      = _BACKEND_DEF_SPIN,
            spin_value  : float     = _SPIN,
            backend     : str       = DEFAULT_BACKEND):
    """
    Apply the Pauli-X (σₓ) operator on the given sites.
    For each site, flip the bit at position (ns-1-site) using binary.flip.
    Parameters
    ----------
    state : int or np.ndarray
        The state to apply the operator to.
    ns : int
        The number of spins in the system. 
    sites : list of int or None
        The sites to apply the operator to. If None, apply to all sites.
    spin : bool, optional
        If True, use the spin convention for flipping the bits.
    spin_value : float, optional
        The value to multiply the state by when flipping the bits.
    backend : str, optional
        The backend to use for the computation.
    Returns
    -------
    int or np.ndarray
        The state after applying the operator.   
    """
    if sites is None:
        sites = list(range(ns))
    
    if isinstance(state, int):
        return _sigma_x_int(state, ns, sites, spin_value)
    elif isinstance(state, np.ndarray):
        return _sigma_x_np(state, ns, sites, spin, spin_value)
    return _sigma_x_jnp(state, ns, sites, spin, spin_value, backend)

# -----------------------------------------------------------------------------
# Sigma-Y (σᵧ) operator
# -----------------------------------------------------------------------------

@maybe_jit
def _sigma_y_int_jnp(state,
                    ns          : int,
                    sites       : Union[List[int], None],
                    spin_value  : float = _SPIN,
                    backend     : str   = DEFAULT_BACKEND):
    """
    σᵧ on an integer state (JAX version).

    For each site, if the bit at (ns-1-site) is set then multiply the coefficient
    by (1j*spin_value), otherwise by (-1j*spin_value); then flip the bit.
    
    Args:
        state (int or JAX array): The state to apply the operator to.
        ns (int): The number of spins in the system.
        sites (list of int or None): The sites to apply the operator to. If None, apply to all sites.
        spin_value (float): The value to multiply the state by when flipping the bits.
        backend (str): Backend flag (unused in this JAX version).
    
    Returns:
        tuple: (new_state, coeff) where new_state is the state after applying the operator,
            and coeff is the accumulated complex coefficient.
    """
    backend = __backend(backend)
    sites_arr = backend.array(sites)

    def body(i, carry):
        state_val, coeff    = carry
        site                = sites_arr[i]
        pos                 = ns - 1 - site
        bitmask             = backend.left_shift(1, pos)
        condition           = (state_val & bitmask) > 0
        factor = lax.cond(condition,
                            lambda _: 1j * spin_value,
                            lambda _: -1j * spin_value,
                            operand=None)
        new_state = flip(state_val, pos, spin_value=spin_value, backend=backend)
        return (new_state, coeff * factor)

    final_state, final_coeff = lax.fori_loop(0, len(sites), body, (state, 1.0 + 0j))
    return final_state, final_coeff

@njit
def _sigma_y_integer(state, ns, sites, spin_value=_SPIN):
    """
    σᵧ on an integer state.
    For each site, if the bit at (ns-1-site) is set then multiply coefficient by I*spin_value,
    otherwise by -I*spin_value; then flip the bit.
    """
    coeff = 1.0 + 0j
    for site in sites:
        pos = ns - 1 - site
        if _binary.check_int(state, pos):
            coeff *= (1j * spin_value)
        else:
            coeff *= (-1j * spin_value)
        state = _binary.flip_int(state, pos)
    out_state       = np.empty(1, dtype=np.int64)
    out_state[0]    = state
    out_coeff       = np.empty(1, dtype=np.complex128)
    out_coeff[0]    = coeff
    return out_state, out_coeff

def _sigma_y_int(state      : int,
                ns          : int,
                sites       : Union[List[int], None],
                spin_value  : float = _SPIN,
                backend     : str = DEFAULT_BACKEND):
    """
    σᵧ on an integer state.
    For each site, if the bit at (ns-1-site) is set then multiply coefficient by I*spin_value,
    otherwise by -I*spin_value; then flip the bit.
    Parameters
    ----------
    state : int
        The state to apply the operator to.
    ns : int
        The number of spins in the system.
    sites : list of int or None
        The sites to apply the operator to. If None, apply to all sites.
    spin_value : float, optional
        The value to multiply the state by when flipping the bits.
    Returns
    -------
    int
        The state after applying the operator.    
    """
    if not isinstance(state, int):
        return _sigma_y_int_jnp(state, ns, sites, spin_value, backend)
    return _sigma_y_integer(state, ns, sites, spin_value)

def _sigma_y_np(state: np.ndarray,
                sites: Union[List[int], None],
                spin: bool = _BACKEND_DEF_SPIN,
                spin_value: float = _SPIN):
    """
    σᵧ on a NumPy array state.
    For each site, use the given site as index.
    Parameters
    ----------
    state : np.ndarray
        The state to apply the operator to.
    sites : list of int or None
        The sites to apply the operator to. If None, apply to all sites.
    spin : bool, optional
        If True, use the spin convention for flipping the bits.
    spin_value : float, optional
        The value to multiply the state by when flipping the bits.
    Returns
    -------
    np.ndarray
        The state after applying the operator    
    """
    coeff = 1.0 + 0j
    for site in sites:
        # For NumPy arrays, we use the site index directly.
        factor  =   1j * spin_value if check(state, site) else -1j * spin_value
        coeff   *=  factor
        state   =   flip(state, site, spin=spin, spin_value=spin_value)
    return state, coeff

@maybe_jit
def _sigma_y_jnp(state,
                ns          : int,
                sites       : Union[List[int], None],
                spin        : bool = _BACKEND_DEF_SPIN,
                spin_value  : float = _SPIN,
                backend     : str = DEFAULT_BACKEND):
    """
    σᵧ on a JAX array state.
    Uses lax.fori_loop.
    Parameters
    ----------
    state : np.ndarray
        The state to apply the operator to.
    ns : int
        The number of spins in the system.
    sites : list of int or None
        The sites to apply the operator to. If None, apply to all sites.    
    spin : bool, optional
        If True, use the spin convention for flipping the bits.
    spin_value : float, optional
        The value to multiply the state by when flipping the bits.
    backend : str, optional
        The backend to use for the computation.
        
    Returns
    -------
    np.ndarray
        The state after applying the operator    
    """
    coeff = 1.0 + 0j

    def body_fun(i, state_val):
        site            = sites[i]
        pos             = ns - 1 - site
        # factor          = 1j * spin_value if check(state_val, pos) else -1j * spin_value
        new_state       = flip(state_val, pos, spin=spin, spin_value=spin_value)
        return new_state
    
    new_state           = lax.fori_loop(0, len(sites), body_fun, state)
    
    # For simplicity, we recompute coeff by a Python loop:
    for site in sites:
        pos     = ns - 1 - site
        coeff   *= (1j * spin_value) if check(state, pos) else (-1j * spin_value)
    return new_state, coeff

def sigma_y(state,
            ns: int,
            sites: Union[List[int], None],
            spin: bool = _BACKEND_DEF_SPIN,
            spin_value: float = _SPIN,
            backend: str = DEFAULT_BACKEND):
    """
    Dispatch for σᵧ.
    """
    if sites is None:
        sites = list(range(ns))
    if isinstance(state, int):
        return _sigma_y_int(state, ns, sites, spin_value)
    elif isinstance(state, np.ndarray):
        return _sigma_y_np(state, sites, spin, spin_value)
    else:
        return _sigma_y_jnp(state, ns, sites, spin, spin_value, backend)

# -----------------------------------------------------------------------------
# Sigma-Z (σ_z) operator
# -----------------------------------------------------------------------------

@maybe_jit
def _sigma_z_int_jnp(state,
                    ns          : int,
                    sites       : Union[List[int], None],
                    spin_value  : float     = _SPIN,
                    backend     : str       = DEFAULT_BACKEND):
    """
    σ_z on an integer state.
    For each site, if the bit at (ns-1-site) is set then multiply by spin_value; else by -spin_value.
    The state is unchanged.
    
    Args:
        state: A JAX integer (or traced array of integers) representing the state.
        ns (int): The number of sites.
        sites (Union[List[int], None]): A list of site indices.
        spin_value (float): The spin value (default _SPIN).
        backend (str): Backend flag (unused in this JAX version).
    
    Returns:
        A tuple (state, coeff) where state is unchanged and coeff is the product
        of the factors determined by the bits in state.
    """
    # Body function for the fori_loop. The loop variable 'i' runs over site indices.
    
    backend = __backend(backend)
    sites   = backend.array(sites)
    
    def body(i, coeff):
        # Since sites is a static Python list, we can extract the site index.
        site        = sites[i]
        # Compute the bit position: (ns - 1 - site)
        pos         = ns - 1 - site
        # Compute the bit mask using JAX operations.
        bitmask     = backend.left_shift(1, pos)
        # Compute the condition: is the bit set? This returns a boolean JAX array.
        condition   = (state & bitmask) > 0
        # Use lax.cond to choose the factor:
        factor      = lax.cond(condition,
                            lambda _: spin_value,
                            lambda _: -spin_value,
                            operand=None)
        # Multiply the accumulator with the factor.
        return coeff * factor

    # Use lax.fori_loop to accumulate the coefficient over all sites.
    coeff = lax.fori_loop(0, len(sites), body, 1.0)
    return state, coeff

@njit
def _sigma_z_integer(state, ns, sites, spin_value=_SPIN):
    """
    σ_z on an integer state.
    For each site, if the bit at (ns-1-site) is set then multiply by spin_value; else by -spin_value.
    The state is unchanged.
    """
    
    coeff = 1.0
    for site in sites:
        pos     = ns - 1 - site
        factor  = spin_value if _binary.check_int(state, pos) else -spin_value
        coeff   *= factor
    out_state       = np.empty(1, dtype=DEFAULT_NP_INT_TYPE)
    out_state[0]    = state
    out_coeff       = np.empty(1, dtype=DEFAULT_NP_FLOAT_TYPE)
    out_coeff[0]    = coeff
    return out_state, out_coeff

def _sigma_z_int(state      : int,
                ns          : int,
                sites       : Union[List[int], None],
                spin_value  : float = _SPIN,
                backend     : str = DEFAULT_BACKEND):
    """
    σ_z on an integer state.
    For each site, if the bit at (ns-1-site) is set then multiply by spin_value; else by -spin_value.
    The state is unchanged.
    """
    if not isinstance(state, int):
        return _sigma_z_int_jnp(state, ns, sites, spin_value, backend)
    return _sigma_z_integer(state, ns, sites, spin_value)

def _sigma_z_np(state: np.ndarray,
                ns: int,
                sites: Union[List[int], None],
                spin: bool = _BACKEND_DEF_SPIN,
                spin_value: float = _SPIN):
    """
    σ_z on a NumPy array state.
    """
    coeff = 1.0
    for site in sites:
        factor = spin_value if check(state, site) else -spin_value
        coeff *= factor
    return state, coeff

@maybe_jit
def _sigma_z_jnp(state,
                ns          : int,
                sites       : Union[List[int], None],
                spin        : bool  = _BACKEND_DEF_SPIN,
                spin_value  : float = _SPIN,
                backend     : str   = DEFAULT_BACKEND):
    """
    σ_z on a JAX array state.
    """
    if sites is None:
        sites = list(range(ns))
    coeff = 1.0
    for site in sites:
        factor = spin_value if check(state, site) else -spin_value
        coeff *= factor
    return state, coeff

def sigma_z(state,
            ns: int,
            sites: Union[List[int], None],
            spin: bool = _BACKEND_DEF_SPIN,
            spin_value: float = _SPIN,
            backend: str = DEFAULT_BACKEND):
    """
    Dispatch for σ_z.
    """
    if sites is None:
        sites = list(range(ns))
    if isinstance(state, int):
        return _sigma_z_integer(state, ns, sites, spin_value)
    elif isinstance(state, np.ndarray):
        return _sigma_z_np(state, ns, sites, spin, spin_value)
    else:
        return _sigma_z_jnp(state, ns, sites, spin, spin_value, backend)

# -----------------------------------------------------------------------------
# Sigma-Plus (σ⁺) operator
# -----------------------------------------------------------------------------

def _sigma_plus_int(state  : int,
                    ns     : int,
                    sites  : Union[List[int], None],
                    spin_value : float = _SPIN):
    """
    σ⁺ (raising) on an integer state.
    For each site, if the bit at (ns-1-site) is set, return 0; else flip.
    """
    coeff = 1.0
    for site in sites:
        pos = ns - 1 - site
        if check(state, pos):
            return state, 0.0
        state, _ = flip(state, pos, spin_value=spin_value)
        coeff *= spin_value
    return state, coeff

def _sigma_plus_np(state    : np.ndarray,
                    ns      : int,
                    sites   : Union[List[int], None],
                    spin    : bool = _BACKEND_DEF_SPIN,
                    spin_value : float = _SPIN):
    """
    σ⁺ on a NumPy array state.
    """
    coeff = 1.0
    for site in sites:
        if check(state, site):
            return state, 0.0
        state = flip(state, site, spin=spin, spin_value=spin_value)
        coeff *= spin_value
    return state, coeff

@maybe_jit
def _sigma_plus_jnp(state,
                    ns      : int,
                    sites   : Union[List[int], None],
                    spin    : bool = _BACKEND_DEF_SPIN,
                    spin_value : float = _SPIN,
                    backend : str = DEFAULT_BACKEND):
    """
    σ⁺ on a JAX array state.
    Uses lax.fori_loop.
    """
    coeff = 1.0
    def body_fun(i, state_val):
        site = sites[i]
        pos  = ns - 1 - site
        if check(state_val, pos):
            return state_val  # Short-circuit; ideally coeff should be set to 0.
        new_state, _ = flip(state_val, pos, spin=spin, spin_value=spin_value)
        return new_state
    new_state = lax.fori_loop(0, len(sites), body_fun, state)
    return new_state, coeff

def sigma_plus(state,
                ns      : int,
                sites   : Union[List[int], None],
                spin    : bool = _BACKEND_DEF_SPIN,
                spin_value : float = _SPIN,
                backend : str = DEFAULT_BACKEND):
    """
    Dispatch for σ⁺.
    """
    if sites is None:
        sites = list(range(ns))
    if isinstance(state, int):
        return _sigma_plus_int(state, ns, sites, spin_value)
    elif isinstance(state, np.ndarray):
        return _sigma_plus_np(state, ns, sites, spin, spin_value)
    return _sigma_plus_jnp(state, ns, sites, spin, spin_value, backend)

@maybe_jit
def _sigma_plus_int_jnp(state, ns, sites, spin_value=_SPIN, backend=DEFAULT_BACKEND):
    backend = __backend(backend)
    sites = backend.array(sites)
    def body(i, carry):
        curr_state, curr_coeff = carry
        pos = ns - 1 - sites[i]
        bitmask = backend.left_shift(1, pos)
        condition = (curr_state & bitmask) > 0
        new_state = lax.cond(condition,
                             lambda _: curr_state,
                             lambda _: flip(curr_state, pos, spin_value=spin_value, backend=backend),
                             operand=None)
        new_coeff = lax.cond(condition,
                             lambda _: 0.0,
                             lambda _: curr_coeff * spin_value,
                             operand=None)
        return (new_state, new_coeff)
    init = (state, 1.0)
    final_state, final_coeff = lax.fori_loop(0, len(sites), body, init)
    return final_state, final_coeff

# -----------------------------------------------------------------------------
# Sigma-Minus (σ⁻) operator
# -----------------------------------------------------------------------------

def _sigma_minus_int(state  : int,
                    ns     : int,
                    sites  : Union[List[int], None],
                    spin_value : float = _SPIN):
    """
    σ⁻ (lowering) on an integer state.
    For each site, if the bit at (ns-1-site) is not set, return 0; else flip.
    """
    coeff = 1.0
    if sites is None:
        sites = list(range(ns))

    for site in sites:
        pos = ns - 1 - site
        if not check(state, pos):
            return state, 0.0
        state, _ = flip(state, pos, spin_value=spin_value)
        coeff *= spin_value
    return state, coeff

def _sigma_minus_np(state   : np.ndarray,
                    ns      : int,
                    sites   : Union[List[int], None],
                    spin    : bool = _BACKEND_DEF_SPIN,
                    spin_value : float = _SPIN):
    """
    σ⁻ on a NumPy array state.
    """
    coeff = 1.0
    for site in sites:
        if not check(state, site):
            return state, 0.0
        state = flip(state, site, spin=spin, spin_value=spin_value)
        coeff *= spin_value
    return state, coeff

@maybe_jit
def _sigma_minus_jnp(state,
                    ns      : int,
                    sites   : Union[List[int], None],
                    spin    : bool = _BACKEND_DEF_SPIN,
                    spin_value : float = _SPIN,
                    backend : str = DEFAULT_BACKEND):
    """
    σ⁻ on a JAX array state.
    """
    coeff = 1.0
    def body_fun(i, state_val):
        site = sites[i]
        pos  = ns - 1 - site
        if not check(state_val, pos):
            return state_val
        new_state, _ = flip(state_val, pos, spin=spin, spin_value=spin_value)
        return new_state
    new_state = lax.fori_loop(0, len(sites), body_fun, state)
    return new_state, coeff

def sigma_minus(state,
                ns      : int,
                sites   : Union[List[int], None],
                spin    : bool = _BACKEND_DEF_SPIN,
                spin_value : float = _SPIN,
                backend : str = DEFAULT_BACKEND):
    """
    Dispatch for σ⁻.
    """
    if sites is None:
        sites = list(range(ns))
    if isinstance(state, int):
        return _sigma_minus_int(state, ns, sites, spin_value)
    elif isinstance(state, np.ndarray):
        return _sigma_minus_np(state, ns, sites, spin, spin_value)
    return _sigma_minus_jnp(state, ns, sites, spin, spin_value, backend)

@maybe_jit
def _sigma_minus_int_jnp(state, ns, sites, spin_value=_SPIN, backend=DEFAULT_BACKEND):
    backend = __backend(backend)
    sites = backend.array(sites)
    def body(i, carry):
        curr_state, curr_coeff = carry
        pos = ns - 1 - sites[i]
        bitmask = backend.left_shift(1, pos)
        condition = (curr_state & bitmask) > 0
        new_state = lax.cond(condition,
                             lambda _: flip(curr_state, pos, spin_value=spin_value, backend=backend),
                             lambda _: curr_state,
                             operand=None)
        new_coeff = lax.cond(condition,
                             lambda _: curr_coeff * spin_value,
                             lambda _: 0.0,
                             operand=None)
        return (new_state, new_coeff)
    init = (state, 1.0)
    final_state, final_coeff = lax.fori_loop(0, len(sites), body, init)
    return final_state, final_coeff

# -----------------------------------------------------------------------------
# Sigma_pm (σ⁺ then σ⁻) operator
# -----------------------------------------------------------------------------

def _sigma_pm_int(state  : int,
                ns     : int,
                sites  : Union[List[int], None],
                spin_value : float = _SPIN):
    """
    Alternating operator: for even indices apply σ⁺, for odd indices apply σ⁻.
    """
    coeff = 1.0
    for i, site in enumerate(sites):
        pos = ns - 1 - site
        if i % 2 == 0:
            if check(state, pos):
                return state, 0.0
            state, _ = flip(state, pos, spin_value=spin_value)
        else:
            if not check(state, pos):
                return state, 0.0
            state, _ = flip(state, pos, spin_value=spin_value)
        coeff *= spin_value
    return state, coeff

def _sigma_pm_np(state   : np.ndarray,
                ns      : int,
                sites   : Union[List[int], None],
                spin    : bool = _BACKEND_DEF_SPIN,
                spin_value : float = _SPIN):
    """
    Alternating operator on a NumPy array state.
    """
    coeff = 1.0
    for i, site in enumerate(sites):
        if i % 2 == 0:
            if check(state, site):
                return state, 0.0
            state = flip(state, site, spin=spin, spin_value=spin_value)
        else:
            if not check(state, site):
                return state, 0.0
            state = flip(state, site, spin=spin, spin_value=spin_value)
        coeff *= spin_value
    return state, coeff

@maybe_jit
def _sigma_pm_jnp(state,
                ns      : int,
                sites   : Union[List[int], None],
                spin    : bool = _BACKEND_DEF_SPIN,
                spin_value : float = _SPIN,
                backend : str = DEFAULT_BACKEND):
    """
    Alternating operator on a JAX array state.
    """
    if sites is None:
        sites = list(range(ns))
    coeff = 1.0
    def body_fun(i, state_val):
        site = sites[i]
        pos  = ns - 1 - site
        if i % 2 == 0:
            if check(state_val, pos):
                return state_val
            new_state, _ = flip(state_val, pos, spin=spin, spin_value=spin_value)
        else:
            if not check(state_val, pos):
                return state_val
            new_state, _ = flip(state_val, pos, spin=spin, spin_value=spin_value)
        return new_state
    new_state = lax.fori_loop(0, len(sites), body_fun, state)
    return new_state, coeff

def sigma_pm(state,
            ns      : int,
            sites   : Union[List[int], None],
            spin    : bool = _BACKEND_DEF_SPIN,
            spin_value : float = _SPIN,
            backend : str = DEFAULT_BACKEND):
    """
    Dispatch for alternating operator: σ⁺ then σ⁻.
    """
    if sites is None:
        sites = list(range(ns))
    if isinstance(state, int):
        return _sigma_pm_int(state, ns, sites, spin_value)
    elif isinstance(state, np.ndarray):
        return _sigma_pm_np(state, ns, sites, spin, spin_value)
    return _sigma_pm_jnp(state, ns, sites, spin, spin_value, backend)

@maybe_jit
def _sigma_pm_int_jnp(state, ns, sites, spin_value=_SPIN, backend=DEFAULT_BACKEND):
    # Alternating operator: even index applies sigma⁺, odd index sigma⁻.
    backend = __backend(backend)
    sites = backend.array(sites)
    def body(i, carry):
        curr_state, curr_coeff = carry
        pos = ns - 1 - sites[i]
        bitmask = backend.left_shift(1, pos)
        # For even indices (i % 2 == 0): require bit not set; for odd indices: require bit set.
        even_branch = lax.cond((curr_state & bitmask) == 0,
                               lambda _: (flip(curr_state, pos, spin_value=spin_value, backend=backend),
                                          curr_coeff * spin_value),
                               lambda _: (curr_state, 0.0),
                               operand=None)
        odd_branch = lax.cond((curr_state & bitmask) > 0,
                              lambda _: (flip(curr_state, pos, spin_value=spin_value, backend=backend),
                                         curr_coeff * spin_value),
                              lambda _: (curr_state, 0.0),
                              operand=None)
        new_state, new_coeff = even_branch if (i % 2 == 0) else odd_branch
        return (new_state, new_coeff)
    init = (state, 1.0)
    final_state, final_coeff = lax.fori_loop(0, len(sites), body, init)
    return final_state, final_coeff

# -----------------------------------------------------------------------------
# Sigma_mp (σ⁻ then σ⁺) operator
# -----------------------------------------------------------------------------

def _sigma_mp_int(state  : int,
                ns     : int,
                sites  : Union[List[int], None],
                spin_value : float = _SPIN):
    """
    Alternating operator: for even indices apply σ⁻, for odd indices apply σ⁺.
    """
    coeff = 1.0
    for i, site in enumerate(sites):
        pos = ns - 1 - site
        if i % 2 == 0:
            if not check(state, pos):
                return state, 0.0
            state, _ = flip(state, pos, spin_value=spin_value)
        else:
            if check(state, pos):
                return state, 0.0
            state, _ = flip(state, pos, spin_value=spin_value)
        coeff *= spin_value
    return state, coeff

def _sigma_mp_np(state   : np.ndarray,
                ns      : int,
                sites   : Union[List[int], None],
                spin    : bool = _BACKEND_DEF_SPIN,
                spin_value : float = _SPIN):
    """
    Alternating operator (σ⁻ then σ⁺) on a NumPy array state.
    """
    coeff = 1.0
    for i, site in enumerate(sites):
        if i % 2 == 0:
            if not check(state, site):
                return state, 0.0
            state = flip(state, site, spin=spin, spin_value=spin_value)
        else:
            if check(state, site):
                return state, 0.0
            state = flip(state, site, spin=spin, spin_value=spin_value)
        coeff *= spin_value
    return state, coeff

@maybe_jit
def _sigma_mp_jnp(state,
                ns      : int,
                sites   : Union[List[int], None],
                spin    : bool = _BACKEND_DEF_SPIN,
                spin_value : float = _SPIN,
                backend : str = DEFAULT_BACKEND):
    """
    Alternating operator (σ⁻ then σ⁺) on a JAX array state.
    """

    coeff = 1.0
    def body_fun(i, state_val):
        site = sites[i]
        pos  = ns - 1 - site
        if i % 2 == 0:
            if not check(state_val, pos):
                return state_val
            new_state, _ = flip(state_val, pos, spin=spin, spin_value=spin_value)
        else:
            if check(state_val, pos):
                return state_val
            new_state, _ = flip(state_val, pos, spin=spin, spin_value=spin_value)
        return new_state
    new_state = lax.fori_loop(0, len(sites), body_fun, state)
    return new_state, coeff

def sigma_mp(state,
            ns      : int,
            sites   : Union[List[int], None],
            spin    : bool = _BACKEND_DEF_SPIN,
            spin_value : float = _SPIN,
            backend : str = DEFAULT_BACKEND):
    """
    Dispatch for alternating operator: σ⁻ then σ⁺.
    """
    if sites is None:
        sites = list(range(ns))
    if isinstance(state, int):
        return _sigma_mp_int(state, ns, sites, spin_value)
    elif isinstance(state, np.ndarray):
        return _sigma_mp_np(state, ns, sites, spin, spin_value)
    return _sigma_mp_jnp(state, ns, sites, spin, spin_value, backend)

@maybe_jit
def _sigma_mp_int_jnp(state, ns, sites, spin_value=_SPIN, backend=DEFAULT_BACKEND):
    # Alternating operator: even index applies sigma⁻, odd index sigma⁺.
    backend = __backend(backend)
    sites = backend.array(sites)
    def body(i, carry):
        curr_state, curr_coeff = carry
        pos = ns - 1 - sites[i]
        bitmask = backend.left_shift(1, pos)
        even_branch = lax.cond((curr_state & bitmask) > 0,
                               lambda _: (flip(curr_state, pos, spin_value=spin_value, backend=backend),
                                          curr_coeff * spin_value),
                               lambda _: (curr_state, 0.0),
                               operand=None)
        odd_branch = lax.cond((curr_state & bitmask) == 0,
                              lambda _: (flip(curr_state, pos, spin_value=spin_value, backend=backend),
                                         curr_coeff * spin_value),
                              lambda _: (curr_state, 0.0),
                              operand=None)
        new_state, new_coeff = even_branch if (i % 2 == 0) else odd_branch
        return (new_state, new_coeff)
    init = (state, 1.0)
    final_state, final_coeff = lax.fori_loop(0, len(sites), body, init)
    return final_state, final_coeff

# -----------------------------------------------------------------------------
# Sigma-K operator (Fourier-transformed spin operator)
# -----------------------------------------------------------------------------

def _sigma_k_int(state : int,
                ns       : int,
                sites    : Union[List[int], None],
                k        : float,
                spin     : bool = _BACKEND_DEF_SPIN,
                spin_value : float = _SPIN):
    """
    Compute the Fourier-transformed spin operator (σₖ) on an integer state.
    For each site, if the bit at (ns-1-site) is set then use +I; else -1,
    multiplied by exp(1j*k*site). Normalize by sqrt(len(sites)).
    """
    total = 0.0 + 0j
    for site in sites:
        pos = ns - 1 - site
        factor = (1j if check(state, pos) else -1.0)
        total += factor * math.exp(1j * k * site)
    norm = math.sqrt(len(sites)) if sites else 1.0
    return state, total / norm

def _sigma_k_np(state : np.ndarray,
                ns       : int,
                sites    : Union[List[int], None],
                k        : float,
                spin     : bool = _BACKEND_DEF_SPIN,
                spin_value : float = _SPIN):
    """
    Compute the Fourier-transformed spin operator (σₖ) on a NumPy array state.
    """
    total = 0.0 + 0j
    for site in sites:
        factor = (1j if check(state, site) else -1.0)
        total += factor * math.exp(1j * k * site)
    norm = math.sqrt(len(sites)) if sites else 1.0
    return state, total / norm

@maybe_jit
def _sigma_k_jnp(state,
                ns       : int,
                sites    : Union[List[int], None],
                k        : float,
                spin     : bool = _BACKEND_DEF_SPIN,
                spin_value : float = _SPIN,
                backend  : str = DEFAULT_BACKEND):
    """
    Compute the Fourier-transformed spin operator (σₖ) on a JAX array state.
    Uses lax.fori_loop.
    """
    total = 0.0 + 0j
    def body_fun(i, total_val):
        site = sites[i]
        pos  = ns - 1 - site
        factor = 1j if check(state, pos) else -1.0
        return total_val + factor * math.exp(1j * k * site)
    total = lax.fori_loop(0, len(sites), body_fun, total)
    norm = math.sqrt(len(sites)) if sites else 1.0
    return state, total / norm

def sigma_k(state,
            ns       : int,
            sites    : Union[List[int], None],
            k        : float,
            spin     : bool = _BACKEND_DEF_SPIN,
            spin_value : float = _SPIN,
            backend  : str = DEFAULT_BACKEND):
    """
    Dispatch for the Fourier-transformed spin operator σₖ.
    """
    if sites is None:
        sites = list(range(ns))
    if isinstance(state, int):
        return _sigma_k_int(state, ns, sites, k, spin, spin_value)
    elif isinstance(state, np.ndarray):
        return _sigma_k_np(state, ns, sites, k, spin, spin_value)
    return _sigma_k_jnp(state, ns, sites, k, spin, spin_value, backend)

@maybe_jit
def _sigma_k_int_jnp(state, ns, sites, k, spin_value=_SPIN, backend=DEFAULT_BACKEND):
    backend = __backend(backend)
    sites = backend.array(sites)
    def body(i, total):
        site = sites[i]
        pos = ns - 1 - site
        bitmask = backend.left_shift(1, pos)
        factor = lax.cond((state & bitmask) > 0,
                          lambda _: 1j,
                          lambda _: -1.0,
                          operand=None)
        return total + factor * math.exp(1j * k * int(site))
    total = lax.fori_loop(0, len(sites), body, 0.0+0j)
    norm = math.sqrt(len(sites)) if len(sites) > 0 else 1.0
    return state, total / norm

################################################################################
# Factory Functions: Wrap the elementary functions in Operator objects.
################################################################################

# -----------------------------------------------------------------------------
# Factory function for sigma-x (σₓ)
# -----------------------------------------------------------------------------

def sig_x(  lattice     : Optional[Lattice]     = None,
            ns          : Optional[int]         = None,
            type_act    : OperatorTypeActing    = OperatorTypeActing.Global,
            sites       : Optional[List[int]]   = [0],
            spin        : bool                  = _BACKEND_DEF_SPIN,
            spin_value  : float                 = _SPIN,
            backend     : str                   = DEFAULT_BACKEND) -> Operator:
    """
    Factory function for σₓ.
    Parameters
    ----------
    lattice : Lattice, optional
        The lattice to use for the operator.
    ns : int, optional
        The number of spins in the system.
    type_act : OperatorTypeActing, optional
        The type of acting for the operator. 
    sites : list of int, optional
        The sites to apply the operator to.
    spin : bool, optional
        If True, use the spin convention for flipping the bits.
    spin_value : float, optional
        The value to multiply the state by when flipping the bits.
    backend : str, optional
        The backend to use for the computation.
    Returns
    -------
    Operator
        The σₓ operator.    
    """
    
    if lattice is None:
        assert ns is not None, "Either lattice or ns must be provided."
    else:
        ns = lattice.ns
    
    if type_act == OperatorTypeActing.Global:
        def op(state):
            return sigma_x(state, ns, sites, spin, spin_value, backend)
        # set the name
        _name = "Sx/"
        for site in sites:
            _name += f"{site}-"
        _name = _name[:-1]
        return Operator(fun     =   op, 
                        eigval  =   1.0, 
                        lattice =   lattice,
                        name    =   _name,
                        typek   =   SymmetryGenerators.Other, modifies=True)
    elif type_act == OperatorTypeActing.Local:
        def op(state, site):
            return sigma_x(state, ns, [site], spin, spin_value, backend)
        return Operator(fun     =   op,
                        eigval  =   1.0,
                        lattice =   lattice,
                        name    =   "Sx/L",
                        typek   =   SymmetryGenerators.Other, modifies=True)
    elif type_act == OperatorTypeActing.Correlation:
        def op(state, site1, site2):
            return sigma_x(state, ns, [site1, site2], spin, spin_value, backend)
        return Operator(fun     =   op,
                        eigval  =   1.0,
                        lattice =   lattice,
                        name    =   "Sx/C",
                        typek   =   SymmetryGenerators.Other, modifies=True)
    else:
        raise ValueError("Invalid OperatorTypeActing")

# -----------------------------------------------------------------------------
# Factory function for sigma-y (σᵧ)
# -----------------------------------------------------------------------------

def sig_y( lattice     : Optional[Lattice]     = None,
           ns          : Optional[int]         = None,
           type_act    : OperatorTypeActing    = OperatorTypeActing.Global,
           sites       : Optional[List[int]]   = [0],
           spin        : bool                  = _BACKEND_DEF_SPIN,
           spin_value  : float                 = _SPIN,
           backend     : str                   = DEFAULT_BACKEND) -> Operator:
    """
    Factory function for σᵧ.
    Parameters
    ----------
    lattice : Lattice, optional
        The lattice to use for the operator.
    ns : int, optional
        The number of spins in the system.
    type_act : OperatorTypeActing, optional
        The type of acting for the operator.
    sites : list of int, optional
        The sites to apply the operator to.
    spin : bool, optional
        If True, use the spin convention for flipping the bits.
    spin_value : float, optional
        The value to multiply the state by when flipping the bits.
    backend : str, optional
        The backend to use for the computation.
    Returns
    -------
    Operator
        The σᵧ operator.
    """
    if lattice is None:
        assert ns is not None, "Either lattice or ns must be provided."
    else:
        ns = lattice.ns
    if type_act == OperatorTypeActing.Global:
        def op(state):
            return sigma_y(state, ns, sites, spin, spin_value, backend)
        _name = "Sy/"
        for site in sites:
            _name += f"{site}-"
        _name = _name[:-1]
        return Operator(fun = op,
                        eigval = 1.0,
                        lattice = lattice,
                        name = _name,
                        typek = SymmetryGenerators.Other, modifies=True)
    elif type_act == OperatorTypeActing.Local:
        def op(state, site):
            return sigma_y(state, ns, [site], spin, spin_value, backend)
        return Operator(fun = op,
                        eigval = 1.0,
                        lattice = lattice,
                        name = "Sy/L",
                        typek = SymmetryGenerators.Other, modifies=True)
    elif type_act == OperatorTypeActing.Correlation:
        def op(state, site1, site2):
            return sigma_y(state, ns, [site1, site2], spin, spin_value, backend)
        return Operator(fun = op,
                        eigval = 1.0,
                        lattice = lattice,
                        name = "Sy/C",
                        typek = SymmetryGenerators.Other, modifies=True)
    else:
        raise ValueError("Invalid OperatorTypeActing")

# -----------------------------------------------------------------------------
# Factory function for sigma_z (σ_z)
# -----------------------------------------------------------------------------

def sig_z( lattice     : Optional[Lattice]     = None,
           ns          : Optional[int]         = None,
           type_act    : OperatorTypeActing    = OperatorTypeActing.Global,
           sites       : Optional[List[int]]   = [0],
           spin        : bool                  = _BACKEND_DEF_SPIN,
           spin_value  : float                 = _SPIN) -> Operator:
    """
    Factory function for σ_z.
    Parameters
    ----------
    lattice : Lattice, optional
        The lattice to use for the operator.
    ns : int, optional
        The number of spins in the system.
    type_act : OperatorTypeActing, optional
        The type of acting for the operator.
    sites : list of int, optional
        The sites to apply the operator to.
    spin : bool, optional
        If True, use the spin convention.
    spin_value : float, optional
        The value used for the spin.
    Returns
    -------
    Operator
        The σ_z operator.
    """
    if lattice is None:
        assert ns is not None, "Either lattice or ns must be provided."
    else:
        ns = lattice.ns
    if type_act == OperatorTypeActing.Global:
        def op(state):
            return sigma_z(state, ns, sites, spin, spin_value)
        _name = "Sz/"
        for site in sites:
            _name += f"{site}-"
        _name = _name[:-1]
        return Operator(fun = op,
                        eigval = 1.0,
                        lattice = lattice,
                        name = _name,
                        typek = SymmetryGenerators.Other, modifies=False)
    elif type_act == OperatorTypeActing.Local:
        def op(state, site):
            return sigma_z(state, ns, [site], spin, spin_value)
        return Operator(fun = op,
                        eigval = 1.0,
                        lattice = lattice,
                        name = "Sz/L",
                        typek = SymmetryGenerators.Other, modifies=False)
    elif type_act == OperatorTypeActing.Correlation:
        def op(state, site1, site2):
            return sigma_z(state, ns, [site1, site2], spin, spin_value)
        return Operator(fun = op,
                        eigval = 1.0,
                        lattice = lattice,
                        name = "Sz/C",
                        typek = SymmetryGenerators.Other, modifies=False)
    else:
        raise ValueError("Invalid OperatorTypeActing")

# -----------------------------------------------------------------------------
# Factory function for sigma-plus (σ⁺)
# -----------------------------------------------------------------------------

def sig_p( lattice    : Optional[Lattice]     = None,
              ns         : Optional[int]         = None,
              type_act   : OperatorTypeActing    = OperatorTypeActing.Global,
              sites      : Optional[List[int]]   = [0],
              spin       : bool                  = _BACKEND_DEF_SPIN,
              spin_value : float                 = _SPIN,
              backend    : str                   = DEFAULT_BACKEND) -> Operator:
    """
    Factory function for σ⁺.
    Parameters
    ----------
    lattice : Lattice, optional
        The lattice to use for the operator.
    ns : int, optional
        The number of spins in the system.
    type_act : OperatorTypeActing, optional
        The type of acting for the operator.
    sites : list of int, optional
        The sites to apply the operator to.
    spin : bool, optional
        If True, use the spin convention.
    spin_value : float, optional
        The value used for the spin.
    backend : str, optional
        The backend to use.
    Returns
    -------
    Operator
        The σ⁺ operator.
    """
    if lattice is None:
        assert ns is not None, "Either lattice or ns must be provided."
    else:
        ns = lattice.ns
    if type_act == OperatorTypeActing.Global:
        def op(state):
            return sigma_plus(state, ns, sites, spin, spin_value, backend)
        _name = "Sp/"
        for site in sites:
            _name += f"{site}-"
        _name = _name[:-1]
        return Operator(fun = op,
                        eigval = 1.0,
                        lattice = lattice,
                        name = _name,
                        typek = SymmetryGenerators.Other, modifies=True)
    elif type_act == OperatorTypeActing.Local:
        def op(state, site):
            return sigma_plus(state, ns, [site], spin, spin_value, backend)
        return Operator(fun = op,
                        eigval = 1.0,
                        lattice = lattice,
                        name = "Sp/L",
                        typek = SymmetryGenerators.Other, modifies=True)
    elif type_act == OperatorTypeActing.Correlation:
        def op(state, site1, site2):
            return sigma_plus(state, ns, [site1, site2], spin, spin_value, backend)
        return Operator(fun = op,
                        eigval = 1.0,
                        lattice = lattice,
                        name = "Sp/C",
                        typek = SymmetryGenerators.Other, modifies=True)
    else:
        raise ValueError("Invalid OperatorTypeActing")

# -----------------------------------------------------------------------------
# Factory function for sigma-minus (σ⁻)
# -----------------------------------------------------------------------------

def sig_m( lattice   : Optional[Lattice]     = None,
               ns        : Optional[int]         = None,
               type_act  : OperatorTypeActing    = OperatorTypeActing.Global,
               sites     : Optional[List[int]]   = [0],
               spin      : bool                  = _BACKEND_DEF_SPIN,
               spin_value: float                 = _SPIN,
               backend   : str                   = DEFAULT_BACKEND) -> Operator:
    """
    Factory function for σ⁻.
    Parameters
    ----------
    lattice : Lattice, optional
        The lattice to use for the operator.
    ns : int, optional
        The number of spins in the system.
    type_act : OperatorTypeActing, optional
        The type of acting for the operator.
    sites : list of int, optional
        The sites to apply the operator to.
    spin : bool, optional
        If True, use the spin convention.
    spin_value : float, optional
        The value used for the spin.
    backend : str, optional
        The backend to use.
    Returns
    -------
    Operator
        The σ⁻ operator.
    """
    if lattice is None:
        assert ns is not None, "Either lattice or ns must be provided."
    else:
        ns = lattice.ns
    if type_act == OperatorTypeActing.Global:
        def op(state):
            return sigma_minus(state, ns, sites, spin, spin_value, backend)
        _name = "Sm/"
        for site in sites:
            _name += f"{site}-"
        _name = _name[:-1]
        return Operator(fun = op,
                        eigval = 1.0,
                        lattice = lattice,
                        name = _name,
                        typek = SymmetryGenerators.Other, modifies=True)
    elif type_act == OperatorTypeActing.Local:
        def op(state, site):
            return sigma_minus(state, ns, [site], spin, spin_value, backend)
        return Operator(fun = op,
                        eigval = 1.0,
                        lattice = lattice,
                        name = "Sm/L",
                        typek = SymmetryGenerators.Other, modifies=True)
    elif type_act == OperatorTypeActing.Correlation:
        def op(state, site1, site2):
            return sigma_minus(state, ns, [site1, site2], spin, spin_value, backend)
        return Operator(fun = op,
                        eigval = 1.0,
                        lattice = lattice,
                        name = "Sm/C",
                        typek = SymmetryGenerators.Other, modifies=True)
    else:
        raise ValueError("Invalid OperatorTypeActing")

# -----------------------------------------------------------------------------
# Factory function for sigma-k (σₖ)
# -----------------------------------------------------------------------------

def sig_k( lattice     : Optional[Lattice]     = None,
           ns          : Optional[int]         = None,
           type_act    : OperatorTypeActing    = OperatorTypeActing.Global,
           sites       : Optional[List[int]]   = None,
           k           : float                 = 0.0,
           spin        : bool                  = _BACKEND_DEF_SPIN,
           spin_value  : float                 = _SPIN,
           backend     : str                   = DEFAULT_BACKEND) -> Operator:
    """
    Factory function for the Fourier-transformed spin operator σₖ.
    Parameters
    ----------
    lattice : Lattice, optional
        The lattice to use for the operator.
    ns : int, optional
        The number of spins in the system.
    type_act : OperatorTypeActing, optional
        The type of acting for the operator.
    sites : list of int, optional
        The sites to apply the operator to. If None, apply to all sites.
    k : float
        The momentum parameter.
    spin : bool, optional
        If True, use the spin convention.
    spin_value : float, optional
        The value used for the spin.
    backend : str, optional
        The backend to use.
    Returns
    -------
    Operator
        The σₖ operator.
    """
    if lattice is None:
        assert ns is not None, "Either lattice or ns must be provided."
    else:
        ns = lattice.ns
    if sites is None:
        sites = list(range(ns))
    if type_act == OperatorTypeActing.Global:
        def op(state):
            return sigma_k(state, ns, sites, k, spin, spin_value, backend)
        _name = "sigk_"
        _name += f"{round(k * ns / (2 * math.pi))}"
        return Operator(fun = op,
                        eigval = 1.0,
                        lattice = lattice,
                        name = _name,
                        typek = SymmetryGenerators.Other, modifies=False)
    elif type_act == OperatorTypeActing.Local:
        def op(state, site):
            return sigma_k(state, ns, [site], k, spin, spin_value, backend)
        return Operator(fun = op,
                        eigval = 1.0,
                        lattice = lattice,
                        name = "sigk/L",
                        typek = SymmetryGenerators.Other, modifies=False)
    elif type_act == OperatorTypeActing.Correlation:
        def op(state, site1, site2):
            return sigma_k(state, ns, [site1, site2], k, spin, spin_value, backend)
        return Operator(fun = op,
                        eigval = 1.0,
                        lattice = lattice,
                        name = "sigk/C",
                        typek = SymmetryGenerators.Other, modifies=False)
    else:
        raise ValueError("Invalid OperatorTypeActing")
    
# -----------------------------------------------------------------------------
# Finalize
# -----------------------------------------------------------------------------

class SpinOperatorTests(GeneralAlgebraicTest):
    """
    A class that implements tests for the spin operators.
    """

    def __init__(self, lattice: Optional[Lattice] = None, ns: Optional[int] = 4, backend='default'):
        super().__init__(backend)
        self.test_count = 1
        if lattice is not None:
            self.lattice = lattice
            self.ns = lattice.ns
        elif ns is not None:
            self.lattice = None
            self.ns = ns

    def change_backend(self, backend: str):
        """ Change the backend. """
        if isinstance(backend, str):
            self.backendstr = backend
            self.backend = __backend(backend)
        else:
            self.backend = backend
            self.backendstr = 'np' if backend == np else 'jnp'
        raise NotImplementedError("Changing backend not implemented for SpinOperatorTests")
    
    # -------------------------------
    
    def test_sig_x_global_int(self, state = 0b0101, ns = 4, sites = 2):
        """
        Test Global σₓ on an integer state.
        For ns=4 and state=0b0101, applying σₓ at site 2 (i.e. flip bit at pos=1)
        should change the state from 0b0101 to 0b0111.
        """
        # state: 0b0101 (bits: [0,1,0,1])
        if not isinstance(state, int):
            state = base2int(state, spin = _BACKEND_DEF_SPIN, spin_value = _SPIN)
        elif state is None:
            state = 0b0101
            ns    = 4
            sites = [2]
            
        if isinstance(sites, int):
            sites = [sites]
        
        # For site 2: pos = ns - 1 - 2 = 1. Flipping bit1: 0b0101 -> 0b0111 (decimal 7)
        op      = sig_x(lattice = self.lattice,
                    ns          = ns,
                    type_act    = OperatorTypeActing.Global,
                    sites       = sites,
                    spin        = _BACKEND_DEF_SPIN,
                    spin_value  = _SPIN,
                    backend     = self.backend)
        
        new_state, coeff = op(state)
        self.__log(f"sig_x_global: {state:04b} -> {new_state:04b}, coeff: {coeff}", self.test_count, color="cyan")
        
        if state == 0b0101:
            expected_state = 7
            expected_coeff = -_SPIN
            assert new_state == expected_state, f"Expected {expected_state}, got {new_state}"
            assert coeff == expected_coeff, f"Expected {expected_coeff}, got {coeff}"
        self.test_count += 1

    # -------------------------------
    
    def test_sig_x_global_np(self, state: Optional[int] = 0b0101, ns: Optional[int] = 4, sites: Union[int, List[int]] = 2):
        """
        Test Global σₓ on a NumPy array state.
        Convert an integer state to a binary vector, apply the operator,
        then convert back to an integer.
        """
        if isinstance(sites, int):
            sites = [sites]
        # Convert integer state to a binary vector (using spin convention)
        if isinstance(state, int):
            base_state = int2base(state, ns, spin=True, spin_value=_SPIN)
        else:
            base_state = state
            
        op = sig_x(lattice       = self.lattice,
                    ns           = ns,
                    type_act     = OperatorTypeActing.Global,
                    sites        = sites,
                    spin         = _BACKEND_DEF_SPIN,
                    spin_value   = _SPIN,
                    backend      = self.backend)
        new_base, coeff = op(base_state)
        self._log(f"sig_x_global_np: {base_state} -> {new_base}, coeff: {coeff}", self.test_count, color="cyan")
        
        if base_state == [0, 1, 0, 1]:
            expected_state = [0, 1, 1, 1]
            expected_coeff = -_SPIN
            assert new_base == expected_state, f"Expected {expected_state}, got {new_base}"
            assert coeff == expected_coeff, f"Expected {expected_coeff}, got {coeff}"
        self.test_count += 1

    # -------------------------------
    def test_sig_y_global_int(self, state: Optional[int] = 0b0101, ns: Optional[int] = 4, sites: Union[int, List[int]] = 2):
        """
        Test Global σᵧ on an integer state.
        For ns=4 and state=0b0101, applying σᵧ at site 2 (flip bit at pos=1)
        should change the state and yield a coefficient of -I * _SPIN.
        """
        if isinstance(sites, int):
            sites = [sites]
        op = sig_y(lattice=self.lattice,
                   ns=ns,
                   type_act=OperatorTypeActing.Global,
                   sites=sites,
                   spin=_BACKEND_DEF_SPIN,
                   spin_value=_SPIN,
                   backend=self.backend)
        new_state, coeff = op(state)
        self.__log(f"sig_y_global_int: {state:04b} -> {new_state:04b}, coeff: {coeff}",
                   self.test_count, color="cyan")
        if state == 0b0101:
            expected_state = 7
            expected_coeff = -1j * _SPIN
            assert new_state == expected_state, f"Expected {expected_state}, got {new_state}"
            assert coeff == expected_coeff, f"Expected coeff {expected_coeff}, got {coeff}"
        self.test_count += 1

    # -------------------------------
    def test_sig_y_global_np(self, state: Optional[int] = 0b0101, ns: Optional[int] = 4, sites: Union[int, List[int]] = 2):
        """
        Test Global σᵧ on a NumPy array state.
        Convert an integer state to a binary array, apply σᵧ, then convert back.
        For state [0,1,0,1] and site 2, expected new base is [0,1,1,1] and coefficient is -1j*_SPIN.
        """
        if isinstance(sites, int):
            sites = [sites]
        if isinstance(state, int):
            base_state = int2base(state, ns, spin=True, spin_value=_SPIN)
        else:
            base_state = state
        op = sig_y(lattice=self.lattice,
                   ns=ns,
                   type_act=OperatorTypeActing.Global,
                   sites=sites,
                   spin=_BACKEND_DEF_SPIN,
                   spin_value=_SPIN,
                   backend=self.backend)
        new_base, coeff = op(base_state)
        new_state = base2int(new_base, spin=True, spin_value=_SPIN)
        self._log(f"sig_y_global_np: {base_state} -> {new_base}, coeff: {coeff}",
                  self.test_count, color="cyan")
        if base_state == [0, 1, 0, 1]:
            expected_state = [0, 1, 1, 1]
            expected_coeff = -1j * _SPIN
            assert new_base == expected_state, f"Expected {expected_state}, got {new_base}"
            assert coeff == expected_coeff, f"Expected coeff {expected_coeff}, got {coeff}"
        self.test_count += 1

    # -------------------------------
    def test_sig_z_global_int(self, state: Optional[int] = 0b0101, ns: Optional[int] = 4, sites: Union[int, List[int]] = 2):
        """
        Test Global σ_z on an integer state.
        For ns=4 and state=0b0101, applying σ_z at site 2 does not change the state.
        For state 0b0101, at pos=ns-1-2 the bit is 0 so the coefficient is -_SPIN.
        """
        if isinstance(sites, int):
            sites = [sites]
        op = sig_z(lattice=self.lattice,
                   ns=ns,
                   type_act=OperatorTypeActing.Global,
                   sites=sites,
                   spin=_BACKEND_DEF_SPIN,
                   spin_value=_SPIN)
        new_state, coeff = op(state)
        self.__log(f"sig_z_global_int: {state:04b} -> {new_state:04b}, coeff: {coeff}",
                   self.test_count, color="cyan")
        if state == 0b0101:
            expected_state = state
            expected_coeff = -_SPIN
            assert new_state == expected_state, f"Expected state {expected_state}, got {new_state}"
            assert coeff == expected_coeff, f"Expected coeff {expected_coeff}, got {coeff}"
        self.test_count += 1

    # -------------------------------
    def test_sig_z_global_np(self, state: Optional[int] = 0b0101, ns: Optional[int] = 4, sites: Union[int, List[int]] = 2):
        """
        Test Global σ_z on a NumPy array state.
        Convert an integer state to a binary array, apply σ_z (which should not flip bits),
        and then convert back.
        For state [0,1,0,1] at site 2, expected new base remains [0,1,0,1] and coefficient is -_SPIN.
        """
        if isinstance(sites, int):
            sites = [sites]
        if isinstance(state, int):
            base_state = int2base(state, ns, spin=True, spin_value=_SPIN)
        else:
            base_state = state
        op = sig_z(lattice=self.lattice,
                   ns=ns,
                   type_act=OperatorTypeActing.Global,
                   sites=sites,
                   spin=_BACKEND_DEF_SPIN,
                   spin_value=_SPIN)
        new_base, coeff = op(base_state)
        self._log(f"sig_z_global_np: {base_state} -> {new_base}, coeff: {coeff}",
                  self.test_count, color="cyan")
        if base_state == [0, 1, 0, 1]:
            expected_state = [0, 1, 0, 1]
            expected_coeff = -_SPIN
            assert new_base == expected_state, f"Expected {expected_state}, got {new_base}"
            assert coeff == expected_coeff, f"Expected coeff {expected_coeff}, got {coeff}"
        self.test_count += 1

    # -------------------------------
    def test_sig_plus_global_int(self, state: Optional[int] = 0b0101, ns: Optional[int] = 4, sites: Union[int, List[int]] = 2):
        """
        Test Global σ⁺ on an integer state.
        For ns=4 and state=0b0101, applying σ⁺ at site 2 should flip the bit if it is 0.
        For state 0b0101, flipping bit at pos=ns-1-2 yields 0b0111 (decimal 7) with coefficient _SPIN.
        """
        if isinstance(sites, int):
            sites = [sites]
        op = sig_plus(lattice=self.lattice,
                      ns=ns,
                      type_act=OperatorTypeActing.Global,
                      sites=sites,
                      spin=_BACKEND_DEF_SPIN,
                      spin_value=_SPIN,
                      backend=self.backend)
        new_state, coeff = op(state)
        self.__log(f"sig_plus_global_int: {state:04b} -> {new_state:04b}, coeff: {coeff}",
                   self.test_count, color="cyan")
        if state == 0b0101:
            expected_state = 7
            expected_coeff = _SPIN
            assert new_state == expected_state, f"Expected {expected_state}, got {new_state}"
            assert coeff == expected_coeff, f"Expected coeff {expected_coeff}, got {coeff}"
        self.test_count += 1

    # -------------------------------
    def test_sig_plus_global_np(self, state: Optional[int] = 0b0101, ns: Optional[int] = 4, sites: Union[int, List[int]] = 2):
        """
        Test Global σ⁺ on a NumPy array state.
        Convert an integer state to a binary array, apply σ⁺, then convert back.
        For state [0,1,0,1] and site 2, expected new base is [0,1,1,1] and coefficient _SPIN.
        """
        if isinstance(sites, int):
            sites = [sites]
        if isinstance(state, int):
            base_state = int2base(state, ns, spin=True, spin_value=_SPIN)
        else:
            base_state = state
        op = sig_plus(lattice=self.lattice,
                      ns=ns,
                      type_act=OperatorTypeActing.Global,
                      sites=sites,
                      spin=_BACKEND_DEF_SPIN,
                      spin_value=_SPIN,
                      backend=self.backend)
        new_base, coeff = op(base_state)
        self._log(f"sig_plus_global_np: {base_state} -> {new_base}, coeff: {coeff}",
                  self.test_count, color="cyan")
        if base_state == [0, 1, 0, 1]:
            expected_state = [0, 1, 1, 1]
            expected_coeff = _SPIN
            assert new_base == expected_state, f"Expected {expected_state}, got {new_base}"
            assert coeff == expected_coeff, f"Expected coeff {expected_coeff}, got {coeff}"
        self.test_count += 1

    # -------------------------------
    def test_sig_minus_global_int(self, state: Optional[int] = 0b0111, ns: Optional[int] = 4, sites: Union[int, List[int]] = 2):
        """
        Test Global σ⁻ on an integer state.
        For ns=4 and state=0b0111, applying σ⁻ at site 2 should flip the bit if it is 1.
        For state 0b0111, flipping bit at pos=ns-1-2 yields 0b0101 (decimal 5) with coefficient _SPIN.
        """
        if isinstance(sites, int):
            sites = [sites]
        op = sig_minus(lattice=self.lattice,
                       ns=ns,
                       type_act=OperatorTypeActing.Global,
                       sites=sites,
                       spin=_BACKEND_DEF_SPIN,
                       spin_value=_SPIN,
                       backend=self.backend)
        new_state, coeff = op(state)
        self.__log(f"sig_minus_global_int: {state:04b} -> {new_state:04b}, coeff: {coeff}",
                   self.test_count, color="cyan")
        if state == 0b0111:
            expected_state = 5
            expected_coeff = _SPIN
            assert new_state == expected_state, f"Expected {expected_state}, got {new_state}"
        self.test_count += 1

    # -------------------------------
    def test_sig_minus_global_np(self, state: Optional[int] = 0b0111, ns: Optional[int] = 4, sites: Union[int, List[int]] = 2):
        """
        Test Global σ⁻ on a NumPy array state.
        Convert an integer state to a binary array, apply σ⁻, then convert back.
        For state [0,1,1,1] and site 2, expected new base is [0,1,0,1] and coefficient _SPIN.
        """
        if isinstance(sites, int):
            sites = [sites]
        if isinstance(state, int):
            base_state = int2base(state, ns, spin=True, spin_value=_SPIN)
        else:
            base_state = state
        op = sig_minus(lattice=self.lattice,
                       ns=ns,
                       type_act=OperatorTypeActing.Global,
                       sites=sites,
                       spin=_BACKEND_DEF_SPIN,
                       spin_value=_SPIN,
                       backend=self.backend)
        new_base, coeff = op(base_state)
        self._log(f"sig_minus_global_np: {base_state} -> {new_base}, coeff: {coeff}",
                  self.test_count, color="cyan")
        if base_state == [0, 1, 1, 1]:
            expected_state = [0, 1, 0, 1]
            expected_coeff = _SPIN
            assert new_base == expected_state, f"Expected {expected_state}, got {new_base}"
            assert coeff == expected_coeff, f"Expected coeff {expected_coeff}, got {coeff}"
        self.test_count += 1

    # -------------------------------
    def test_sig_k_global_int(self, state: Optional[int] = 0b0101, ns: Optional[int] = 4, sites: Optional[List[int]] = None, k: float = 0.0):
        """
        Test Global σₖ on an integer state.
        For ns=4 and state=0b0101 over sites [0,1,2,3] with k=0,
        the operator does not change the state but produces a coefficient.
        For state 0b0101, expected coefficient (for k=0) is (1j - 1.0) (normalized by sqrt(4)=2).
        """
        if sites is None:
            sites = list(range(ns))
        op = sig_k(lattice=self.lattice,
                   ns=ns,
                   type_act=OperatorTypeActing.Global,
                   sites=sites,
                   k=k,
                   spin=_BACKEND_DEF_SPIN,
                   spin_value=_SPIN,
                   backend=self.backend)
        new_state, coeff = op(state)
        self.__log(f"sig_k_global_int: {state:04b} -> {new_state:04b}, coeff: {coeff}",
                   self.test_count, color="cyan")
        if state == 0b0101:
            expected_state = state
            expected_coeff = 1j - 1.0  # For k=0, normalized by sqrt(4)=2; adjust as needed.
            assert new_state == expected_state, f"Expected state {expected_state}, got {new_state}"
            assert abs(coeff - expected_coeff) < 1e-6, f"Expected coeff {expected_coeff}, got {coeff}"
        self.test_count += 1

    # -------------------------------
    def test_sig_k_global_np(self, state: Optional[int] = 0b0101, ns: Optional[int] = 4, sites: Optional[List[int]] = None, k: float = 0.0):
        """
        Test Global σₖ on a NumPy array state.
        Convert an integer state to a binary array, apply σₖ, then convert back.
        For state [0,1,0,1] over sites [0,1,2,3] with k=0, the coefficient should be (1j - 1.0).
        """
        if sites is None:
            sites = list(range(ns))
        if isinstance(state, int):
            base_state = int2base(state, ns, spin=True, spin_value=_SPIN)
        else:
            base_state = state
        op = sig_k(lattice=self.lattice,
                   ns=ns,
                   type_act=OperatorTypeActing.Global,
                   sites=sites,
                   k=k,
                   spin=_BACKEND_DEF_SPIN,
                   spin_value=_SPIN,
                   backend=self.backend)
        new_base, coeff = op(base_state)
        self._log(f"sig_k_global_np: {base_state} -> {new_base}, coeff: {coeff}",
                  self.test_count, color="cyan")
        if base_state == [0, 1, 0, 1]:
            expected_state = base_state
            expected_coeff = 1j - 1.0  # adjust as needed
            assert new_base == expected_state, f"Expected {expected_state}, got {new_base}"
            assert abs(coeff - expected_coeff) < 1e-6, f"Expected coeff {expected_coeff}, got {coeff}"
        self.test_count += 1

    # -------------------------------
    def add_tests(self):
        """
        Adds all spin operator tests.
        """
        self.tests.append(self.test_sig_x_global_int)
        self.tests.append(self.test_sig_x_global_np)
        self.tests.append(self.test_sig_y_global_int)
        self.tests.append(self.test_sig_y_global_np)
        self.tests.append(self.test_sig_z_global_int)
        self.tests.append(self.test_sig_z_global_np)
        self.tests.append(self.test_sig_plus_global_int)
        self.tests.append(self.test_sig_plus_global_np)
        self.tests.append(self.test_sig_minus_global_int)
        self.tests.append(self.test_sig_minus_global_np)
        self.tests.append(self.test_sig_k_global_int)
        self.tests.append(self.test_sig_k_global_np)

# -----------------------------------------------------------------------------