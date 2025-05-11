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
import numba
from typing import List, Union, Optional, Callable
from functools import partial

################################################################################
from Algebra.Operator.operator import (
    Operator, OperatorTypeActing, SymmetryGenerators, 
    create_operator, ensure_operator_output_shape_numba
)
################################################################################
from general_python.common.tests import GeneralAlgebraicTest
from general_python.lattices.lattice import Lattice
from general_python.algebra.utils import DEFAULT_BACKEND, get_backend, maybe_jit
from general_python.algebra.utils import DEFAULT_NP_INT_TYPE, DEFAULT_NP_FLOAT_TYPE, DEFAULT_NP_CPX_TYPE
from general_python.common.binary import BACKEND_REPR as _SPIN, BACKEND_DEF_SPIN, JAX_AVAILABLE
import general_python.common.binary as _binary
from general_python.common.binary import (
    flip, flip_all, check, base2int, int2base, int2binstr
)
################################################################################

if JAX_AVAILABLE:
    import jax
    import Algebra.Operator.operators_spin_jax as jaxpy
    import jax.numpy as jnp
    # sigma x
    from Algebra.Operator.operators_spin_jax import sigma_x_int_jnp, sigma_x_jnp, sigma_x_int_static_jnp, sigma_x_inv_static_jnp
    # sigma y
    from Algebra.Operator.operators_spin_jax import sigma_y_int_jnp, sigma_y_jnp
    # sigma z
    from Algebra.Operator.operators_spin_jax import sigma_z_int_jnp, sigma_z_jnp
    # sigma plus
    from Algebra.Operator.operators_spin_jax import sigma_plus_int_jnp, sigma_plus_jnp
    # sigma minus
    from Algebra.Operator.operators_spin_jax import sigma_minus_int_jnp, sigma_minus_jnp
    # sigma pm
    from Algebra.Operator.operators_spin_jax import sigma_pm_int_jnp, sigma_pm_jnp
    # sigma mp
    from Algebra.Operator.operators_spin_jax import sigma_mp_int_jnp, sigma_mp_jnp
    # sigma k
    from Algebra.Operator.operators_spin_jax import sigma_k_int_jnp, sigma_k_jnp
else:
    sigma_x_int_jnp     = sigma_x_jnp       = None
    sigma_y_int_jnp     = sigma_y_jnp       = None
    sigma_z_int_jnp     = sigma_z_jnp       = None
    sigma_plus_int_jnp  = sigma_plus_jnp    = None
    sigma_minus_int_jnp = sigma_minus_jnp   = None
    sigma_pm_int_jnp    = sigma_pm_jnp      = None
    sigma_mp_int_jnp    = sigma_mp_jnp      = None
    sigma_k_int_jnp     = sigma_k_jnp       = None

################################################################################
#! Standard Pauli matrices
################################################################################

# Define the Pauli matrices for reference
_SIG_0 = np.array([[1, 0],
                [0, 1]], dtype=float)
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
#! Sigma-X (σₓ) operator
# -----------------------------------------------------------------------------

@numba.njit
def sigma_x_int_np(state, ns, sites, spin: bool = BACKEND_DEF_SPIN, spin_value=_SPIN):
    """
    Apply the Pauli-X (σₓ) operator on the given sites.
    For each site, flip the bit at position (ns-1-site) using binary.flip.
    """
    out_state       = np.empty(1, dtype=DEFAULT_NP_INT_TYPE)
    out_coeff       = np.empty(1, dtype=DEFAULT_NP_FLOAT_TYPE)
    coeff           = 1.0
    for site in sites:
        pos     = ns - 1 - site
        state   = _binary.flip_int(state, pos)
        coeff   *= spin_value
    out_state[0]    = state
    out_coeff[0]    = coeff
    # return ensure_operator_output_shape_numba(out_state, out_coeff)
    return out_state, out_coeff

def sigma_x_int(state  : int,
            ns          : int,
            sites       : Union[List[int], None],
            spin_value  : float     = _SPIN):
    """
    Apply the Pauli-X (σₓ) operator on the given sites.
    For each site, flip the bit at position (ns-1-site) using binary.flip.
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
    backend : str, optional
        The backend to use for the computation.
    Returns
    -------
    """
    if not isinstance(state, (int, np.integer)):
        return sigma_x_int_jnp(state, ns, sites, spin_value)
    return sigma_x_int_np(state, ns, sites, spin_value)

@numba.njit
def sigma_x_np(state    : np.ndarray,
            sites       : Union[List[int], None],
            spin        : bool  = BACKEND_DEF_SPIN,
            spin_value  : float = _SPIN):
    """
    Apply the Pauli-X (σₓ) operator on the given sites.
    For each site, flip the bit at position (ns-1-site) using binary.flip.
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
        The state after applying the operator.
    """
    coeff   = np.ones(1, dtype=DEFAULT_NP_FLOAT_TYPE)
    out     = state.copy()
    for site in sites:
        out     = _binary.flip_array_np_nspin(out, site) if not spin else _binary.flip_array_np_spin(out, site)
        coeff  *= spin_value
    return ensure_operator_output_shape_numba(out, coeff)
    # return out, coeff

def sigma_x(state,
            ns          : int,
            sites       : Union[List[int], None],
            spin        : bool      = BACKEND_DEF_SPIN,
            spin_value  : float     = _SPIN):
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
    
    if isinstance(state, (int, np.integer)):
        return sigma_x_int_np(state, ns, sites, spin_value)
    elif isinstance(state, np.ndarray):
        return sigma_x_np(state, sites, spin, spin_value)
    return sigma_x_jnp(state, ns, sites, spin, spin_value)

# -----------------------------------------------------------------------------
#! Sigma-Y (σᵧ) operator
# -----------------------------------------------------------------------------

@numba.njit
def sigma_y_int_np_real(state, 
                        ns, 
                        sites,
                        spin        : bool  = BACKEND_DEF_SPIN,
                        spin_value  : float = _SPIN,
                        ):
    """
    σᵧ on an integer state.
    For each site, if the bit at (ns-1-site) is set then multiply coefficient by I*spin_value,
    otherwise by -I*spin_value; then flip the bit.
    """
    out_state       = np.empty(1, dtype=DEFAULT_NP_INT_TYPE)
    out_coeff       = np.empty(1, dtype=DEFAULT_NP_FLOAT_TYPE)
    coeff           = 1.0 + 0j
    for site in sites:
        pos     = ns - 1 - site
        bit     = _binary.check_int(state, pos)
        coeff   *= (2 * bit - 1.0) * 1.0j * spin_value
        state   = _binary.flip_int(state, pos)
        
    # Create output arrays
    out_state[0] = state
    out_coeff[0] = coeff.real
    # return ensure_operator_output_shape_numba(out_state, out_coeff)
    return out_state, out_coeff

@numba.njit
def sigma_y_int_np(state, ns, sites, spin: bool = BACKEND_DEF_SPIN, spin_value=_SPIN):
    """
    σᵧ on an integer state.
    For each site, if the bit at (ns-1-site) is set then multiply coefficient by I*spin_value,
    otherwise by -I*spin_value; then flip the bit.
    """
    out_state       = np.empty(1, dtype=DEFAULT_NP_INT_TYPE)
    out_coeff       = np.empty(1, dtype=DEFAULT_NP_CPX_TYPE)
    coeff           = 1.0 + 0j
    for site in sites:
        pos     = ns - 1 - site
        bit     = _binary.check_int(state, pos)
        coeff   *= (2 * bit - 1.0) * 1.0j * spin_value
        state   = _binary.flip_int(state, pos)
        
    # Create output arrays
    out_state[0] = state
    out_coeff[0] = coeff
    # return ensure_operator_output_shape_numba(out_state, out_coeff)
    return out_state, out_coeff

def sigma_y_int(state       : int,
                ns          : int,
                sites       : Union[List[int], None],
                spin_value  : float = _SPIN):
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
    if not isinstance(state, (int, np.integer)):
        return sigma_y_int_jnp(state, ns, sites, spin_value)
    if len(sites) % 2 == 1:
        return sigma_y_int_np(state, ns, sites, spin_value)
    return sigma_y_int_np_real(state, ns, sites, spin_value)

@numba.njit
def sigma_y_np_real(state       : np.ndarray,
                    sites       : Union[List[int], None],
                    spin        : bool  = BACKEND_DEF_SPIN,
                    spin_value  : float = _SPIN):
    """
    σᵧ on a NumPy array state.
    For each site, use the given site as index.
    Parameters
    ----------
    state : np.ndarray
        The state to apply the operator to.
    sites : list of int or None
        The sites to apply the operator to. If None, apply to all sites.
    spin_value : float, optional
        The value to multiply the state by when flipping the bits.
    Returns
    -------
    np.ndarray
        The state after applying the operator    
    """
    coeff = 1.0 + 0j
    out   = state.copy()
    for site in sites:
        # For NumPy arrays, we use the site index directly.
        factor  =  (2.0 * _binary.check_arr_np(state, site) - 1.0) * 1.0j * spin_value
        coeff  *=  factor
        out     =  _binary.flip_array_np_nspin(out, site) if not spin else _binary.flip_array_np_spin(out, site)
    return ensure_operator_output_shape_numba(out, coeff.real)
    # return out, coeff.real

@numba.njit
def sigma_y_np(state        : np.ndarray,
                sites       : Union[List[int], None],
                spin        : bool  = BACKEND_DEF_SPIN,
                spin_value  : float = _SPIN):
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
    coeff   = np.ones(1, dtype=DEFAULT_NP_CPX_TYPE)
    out     = state.copy()
    for site in sites:
        bit     =   _binary.check_arr_np(state, site)
        factor  =   (2.0 * bit - 1.0) * 1.0j * spin_value
        coeff  *=   factor
        out     =   _binary.flip_array_np_nspin(out, site) if not spin else _binary.flip_array_np_spin(out, site)
    return ensure_operator_output_shape_numba(out, coeff)

def sigma_y(state,
            ns              : int,
            sites           : Union[List[int], None],
            spin            : bool  = BACKEND_DEF_SPIN,
            spin_value      : float = _SPIN):
    """
    Dispatch for σᵧ.
    """
    if sites is None:
        sites = list(range(ns))
    if isinstance(state, (int, np.integer)):
        return sigma_y_int_np(state, ns, sites, spin_value)
    elif isinstance(state, np.ndarray):
        return sigma_y_np(state, sites, spin, spin_value)
    else:
        return sigma_y_jnp(state, sites, spin_value)

# -----------------------------------------------------------------------------
#! Sigma-Z (σ_z) operator
# -----------------------------------------------------------------------------

@numba.njit
def sigma_z_int_np(state     : int,
                ns           : int,
                sites        : Union[List[int], None],
                spin         : bool = BACKEND_DEF_SPIN,
                spin_value   : bool = _SPIN):
    """
    σ_z on an integer state.
    For each site, if the bit at (ns-1-site) is set then multiply by spin_value; else by -spin_value.
    The state is unchanged.
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
    float
        The coefficient after applying the operator.
    """
    out_state       = np.empty(1, dtype=DEFAULT_NP_INT_TYPE)
    out_coeff       = np.empty(1, dtype=DEFAULT_NP_FLOAT_TYPE)
    out_state[0]    = state
    coeff           = 1.0
    for site in sites:
        pos         =  ns - 1 - site
        bit         =  _binary.check_int(state, pos)
        coeff      *=  (2.0 * bit - 1.0) * spin_value
    out_coeff[0]    = coeff
    # return ensure_operator_output_shape_numba(out_state, out_coeff)
    return out_state, out_coeff

def sigma_z_int(state       : int,
                ns          : int,
                sites       : Union[List[int], None],
                spin_value  : float = _SPIN):
    """
    σ_z on an integer state.
    For each site, if the bit at (ns-1-site) is set then multiply by spin_value; else by -spin_value.
    The state is unchanged.
    """
    if not isinstance(state, (int, np.integer)):
        return sigma_z_int_jnp(state, ns, sites, spin_value)
    return sigma_z_int_np(state, ns, sites, spin_value)

@numba.njit
def sigma_z_np(state        : np.ndarray,
                sites       : Union[List[int], None],
                spin        : bool  = BACKEND_DEF_SPIN,
                spin_value  : float = _SPIN):
    """
    σ_z on a NumPy array state.
    Parameters
    ----------
    state : np.ndarray
        The state to apply the operator to.
    sites : list of int or None
        The sites to apply the operator to. If None, apply to all sites.
    spin_value : float, optional
        The value to multiply the state by when flipping the bits.
    Returns
    -------
    np.ndarray
        The state after applying the operator.
    float
        The coefficient after applying the operator.
    """
    coeff   = np.ones(1, dtype=DEFAULT_NP_FLOAT_TYPE)
    for site in sites:
        bit     = _binary.check_arr_np(state, site)
        coeff  *= (2 * bit - 1.0) * spin_value
    return ensure_operator_output_shape_numba(state, coeff)
    # return state, coeff

def sigma_z(state,
            ns          : int,
            sites       : Union[List[int], None],
            spin        : bool  = BACKEND_DEF_SPIN,
            spin_value  : float = _SPIN):
    """
    Dispatch for σ_z.
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
    Returns
    -------
    int or np.ndarray
        The state after applying the operator.
    float
        The coefficient after applying the operator.
    """
    if sites is None:
        sites = list(range(ns))
    if isinstance(state, (int, np.integer, jnp.integer)):
        return sigma_z_int_np(state, ns, sites, spin, spin_value)
    elif isinstance(state, np.ndarray):
        return sigma_z_np(state, sites, spin, spin_value)
    return sigma_z_jnp(state, ns, sites, spin, spin_value)

# -----------------------------------------------------------------------------
#! Sigma-Plus (σ⁺) operator
# -----------------------------------------------------------------------------

@numba.njit
def sigma_plus_int_np(state         : int,
                    ns              : int,
                    sites           : List[int],
                    spin            : bool  = BACKEND_DEF_SPIN,
                    spin_value      : float = _SPIN):
    """
    σ⁺ |state⟩  on an *integer* spin string.
    Returns (len -1 array new_state, len -1 array coeff).  Zero coeff ⇒ annihilation.
    """
    new_state = state
    coeff     = 1.0
    for site in sites:
        pos = ns - 1 - site
        if _binary.check_int(new_state, pos):
            coeff = 0.0
            break
        new_state = _binary.flip_int(new_state, pos)
        coeff *= spin_value

    out_state            = np.empty(1, dtype=DEFAULT_NP_INT_TYPE)
    out_coeff            = np.empty(1, dtype=DEFAULT_NP_FLOAT_TYPE)
    out_state[0]         = new_state
    out_coeff[0]         = coeff
    # return ensure_operator_output_shape_numba(out_state, out_coeff)
    return out_state, out_coeff

@numba.njit
def sigma_plus_np(state         : np.ndarray,
                sites           : List[int],
                spin            : bool  = BACKEND_DEF_SPIN,
                spin_value      : float = _SPIN):
    """
    σ⁺ |state⟩ on a NumPy array representation (0/1 occupation).
    """
    coeff = np.ones(1, dtype=DEFAULT_NP_FLOAT_TYPE)
    out   = state.copy()
    for site in sites:
        bit     = _binary.check_arr_np(out, site)
        if bit: # annihilation
            coeff *= 0.0
            break
        out     = _binary.flip_array_np_nspin(out, site) if not spin else _binary.flip_array_np_spin(out, site)
        coeff  *= spin_value
    return ensure_operator_output_shape_numba(out, coeff)
    # return out, coeff

def sigma_plus(state,
            ns          : int,
            sites       : Union[List[int], None],
            spin        : bool  = BACKEND_DEF_SPIN,
            spin_value  : float = _SPIN):
    if sites is None:
        sites = list(range(ns))
    if isinstance(state, (int, np.integer)):
        return sigma_plus_int_np(int(state), ns, sites, spin, spin_value)
    if isinstance(state, np.ndarray):
        return sigma_plus_np(state, sites, spin, spin_value)
    return sigma_plus_jnp(state, ns, sites, spin, spin_value)

# -----------------------------------------------------------------------------
#! Sigma-Minus (σ⁻) operator
# -----------------------------------------------------------------------------

@numba.njit
def sigma_minus_int_np(state        : int,
                    ns              : int,
                    sites           : List[int],
                    spin            : bool  = BACKEND_DEF_SPIN,
                    spin_value      : float = _SPIN):
    """
    Applies the spin lowering (σ⁻) operator to the specified sites of a quantum spin state represented as an integer.

    Parameters:
        state (int):
            The integer representation of the quantum spin state.
        ns (int):
            The total number of sites (spins) in the system.
        sites (List[int]):
            List of site indices (0-based, leftmost is 0) where the σ⁻ operator is applied.
        spin (bool, optional):
            Indicates if the system uses spin representation. Defaults to BACKEND_DEF_SPIN.
        spin_value (float, optional):
            The value to multiply the coefficient by when lowering a spin. Defaults to _SPIN.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - out_state: A NumPy array containing the new state(s) as integer(s) after applying σ⁻.
            - out_coeff: A NumPy array containing the corresponding coefficient(s) for the new state(s).

    Notes:
        - If any of the specified sites is already in the |↓⟩ (spin-down, 0) state, the coefficient is set to 0 and the state is not changed.
        - The function assumes that the spin state is encoded in binary, with each bit representing a site.
    """

    new_state = state
    coeff     = 1.0
    for site in sites:
        pos = ns - 1 - site
        if _binary.check_int(new_state, pos) == 0: # trying to lower |↓⟩ ⇒ 0
            coeff = 0.0
            break
        new_state = _binary.flip_int(new_state, pos)  # 1 → 0
        coeff    *= spin_value

    out_state            = np.empty(1, dtype=DEFAULT_NP_INT_TYPE)
    out_coeff            = np.empty(1, dtype=DEFAULT_NP_FLOAT_TYPE)
    out_state[0]         = new_state
    out_coeff[0]         = coeff
    # return ensure_operator_output_shape_numba(out_state, out_coeff)
    return out_state, out_coeff

@numba.njit
def sigma_minus_np(state        : np.ndarray,
                sites           : List[int],
                spin            : bool  = BACKEND_DEF_SPIN,
                spin_value      : float = _SPIN):
    """
    Applies the sigma-minus (lowering) operator to the specified sites of a quantum state.

    This function operates on a numpy array representing the quantum state, applying the lowering operator
    to each site in the provided list. If the occupation at a site is 0, the operation annihilates the state
    and returns a coefficient of 0. Otherwise, the occupation is flipped from 1 to 0, and the coefficient is
    multiplied by the given spin value.

    Args:
        state (np.ndarray):
            The quantum state represented as a numpy array.
        sites (List[int]):
            List of site indices where the sigma-minus operator is applied.
        spin (bool, optional):
            Indicates whether to use spinful or spinless operations. Defaults to BACKEND_DEF_SPIN.
        spin_value (float, optional):
            The value to multiply the coefficient by for each successful lowering operation. Defaults to _SPIN.

    Returns:
        Tuple[np.ndarray, float]: The updated quantum state and the resulting coefficient after applying the operator.
    """

    coeff = np.ones(1, dtype=DEFAULT_NP_FLOAT_TYPE)
    out   = state.copy()
    for site in sites:
        bit     = _binary.check_arr_np(out, site)
        if bit == 0:
            coeff *= 0.0
            break
        out     = _binary.flip_array_np_spin(out, site) if spin else _binary.flip_array_np_nspin(out, site)
        coeff  *= spin_value
    return ensure_operator_output_shape_numba(out, coeff)
    # return out, coeff

def sigma_minus(state,
                ns          : int,
                sites       : Union[List[int], None],
                spin        : bool  = BACKEND_DEF_SPIN,
                spin_value  : float = _SPIN):
    if sites is None:
        sites = list(range(ns))
    if isinstance(state, (int, np.integer)):
        return sigma_minus_int_np(int(state), ns, sites, spin, spin_value)
    if isinstance(state, np.ndarray):
        return sigma_minus_np(state, sites, spin, spin_value)
    return sigma_minus_jnp(state, ns, sites, spin, spin_value)

# -----------------------------------------------------------------------------
#! Sigma_pm (σ⁺ then σ⁻) operator and Sigma_mp (σ⁻ then σ⁺) operator
# -----------------------------------------------------------------------------

@numba.njit
def _sigma_pm_int_core(state        : int,
                        ns          : int,
                        sites       : List[int],
                        start_up    : bool,
                        spin        : bool  = BACKEND_DEF_SPIN,
                        spin_val    : float = _SPIN):
    """
    Core for σ_pm / σ_mp alternating flips on integer states.
    start_up = True  → even  = σ⁺ , odd = σ⁻
    start_up = False → even  = σ⁻ , odd = σ⁺
    """
    new_state   = state
    coeff       = 1.0
    for i, site in enumerate(sites):
        pos     = ns - 1 - site
        bit     = _binary.check_int(new_state, pos)
        need_up = (i % 2 == 0) == start_up

        if need_up: # σ⁺
            if bit == 1:
                coeff *= 0.0
                break
            new_state = _binary.flip_int(new_state, pos)
        else: # σ⁻
            if bit == 0:
                coeff *= 0.0
                break
            new_state = _binary.flip_int(new_state, pos)
        coeff *= spin_val
    return new_state, coeff

@numba.njit
def sigma_pm_int_np(state       : int,
                    ns          : int,
                    sites       : List[int],
                    spin        : bool  = BACKEND_DEF_SPIN,
                    spin_val    : float = _SPIN):
    """
    Applies the spin raising or lowering operator (σ⁺ or σ⁻) to a given quantum state represented as an integer.

    Parameters:
        state (int):
            The integer representation of the quantum state to which the operator is applied.
        ns (int):
            The number of sites (spins) in the system.
        sites (List[int]):
            The list of site indices where the operator acts.
        spin_val (float, optional):
            The value of the spin (default is _SPIN).

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - out_state: A NumPy array containing the new state(s) as integer(s) after applying the operator.
            - out_coeff: A NumPy array containing the corresponding coefficient(s) for each new state.
    """
    new_state, coeff     = _sigma_pm_int_core(state, ns, sites, True, spin, spin_val)
    out_state            = np.empty((1,1), dtype=DEFAULT_NP_INT_TYPE)
    out_coeff            = np.empty(1, dtype=DEFAULT_NP_FLOAT_TYPE)
    out_state[0,0]       = new_state
    out_coeff[0]         = coeff
    # return ensure_operator_output_shape_numba(out_state, out_coeff)
    return out_state, out_coeff

@numba.njit
def _sigma_pm_np_core(state     : np.ndarray,
                    sites       : List[int],
                    start_up    : bool,
                    spin        : bool  = BACKEND_DEF_SPIN,
                    spin_val    : float = _SPIN):
    """
    Applies a sequence of spin raising or lowering operators (σ⁺ or σ⁻) to specified sites of a quantum state array.

    Parameters:
        state (np.ndarray):
            The quantum state represented as a NumPy array of bits (0 for spin-down, 1 for spin-up).
        sites (List[int]):
            List of site indices where the spin operators are to be applied.
        start_up (bool):
            If True, the sequence starts with a raising operator (σ⁺); if False, starts with lowering (σ⁻).
        spin_val (float):
            The coefficient to multiply for each successful spin flip.

    Returns:
        Tuple[np.ndarray, float]: 
            - The modified state array after applying the operators (may be unchanged if operation is invalid).
            - The resulting coefficient (0.0 if the operation is not allowed by spin selection rules, otherwise the product of spin_val for each flip).
    """
    coeff = np.ones(1, dtype=DEFAULT_NP_FLOAT_TYPE)
    out   = state.copy()
    for i, site in enumerate(sites):
        bit      = _binary.check_arr_np(out, site)
        need_up  = (i % 2 == 0) == start_up
        if need_up:
            if bit == 1:
                coeff *= 0.0
                break
        else:
            if bit == 0:
                coeff *= 0.0
                break
        out     = _binary.flip_array_np_nspin(out, site) if not spin else _binary.flip_array_np_spin(out, site)
        coeff  *= spin_val
    return ensure_operator_output_shape_numba(out, coeff)
    # return out, coeff

@numba.njit
def sigma_pm_np(state       : np.ndarray,
                sites       : List[int],
                spin        : bool  = BACKEND_DEF_SPIN,
                spin_val    : float = _SPIN):
    """σ⁺ σ⁻ alternating on NumPy array (start σ⁺)."""
    return _sigma_pm_np_core(state, sites, True, spin, spin_val)

def sigma_pm(state,
            ns          : int,
            sites       : Union[List[int], None],
            spin        : bool  = BACKEND_DEF_SPIN,
            spin_value  : float = _SPIN):
    if sites is None:
        sites = list(range(ns))
    if isinstance(state, (int, np.integer)):
        return sigma_pm_int_np(int(state), ns, sites, spin, spin_value)
    if isinstance(state, np.ndarray):
        return sigma_pm_np(state, sites, spin, spin_value)
    return sigma_pm_jnp(state, ns, sites, spin, spin_value)

@numba.njit
def sigma_mp_int_np(state      : int,
                    ns         : int,
                    sites      : List[int],
                    spin       : bool  = BACKEND_DEF_SPIN,
                    spin_val   : float = _SPIN):
    """Alternating σ⁻ σ⁺ starting with σ⁻ on even index (integer state)."""
    new_state, coeff     = _sigma_pm_int_core(state, ns, sites, False, spin, spin_val)
    out_state            = np.empty(1, dtype=DEFAULT_NP_INT_TYPE)
    out_coeff            = np.empty(1, dtype=DEFAULT_NP_FLOAT_TYPE)
    out_state[0]         = new_state
    out_coeff[0]         = coeff
    # return ensure_operator_output_shape_numba(out_state, out_coeff)
    return out_state, out_coeff

@numba.njit
def sigma_mp_np(state       : np.ndarray,
                sites       : List[int],
                spin        : bool  = BACKEND_DEF_SPIN,  
                spin_val    : float = _SPIN):
    """σ⁻ σ⁺ alternating on NumPy array (start σ⁻)."""
    return _sigma_pm_np_core(state, sites, False, spin, spin_val)

def sigma_mp(state,
            ns          : int,
            sites       : Union[List[int], None],
            spin        : bool  = BACKEND_DEF_SPIN,
            spin_value  : float = _SPIN):
    if sites is None:
        sites = list(range(ns))
    if isinstance(state, (int, np.integer)):
        return sigma_mp_int_np(int(state), ns, sites, spin, spin_value)
    if isinstance(state, np.ndarray):
        return sigma_mp_np(state, sites, spin, spin_value)
    return sigma_mp_jnp(state, ns, sites, spin, spin_value)

# -----------------------------------------------------------------------------
#! Sigma-K operator (Fourier-transformed spin operator)
# -----------------------------------------------------------------------------

@numba.njit
def sigma_k_int_np(state    : int,
                ns          : int,
                sites       : List[int],
                k           : float,
                spin        : bool  = BACKEND_DEF_SPIN,
                spin_value  : float = _SPIN):
    """
    Applies the momentum-space spin operator (σ_k) to a given quantum state represented as an integer.

    This function computes the action of the σ_k operator (sum over sites of σ_z * exp(i k r)) on a basis state
    encoded as an integer, for a specified set of sites and momentum k. The result is returned as arrays of output
    states and their corresponding coefficients.

    Args:
        state (int):
            Integer representation of the quantum basis state.
        ns (int):
            Total number of sites in the system.
        sites (List[int]):
            List of site indices on which to apply the operator.
        k (float):
            Momentum value (in radians) for the Fourier transform.
        spin (bool, optional):
            If True, use spin-1/2 convention for σ_z; otherwise, use occupation number. Defaults to BACKEND_DEF_SPIN.
        spin_value (float, optional):
            Value to scale the spin operator. Defaults to _SPIN.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - out_state: Array of output state(s) (as integers).
            - out_coeff: Array of corresponding coefficients (complex numbers).

    Notes:
        - The function assumes that the binary representation of `state` encodes the occupation/spin configuration.
        - The normalization is by sqrt(len(sites)), or 1.0 if `sites` is empty.
    """

    accum = 0.0 + 0.0j
    for i in sites:
        pos        = ns - 1 - i
        bit        = _binary.check_int(state, pos)
        sigma_z_i  = (2.0 * bit - 1.0) * spin_value
        accum     += sigma_z_i * (np.cos(k * i) + 1j * np.sin(k * i))
    norm          = np.sqrt(len(sites)) if sites else 1.0
    out_state     = np.empty(1, dtype=DEFAULT_NP_INT_TYPE)
    out_coeff     = np.empty(1, dtype=DEFAULT_NP_CPX_TYPE)
    out_state[0]  = state
    out_coeff[0]  = accum / norm
    # return ensure_operator_output_shape_numba(out_state, out_coeff)
    return out_state, out_coeff

@numba.njit
def sigma_k_np(state    : np.ndarray,
            sites       : List[int],
            k           : float,
            spin        : bool  = BACKEND_DEF_SPIN,
            spin_value  : float = _SPIN):
    """
    σₖ |state⟩  for a NumPy spin/occupation array (same formula as above).
    """
    accum = np.zeros(1, dtype=DEFAULT_NP_CPX_TYPE)
    for i in sites:
        bit        = _binary.check_arr_np(state, i)
        sigma_z_i  = (2.0 * bit - 1.0) * spin_value
        accum     += sigma_z_i * np.exp(1j * k * i)
    norm = np.sqrt(len(sites)) if len(sites) > 0 else 1.0
    return ensure_operator_output_shape_numba(state, accum / norm)
    # return state, accum / norm

def sigma_k(state,
            ns          : int,
            sites       : Union[List[int], None],
            k           : float,
            spin        : bool  = BACKEND_DEF_SPIN,
            spin_value  : float = _SPIN):
    if sites is None:
        sites = list(range(ns))
    if isinstance(state, (int, np.integer)):
        return sigma_k_int_np(int(state), ns, sites, k, spin, spin_value)
    if isinstance(state, np.ndarray):
        return sigma_k_np(state, sites, k, spin, spin_value)
    return sigma_k_jnp(state, ns, sites, k, spin, spin_value)

################################################################################
#! Factory Functions: Wrap the elementary functions in Operator objects.
################################################################################

# -----------------------------------------------------------------------------
#! Factory function for sigma-x (σₓ)
# -----------------------------------------------------------------------------

def sig_x(  lattice     : Optional[Lattice]     = None,
            ns          : Optional[int]         = None,
            type_act    : OperatorTypeActing    = OperatorTypeActing.Global,
            sites       : Optional[List[int]]   = None,
            spin        : bool                  = BACKEND_DEF_SPIN,
            spin_value  : float                 = _SPIN) -> Operator:
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
    if JAX_AVAILABLE:
        # only sites fun
        if type_act == OperatorTypeActing.Global:
            jnp_fun = partial(sigma_x_int_static_jnp, spin=spin, spin_value=spin_value)
        else:
            jnp_fun = partial(sigma_x_jnp, spin=spin, spin_value=spin_value)
    else:
        jnp_fun = None
        
    return create_operator(
        type_act    = type_act,
        op_func_int = sigma_x_int_np,
        op_func_np  = sigma_x_np,
        op_func_jnp = jnp_fun,
        lattice     = lattice,
        ns          = ns,
        sites       = sites,
        extra_args  = (spin, spin_value),
        name        = "Sx",
        modifies    = True
    )

# -----------------------------------------------------------------------------
#! Factory function for sigma-y (σᵧ)
# -----------------------------------------------------------------------------

def sig_y( lattice     : Optional[Lattice]     = None,
            ns         : Optional[int]         = None,
            type_act   : OperatorTypeActing    = OperatorTypeActing.Global,
            sites      : Optional[List[int]]   = None,
            spin       : bool                  = BACKEND_DEF_SPIN,
            spin_value : float                 = _SPIN) -> Operator:
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
    Returns
    -------
    Operator
        The σᵧ operator.
    """
    
    np_fun  = sigma_y_np
    int_fun = sigma_y_int_np
    
    if JAX_AVAILABLE:
        # only sites fun
        jnp_fun = partial(sigma_y_jnp, spin=spin, spin_value=spin_value)
    else:
        jnp_fun = None
    
    if type_act == OperatorTypeActing.Global:
        if sites is not None and len(sites) % 2 == 1:
            np_fun  = sigma_y_np_real
            int_fun = sigma_y_int_np_real
    elif type_act == OperatorTypeActing.Correlation:
        np_fun  = sigma_y_np_real
        int_fun = sigma_y_int_np_real
    
    return create_operator(
        type_act    = type_act,
        op_func_int = int_fun,
        op_func_np  = np_fun,
        op_func_jnp = jnp_fun,
        lattice     = lattice,
        ns          = ns,
        sites       = sites,
        extra_args  = (spin, spin_value),
        name        = "Sy",
        modifies    = True
    )

# -----------------------------------------------------------------------------
#! Factory function for sigma_z (σ_z)
# -----------------------------------------------------------------------------

def sig_z(  lattice     : Optional[Lattice]     = None,
            ns          : Optional[int]         = None,
            type_act    : OperatorTypeActing    = OperatorTypeActing.Global,
            sites       : Optional[List[int]]   = None,
            spin        : bool                  = BACKEND_DEF_SPIN,
            spin_value  : float                 = _SPIN) -> Operator:
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
    
    if JAX_AVAILABLE:
        # only sites fun
        jnp_fun = partial(sigma_z_jnp, spin=spin, spin_value=spin_value)
    else:
        jnp_fun = None
    
    return create_operator(
        type_act    = type_act,
        op_func_int = sigma_z_int_np,
        op_func_np  = sigma_z_np,
        op_func_jnp = jnp_fun,
        lattice     = lattice,
        ns          = ns,
        sites       = sites,
        extra_args  = (spin, spin_value),
        name        = "Sz",
        modifies    = False
    )

# -----------------------------------------------------------------------------
#! Factory function for sigma-plus (σ⁺)
# -----------------------------------------------------------------------------

def sig_p(  lattice     : Optional[Lattice]     = None,
            ns          : Optional[int]         = None,
            type_act    : OperatorTypeActing    = OperatorTypeActing.Global,
            sites       : Optional[List[int]]   = None,
            spin        : bool                  = BACKEND_DEF_SPIN,
            spin_value  : float                 = _SPIN) -> Operator:
    """
    Factory for the spin‑raising operator σ⁺.
    """
    if JAX_AVAILABLE:
        # only sites fun
        jnp_fun = partial(sigma_plus_jnp, spin=spin, spin_value=spin_value)
    else:
        jnp_fun = None
    return create_operator(
        type_act    = type_act,
        op_func_int = sigma_plus_int_np,
        op_func_np  = sigma_plus_np,
        op_func_jnp = jnp_fun,
        lattice     = lattice,
        ns          = ns,
        sites       = sites,
        extra_args  = (spin, spin_value),          # create_operator injects `spin`
        name        = "Sp",
        modifies    = True                    # σ⁺ flips bits
    )

# -----------------------------------------------------------------------------
#! Factory function for sigma-minus (σ⁻)
# -----------------------------------------------------------------------------

def sig_m(  lattice     : Optional[Lattice]     = None,
            ns          : Optional[int]         = None,
            type_act    : OperatorTypeActing    = OperatorTypeActing.Global,
            sites       : Optional[List[int]]   = None,
            spin        : bool                  = BACKEND_DEF_SPIN,
            spin_value  : float                 = _SPIN) -> Operator:
    """
    Factory for the spin‑lowering operator σ⁻.
    """
    
    if JAX_AVAILABLE:
        # only sites fun
        jnp_fun = partial(sigma_minus_jnp, spin=spin, spin_value=spin_value)
    else:
        jnp_fun = None
    
    return create_operator(
        type_act    = type_act,
        op_func_int = sigma_minus_int_np,
        op_func_np  = sigma_minus_np,
        op_func_jnp = jnp_fun,
        lattice     = lattice,
        ns          = ns,
        sites       = sites,
        extra_args  = (spin, spin_value),
        name        = "Sm",
        modifies    = True
    )

# -----------------------------------------------------------------------------
#! Factory function for sigma-pm (σ⁺ then σ⁻)
# -----------------------------------------------------------------------------

def sig_pm( lattice     : Optional[Lattice]     = None,
            ns          : Optional[int]         = None,
            type_act    : OperatorTypeActing    = OperatorTypeActing.Global,
            sites       : Optional[List[int]]   = None,
            spin        : bool                  = BACKEND_DEF_SPIN,
            spin_value  : float                 = _SPIN) -> Operator:
    """
    Factory for the alternating operator: even‑indexed sites σ⁺, odd‑indexed σ⁻.
    """
    if JAX_AVAILABLE:
        # only sites fun
        jnp_fun = partial(sigma_pm_jnp, spin=spin, spin_value=spin_value)
    else:
        jnp_fun = None
    
    return create_operator(
        type_act    = type_act,
        op_func_int = sigma_pm_int_np,
        op_func_np  = sigma_pm_np,
        op_func_jnp = jnp_fun,
        lattice     = lattice,
        ns          = ns,
        sites       = sites,
        extra_args  = (spin, spin_value),
        name        = "Spm",
        modifies    = True
    )
    
# -----------------------------------------------------------------------------
#! Factory function for sigma-mp (σ⁻ then σ⁺)
# -----------------------------------------------------------------------------

def sig_mp( lattice     : Optional[Lattice]     = None,
            ns          : Optional[int]         = None,
            type_act    : OperatorTypeActing    = OperatorTypeActing.Global,
            sites       : Optional[List[int]]   = None,
            spin        : bool                  = BACKEND_DEF_SPIN,
            spin_value  : float                 = _SPIN) -> Operator:
    """
    Factory for the alternating operator: even‑indexed sites σ⁻, odd‑indexed σ⁺.
    """
    if JAX_AVAILABLE:
        # only sites fun
        jnp_fun = partial(sigma_mp_jnp, spin=spin, spin_value=spin_value)
    else:
        jnp_fun = None
    
    return create_operator(
        type_act    = type_act,
        op_func_int = sigma_mp_int_np,
        op_func_np  = sigma_mp_np,
        op_func_jnp = jnp_fun,
        lattice     = lattice,
        ns          = ns,
        sites       = sites,
        extra_args  = (spin, spin_value),
        name        = "Smp",
        modifies    = True
    )

# -----------------------------------------------------------------------------
#! Factory function for sigma-k (σₖ)
# -----------------------------------------------------------------------------

def sig_k(  k           : float,
            lattice     : Optional[Lattice]     = None,
            ns          : Optional[int]         = None,
            type_act    : OperatorTypeActing    = OperatorTypeActing.Global,
            sites       : Optional[List[int]]   = None,
            spin        : bool                  = BACKEND_DEF_SPIN,
            spin_value  : float                 = _SPIN) -> Operator:
    r"""
    Factory for the momentum‑space operator  

        σₖ = (1/√N)\,\sum_{i∈\text{sites}} σ_z(i)\,e^{\,ik i}.
    """
    if JAX_AVAILABLE:
        # only sites fun
        jnp_fun = partial(sigma_k_jnp, spin=spin, spin_value=spin_value, k=k)
    else:
        jnp_fun = None
    
    return create_operator(
        type_act    = type_act,
        op_func_int = sigma_k_int_np,
        op_func_np  = sigma_k_np,
        op_func_jnp = jnp_fun,
        lattice     = lattice,
        ns          = ns,
        sites       = sites,
        extra_args  = (k, spin, spin_value),
        name        = f"Sk(k={k:.3g})",
        modifies    = False # σₖ leaves the state unchanged
    )
    
# -----------------------------------------------------------------------------
#! Finalize
# -----------------------------------------------------------------------------

def test_spin_operator_matrices(nh = 4, site = 0):
    """
    Test the equality between built-in operator matrices and explicitly constructed matrices
    using Kronecker products for spin operators.
    This function constructs representations for Pauli spin operators (sig_x, sig_y, and sig_z)
    using both the operator’s built-in methods and explicit Kronecker product expansion.
    It then compares these two implementations for consistency.
    Parameters:
        nh (int, optional): The Hilbert space dimension for the subsystem.
            Typically, nh should be a power of 2 (e.g., 4 represents a two-qubit system).
            Default is 4.
        site (int, optional): The index (position) of the site where the active spin operator
            is applied, with all other sites receiving the identity operator.
            Default is 0.
    Functionality:
        - Constructs a dense identity matrix for the first subsystem.
        - Computes the number of sites (ns) as the logarithm base 2 of nh.
        - Generates operator matrices for sig_x, sig_y, and sig_z using built-in methods.
        - Constructs explicit matrix representations of the operators using successive Kronecker products:
            * For the target site, the appropriate Pauli matrix (scaled by 0.5) is used.
            * For non-target sites, the identity matrix (_SIG_0) is used.
            * Note: The sign for sig_z is inverted (i.e., -_SIG_Z * 0.5) at the active site.
        - Compares the built-in and explicit matrices using numpy.allclose.
        - Prints whether the matrices are equal.
        - If the Hilbert space is small (nh < 64), prints both sets of matrices for visual verification,
            using a MatrixPrinter utility if available.
    Returns:
        None
    """
    
    from general_python.common.plot import MatrixPrinter
    
    # Set the Hilbert space dimension (for example, 4) and construct an identity.
    # Create a dense identity for the first subsystem.
    ns                  = np.log2(nh)
    # Build the operator matrices via the operator’s built-in method.
    sig_x_op            = sig_x(ns = ns, type_act = OperatorTypeActing.Global, sites = [0])
    sig_y_op            = sig_y(ns = ns, type_act = OperatorTypeActing.Global, sites = [0])
    sig_z_op            = sig_z(ns = ns, type_act = OperatorTypeActing.Global, sites = [0])
    sig_x_op_mat        = sig_x_op.matrix(dim=nh, matrix_type='sparse', use_numpy=True)
    sig_y_op_mat        = sig_y_op.matrix(dim=nh, matrix_type='sparse', use_numpy=True)
    sig_z_op_mat        = sig_z_op.matrix(dim=nh, matrix_type='sparse', use_numpy=True)
    
    # Build the same matrices via explicit Kronecker products.
    # Note: The Kronecker product is not the most efficient way to build these matrices,
    # but it is useful for testing purposes.
    
    out_sig_x           = _SIG_X * 0.5 if site == 0 else _SIG_0
    out_sig_y           = _SIG_Y * 0.5 if site == 0 else _SIG_0
    out_sig_z           = -_SIG_Z * 0.5 if site == 0 else _SIG_0
    for i in range(1, int(ns)):
        if i == site:
            out_sig_x = np.kron(out_sig_x, _SIG_X * 0.5)
            out_sig_y = np.kron(out_sig_y, _SIG_Y * 0.5)
            out_sig_z = np.kron(out_sig_z, -_SIG_Z * 0.5)
        else:
            out_sig_x = np.kron(out_sig_x, _SIG_0)
            out_sig_y = np.kron(out_sig_y, _SIG_0)
            out_sig_z = np.kron(out_sig_z, _SIG_0)
    
    # For this test we compare the operator's built-in result to one of the explicit ones.
    # (Choose which one matches your intended ordering.)
    is_equal_x = np.allclose(sig_x_op_mat.todense(), out_sig_x)
    print("Are the two matrices (sig_x) equal?", is_equal_x)
    is_equal_y = np.allclose(sig_y_op_mat.todense(), out_sig_y)
    print("Are the two matrices (sig_y) equal?", is_equal_y)
    is_equal_z = np.allclose(sig_z_op_mat.todense(), out_sig_z)
    print("Are the two matrices (sig_z) equal?", is_equal_z)
    
    # If the dimension is small, print out the matrices.
    if nh < 64:
        try:
            
            print("Matrix from operator (sig_x):")
            MatrixPrinter.print_matrix(out_sig_x.todense())
            print("Matrix from tensor (sig_x):")
            MatrixPrinter.print_matrix(sig_x_op_mat.todense())
            print("Matrix from operator (sig_y):")
            MatrixPrinter.print_matrix(out_sig_y.todense())
            print("Matrix from tensor (sig_y):")
            MatrixPrinter.print_matrix(sig_y_op_mat.todense())
            print("Matrix from operator (sig_z):")
            MatrixPrinter.print_matrix(out_sig_z.todense())
            print("Matrix from tensor (sig_z):")
            MatrixPrinter.print_matrix(sig_z_op_mat.todense())
        except ImportError:
            print("Matrix from operator (sig_x):")
            print(out_sig_x)
            print("Matrix from tensor (sig_x):")
            print(sig_x_op_mat.todense())
            print("Matrix from operator (sig_y):")
            print(out_sig_y)
            print("Matrix from tensor (sig_y):")
            print(sig_y_op_mat.todense())
            print("Matrix from operator (sig_z):")
            print(out_sig_z)
            print("Matrix from tensor (sig_z):")
            print(sig_z_op_mat.todense())

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
            self.backend = get_backend(backend)
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
        if not isinstance(state, (int, np.integer)):
            state = base2int(state, spin = BACKEND_DEF_SPIN, spin_value = _SPIN)
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
                    spin        = BACKEND_DEF_SPIN,
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
        if isinstance(state, (int, np.integer)):
            base_state = int2base(state, ns, spin=True, spin_value=_SPIN)
        else:
            base_state = state
            
        op = sig_x(lattice       = self.lattice,
                    ns           = ns,
                    type_act     = OperatorTypeActing.Global,
                    sites        = sites,
                    spin         = BACKEND_DEF_SPIN,
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
                   spin=BACKEND_DEF_SPIN,
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
        if isinstance(state, (int, np.integer)):
            base_state = int2base(state, ns, spin=True, spin_value=_SPIN)
        else:
            base_state = state
        op = sig_y(lattice=self.lattice,
                   ns=ns,
                   type_act=OperatorTypeActing.Global,
                   sites=sites,
                   spin=BACKEND_DEF_SPIN,
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
                   spin=BACKEND_DEF_SPIN,
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
        if isinstance(state, (int, np.integer)):
            base_state = int2base(state, ns, spin=True, spin_value=_SPIN)
        else:
            base_state = state
        op = sig_z(lattice=self.lattice,
                   ns=ns,
                   type_act=OperatorTypeActing.Global,
                   sites=sites,
                   spin=BACKEND_DEF_SPIN,
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
                      spin=BACKEND_DEF_SPIN,
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
        if isinstance(state, (int, np.integer)):
            base_state = int2base(state, ns, spin=True, spin_value=_SPIN)
        else:
            base_state = state
        op = sig_plus(lattice=self.lattice,
                      ns=ns,
                      type_act=OperatorTypeActing.Global,
                      sites=sites,
                      spin=BACKEND_DEF_SPIN,
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
                       spin=BACKEND_DEF_SPIN,
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
        if isinstance(state, (int, np.integer)):
            base_state = int2base(state, ns, spin=True, spin_value=_SPIN)
        else:
            base_state = state
        op = sig_minus(lattice=self.lattice,
                       ns=ns,
                       type_act=OperatorTypeActing.Global,
                       sites=sites,
                       spin=BACKEND_DEF_SPIN,
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
                   spin=BACKEND_DEF_SPIN,
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
        if isinstance(state, (int, np.integer)):
            base_state = int2base(state, ns, spin=True, spin_value=_SPIN)
        else:
            base_state = state
        op = sig_k(lattice=self.lattice,
                   ns=ns,
                   type_act=OperatorTypeActing.Global,
                   sites=sites,
                   k=k,
                   spin=BACKEND_DEF_SPIN,
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