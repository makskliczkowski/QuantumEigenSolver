"""
file        : Algebra/Operator/operators_spin_jax.py

This module implements spin operators for quantum systems using the JAX library.
It includes functions for sigma_x, sigma_y, sigma_z, sigma_plus (raising),
sigma_minus (lowering), their products, and a Fourier-transformed sigma_k operator.
The implementation is based on the provided C++ code and uses a general Operator class.

Author      : Maksymilian Kliczkowski, WUST, Poland
Date        : February 2025
Version     : 1.0
"""

import math
from typing import List, Union
from functools import partial

################################################################################
from Algebra.Operator.operator import Operator, OperatorTypeActing, SymmetryGenerators
################################################################################

################################################################################
from general_python.common.tests import GeneralAlgebraicTest
from general_python.lattices.lattice import Lattice
from general_python.algebra.utils import DEFAULT_BACKEND, get_backend
from general_python.common.binary import BACKEND_REPR as _SPIN, BACKEND_DEF_SPIN, JAX_AVAILABLE

# binary operations
import general_python.common.binary as _binary

# JAX imports
if JAX_AVAILABLE:
    import jax
    from jax import lax
    from jax import numpy as jnp

    # transform the matrices to JAX arrays
    _SIG_0_jnp = jnp.array([[1, 0],
                    [0, 1]], dtype=float)
    _SIG_X_jnp = jnp.array([[0, 1],
                    [1, 0]], dtype=float)
    _SIG_Y_jnp = jnp.array([[0, -1j],
                    [1j, 0]], dtype=complex)
    _SIG_Z_jnp = jnp.array([[1,  0],
                    [0, -1]], dtype=float)
    _SIG_P_jnp = jnp.array([[0, 1],
                    [0, 0]], dtype=float)
    _SIG_M_jnp = jnp.array([[0, 0],
                    [1, 0]], dtype=float)

# -----------------------------------------------------------------------------
#! Sigma-X (σₓ) operator
# -----------------------------------------------------------------------------

if JAX_AVAILABLE:
    
    @partial(jax.jit, static_argnums=(1, 2))
    def sigma_x_int_jnp(state,
                        ns,
                        sites,
                        spin        : bool  = BACKEND_DEF_SPIN,
                        spin_value  : float = _SPIN):
        """
        Apply the Pauli-X (σₓ) operator on the given sites.
        For each site, flip the bit at position (ns-1-site) using a JAX-compatible flip function.
        
        Args:
            state: 
                A JAX integer (or traced array) representing the state.
            ns (int): 
                Number of sites.
            sites (Union[List[int], None]): 
                A list of site indices.
            spin_value (float): 
                Spin value (default _SPIN).
        
        Returns:
            A tuple (state, coeff) with the updated state and accumulated coefficient.
        """
        sites = jnp.asarray(sites)
        def body(i, carry):
            curr_state, curr_coeff  = carry
            # sites is static, so extract the site.
            site                    = sites[i]
            site                    = ns - 1 + site
            # flip is assumed to be a JAX-compatible function that flips the bit at position pos.
            new_state               = _binary.flip_int_traced_jax(curr_state, site)
            new_coeff               = curr_coeff * spin_value
            return (new_state, new_coeff)

        num_sites   = len(sites)
        init        = (state, 1.0)
        final_state, final_coeff = lax.fori_loop(0, num_sites, body, init)
        return final_state, final_coeff

    @partial(jax.jit, static_argnums=(2,))
    def sigma_x_jnp(state,
                    sites       : Union[List[int], None],
                    spin        : bool = BACKEND_DEF_SPIN,
                    spin_value  : float = _SPIN):
        """
        Apply the Pauli-X (σₓ) operator on a JAX array state.
        For each site, flip the bit at position (ns-1-site) using binary.flip.
        
        This version is optimized by using a JAX for-loop (lax.fori_loop)
        to avoid Python-level loops and enable JIT compilation.
        Parameters:
            state (jax.numpy.ndarray):
                The state to be modified.
            ns (int):
                Number of sites.
            sites (Union[List[int], None]):
                A list of site indices to flip.
        """
        
        # Pre-calculate the overall coefficient as spin_value raised to the number of flips.
        sites_arr   = jnp.array(sites)
        coeff       = spin_value ** sites_arr.shape[0]
        
        def apply_spin_flip(selected_elements):
            return -selected_elements

        def apply_nspin_flip(selected_elements):
            return 1.0 - selected_elements
        
        def update_state(current_state, indices_to_update):
            return lax.cond(
                spin,
                lambda s: s.at[indices_to_update].apply(apply_spin_flip),
                lambda s: s.at[indices_to_update].apply(apply_nspin_flip),
                current_state
            )
        
        new_state = lax.cond(
            sites_arr.shape[0] > 0,
            lambda s: update_state(s, sites_arr),
            lambda s: s,
            state
        )
        
        return new_state, coeff

# -----------------------------------------------------------------------------
#! Sigma-Y (σᵧ) operator
# -----------------------------------------------------------------------------

if JAX_AVAILABLE:
    
    @partial(jax.jit, static_argnums=(1, 2))
    def sigma_y_int_jnp(state,
                        ns          : int,
                        sites       : Union[List[int], None],
                        spin        : bool = BACKEND_DEF_SPIN,
                        spin_value  : float = _SPIN):
        """
        σᵧ on an integer state (JAX version).

        For each site, if the bit at (ns-1-site) is set then multiply the coefficient
        by (1j*spin_value), otherwise by (-1j*spin_value); then flip the bit.
        
        Args:
            state (int or JAX array) :
                The state to apply the operator to.
            ns (int) :
                The number of spins in the system.
            sites (list of int or None) :
                The sites to apply the operator to. If None, apply to all sites.
            spin_value (float) :
                The value to multiply the state by when flipping the bits.
        
        Returns:
            tuple: (new_state, coeff) where new_state is the state after applying the operator,
                and coeff is the accumulated complex coefficient.
        """
        sites_arr = jnp.array(sites)

        def body(i, carry):
            state_val, coeff    = carry
            site                = sites_arr[i]
            pos                 = ns - 1 - site
            bitmask             = jnp.left_shift(1, pos)
            condition           = (state_val & bitmask) > 0
            factor = lax.cond(condition,
                                lambda _: 1j * spin_value,
                                lambda _: -1j * spin_value,
                                operand=None)
            new_state = _binary.flip_int_traced_jax(state_val, pos)
            return (new_state, coeff * factor)

        final_state, final_coeff = lax.fori_loop(0, sites_arr.shape[0], body, (state, 1.0 + 0j))
        return final_state, final_coeff

    @partial(jax.jit, static_argnums=(2,))
    def sigma_y_jnp(state,
                    sites       : Union[List[int], None],
                    spin        : bool = BACKEND_DEF_SPIN,
                    spin_value  : float = _SPIN):
        """
        σᵧ on a JAX array state.
        Uses lax.fori_loop.
        
        Parameters:
            state (np.ndarray) : 
                The state to apply the operator to.
            ns (int) : 
                The number of spins in the system.
            sites (list of int or None) : 
                The sites to apply the operator to. If None, apply to all sites.    
            spin (bool) : 
                If True, use the spin convention for flipping the bits.
            spin_value (float) : 
                The value to multiply the state by when flipping the bits.
            
        Returns:
            tuple: (new_state, coeff) where new_state is the state after applying the operator
                and coeff is the accumulated coefficient.
        """
        iscpx = sites.shape[0] % 2 == 1
        coeff = 1.0 + 0j if iscpx else 1.0

        def body_fun(i, state_val):
            site            = sites[i]
            # factor          = 1j * spin_value if check(state_val, pos) else -1j * spin_value
            new_state       = _binary.flip_array_jax_spin(state_val, site) if spin else _binary.flip_array_jax_nspin(state_val, site)
            return new_state
        
        new_state           = lax.fori_loop(0, len(sites), body_fun, state)
        
        # If spin_value is real, keep coeff real; only use complex if needed
        sites_arr   = jnp.asarray(sites)
        def body(i, coeff):
            site    = sites_arr[i]
            bit     = _binary.check_arr_jax(state, site)
            factor  = (2 * bit - 1.0) * spin_value if spin else spin_value * bit
            if iscpx:
                factor = factor * 1.0j
            return coeff * factor
        coeff = lax.fori_loop(0, len(sites), body, 1.0)
        
        # apply the sign if 1j * 1j = -1 for even but not twice even 
        if sites.shape[0] % 4 != 0 and not iscpx:
            coeff = -coeff
        
        return new_state, coeff

# -----------------------------------------------------------------------------
#! Sigma-Z (σ_z) operator
# -----------------------------------------------------------------------------

if JAX_AVAILABLE:

    @partial(jax.jit, static_argnums=(1, 2))
    def sigma_z_int_jnp(state,
                        ns          : int,
                        sites       : Union[List[int], None],
                        spin        : bool      = BACKEND_DEF_SPIN,
                        spin_value  : float     = _SPIN):
        """
        σ_z on an integer state.
        For each site, if the bit at (ns-1-site) is set then multiply by spin_value; else by -spin_value.
        The state is unchanged.
        
        Args:
            state :
                A JAX integer (or traced array of integers) representing the state.
            ns (int) :
                The number of sites.
            sites (Union[List[int], None]) :
                A list of site indices.
            spin (bool) :
                If True, use the spin convention for flipping the bits.
            spin_value (float) :
                The spin value (default _SPIN).
        
        Returns:
            A tuple (state, coeff) where state is unchanged and coeff is the product
            of the factors determined by the bits in state.
        """
        # Body function for the fori_loop. The loop variable 'i' runs over site indices.
        
        sites = jnp.array(sites)
        
        def body(i, coeff):
            # Since sites is a static Python list, we can extract the site index.
            site        = sites[i]
            # Compute the bit position: (ns - 1 - site)
            pos         = ns - 1 - site
            # Compute the bit mask using JAX operations.
            bitmask     = jnp.left_shift(1, pos)
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

    @partial(jax.jit, static_argnums=(2,))
    def sigma_z_jnp(state,
                    sites       : Union[List[int], None],
                    spin        : bool = BACKEND_DEF_SPIN,
                    spin_value  : float = _SPIN):
        """
        σ_z on a JAX array state.
        
        Parameters:
            state (np.ndarray) :
                The state to apply the operator to.
            ns (int) :
                The number of spins in the system.
            sites (list of int or None) :
                The sites to apply the operator to. If None, apply to all sites.
            spin (bool) :
                If True, use the spin convention for flipping the bits.
            spin_value (float) :
                The value to multiply the state by when flipping the bits.
            
        Returns:
            tuple: (state, coeff) where state is unchanged and coeff is the accumulated coefficient.
        """        
        coeff       = 1.0
        sites       = jnp.array(sites)
        for site in sites:
            bit     =   _binary.check_arr_jax(state, site)
            factor  =   (2 * bit - 1.0) * spin_value if spin else spin_value * bit
            coeff   *=  factor
        return state, coeff

# -----------------------------------------------------------------------------
#! Sigma-Plus (σ⁺) operator
# -----------------------------------------------------------------------------
if JAX_AVAILABLE:
    
    @jax.jit
    def sigma_plus_int_jnp(state, 
                        ns          : int, 
                        sites       : Union[List[int], None], 
                        spin        : bool = BACKEND_DEF_SPIN,
                        spin_value  : float = _SPIN):
        """
        Apply the raising operator σ⁺ on an integer state (JAX version).
        """
        sites = jnp.array(sites)
        def body(i, carry):
            curr_state, curr_coeff  = carry
            pos                     = ns - 1 - sites[i]
            bitmask                 = jnp.left_shift(1, pos)
            condition               = (curr_state & bitmask) > 0
            new_state               = lax.cond(condition,
                                        lambda _: curr_state,
                                        lambda _: _binary.flip_int_traced_jax(curr_state, pos) if spin else _binary.flip_int_traced_jax(curr_state, pos),
                                        operand=None)
            new_coeff = lax.cond(condition,
                                    lambda _: 0.0,
                                    lambda _: curr_coeff * spin_value,
                                    operand = None)
            return (new_state, new_coeff)
        init = (state, 1.0)
        final_state, final_coeff = lax.fori_loop(0, len(sites), body, init)
        return final_state, final_coeff

    @jax.jit
    def sigma_plus_jnp(state,
                        sites       : Union[List[int], None],
                        spin        : bool = BACKEND_DEF_SPIN,
                        spin_value  : float = _SPIN):
        """
        σ⁺ on a JAX array state.
        Uses lax.fori_loop.
        """
        def body_fun(i, state_val):
            state_in, coeff_in  = state_val
            site                = sites[i]
            coeff_new           = jax.lax.cond(_binary.check_arr_jax(state_in, site),
                                                lambda _: 0.0,
                                                lambda _: coeff_in * spin_value,
                                                operand=None)
            new_state           = jax.lax.cond(spin,
                                                lambda _: _binary.flip_array_jax_spin(state_in, site),
                                                lambda _: _binary.flip_array_jax_nspin(state_in, site),
                                                operand=None)
            return new_state, coeff_new
        new_state, coeff = lax.fori_loop(0, len(sites), body_fun, (state, 1.0))
        return new_state, coeff

# -----------------------------------------------------------------------------
#! Sigma-Minus (σ⁻) operator
# -----------------------------------------------------------------------------

if JAX_AVAILABLE:
    
    @jax.jit
    def sigma_minus_int_jnp(state, 
                            ns, 
                            sites, 
                            spin        : bool = BACKEND_DEF_SPIN,
                            spin_value  : float = _SPIN):
        sites_arr = jnp.array(sites)
        def body(i, carry):
            curr_state, curr_coeff = carry
            pos         = ns - 1 - sites_arr[i]
            bitmask     = jnp.left_shift(1, pos)
            condition   = (curr_state & bitmask) > 0
            new_state   = lax.cond(condition,
                                    lambda _: _binary.flip_int_traced_jax(curr_state, pos) if spin else _binary.flip_int_traced_jax(curr_state, pos),
                                    lambda _: curr_state,
                                    operand=None)
            new_coeff   = lax.cond(condition,
                                    lambda _: curr_coeff * spin_value,
                                    lambda _: 0.0,
                                    operand=None)
            return (new_state, new_coeff)
        init = (state, 1.0)
        final_state, final_coeff = lax.fori_loop(0, len(sites), body, init)
        return final_state, final_coeff

    @jax.jit
    def sigma_minus_jnp(state,
                        sites       : Union[List[int], None],
                        spin        : bool = BACKEND_DEF_SPIN,
                        spin_value  : float = _SPIN):
        """
        σ⁻ on a JAX array state.
        """
        def body_fun(i, state_val):
            state_in, coeff_in  = state_val
            site                = sites[i]
            coeff_new           = jax.lax.cond(_binary.check_arr_jax(state_in, site),
                                                lambda _: 0.0,
                                                lambda _: coeff_in * spin_value,
                                                operand=None)
            new_state           = jax.lax.cond(spin,
                                                lambda _: _binary.flip_array_jax_spin(state_in, site),
                                                lambda _: _binary.flip_array_jax_nspin(state_in, site),
                                                operand=None)
            return new_state, coeff_new
        new_state, coeff = lax.fori_loop(0, len(sites), body_fun, (state, 1.0))
        return new_state, coeff

# -----------------------------------------------------------------------------
#! Sigma_pm (σ⁺ then σ⁻) operator
# -----------------------------------------------------------------------------

if JAX_AVAILABLE:
    
    @jax.jit
    def sigma_pm_jnp(state, sites, spin: bool = BACKEND_DEF_SPIN, spin_value: float = _SPIN):
        coeff = 1.0
        
        def flip_func(state_val, pos):
            return jax.lax.cond(
                spin,
                lambda _: _binary.flip_array_jax_spin(state_val, pos),
                lambda _: _binary.flip_array_jax_nspin(state_val, pos),
                operand=None
            )
        
        def body_fun(i, state_val):
            site = sites[i]
            pos  = site

            def even_branch(_):
            # If bit is set, return state_val; else flip
                return jax.lax.cond(
                    _binary.check_arr_jax(state_val, pos),
                    lambda _: state_val,
                    lambda _: flip_func(state_val, pos),
                    operand=None
                )

            def odd_branch(_):
            # If bit is not set, return state_val; else flip
                return jax.lax.cond(
                    _binary.check_arr_jax(state_val, pos),
                    lambda _: flip_func(state_val, pos),
                    lambda _: state_val,
                    operand=None
                )

            return jax.lax.cond(
                (i % 2) == 0,
                even_branch,
                odd_branch,
                operand=None
                )

        new_state = lax.fori_loop(0, len(sites), body_fun, state)
        return new_state, coeff

    @jax.jit
    def sigma_pm_int_jnp(state, sites, spin: bool = BACKEND_DEF_SPIN, spin_value: float = _SPIN):
        sites = jnp.array(sites)
        def body(i, carry):
            curr_state, curr_coeff  = carry
            pos                     = sites[i]
            bitmask                 = jnp.left_shift(1, pos)
            even_branch             = lax.cond((curr_state & bitmask) == 0,
                                lambda _: (_binary.flip_int_traced_jax(curr_state, pos) if spin else _binary.flip_int_traced_jax(curr_state, pos),
                                            curr_coeff * spin_value),
                                lambda _: (curr_state, 0.0),
                                operand=None)
            odd_branch              = lax.cond((curr_state & bitmask) > 0,
                                lambda _: (_binary.flip_int_traced_jax(curr_state, pos) if spin else _binary.flip_int_traced_jax(curr_state, pos),
                                            curr_coeff * spin_value),
                                lambda _: (curr_state, 0.0),
                                operand=None)
            new_state, new_coeff = even_branch if (i % 2 == 0) else odd_branch
            return (new_state, new_coeff)
        init = (state, 1.0)
        final_state, final_coeff = lax.fori_loop(0, len(sites), body, init)
        return final_state, final_coeff

# -----------------------------------------------------------------------------
#! Sigma_mp (σ⁻ then σ⁺) operator
# -----------------------------------------------------------------------------

if JAX_AVAILABLE:

    @jax.jit
    def sigma_mp_int_jnp(state, sites, spin: bool = BACKEND_DEF_SPIN, spin_value: float = _SPIN):
        sites = jnp.array(sites)
        def body(i, carry):
            curr_state, curr_coeff  = carry
            pos                     = sites[i]
            bitmask                 = jnp.left_shift(1, pos)
            even_branch             = lax.cond((curr_state & bitmask) > 0,
                                lambda _: (_binary.flip_int_traced_jax(curr_state, pos) if spin else _binary.flip_int_traced_jax(curr_state, pos), curr_coeff * spin_value),
                                lambda _: (curr_state, 0.0),
                                operand=None)
            odd_branch              = lax.cond((curr_state & bitmask) == 0,
                                lambda _: (_binary.flip_int_traced_jax(curr_state, pos) if spin else _binary.flip_int_traced_jax(curr_state, pos), curr_coeff * spin_value),
                                lambda _: (curr_state, 0.0),
                                operand=None)
            new_state, new_coeff    = even_branch if (i % 2 == 0) else odd_branch
            return (new_state, new_coeff)
        init = (state, 1.0)
        final_state, final_coeff = lax.fori_loop(0, len(sites), body, init)
        return final_state, final_coeff

    @jax.jit
    def sigma_mp_jnp(state,
                    sites   : Union[List[int], None],
                    spin    : bool = BACKEND_DEF_SPIN,
                    spin_value : float = _SPIN):
        """
        Alternating operator (σ⁻ then σ⁺) on a JAX array state.
        """
        def body_fun(i, state_val):
            state_in, coeff_in = state_val
            site = sites[i]
            def even_branch(_):
                # σ⁻: only act if bit is set
                coeff_new = jax.lax.cond(_binary.check_arr_jax(state_in, site),
                                         lambda _: coeff_in * spin_value,
                                         lambda _: 0.0,
                                         operand=None)
                new_state = jax.lax.cond(spin,
                                         lambda _: _binary.flip_array_jax_spin(state_in, site),
                                         lambda _: _binary.flip_array_jax_nspin(state_in, site),
                                         operand=None)
                return new_state, coeff_new

            def odd_branch(_):
                # σ⁺: only act if bit is not set
                coeff_new = jax.lax.cond(_binary.check_arr_jax(state_in, site),
                                         lambda _: 0.0,
                                         lambda _: coeff_in * spin_value,
                                         operand=None)
                new_state = jax.lax.cond(spin,
                                         lambda _: _binary.flip_array_jax_spin(state_in, site),
                                         lambda _: _binary.flip_array_jax_nspin(state_in, site),
                                         operand=None)
                return new_state, coeff_new

            new_state, coeff_new = jax.lax.cond(
                (i % 2) == 0,
                even_branch,
                odd_branch,
                operand=None
            )
            return new_state, coeff_new
        new_state, coeff = lax.fori_loop(0, len(sites), body_fun, (state, 1.0))
        return new_state, coeff

# -----------------------------------------------------------------------------
#! Sigma-K (σₖ) operator
# -----------------------------------------------------------------------------

if JAX_AVAILABLE:
    @jax.jit
    def sigma_k_int_jnp(state, 
                        ns          : int, 
                        sites       : Union[List[int], None], 
                        k           : float, 
                        spin        : bool = BACKEND_DEF_SPIN, 
                        spin_value  : float = _SPIN):
        
        sites = jnp.asarray(sites)
        def body(i, total):
            site    = sites[i]
            pos     = ns - 1 - site
            bitmask = jnp.left_shift(1, pos)
            factor  = lax.cond((state & bitmask) > 0,
                            lambda _: 1j,
                            lambda _: -1.0,
                            operand=None)
            return total + factor * math.exp(1j * k * site)
        total   = lax.fori_loop(0, len(sites), body, 0.0+0j)
        norm    = math.sqrt(len(sites)) if len(sites) > 0 else 1.0
        return state, total / norm

    @jax.jit
    def sigma_k_jnp(state,
                    ns          : int,
                    sites       : Union[List[int], None],
                    k           : float,
                    spin        : bool = BACKEND_DEF_SPIN,
                    spin_value  : float = _SPIN):
        """
        Compute the Fourier-transformed spin operator (σₖ) on a JAX array state.
        Uses lax.fori_loop.
        Parameters:
            state (np.ndarray) :
                The state to apply the operator to.
            ns (int) :
                The number of spins in the system.
            sites (list of int or None) :
                The sites to apply the operator to. If None, apply to all sites.
            k (float) :
                The wave vector for the Fourier transform.
            spin (bool) :
                If True, use the spin convention for flipping the bits.
            spin_value (float) :
                The value to multiply the state by when flipping the bits.
        Returns:
            tuple: (state, coeff) where state is unchanged and coeff is the accumulated coefficient.
        """
        total = 0.0 + 0j
        def body_fun(i, total_val):
            site    = sites[i]
            pos     = site
            bit     = _binary.check_arr_jax(state, pos)
            factor  = (2 * bit - 1.0) * spin_value if spin else spin_value * bit
            return total_val + factor * math.exp(1j * k * site)
        total   = lax.fori_loop(0, len(sites), body_fun, total)
        norm    = math.sqrt(len(sites)) if sites else 1.0
        return state, total / norm

# -----------------------------------------------------------------------------

if not JAX_AVAILABLE:
    sigma_x_int_jnp = None
    sigma_x_jnp = None
    sigma_y_int_jnp = None
    sigma_y_jnp = None
    sigma_z_int_jnp = None
    sigma_z_jnp = None
    sigma_plus_int_jnp = None
    sigma_plus_jnp = None
    sigma_minus_int_jnp = None
    sigma_minus_jnp = None
    sigma_pm_jnp = None
    sigma_mp_jnp = None
    sigma_k_int_jnp = None
    sigma_k_jnp = None
    sigma_pm_int_jnp = None
    sigma_mp_int_jnp = None
    sigma_k_int_jnp = None
    sigma_k_jnp = None
    