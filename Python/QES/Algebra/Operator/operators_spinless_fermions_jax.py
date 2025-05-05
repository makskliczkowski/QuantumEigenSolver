'''
file    : Algebra/Operator/operators_spinless_fermions_jax.py
desc    : Spinless fermions operators (Jordan-Wigner) with JAX. For documentation
          see `operators_spinless_fermions.py`.
'''

import math
import numpy as np
import numba
from typing import List, Union, Optional, Callable

from general_python.algebra.utils import JAX_AVAILABLE, Array, DEFAULT_JP_FLOAT_TYPE, DEFAULT_JP_CPX_TYPE, DEFAULT_JP_INT_TYPE
import general_python.common.binary as _binary


if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
    from jax import jit, lax
    _DEFAULT_INT    = DEFAULT_JP_INT_TYPE
    _DEFAULT_FLOAT  = DEFAULT_JP_FLOAT_TYPE
    _DEFAULT_CPX    = DEFAULT_JP_CPX_TYPE
    _flip_jnp       = _binary.flip_int_traced_jax
    _bit_jnp        = _binary.check_int_traced_jax
    
    @jit
    def _popcount_mask_jnp(x: jnp.int64, mask_bits: jnp.int64) -> jnp.int64:
        """Number of set bits in x & mask_bits (JIT-safe)."""
        return jnp.bit_count(x & mask_bits)
    
    #============================================================================
    #! Jordan–Wigner parity  (-1)^{# occupied left of `site`}
    #============================================================================
    
    @jit
    def f_parity_int_jnp(state: jnp.int64, ns: int, site: int) -> jnp.float64:
        shift      = ns - site
        mask_bits  = (jnp.int64(1) << jnp.int64(shift)) - jnp.int64(1)
        parity     = _popcount_mask_jnp(state, mask_bits) & jnp.int64(1)
        return jnp.where(parity == 1, -1.0, 1.0)

    @jit
    def f_parity_np_jnp(state: jnp.ndarray, site: int) -> jnp.float64:
        parity = jnp.bitwise_and(jnp.sum(state[:site]), 1)
        return jnp.where(parity == 1, -1.0, 1.0)
    
    #============================================================================
    
    @jit
    def c_dag_int_jnp(state: jnp.int64,
                    ns   : int,
                    site : int,
                    pref : float = 1.0):
        pos        = ns - 1 - site
        bit        = _bit_jnp(state, pos)

        sign       = f_parity_int_jnp(state, ns, site)
        new_state  = lax.select(bit == 1, state, _flip_jnp(state, pos))
        coeff_val  = lax.select(bit == 1, 0.0, sign * pref)

        return (jnp.array([new_state], dtype=_DEFAULT_INT),
                jnp.array([coeff_val], dtype=_DEFAULT_FLOAT))

    @jit
    def c_int_jnp(state: jnp.int64,
                ns   : int,
                site : int,
                pref : float = 1.0):
        pos        = ns - 1 - site
        bit        = _bit_jnp(state, pos)

        sign       = f_parity_int_jnp(state, ns, site)
        new_state  = lax.select(bit == 0, state, _flip_jnp(state, pos))
        coeff_val  = lax.select(bit == 0, 0.0, sign * pref)

        return (jnp.array([new_state], dtype=_DEFAULT_INT),
                jnp.array([coeff_val], dtype=_DEFAULT_FLOAT))

    @jit
    def c_dag_jnp(state: jnp.ndarray,
                site : int,
                pref : float = 1.0):
        bit        = state[site]
        sign       = f_parity_np_jnp(state, site)

        new_state  = lax.cond(bit == 1,
                            lambda s: s,
                            lambda s: s.at[site].set(1), state)

        coeff_val  = lax.select(bit == 1, 0.0, sign * pref)
        return new_state, coeff_val

    @jit
    def c_jnp(state: jnp.ndarray,
            site : int,
            pref : float = 1.0):
        bit        = state[site]
        sign       = f_parity_np_jnp(state, site)

        new_state  = lax.cond(bit == 0,
                            lambda s: s,
                            lambda s: s.at[site].set(0), state)

        coeff_val  = lax.select(bit == 0, 0.0, sign * pref)
        return new_state, coeff_val

    # ========================================================================
    #! JAX vectorised operators
    # ========================================================================
    
    f_parity_int_vec = jax.vmap(f_parity_int_jnp, in_axes=(None, None, 0))

    @jit
    def c_k_int_jnp(state     : jnp.int64,
                    ns        : int,
                    sites     : jnp.ndarray,
                    k         : float,
                    pref      : float = 1.0):
        """
        Momentum-space annihilation  c_k  on integer basis label.
        Output arrays have length = len(sites); coeff==0 for inactive sites.
        """
        pos            = ns - 1 - sites
        bits           = _bit_jnp(state, pos)               # vectorised via broadcasting
        signs          = f_parity_int_vec(state, ns, sites)

        new_states     = state ^ (jnp.int64(1) << pos)      # flipped regardless; masked later
        coeffs         = signs * pref * jnp.exp(-1j * k * sites)
        coeffs         = coeffs * (1.0 - bits)              # zero where bit==1 is False

        norm           = jnp.sqrt(jnp.maximum(jnp.count_nonzero(coeffs), 1))
        return new_states.astype(_DEFAULT_INT), coeffs / norm


    @jit
    def c_k_dag_int_jnp(state : jnp.int64,
                        ns    : int,
                        sites : jnp.ndarray,
                        k     : float,
                        pref  : float = 1.0):
        """
        Momentum-space creation  c_k†  on integer basis label.
        """
        pos            = ns - 1 - sites
        bits           = _bit_jnp(state, pos)
        signs          = f_parity_int_vec(state, ns, sites)

        new_states     = state ^ (jnp.int64(1) << pos)
        coeffs         = signs * pref * jnp.exp(1j * k * sites)
        coeffs         = coeffs * bits                          # only where bit==1

        norm           = jnp.sqrt(jnp.maximum(jnp.count_nonzero(coeffs), 1))
        return new_states.astype(_DEFAULT_INT), coeffs / norm

    # ========================================================================
    #! JAX vectorised operators
    # ========================================================================
    
    f_parity_np_vec = jax.vmap(f_parity_np_jnp, in_axes=(None, 0))

    @jit
    def c_k_jnp(state    : jnp.ndarray,
                sites    : jnp.ndarray,
                k        : float,
                pref     : float = 1.0):
        """
        Momentum-space annihilation  c_k  on occupation array.
        """
        bits           = state[sites]
        signs          = f_parity_np_vec(state, sites)

        flipped        = state.at[sites].set(0)                          # mass flip
        new_states     = jax.vmap(lambda idx: state.at[idx].set(0))(sites)

        coeffs         = signs * pref * jnp.exp(-1j * k * sites)
        coeffs         = coeffs * bits                                   # only occupied

        norm           = jnp.sqrt(jnp.maximum(jnp.count_nonzero(coeffs), 1))
        return new_states, coeffs / norm

    @jit
    def c_k_dag_jnp(state  : jnp.ndarray,
                    sites  : jnp.ndarray,
                    k      : float,
                    pref   : float = 1.0):
        """
        Momentum-space creation  c_k†  on occupation array.
        """
        bits           = state[sites]
        signs          = f_parity_np_vec(state, sites)

        new_states     = jax.vmap(lambda idx: state.at[idx].set(1))(sites)

        coeffs         = signs * pref * jnp.exp(1j * k * sites)
        coeffs         = coeffs * (1.0 - bits)                            # only empty

        norm           = jnp.sqrt(jnp.maximum(jnp.count_nonzero(coeffs), 1))
        return new_states, coeffs / norm

    # ========================================================================
    
else:
    jax                 = None
    jnp                 = None
    lax                 = None
    jit                 = None
    c_jnp               = None
    c_dag_jnp           = None
    c_k_jnp             = None
    c_k_dag_jnp         = None
    f_parity_int_jnp    = None
    f_parity_np_jnp     = None
    f_parity_int_vec    = None
    f_parity_np_vec     = None

#============================================================================
#! End of JAX operators
#============================================================================