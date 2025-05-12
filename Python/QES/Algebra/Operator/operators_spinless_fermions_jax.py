'''
Spinless Fermion Operators (Jordan-Wigner) with JAX
===================================================

This module implements creation and annihilation operators for spinless fermions using the Jordan-Wigner transformation, with JAX acceleration for efficient computation and automatic differentiation. The operators are provided for both integer basis labels (bit representations) and occupation number arrays.

Main Features:
--------------

- JAX-accelerated implementations of fermionic creation (`c†`) and annihilation (`c`) operators.
- Support for both integer (bitstring) and array-based state representations.
- Jordan-Wigner parity calculation for correct fermionic sign structure.
- Vectorized (batched) versions of operators for efficient application to multiple sites or states.
- Momentum-space operators (`c_k`, `c_k†`) for Fourier-transformed basis.
- All functions are JIT-compiled with JAX for high performance.

Functions:
----------

- `f_parity_int_jnp(state, ns, site)`: Computes the Jordan-Wigner parity for a given site in an integer-encoded state.
- `f_parity_np_jnp(state, site)`: Computes the Jordan-Wigner parity for a given site in an occupation array.
- `c_dag_int_jnp(state, ns, site, pref=1.0)`: Applies the creation operator to an integer-encoded state.
- `c_int_jnp(state, ns, site, pref=1.0)`: Applies the annihilation operator to an integer-encoded state.
- `c_dag_jnp(state, site, pref=1.0)`: Applies the creation operator to an occupation array.
- `c_jnp(state, site, pref=1.0)`: Applies the annihilation operator to an occupation array.
- `c_k_int_jnp(state, ns, sites, k, pref=1.0)`: Applies the momentum-space annihilation operator to an integer-encoded state.
- `c_k_dag_int_jnp(state, ns, sites, k, pref=1.0)`: Applies the momentum-space creation operator to an integer-encoded state.
- `c_k_jnp(state, sites, k, pref=1.0)`: Applies the momentum-space annihilation operator to an occupation array.
- `c_k_dag_jnp(state, sites, k, pref=1.0)`: Applies the momentum-space creation operator to an occupation array.

Notes:
------

- All JAX-based functions require JAX to be installed and available.
- If JAX is not available, all operator functions are set to `None`.
- The module relies on utility functions for bit manipulation and type definitions from `general_python.algebra.utils` and `general_python.common.binary`.
- For further documentation, see the corresponding `operators_spinless_fermions.py` file.

file    : Algebra/Operator/operators_spinless_fermions_jax.py

desc    : Spinless fermions operators (Jordan-Wigner) with JAX. For documentation
          see `operators_spinless_fermions.py`.
'''

import math
import numpy as np
import numba
from typing import List, Union, Optional, Callable, Sequence
from functools import partial

from general_python.algebra.utils import JAX_AVAILABLE, Array, DEFAULT_JP_FLOAT_TYPE, DEFAULT_JP_CPX_TYPE, DEFAULT_JP_INT_TYPE
from Algebra.Operator.operator import ensure_operator_output_shape_jax
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
    #! Jordan-Wigner parity  (-1)^{# occupied left of `site`}
    #============================================================================
    
    @jit
    def f_parity_int_jnp(state: jnp.int64, ns: int, site: int) -> jnp.float64:
        shift      = ns - site
        mask_bits  = (jnp.int64(1) << jnp.int64(shift)) - jnp.int64(1)
        parity     = _popcount_mask_jnp(state, mask_bits) & jnp.int64(1)
        return jnp.where(parity == 1, -1.0, 1.0)

    @jit
    def f_parity_np_jnp(state: jnp.ndarray, site: int) -> jnp.float32:
        """
        (-1)**( # occupied sites to the left of *site* )   for an occupation array.

        Works even if `state` was inadvertently created as float32/float64.
        """
        mask      = (jnp.arange(state.shape[0]) < site)         # bool, static shape
        occ_left  = state * mask                                # same dtype as state
        # ---- make sure we feed integers to the bit‑wise op --------------------
        occ_i32   = occ_left.astype(jnp.int32)
        cnt       = jnp.sum(occ_i32, dtype=jnp.int32)
        parity    = cnt & 1                                     # integer 0 or 1
        return jnp.where(parity == 1, -1.0, 1.0)
    
    #============================================================================
    
    @jit
    def c_dag_int_jnp(state : jnp.int64,
                    ns      : int,
                    sites   : jnp.ndarray,
                    pref    : float = 1.0):
        pos   = ns - 1 - sites
        bits  = (state >> pos) & jnp.int64(1)
        any_occ = jnp.any(bits == 1)

        def _abort(_):
            return (jnp.array([state],  dtype=_DEFAULT_INT),
                    jnp.array([0.0],    dtype=_DEFAULT_FLOAT))

        def _apply(_):
            def body(carry, site):
                st, coeff = carry
                sign      = f_parity_int_jnp(st, ns, site)
                st        = _flip_jnp(st, ns - 1 - site)
                return (st, coeff * sign), None

            (new_state, coeff), _ = lax.scan(body,
                                            init=(state, 1.0),
                                            xs=sites)
            n_sites = sites.shape[0]
            coeff  *= pref ** n_sites
            return (jnp.array([new_state], dtype=_DEFAULT_INT),
                    jnp.array([coeff],     dtype=_DEFAULT_FLOAT))
        # st, coeff = lax.cond(any_occ, _abort, _apply, operand=None)
        # return ensure_operator_output_shape_jax(st, coeff)
        return lax.cond(any_occ, _abort, _apply, operand=None)
    
    @jit
    def c_int_jnp(state : jnp.int64,
                ns      : int,
                sites   : jnp.ndarray,
                pref    : float = 1.0):
        pos         = ns - 1 - sites
        bits        = (state >> pos) & jnp.int64(1)
        any_empty   = jnp.any(bits == 0)

        def _abort(_):
            return (jnp.array([state],  dtype=_DEFAULT_INT),
                    jnp.array([0.0],    dtype=_DEFAULT_FLOAT))

        def _apply(_):
            def body(carry, site):
                st, coeff = carry
                sign      = f_parity_int_jnp(st, ns, site)
                st        = _flip_jnp(st, ns - 1 - site)
                return (st, coeff * sign), None

            (new_state, coeff), _ = lax.scan(body,
                                            init    = (state, 1.0),
                                            xs      = sites)
            n_sites = sites.shape[0]
            coeff  *= pref ** n_sites
            return (jnp.array([new_state], dtype=_DEFAULT_INT),
                    jnp.array([coeff],     dtype=_DEFAULT_FLOAT))
        # st, coeff = lax.cond(any_empty, _abort, _apply, operand=None)
        # return ensure_operator_output_shape_jax(st, coeff)
        return lax.cond(any_empty, _abort, _apply, operand=None)
    
    # @jit
    def c_dag_jnp(state: jnp.ndarray,
                  sites: jnp.ndarray,
                  pref : float = 1.0):
        """
        In-place creation on an occupation array (1 → already occupied).
        """
        bits      = jnp.take(state, sites)
        any_occ   = jnp.any(bits == 1)

        def _abort(_):
            return state, 0.0

        def _apply(_):
            def body(carry, site):
                st, coeff = carry
                sign      = f_parity_np_jnp(st, site)
                st        = st.at[site].set(1)
                return (st, coeff * sign), None

            (new_state, coeff), _ = lax.scan(body,
                                             init=(state, 1.0),
                                             xs=sites)
            n_sites = sites.shape[0]
            coeff  *= pref ** n_sites
            return new_state, coeff
        st, coeff = lax.cond(any_occ, _abort, _apply, operand=None)
        return ensure_operator_output_shape_jax(st, coeff)
        # return lax.cond(any_occ, _abort, _apply, operand=None)

    # @jit
    def c_jnp(state : jnp.ndarray,
            sites   : jnp.ndarray,
            pref    : float = 1.0):
        bits        = jnp.take(state, sites)
        any_empty   = jnp.any(bits == 0)
        def _abort(_):
            return state, 0.0

        def _apply(_):
            def body(carry, site):
                st, coeff = carry
                sign      = f_parity_np_jnp(st, site)
                st        = st.at[site].set(0)
                return (st, coeff * sign), None

            (new_state, coeff), _ = lax.scan(body,
                                            init    =   (state, 1.0),
                                            xs      =   sites)
            n_sites = sites.shape[0]
            coeff  *= pref ** n_sites
            return new_state, coeff
        
        st, coeff = lax.cond(any_empty, _abort, _apply, operand=None)
        return ensure_operator_output_shape_jax(st, coeff)
    
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

    # @jit
    def c_k_jnp(state    : jnp.ndarray,
                sites    : jnp.ndarray,
                k        : float,
                pref     : float = 1.0):
        """
        Momentum-space annihilation  c_k  on occupation array.
        """
        bits           = jnp.take(state, sites)
        signs          = f_parity_np_vec(state, sites)

        flipped        = state.at[sites].set(0)                          # mass flip
        new_states     = jax.vmap(lambda idx: state.at[idx].set(0))(sites)

        coeffs         = signs * pref * jnp.exp(-1j * k * sites)
        coeffs         = coeffs * bits                                   # only occupied

        norm           = jnp.sqrt(jnp.maximum(jnp.count_nonzero(coeffs), 1))
        return new_states, coeffs / norm

    # @jit
    def c_k_dag_jnp(state  : jnp.ndarray,
                    sites  : jnp.ndarray,
                    k      : float,
                    pref   : float = 1.0):
        """
        Momentum-space creation  c_k†  on occupation array.
        """
        bits           = jnp.take(state, sites)
        signs          = f_parity_np_vec(state, sites)

        new_states     = jax.vmap(lambda idx: state.at[idx].set(1))(sites)

        coeffs         = signs * pref * jnp.exp(1j * k * sites)
        coeffs         = coeffs * (1.0 - bits)                            # only empty

        norm           = jnp.sqrt(jnp.maximum(jnp.count_nonzero(coeffs), 1))
        return ensure_operator_output_shape_jax(new_states, coeffs / norm)
        # return new_states, coeffs / norm

    # ========================================================================
    
    @jax.jit
    def n_int_jax(state     : int | jnp.ndarray,
                ns          : int,
                sites       : Sequence[int],
                prefactor   : float = 1.0):
        """
        JAX version of the number operator acting on an **integer‑encoded** state.

        Parameters
        ----------
        state : int | jax.Array
            Basis state encoded as an integer (Jordan–Wigner binary).
        ns : int
            Total number of sites.
        sites : Sequence[int]
            Target site indices.
        prefactor : float, default 1.0
            Multiplicative scale (applied as prefactor**len(sites)).

        Returns
        -------
        out_state : jax.Array(shape=(1,), dtype=_DEFAULT_INT)
            Same integer state, wrapped in a length‑1 array.
        out_coeff : jax.Array(shape=(1,), dtype=_DEFAULT_FLOAT)
            1·prefactor**n  if **all** sites are occupied, else 0.
        """
        state_arr     = jnp.asarray(state, dtype=_DEFAULT_INT)

        # bit positions to test:  pos = ns‑1‑site
        pos           = ns - 1 - jnp.asarray(sites, dtype=_DEFAULT_INT)
        # occupancy of each requested site
        occ           = (state_arr >> pos) & 1
        coeff_val     = jnp.prod(occ, dtype=_DEFAULT_FLOAT) * (prefactor ** len(sites))

        return jnp.asarray([state_arr], dtype=_DEFAULT_INT), jnp.asarray([coeff_val],
                                                                        dtype=_DEFAULT_FLOAT)

    # @jax.jit
    def n_jax(state     : jnp.ndarray,
            sites     : jnp.ndarray,
            prefactor : float = 1.0):
        r"""
        Number operator acting on a **JAX occupation vector** (0/1, immutable).

        Parameters
        ----------
        state : jax.Array(shape=(ns,), dtype=int32 or bool)
            Occupation numbers.
        sites : jax.Array(shape=(m,), dtype=int32)
            Indices where n_i is measured.
        prefactor : float, default 1.0
            Overall scale (raised to ``m = sites.size``).

        Returns
        -------
        tuple
            (state, amplitude) with

            * **state** – unchanged input array (not copied);
            * **amplitude** – :math:`\Bigl(\prod_i \mathrm{occ}_i\Bigr)\,
            \text{prefactor}^{\,m}`.
        """
        occ_prod  = jnp.prod(state[sites], dtype=_DEFAULT_FLOAT)
        coeff_val = occ_prod * (prefactor ** sites.size)
        return ensure_operator_output_shape_jax(state, coeff_val)
        # return state, coeff_val
    
else:
    jax = jnp = lax = jit = None
    c_jnp = c_dag_jnp = c_k_jnp = c_k_dag_jnp = None
    f_parity_int_jnp = f_parity_np_jnp = f_parity_int_vec = f_parity_np_vec = None
    c_int_jnp = c_dag_int_jnp = None
    n_jax = n_int_jax = None

#============================================================================
#! End of JAX operators
#============================================================================