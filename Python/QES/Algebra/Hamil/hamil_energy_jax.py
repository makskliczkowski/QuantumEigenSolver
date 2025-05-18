'''
Module      : Algebra/hamil_energy_jax.py
Author      : Maksymilian Kliczkowski
Date        : 2025-04-01
Description : This module implements functions for efficient
                Hamiltonian energy calculations using JAX.

-------------------------
This module implements functions for efficient Hamiltonian energy calculations using JAX.
It provides routines to compute local energy contributions of a quantum state by combining
various operator terms. The module distinguishes between operators that modify the state
with explicit site indices and those that do not, and it aggregates their contributions
using JAX's control flow primitives such as lax.scan, lax.switch, and lax.fori_loop.

The key functionalities include:
    - local_energy_jax_nonmod_sites:
        Computes the local energy contribution for a given state from operators that
        require site indices. It uses a loop (via lax.scan) to process each operator,
        applying corresponding multipliers and combining their contributions.
    - local_energy_jax_nonmod_nosites:
        Computes the local energy contribution for operators that do not involve site
        indices. Similar in structure to the sites version, it iterates over the provided
        operators and aggregates the energy contributions.
    - local_energy_jax_nonmod:
        Aggregates the local energy contributions by combining the results from the
        non-site and site-based operator evaluations.
    - _pad_indices_2d:
        Pads a list of index lists to form a uniform 2D array, returning both the padded
        indices and a boolean mask indicating the valid entries. This is crucial for handling
        variable-length site index lists during energy evaluation.
    - local_energy_jax_wrap:
        Provides a higher-level JIT-compiled wrapper function that prepares, unpacks, and
        flattens the operator terms. It preallocates output arrays for both the updated
        states and their corresponding energy contributions, and it loops through different
        operator groups (those modifying the state with or without sites, and those that do not
        modify the state) to update the energy and state arrays accordingly.
Usage:
    The functions in this module are designed for use within quantum eigen solvers or
    similar applications where one needs to compute and differentiate through local energy
    contributions. The module leverages JAX's support for just-in-time (JIT) compilation and
    automatic differentiation to enable fast, efficient computation on modern hardware.
Notes:
    - Operator terms should be provided with their corresponding multipliers.
    - Site-dependent operators use padded indices and masks to handle variable-length inputs.
    - The wrapper function, local_energy_jax_wrap, combines all operator contributions into a
        single callable that returns aggregated state updates and energy contributions.
        
This structured approach allows seamless integration of various Hamiltonian contributions while
ensuring optimal performance and differentiability using the JAX ecosystem.
'''

################################################################################

import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple, Optional, List, Callable, Union
from Algebra.Hamil.hamil_energy_helper import unpack_operator_terms, flatten_operator_terms

################################################################################

# ------------------------------------------------------------------------------

@partial(jax.jit, static_argnums=(1,))
def local_energy_jax_nonmod_nosites(state       : jnp.ndarray,
                                functions       : Callable,
                                multipliers     : List[Union[float, complex]]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute local energy contributions for a given state without site indices using JAX control flow.
    
    Parameters:
        state:
            1D jnp.ndarray of shape (state_dim,).
        functions:
            Local operator functions.
        multipliers:
            List of multipliers for the local operator.
    ---
    Returns:
        A tuple (local_state, local_energy) where:
            local_state (jnp.ndarray):
                jnp.ndarray of shape (1, state_dim) representing the state.
            local_energy (jnp.ndarray):
                jnp.ndarray of shape (1,) representing the energy contribution.
    """
    
    num_local               = len(functions)
    if num_local == 0:
        return jnp.array([0.0], dtype=multipliers.dtype)
    
    # Process local operators by summing their contributions using lax.scan.
    def local_scan_fn(carry, i):
        _, op_energy        = jax.lax.switch(i, functions, state)
        return carry + jnp.squeeze(op_energy * multipliers[i]), 0.0
    
    local_total, _          = jax.lax.scan(local_scan_fn, 0.0, jnp.arange(num_local))
    return jnp.array([local_total], dtype=multipliers.dtype)

# ------------------------------------------------------------------------------

@partial(jax.jit, static_argnums=(1, 3))
def local_energy_jax_nonmod(state               : jnp.ndarray,
                            functions_no_sites  : List[Callable],
                            mult_no_sites       : List[Union[float, complex]],
                            functions_sites     : List[Callable],
                            mult_sites          : List[Union[float, complex]]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute local energy contributions for a given state using JAX control flow.
    This function computes the local energy contributions for a given state using JAX control flow.
    """
    
    # Process the local operators without sites
    e1 = local_energy_jax_nonmod_nosites(state, functions_no_sites, mult_no_sites)
    # Process the local operators with sites
    e2 = local_energy_jax_nonmod_nosites(state, functions_sites, mult_sites)
    
    # Combine the local energies
    local_energy = e1 + e2
    return local_energy

################################################################################

def local_energy_jax_wrap(ns                        : int,
                        operator_terms_list         : List,
                        operator_terms_list_ns      : List,
                        operator_terms_list_nmod    : List,
                        operator_terms_list_nmod_ns : List,
                        n_max                       : Optional[int] = 1,
                        dtype                       : Optional[jnp.dtype] = jnp.float32) -> Callable:
    '''
    Wrap the local energy computation function in a JAX JIT-compiled callable.
    This callable shall only accept a state array and return the aggregated states and energies.
    
    ---
    This function prepares and returns a JIT-compiled function that computes local energy contributions
    based on provided operator terms and local operator terms for a given state array. It unpacks the 
    operator terms, allocates arrays for storing aggregated state updates and energies, and iterates 
    over the number of sites to compute local contributions via nested calls to lower-level energy routines.
    
    ---
    Parameters:
        ns (int):
            The number of sites (this is also handled by the unpacking function).
        operator_terms_list (List):
            A list of operator terms that modify the state with sites.
        operator_terms_list_ns (List):
            A list of operator terms that modify the state without sites.
        operator_terms_list_nmod (List):
            A list of operator terms that do not modify the state with sites.
        operator_terms_list_nmod_ns (List):
            A list of operator terms that do not modify the state without sites.
        n_max (int, optional):
            The maximum number of sites to consider for the local energy computation.
            Default is 1.
        dtype (jnp.dtype, optional):
            The data type for the JAX arrays. Default is jnp.float32.
        #! Note: The dtype is not used in the function, but it is passed to the JAX arrays.
            It is recommended to use jnp.float32 or jnp.float64 for numerical stability.
    
    ---
    Example:
        # Define operator terms
        operator_terms = [
            (lambda x, sites: (x + 1, 0.5), [[0, 1]], [1.0]),
            (lambda x, sites: (x - 1, 0.5), [[2, 3]], [1.0])
        ]
        # use the wrapper function
        wrapped_function = local_energy_jax_wrap(4, operator_terms, [], [], [], 1)
        # Call the wrapped function with a state
        state                       = jnp.array([1.0, 2.0, 3.0, 4.0])
        all_states, all_energies    = wrapped_function(state)
        print(all_states)
    
    Returns:
        Callable:
            A JIT-compiled function that computes local energy contributions for a given state.
            
    '''
    
    # check if dtype is jax type
    if not isinstance(dtype, jnp.dtype):
        dtype = jnp.dtype(dtype)
        
    # unpack the operator terms
    _op_f_mod_sites, _op_i_mod_sites, _op_m_mod_sites           = unpack_operator_terms(ns, operator_terms_list)
    _op_f_mod_nosites, _op_i_mod_nosites, _op_m_mod_nosites     = unpack_operator_terms(ns, operator_terms_list_ns)
    _op_f_nmod_sites, _op_i_nmod_sites, _op_m_nmod_sites        = unpack_operator_terms(ns, operator_terms_list_nmod)
    _op_f_nmod_nosites, _op_i_nmod_nosites, _op_m_nmod_nosites  = unpack_operator_terms(ns, operator_terms_list_nmod_ns)
        
    # flatten the operator terms for all the operators
    _op_f_mod_sites, _op_i_mod_sites, _op_m_mod_sites           = flatten_operator_terms(_op_f_mod_sites, _op_i_mod_sites, _op_m_mod_sites)
    _op_f_mod_nosites, _op_i_mod_nosites, _op_m_mod_nosites     = flatten_operator_terms(_op_f_mod_nosites, _op_i_mod_nosites, _op_m_mod_nosites)
    _op_f_nmod_sites, _op_i_nmod_sites, _op_m_nmod_sites        = flatten_operator_terms(_op_f_nmod_sites, _op_i_nmod_sites, _op_m_nmod_sites)
    _op_f_nmod_nosites, _op_i_nmod_nosites, _op_m_nmod_nosites  = flatten_operator_terms(_op_f_nmod_nosites, _op_i_nmod_nosites, _op_m_nmod_nosites)
    
    # check the dtype of the operator terms and cast to highest type
    
    
    # change all multipliers to jax arrays
    _op_m_mod_sites                                             = jnp.array(_op_m_mod_sites, dtype=dtype)
    _op_m_mod_nosites                                           = jnp.array(_op_m_mod_nosites, dtype=dtype)
    _op_m_nmod_sites                                            = jnp.array(_op_m_nmod_sites, dtype=dtype)
    _op_m_nmod_nosites                                          = jnp.array(_op_m_nmod_nosites, dtype=dtype)
    
    # handle the preallocation for the arrays
    total_rows_sites                                            = len(_op_f_mod_sites) * n_max
    total_rows_nosites                                          = len(_op_f_mod_nosites) * n_max
    total_rows                                                  = total_rows_sites + total_rows_nosites + 1 # +1 for the original state
    
    # Convert functions to tuples
    def operator_wrap(f, sites):
        static_sites = jnp.asarray(sites, dtype=jnp.int32)
        
        @jax.jit
        def wrapped_operator(state):
            return f(state, *static_sites)
        return wrapped_operator
    
    _op_f_mod_sites_wrap                                        = tuple(operator_wrap(_op_f_mod_sites[i], _op_i_mod_sites[i]) for i in range(len(_op_f_mod_sites)))
    _op_f_nmod_sites_wrap                                       = tuple(operator_wrap(_op_f_nmod_sites[i], _op_i_nmod_sites[i]) for i in range(len(_op_f_nmod_sites)))
    _op_f_mod_nosites                                           = tuple(_op_f_mod_nosites)
    _op_f_nmod_nosites                                          = tuple(_op_f_nmod_nosites)
    
    # create a wrapper function
    def init_wrapper(
                # modifies with sites
                _op_f_mod_sites_                : List[Callable],
                _op_m_mod_sites_                : List[Union[float, complex]],
                # modifies without sites
                _op_f_mod_nosites_              : List[Callable],
                _op_m_mod_nosites_              : List[Union[float, complex]],
                # not modifies with sites
                _op_f_nmod_sites_               : List[Callable],
                _op_m_nmod_sites_               : List[Union[float, complex]],
                # not modifies without sites
                _op_f_nmod_nosites_             : List[Callable],
                _op_m_nmod_nosites_             : List[Union[float, complex]]) -> Tuple[jnp.ndarray, jnp.ndarray]:
                
        @jax.jit
        def wrapper(state: jnp.ndarray):
            
            state_dim           = state.shape[0]

            # Preallocate arrays for the aggregated states and energies
            all_states          = jnp.zeros((total_rows, state_dim), dtype=state.dtype)
            all_energies        = jnp.zeros((total_rows,), dtype=dtype)
            
            # handle the diagnal state
            diagonal_energy     = local_energy_jax_nonmod(
                                    state,
                                    _op_f_nmod_nosites_,
                                    _op_m_nmod_nosites_,
                                    _op_f_nmod_sites_,
                                    _op_m_nmod_sites_).reshape((1,))
            all_states          = jax.lax.dynamic_update_slice(all_states, state.reshape(1, -1), (0, 0))
            all_energies        = jax.lax.dynamic_update_slice(all_energies, diagonal_energy, (0,))
            # #! Modifies the states and accepts sites
            def body_fun_op(_op_f, _op_m):
                def body_fun(i, carry):
                    all_states_acc, all_energies_acc, start = carry
                    # use jax.lax.switch
                    site_states, site_energies  = jax.lax.switch(i, _op_f, state)
                    site_energies               = jnp.reshape(site_energies, (-1,)) * _op_m[i]
                    site_energies               = site_energies.astype(all_energies_acc.dtype)
                    # site_energies               = (site_energies * _op_m[i]).astype(all_energies_acc.dtype)
                    
                    # update the state and energy arrays
                    end_idx                     = start + site_energies.shape[0]
                    all_states_acc              = jax.lax.dynamic_update_slice(all_states_acc, site_states.reshape(1, state_dim), (start, 0))
                    all_energies_acc            = jax.lax.dynamic_update_slice(all_energies_acc, site_energies, (start,))
                    return (all_states_acc, all_energies_acc, end_idx)
                return body_fun
            
            # Initialize the starting row index after the diagonal element (row 0)
            start_row_after_diag    = 1
            num_mod_sites           = len(_op_f_mod_sites)
            if num_mod_sites > 0:
                # Run the loop, the final carry contains the updated arrays and the next available row index
                all_states, all_energies, next_start_row = jax.lax.fori_loop(
                                                                0,
                                                                num_mod_sites,
                                                                body_fun_op(_op_f_mod_sites_, _op_m_mod_sites_),
                                                                (all_states, all_energies, start_row_after_diag))
            else:
                next_start_row = start_row_after_diag # No rows were added
            
            num_mod_nosites         = len(_op_f_mod_nosites)
            if num_mod_nosites > 0:
                # Run the loop, the final carry contains the updated arrays and the next available row index
                all_states, all_energies, final_start_row = jax.lax.fori_loop(
                                                                    0,
                                                                    num_mod_nosites,
                                                                    body_fun_op(_op_f_mod_nosites_, _op_m_mod_nosites_),
                                                                    (all_states, all_energies, next_start_row))
            else:
                final_start_row = next_start_row
            return all_states, all_energies, final_start_row
        return wrapper
    
    initial_wrapper = init_wrapper(
        _op_f_mod_sites_wrap, _op_m_mod_sites,
        _op_f_mod_nosites, _op_m_mod_nosites,
        _op_f_nmod_sites_wrap, _op_m_nmod_sites,
        _op_f_nmod_nosites, _op_m_nmod_nosites)
    
    # Create a wrapper function that accepts the state and calls the initial wrapper
    @jax.jit
    def final_wrapper(state: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        all_states, all_energies, final_row = initial_wrapper(state)
        return all_states, all_energies
    
    return final_wrapper

################################################################################