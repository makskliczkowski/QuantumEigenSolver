'''
file : QES/Algebra/hamil_energy_jax.py
This module contains the implementation of the Hamiltonian energy calculation
using JAX.
'''

################################################################################

import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple, Optional, List, Callable
from Algebra.hamil_energy_helper import unpack_operator_terms

#################################################################################

@jax.jit
def local_energy_arr_jax(state      : jnp.ndarray,
                        nonloc_funcs: List[Callable],
                        nonloc_sites: List,
                        nonloc_mult : List[float],
                        local_funcs : List[Callable],
                        local_sites : List,
                        local_mult  : List[float]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute non-local and local energy contributions for a given site using JAX control flow.
    
    Assumptions:
        - Each non-local operator function returns a 2D array of shape (n, state_dim)
            and a corresponding value array of shape (n,). (n is fixed for each operator - usually n = 1)
        - Each local operator returns a scalar energy contribution.
    
    ---
    Parameters:
        state           : 1D jnp.ndarray of shape (state_dim,).
        nonloc_funcs    : List of non-local operator functions.
        nonloc_sites    : List of corresponding operator arguments.
        nonloc_mult     : List of multipliers for non-local operators.
        local_funcs     : List of local operator functions.
        local_sites     : List of corresponding local operator arguments.
        local_mult      : List of multipliers for local operators.
    
    ---
    Returns:
        A tuple (site_states, site_energies) where:
            site_states     : jnp.ndarray concatenating outputs from all non-local operators and the local part.
            site_energies   : jnp.ndarray concatenating corresponding energy values.
            
    ---
    Notes:
        - the local part simply returns the original state with its aggregated local energy.
        - the non-local part returns a stack of states and energies, where each state is a row.
    """
    
    # Process non-local operators using lax.scan.
    ns = len(nonloc_funcs)
    def nonloc_scan_fn(_, i):
        # i is an index into the list of non-local operators.
        new_states, op_energy   = nonloc_funcs[i](state, nonloc_sites[i])
        # Multiply energy contributions by the corresponding multiplier.
        op_energy               = op_energy * nonloc_mult[i]
        # We assume new_states is already 2D.
        return None, (new_states, op_energy)
    _, nonloc_out = jax.lax.scan(nonloc_scan_fn, None, jnp.arange(ns, dtype=jnp.int32))
    
    # nonloc_out is a tuple (states_stack, energies_stack)
    # where states_stack has shape (ns, n, state_dim) and energies_stack has shape (ns, n).
    nonloc_states, nonloc_energies = nonloc_out
    
    # flatten over the operator axis.
    nonloc_states_flat      = nonloc_states.reshape(-1, state.shape[0])
    nonloc_energies_flat    = nonloc_energies.reshape(-1)
    
    # Process local operators by summing their contributions using lax.scan.
    num_local               = len(local_funcs)
    def local_scan_fn(carry, i):
        _, op_energy = local_funcs[i](state, local_sites[i])
        return carry + op_energy * local_mult[i], 0.0 # We return 0.0 as the second output to match the non-local scan.
    local_total, _          = jax.lax.scan(local_scan_fn, 0.0, jnp.arange(num_local))
    
    # For the local part, we add one row: the original state with the aggregated local energy.
    local_state             = state[None, :]  # shape (1, state_dim)
    local_energy            = jnp.array([local_total])
    
    # Concatenate non-local and local outputs.
    site_states             = jnp.concatenate([nonloc_states_flat, local_state], axis=0)
    site_energies           = jnp.concatenate([nonloc_energies_flat, local_energy], axis=0)
    
    return site_states, site_energies

##################################################################################

def local_energy_jax_wrap(ns                        : int,
                        operator_terms_list         : List,
                        operator_local_list         : List,
                        n_max                       : Optional[int] = 1,
                        dtype                       : Optional[jnp.dtype] = jnp.float32) -> Callable:
    '''
    Wrap the local energy computation function in a JAX JIT-compiled callable.
    This function prepares and returns a JIT-compiled function that computes local energy contributions
    based on provided operator terms and local operator terms for a given state array. It unpacks the 
    operator terms, allocates arrays for storing aggregated state updates and energies, and iterates 
    over the number of sites to compute local contributions via nested calls to lower-level energy routines.
    
    ---
    Parameters:
        ns (int):
            The number of sites (this is also handled by the unpacking function).
        operator_terms_list (List):
            A list containing operator terms functions and related data for non-local interactions
            (e.g., hopping, pairing) - each term is a tuple of (functions, list of lists of indices, list of multipliers).
        operator_local_list (List):
            A list containing operator terms functions and related data for local interactions.
            (e.g., on-site interactions) - each term is a tuple of (functions, list of lists of indices, list of multipliers).
            
            Note: May be empty.
        n_max (Optional[int], default=1):
            The maximum number of non-local operator contributions per site. By default it is one - 
            a single operator returns just a single state. It however can be multiple sites and this needs to 
            be handled accordingly (like sums of local operators etc.)
        dtype (Optional[jnp.dtype], default=jnp.float32):
            The data type used for energy calculations. It is usually taken from the Hamiltonian class.
    
    ---
    Example:
        >>> wrapped_function = local_energy_jax_wrap(ns, operator_terms_list, operator_local_list, n_max=1, dtype=jnp.float32)
    
    ---
    Returns:
        Callable:
            A JIT-compiled function that accepts a state array (jnp.ndarray) and returns a tuple:
                - jnp.ndarray: Aggregated states array.
                - jnp.ndarray: Aggregated energies array.
    '''
    # unpack the operator terms
    operator_funcs, operator_indices, operator_mult = unpack_operator_terms(ns, operator_terms_list)
    operator_local_funcs, operator_local_indices, operator_local_mult = unpack_operator_terms(ns, operator_local_list)
    
    # handle the preallocation for the arrays
    rows_per_site   = (len(operator_funcs[0]) * n_max) + 1
    total_rows      = rows_per_site * ns
    
    # create a wrapper function
    @partial(jax.jit, static_argnums=(1, 2, 3, 4, 5, 6, 7, 8, 9))
    def wrapper(state                   : jnp.ndarray,
                nonloc_funcs_unpacked   : List[List[Callable]],
                nonloc_sites_unpacked   : List[List[int]],
                nonloc_mult_unpacked    : List[float],
                local_funcs_unpacked    : List[List[Callable]],
                local_sites_unpacked    : List[List[int]],
                local_mult_unpacked     : List[float],
                ns                      : int,
                rows_per_site           : int,
                state_dim               : int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        
        # Preallocate arrays for the aggregated output.
        all_states      = jnp.zeros((total_rows, state_dim), dtype=state.dtype)
        all_energies    = jnp.zeros((total_rows,), dtype=dtype)
        
        def body_fun(i, carry):
            all_states_acc, all_energies_acc = carry
            site_states, site_energies = local_energy_arr_jax(
                state,
                nonloc_funcs_unpacked[i],
                nonloc_sites_unpacked[i],
                nonloc_mult_unpacked[i],
                local_funcs_unpacked[i],
                local_sites_unpacked[i],
                local_mult_unpacked[i]
            )
            
            start               = i * rows_per_site
            all_states_acc      = jax.lax.dynamic_update_slice(all_states_acc, site_states, (start, 0))
            all_energies_acc    = jax.lax.dynamic_update_slice(all_energies_acc, site_energies, (start,))
            return (all_states_acc, all_energies_acc)
        
        all_states, all_energies = jax.lax.fori_loop(0, ns, body_fun, (all_states, all_energies))
        return all_states, all_energies
    
    # Return the wrapper function
    def final_wrapper(state: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Final wrapper function to call the JAX compiled function.
        
        ---
        Params:
        state (jnp.ndarray): 
            Input state array.
        
        ---
        Returns:
        Tuple[jnp.ndarray, jnp.ndarray]:
            Padded states and energies.
        """
        return wrapper(
            state,
            operator_funcs,
            operator_indices,
            operator_mult,
            operator_local_funcs,
            operator_local_indices,
            operator_local_mult,
            ns,
            rows_per_site,
            state.shape[1]
        )
    
    return jax.jit(final_wrapper)

#################################################################################