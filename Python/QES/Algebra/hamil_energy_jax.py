'''
file : QES/Algebra/hamil_energy_jax.py
This module contains the implementation of the Hamiltonian energy calculation
using JAX.
'''

################################################################################

import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple, Optional, List, Callable, Union
from Algebra.hamil_energy_helper import unpack_operator_terms, flatten_operator_terms

#################################################################################

@partial(jax.jit, static_argnums=(1,))
def local_energy_jax_nonmod_sites(state         : jnp.ndarray,
                                functions       : Callable,
                                sites_pad       : jnp.ndarray,
                                sites_mask      : jnp.ndarray,
                                multipliers     : List[Union[float, complex]]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute local energy contributions for a given state using JAX control flow.
    
    Parameters:
        state:
            1D jnp.ndarray of shape (state_dim,).
        functions:
            Local operator functions.
        sites_pad:
            Padded indices for the local operator functions.
        sites_mask:
            Mask for the padded indices.
        multiplier:
            Multiplier for the local operator.
    ---
    Returns:
        A tuple (site_states, site_energies) where:
            site_states (jnp.ndarray):
                jnp.ndarray of shape (1, state_dim) representing the state.
            site_energies (jnp.ndarray):
                jnp.ndarray of shape (1,) representing the energy contribution.
    """
    
    # Process local operators by summing their contributions using lax.scan.
    num_local               = len(functions)
    
    if num_local == 0:
        return jnp.array([0.0])
    
    def local_scan_fn(carry, i):
        sites_valid         = sites_pad[i][sites_mask[i]]
        operands            = (state, sites_valid)
        _, op_energy        = jax.lax.switch(i, functions, operands)
        # _, op_energy = functions[i](state, sites[i])
        term_contribution   = jnp.squeeze(op_energy * multipliers[i])
        return carry + term_contribution, 0.0
    
    local_total, _                  = jax.lax.scan(local_scan_fn, 0.0, jnp.arange(num_local))
    
    # For the local part, we add one row: the original state with the aggregated local energy.
    local_energy            = jnp.array([local_total])
    
    return local_energy

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
    def local_scan_fn(carry, i):
        # Process local operators by summing their contributions using lax.scan.
        operands            = state
        _, op_energy        = jax.lax.switch(i, functions, operands)
        # _, op_energy = functions[i](state)
        total_energy        = jnp.squeeze(op_energy * multipliers[i])
        return carry + total_energy, 0.0
    
    local_total, _          = jax.lax.scan(local_scan_fn, 0.0, jnp.arange(num_local))
    local_energy            = jnp.array([local_total])
    
    return local_energy

@partial(jax.jit, static_argnums=(1, 3))
def local_energy_jax_nonmod(state               : jnp.ndarray,
                            functions_no_sites  : List[Callable],
                            mult_no_sites       : List[Union[float, complex]],
                            functions_sites     : List[Callable],
                            sites_pad           : List[List[int]],
                            sites_mask          : List[List[bool]],
                            mult_sites          : List[Union[float, complex]]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute local energy contributions for a given state using JAX control flow.
    This function computes the local energy contributions for a given state using JAX control flow.
    """
    
    # Process the local operators without sites
    e1 = local_energy_jax_nonmod_nosites(state, functions_no_sites, mult_no_sites)
    # Process the local operators with sites
    e2 = local_energy_jax_nonmod_sites(state, functions_sites, sites_pad, sites_mask, mult_sites)
    
    # Combine the local energies
    local_energy = e1 + e2
    return local_energy

##################################################################################

def _pad_indices_2d(indices: List[List[int]], pad_value = -1.0) -> jnp.ndarray:
    """
    Pad the indices with zeros to ensure each sublist has the same length.
    
    ---
    Parameters:
        indices (List[List[int]]):
            List of lists of indices to be padded.
    
    ---
    Returns:
        jnp.ndarray:
            A 2D JAX array with padded indices.
    """
    
    if not indices:
        # If the input is empty, return an empty array.
        return jnp.array([[]], dtype=jnp.int32), jnp.array([[]], dtype=jnp.bool_)
    
    num_terms = len(indices)
    if num_terms == 0:
        # Return arrays with shape (0, 0) - represents 0 terms, 0 indices/term max length
        return  jnp.array([], dtype=jnp.int32).reshape(0, 0), \
                jnp.array([], dtype=jnp.bool_).reshape(0, 0)
    

    # Calculate max length only if there are terms
    max_idx_length = max(len(idx_list) for idx_list in indices) if indices else 0

    if max_idx_length == 0 and num_terms > 0 :
        # Handle list of empty lists e.g. [ [], [], [] ]
        # Shape (num_terms, 0)
        return  jnp.zeros((num_terms, 0), dtype=jnp.int32), \
                jnp.zeros((num_terms, 0), dtype=jnp.bool_)

    padded_data_list    = []
    mask_list           = []
    for inner in indices:
        current_length  = len(inner)
        padding_needed  = max_idx_length - current_length
        padded_inner    = inner + [pad_value] * padding_needed
        mask_inner      = [True] * current_length + [False] * padding_needed
        padded_data_list.append(padded_inner)
        mask_list.append(mask_inner)

    return  jnp.array(padded_data_list, dtype=jnp.int32), \
            jnp.array(mask_list, dtype=jnp.bool_)

# --------------------------------------------------------------------------------

def local_energy_jax_wrap(ns                        : int,
                        operator_terms_list         : List,
                        operator_terms_list_ns      : List,
                        operator_terms_list_nmod    : List,
                        operator_terms_list_nmod_ns : List,
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
    
    # check if dtype is jax type
    if not isinstance(dtype, jnp.dtype):
        dtype = jnp.dtype(dtype)
        
    # unpack the operator terms
    _op_f_mod_sites, _op_i_mod_sites, _op_m_mod_sites           = unpack_operator_terms(ns, operator_terms_list)
    _op_f_mod_nosites, _op_i_mod_nosites, _op_m_mod_nosites     = unpack_operator_terms(ns, operator_terms_list_ns)
    _op_f_nmod_sites, _op_i_nmod_sites, _op_m_nmod_sites        = unpack_operator_terms(ns, operator_terms_list_nmod)
    _op_f_nmod_nosites, _op_i_nmod_nosites, _op_m_nmod_nosites  = unpack_operator_terms(ns, operator_terms_list_nmod_ns)
        
    # flatten the operator terms that modify the states
    _op_f_mod_sites, _op_i_mod_sites, _op_m_mod_sites           = flatten_operator_terms(_op_f_mod_sites, _op_i_mod_sites, _op_m_mod_sites)
    _op_f_mod_nosites, _op_i_mod_nosites, _op_m_mod_nosites     = flatten_operator_terms(_op_f_mod_nosites, _op_i_mod_nosites, _op_m_mod_nosites)
    _op_f_nmod_sites, _op_i_nmod_sites, _op_m_nmod_sites        = flatten_operator_terms(_op_f_nmod_sites, _op_i_nmod_sites, _op_m_nmod_sites)
    _op_f_nmod_nosites, _op_i_nmod_nosites, _op_m_nmod_nosites  = flatten_operator_terms(_op_f_nmod_nosites, _op_i_nmod_nosites, _op_m_nmod_nosites)
    _op_i_mod_sites_padded, _op_i_mod_sites_mask                = _pad_indices_2d(_op_i_mod_sites)
    _op_i_nmod_sites_padded, _op_i_nmod_sites_mask              = _pad_indices_2d(_op_i_nmod_sites)
    
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
    _op_f_mod_sites                                            = tuple(_op_f_mod_sites)
    _op_f_mod_nosites                                          = tuple(_op_f_mod_nosites)
    _op_f_nmod_sites                                           = tuple(_op_f_nmod_sites)
    _op_f_nmod_nosites                                         = tuple(_op_f_nmod_nosites)
    
    # create a wrapper function
    def init_wrapper(
                # modifies with sites
                _op_f_mod_sites_                : List[Callable],
                _op_i_mod_sites_padded_          : jnp.ndarray,
                _op_i_mod_sites_mask_           : jnp.ndarray,
                _op_m_mod_sites_                : List[Union[float, complex]],
                # modifies without sites
                _op_f_mod_nosites_              : List[Callable],
                _op_m_mod_nosites_              : List[Union[float, complex]],
                # not modifies with sites
                _op_f_nmod_sites_               : List[Callable],
                _op_i_nmod_sites_padded_        : jnp.ndarray,
                _op_i_nmod_sites_mask_          : jnp.ndarray,
                _op_m_nmod_sites_               : List[Union[float, complex]],
                # not modifies without sites
                _op_f_nmod_nosites_             : List[Callable],
                _op_m_nmod_nosites_             : List[Union[float, complex]]) -> Tuple[jnp.ndarray, jnp.ndarray]:
                
        @jax.jit
        def wrapper(state: jnp.ndarray):
            
            state_dim           = state.shape[0]

            # Preallocate arrays for the aggregated states and energies
            all_states          = jnp.zeros((total_rows, state_dim), dtype=dtype)
            all_energies        = jnp.zeros((total_rows,), dtype=dtype)
            
            # handle the diagnal state
            diagonal_energy     = local_energy_jax_nonmod(
                                    state,
                                    _op_f_nmod_nosites_,
                                    _op_m_nmod_nosites_,
                                    _op_f_nmod_sites_,
                                    _op_i_nmod_sites_padded_,
                                    _op_i_nmod_sites_mask_,
                                    _op_m_nmod_sites_
                                )
            diagonal_energy     = diagonal_energy.reshape((1,))
            all_states          = jax.lax.dynamic_update_slice(all_states, state.reshape(1, -1), (0, 0))
            all_energies        = jax.lax.dynamic_update_slice(all_energies, diagonal_energy, (0,))
            
            # #! Modifies the states and accepts sites
            def body_fun_sites(i, carry):
                all_states_acc_sites, all_energies_acc_sites, start = carry
                
                # the masks for the non-local sites
                masks                   = _op_i_mod_sites_mask_[i]
                sites_in                = _op_i_mod_sites_padded_[i][masks]
                
                # use jax.lax.switch
                operands                = (state, sites_in)
                site_states, site_energies = jax.lax.switch(i, _op_f_mod_sites_, operands)
                site_energies           = site_energies * _op_m_mod_sites_[i]
                
                # Reshape update arrays to match operand rank
                site_states_reshaped  = site_states.reshape(1, state_dim)   # Shape (1, state_dim)
                site_energies_reshaped = jnp.reshape(site_energies, (1,))   # Shape (1,)
                
                # update the state and energy arrays
                start_idx               = start + 1
                all_states_acc_sites    = jax.lax.dynamic_update_slice(all_states_acc_sites, site_states_reshaped, (start_idx, 0))
                all_energies_acc_sites  = jax.lax.dynamic_update_slice(all_energies_acc_sites, site_energies_reshaped, (start_idx,))
                return (all_states_acc_sites, all_energies_acc_sites, start_idx)
            
            # Initialize the starting row index after the diagonal element (row 0)
            start_row_after_diag    = 1
            loop_carry_init_sites   = (all_states, all_energies, start_row_after_diag)
            num_mod_sites           = len(_op_f_mod_sites)
            if num_mod_sites > 0:
                # Run the loop, the final carry contains the updated arrays and the next available row index
                all_states, all_energies, next_start_row = jax.lax.fori_loop(
                                                                0,
                                                                num_mod_sites,
                                                                body_fun_sites,
                                                                loop_carry_init_sites)
            else:
                next_start_row = start_row_after_diag # No rows were added
            
            # #! Modifies the states and does not accept sites
            def body_fun_nosites(i, carry):
                all_states_acc_nosites, all_energies_acc_nosites, start = carry
                
                # use jax.lax.switch
                operands                    = state
                site_states, site_energies  = jax.lax.switch(i, _op_f_mod_nosites_, operands)
                site_energies               = site_energies * _op_m_mod_nosites_[i]
                
                # Reshape update arrays to match operand rank
                site_states_reshaped  = site_states.reshape(1, state_dim)   # Shape (1, state_dim)
                site_energies_reshaped = jnp.reshape(site_energies, (1,))   # Shape (1,)
                
                # update the state and energy arrays
                start_idx                   = start + 1
                all_states_acc_nosites      = jax.lax.dynamic_update_slice(all_states_acc_nosites, site_states_reshaped, (start_idx, 0))
                all_energies_acc_nosites    = jax.lax.dynamic_update_slice(all_energies_acc_nosites, site_energies_reshaped, (start_idx,))
                return (all_states_acc_nosites, all_energies_acc_nosites, start_idx)
            
            loop_carry_init_nosites = (all_states, all_energies, next_start_row)
            num_mod_nosites         = len(_op_f_mod_nosites)
            if num_mod_nosites > 0:
                # Run the loop, the final carry contains the updated arrays and the next available row index
                all_states, all_energies, final_start_row = jax.lax.fori_loop(
                                                                    0,
                                                                    num_mod_nosites,
                                                                    body_fun_nosites,
                                                                    loop_carry_init_nosites)
            else:
                final_start_row = next_start_row
            return all_states, all_energies, final_start_row
        return wrapper
    
    initial_wrapper = init_wrapper(
        _op_f_mod_sites,
        _op_i_mod_sites_padded,
        _op_i_mod_sites_mask,
        _op_m_mod_sites,
        _op_f_mod_nosites,
        _op_m_mod_nosites,
        _op_f_nmod_sites,
        _op_i_nmod_sites_padded,
        _op_i_nmod_sites_mask,
        _op_m_nmod_sites,
        _op_f_nmod_nosites,
        _op_m_nmod_nosites)
    
    # Create a wrapper function that accepts the state and calls the initial wrapper
    def final_wrapper(state: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Final wrapper function that calls the init_wrapper and returns the result.
        
        ---
        Parameters:
            state (jnp.ndarray):
                The input state array.
        
        ---
        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]:
                The aggregated states and energies arrays.
        """
        
        # Call the init_wrapper with the unpacked operator terms
        all_states, all_energies, final_row = initial_wrapper(state)
        
        # Return the final states and energies
        return all_states[:final_row], all_energies[:final_row]
    
    return final_wrapper

#################################################################################