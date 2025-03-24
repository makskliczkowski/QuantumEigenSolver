'''
file    : Algebra/hamil_energy.py
author  :   
'''

################################################################################

from typing import Callable, Tuple, Optional, Any
import numba.typed
import numpy as np
import numba

#################################################################################

# @numba.njit
def process_term_nonlocal_numba(k_map, sites_args, multiplier, op_func):
    '''
    Process a single non-local operator term using Numba.
    '''
    new_states, op_values = op_func(k_map, *sites_args)
    return new_states, multiplier * op_values

# @numba.njit
def process_term_local_numba(k_map, sites_args, multiplier, op_func):
    '''
    Process a single local operator term using Numba.
    '''
    _, op_values = op_func(k_map, *sites_args)
    return op_values * multiplier

#################################################################################
#! INTEGER REPRESENTATION
#################################################################################

# @numba.njit
def local_energy_int(k_map : np.int64,
                    operator_terms_func,
                    operator_terms_site,
                    operator_temrs_mult,
                    loc_operator_func,
                    loc_operator_site,
                    loc_operator_mult) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the non-local and local energy contributions for a given state map by applying a set of operator functions.
    Parameters:
        - k_map (np.int64) : Current state represented as an integer mapping.
        - operator_terms_func (iterable of function) : Functions corresponding to non-local operator terms.
        - operator_terms_site (iterable) : Site-specific arguments for the non-local operator terms.
        - operator_terms_mult (iterable) : Multipliers for the non-local operator terms.
        - loc_operator_func (iterable of function) : Functions corresponding to local operator terms.
        - loc_operator_site (iterable) : Site-specific arguments for the local operator terms.
        - loc_operator_mult (iterable) : Multipliers for the local operator terms.
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - A concatenated array of new states resulting from applying both non-local and local operators.
            - A concatenated array of corresponding energy contributions (operator values).
    Notes:
        - The function processes non-local operator terms using `process_term_nonlocal_numba`.
        - The function processes local operator terms using `process_term_local_numba`.
        - A local energy contribution is added as a state if any local operators are provided.
        - There is a potential typo in the parameter name "operator_temrs_mult" which likely should be "operator_terms_mult".
    """

    num_nonloc  = len(operator_terms_func)
    states_list = numba.typed.List()
    values_list = numba.typed.List()
    
    for ii in range(num_nonloc):
        op_func                 = operator_terms_func[ii]
        sites_args              = operator_terms_site[ii]
        multiplier              = operator_temrs_mult[ii]
        new_states, op_values   = process_term_nonlocal_numba(k_map, sites_args, multiplier, op_func)
        states_list.append(new_states)
        values_list.append(op_values)

    # locals
    num_locals                  = len(loc_operator_func)
    local_value                 = 0.0
    for ii in range(num_locals):
        op_func                 = loc_operator_func[ii]
        sites_args              = loc_operator_site[ii]
        multiplier              = loc_operator_mult[ii]
        local_value             += process_term_local_numba(k_map, sites_args, multiplier, op_func)[0]
    if num_locals > 0:
        states_list.append(np.array(k_map))
        values_list.append(np.array(local_value))
        
    # Concatenate the resulting arrays.
    if states_list:
        return np.concatenate(states_list), np.concatenate(values_list)
    else:
        return np.array([]), np.array([])

################################################################################

def local_energy_int_wrap(ns : int,
                        operator_terms_list,
                        local_operators_list):
    '''                        
    Creates a wrapper function to compute the local energy interaction for a given state 
    and site index using operator terms and local operators.
    
    Parameters
        - ns : int
            The number of sites in the system.
        - operator_terms_list : list
            A list of operator terms to be applied. If the list contains a single element, 
            it is broadcasted to all sites.
        - local_operators_list : list
            A list of local operators corresponding to each site. If the list contains a 
            single element, it is broadcasted to all sites.
        - function
            A wrapper function that takes two arguments:
                - k_map (int): The current state (e.g., an integer representing a spin basis state).
                - i (int): The site index where the operator acts locally.
            The wrapper function returns:
                - new_states (np.ndarray): Concatenated array of new states produced by each operator term.
                - new_values (np.ndarray): Concatenated array of corresponding operator values (each multiplied by its multiplier).
    ----------
    Returns
    ----------
        - wrapper : function
            A function that takes a state and site index as input and returns the new states and operator values.
    '''
    
    if len(operator_terms_list) == 1:
        operator_terms_list     = [operator_terms_list for _ in range(ns)]
        for i in range(len(operator_terms_list, ns)):
            operator_terms_list.append(([],[],[]))
            
    if len(local_operators_list) != ns:
        for i in range(len(local_operators_list, ns)):
            local_operators_list.append(([],[],[]))
    
    # unpack nonlocal
    nonlocal_func_unpacked  = [[] for _ in range(ns)]
    nonlocal_site_unpacked  = [[] for _ in range(ns)]
    nonlocal_mult_unpacked  = [[] for _ in range(ns)]
    for i in range(ns):
        nonlocal_i_size     = len(operator_terms_list[i])
        for k in range(nonlocal_i_size):
            nonlocal_func_unpacked[i].append(operator_terms_list[i][k][0])
            nonlocal_site_unpacked[i].append(operator_terms_list[i][k][1])
            nonlocal_mult_unpacked[i].append(operator_terms_list[i][k][2])
    # locals
    local_func_unpacked     = [[] for _ in range(ns)]
    local_site_unpacked     = [[] for _ in range(ns)]
    local_mult_unpacked     = [[] for _ in range(ns)]
    for i in range(ns):
        local_i_size        = len(local_operators_list[i])
        for k in range(local_i_size):
            local_func_unpacked[i].append(local_operators_list[i][k][0])
            local_site_unpacked[i].append(local_operators_list[i][k][1])
            local_mult_unpacked[i].append(local_operators_list[i][k][2])

    # @numba.njit(nopytho)
    def wrapper(k_map, i):
        """
        Wrapper function to apply operator terms to a state in an optimized manner.
        ----------
        k_map           : int
            The current state (for example, an integer representing a spin basis state).
        i               : int
            The site index for the operator terms. (i.e. the site where the operator acts locally)
        
        Returns
        -------
        new_states : np.ndarray
            Concatenated array of new states produced by each operator term.
        new_values : np.ndarray
            Concatenated array of corresponding operator values (each multiplied by its multiplier).
        """
        return local_energy_int(k_map,
                                nonlocal_func_unpacked[i],
                                nonlocal_site_unpacked[i],
                                nonlocal_mult_unpacked[i],
                                local_func_unpacked[i],
                                local_site_unpacked[i],
                                local_mult_unpacked[i])
    return wrapper

################################################################################
#! NUMPY ARRAY REPRESENTATION
################################################################################

@numba.njit
def process_term_nonlocal_arr(state         : np.ndarray,
                                sites_args  : Any,
                                multiplier  : float,
                                op_func     : Callable) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a non-local operator to the given array state.
    
    Parameters:
        state      : 1D numpy array representing the current state.
        sites_args : Site-specific argument(s) for the operator.
        multiplier : Scalar multiplier.
        op_func    : Function that takes (state, sites_args) and returns (new_state, op_value).
                    new_state may have shape (m, n) with m>=1.
    
    Returns:
        A tuple (new_state, op_value) where new_state is guaranteed to be 2D and
        op_value is scaled by the multiplier.
    """
    new_state, op_value = op_func(state, sites_args)
    if new_state.ndim == 1:
        new_state = new_state.reshape(1, new_state.shape[0])
    return new_state, multiplier * op_value

@numba.njit
def process_term_local_arr(state        : np.ndarray,
                            sites_args  : Any,
                            multiplier  : float,
                            op_func     : Callable) -> float:
    """
    Apply a local operator to the given array state.
    
    Parameters:
        state      : 1D numpy array representing the current state.
        sites_args : Site-specific argument(s) for the operator.
        multiplier : Scalar multiplier.
        op_func    : Function that takes (state, sites_args) and returns (new_state, op_value).
    
    Returns:
        The operator value (scaled by multiplier) as a float.
    """
    _, op_value = op_func(state, sites_args)
    return multiplier * op_value

@numba.njit
def local_energy_arr(state: np.ndarray,
                    operator_terms_func,
                    operator_terms_site,
                    operator_terms_mult,
                    loc_operator_func,
                    loc_operator_site,
                    loc_operator_mult) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute non-local and local energy contributions for an array state.
    
    Parameters:
        state               : 1D numpy array representing the current state.
        operator_terms_func : Typed list of functions for non-local operator terms.
        operator_terms_site : Typed list of site-specific arguments for non-local terms.
        operator_terms_mult : Typed list of multipliers for non-local terms.
        loc_operator_func   : Typed list of functions for local operator terms.
        loc_operator_site   : Typed list of site-specific arguments for local terms.
        loc_operator_mult   : Typed list of multipliers for local terms.
    
    Returns:
        Tuple (new_states, energy_values) where:
            - new_states is a 2D numpy array where each row is one state produced by any operator term.
            - energy_values is a 1D numpy array of the corresponding energy contributions.
    """
    # Create typed lists to accumulate results.
    states_list = numba.typed.List()
    values_list = numba.typed.List()
    
    num_nonloc  = len(operator_terms_func)
    
    # Process non-local operator terms.
    for i in range(num_nonloc):
        new_states_term, op_values_term = process_term_nonlocal_arr(state,
                                                                operator_terms_site[i],
                                                                operator_terms_mult[i],
                                                                operator_terms_func[i])
        # new_states_term may have several rows.
        states_list.append(new_states_term)
        # Ensure op_values_term is a 1D array.
        if op_values_term.ndim == 0:
            op_values_term = np.array([op_values_term])
        elif op_values_term.ndim == 1:
            pass
        else:
            op_values_term = op_values_term.reshape(-1)
        values_list.append(op_values_term)
    
    # Process local operator terms.
    num_locals  = len(loc_operator_func)
    local_value = 0.0
    for i in range(num_locals):
        local_value += process_term_local_arr(state,
                                            loc_operator_site[i],
                                            loc_operator_mult[i],
                                            loc_operator_func[i])
    if num_locals > 0:
        # Append the original state (as one row) and the local energy contribution.
        states_list.append(state.reshape(1, state.shape[0]))
        values_list.append(np.array([local_value]))
    
    # Concatenate the accumulated states and values.
    new_states_out      = np.concatenate(states_list, axis=0)
    energy_values_out   = np.concatenate(values_list)
    return new_states_out, energy_values_out

def local_energy_np_wrap(ns                 : int,
                    operator_terms_list     : list,
                    local_operators_list    : list):
    """
    Create a wrapper function that computes local energy interactions for all sites,
    looping over i in ns and including all operator terms.
    
    Parameters:
        ns                   : Number of sites.
        operator_terms_list  : Per-site list of tuples (op_func, sites_args, multiplier) for non-local terms.
                                If a single list is provided, it is broadcast to all sites.
        local_operators_list : Per-site list of tuples (op_func, sites_args, multiplier) for local terms.
                                If a single list is provided, it is broadcast to all sites.
    
    Returns:
        A function that takes a state (1D numpy array) and returns a tuple:
            (all_new_states, all_energy_values)
        where the results are aggregated over all sites.
    """
    # Broadcast non-local operator terms if only one set is provided.
    if len(operator_terms_list) == 1:
        operator_terms_list = operator_terms_list * ns
    # Ensure the local operators list has one entry per site.
    if len(local_operators_list) == 1:
        local_operators_list = local_operators_list * ns
    elif len(local_operators_list) < ns:
        for _ in range(ns - len(local_operators_list)):
            local_operators_list.append(([], [], []))
    
    # Unpack non-local operator components for each site.
    nonlocal_func_unpacked  = numba.typed.List()
    nonlocal_site_unpacked  = numba.typed.List()
    nonlocal_mult_unpacked  = numba.typed.List()
    for i in range(ns):
        funcs = numba.typed.List()
        sites = numba.typed.List()
        mults = numba.typed.List()
        for term in operator_terms_list[i]:
            funcs.append(term[0])
            sites.append(term[1])
            mults.append(term[2])
        nonlocal_func_unpacked.append(funcs)
        nonlocal_site_unpacked.append(sites)
        nonlocal_mult_unpacked.append(mults)
    
    # Unpack local operator components for each site.
    local_func_unpacked = numba.typed.List()
    local_site_unpacked = numba.typed.List()
    local_mult_unpacked = numba.typed.List()
    for i in range(ns):
        funcs = numba.typed.List()
        sites = numba.typed.List()
        mults = numba.typed.List()
        for term in local_operators_list[i]:
            funcs.append(term[0])
            sites.append(term[1])
            mults.append(term[2])
        local_func_unpacked.append(funcs)
        local_site_unpacked.append(sites)
        local_mult_unpacked.append(mults)
    
    @numba.njit
    def wrapper(state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Use typed lists to accumulate the per-site outputs.
        all_states_list     = numba.typed.List()
        all_energies_list   = numba.typed.List()
        
        for i in range(ns):
            new_states, energies = local_energy_arr(
                state,
                nonlocal_func_unpacked[i],
                nonlocal_site_unpacked[i],
                nonlocal_mult_unpacked[i],
                local_func_unpacked[i],
                local_site_unpacked[i],
                local_mult_unpacked[i]
            )
            all_states_list.append(new_states)
            all_energies_list.append(energies)
        
        all_new_states = np.concatenate(all_states_list, axis=0)
        all_energy_values = np.concatenate(all_energies_list)
        return all_new_states, all_energy_values
    return wrapper

################################################################################