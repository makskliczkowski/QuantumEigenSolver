'''
file    : Algebra/hamil_energy.py
author  :   
'''

################################################################################

import numba
import numpy as np
import numba.typed

from general_python.algebra.utils import _JAX_AVAILABLE
from typing import Callable, Tuple, Optional, Any, List
from Algebra.hamil_energy_helper import unpack_operator_terms, flatten_operator_terms

if _JAX_AVAILABLE:
    from Algebra.hamil_energy_jax import local_energy_jax_wrap

#################################################################################
#! INTEGER REPRESENTATION
#################################################################################

# @numba.njit
def local_energy_int(k_map              : np.int64,
                    _op_f_mod_sites     : Tuple[Callable],
                    _op_i_mod_sites     : Tuple[List[int]],
                    _op_m_mod_sites     : Tuple[float],
                    _op_f_mod_nosites   : Tuple[Callable],
                    _op_m_mod_nosites   : Tuple[float],
                    _op_f_nmod_sites    : Tuple[Callable],
                    _op_i_nmod_sites    : Tuple[List[int]],
                    _op_m_nmod_sites    : Tuple[float],
                    _op_f_nmod_nosites  : Tuple[Callable],
                    _op_m_nmod_nosites  : Tuple[float],
                    dtype               : float) -> Tuple[np.ndarray, np.ndarray]:

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

    num_mod_sites               = len(_op_f_mod_sites) if _op_f_mod_sites is not None else 0
    num_mod_nosites             = len(_op_f_mod_nosites) if _op_f_mod_nosites is not None else 0
    num_nmod_sites              = len(_op_f_nmod_sites) if _op_f_nmod_sites is not None else 0
    num_nmod_nosites            = len(_op_f_nmod_nosites) if _op_f_nmod_nosites is not None else 0
    
    states_list                 = numba.typed.List.empty_list(numba.types.Array(numba.types.int64, 1, 'A'))
    if dtype == np.float32:
        values_list             = numba.typed.List.empty_list(numba.types.float32[:])
    elif dtype == np.float64:
        values_list             = numba.typed.List.empty_list(numba.types.float64[:])
    elif dtype == np.complex64:
        values_list             = numba.typed.List.empty_list(numba.types.complex64[:])
    elif dtype == np.complex128:
        values_list             = numba.typed.List.empty_list(numba.types.complex128[:])
    else:
        raise ValueError(f"Unsupported dtype: {dtype}. Supported types are float32, float64, complex64, and complex128.")
    
    if num_mod_sites > 0:
        for ii in range(num_mod_sites):
            op_func                 = _op_f_mod_sites[ii]
            sites_args              = _op_i_mod_sites[ii]
            multiplier              = _op_m_mod_sites[ii]
            new_states, op_values   = op_func(k_map, *sites_args)
            states_list.append(new_states)
            values_list.append(op_values * multiplier)
    
    if num_mod_nosites > 0:
        for ii in range(num_mod_nosites):
            op_func                 = _op_f_mod_nosites[ii]
            multiplier              = _op_m_mod_nosites[ii]
            new_states, op_values   = op_func(k_map)
            states_list.append(new_states)
            values_list.append(op_values * multiplier)
        
    # Non-modifying operators
    value = np.zeros((1,), dtype=dtype)
    if num_nmod_sites > 0:
        for ii in range(num_nmod_sites):
            op_func             = _op_f_nmod_sites[ii]
            sites_args          = _op_i_nmod_sites[ii]
            multiplier          = _op_m_nmod_sites[ii]
            _, op_value         = op_func(k_map, *sites_args)
            op_value            = multiplier * op_value
            op_value            = np.asarray(op_value, dtype=dtype)
            value               += np.sum(op_value)
    
    if num_nmod_nosites > 0:
        for ii in range(num_nmod_nosites):
            op_func             = _op_f_nmod_nosites[ii]
            multiplier          = _op_m_nmod_nosites[ii]
            _, op_value         = op_func(k_map)
            op_value            = multiplier * op_value
            op_value            = np.asarray(op_value, dtype=dtype)
            value               += np.sum(op_value)
    # Append the local energy contribution to the states and values lists.
    if value.shape[0] > 0:
        states_list.append(np.array([k_map], dtype=np.int64))
        values_list.append(value)
        
    # Concatenate the resulting arrays.
    if states_list:
        return np.concatenate(states_list), np.concatenate(values_list)
    else:
        return np.array([]), np.array([])

################################################################################

def local_energy_int_wrap(ns            : int,
                        _op_mod_sites   : List,
                        _op_mod_nosites : List,
                        _op_nmod_sites  : List,
                        _op_nmod_nosites: List,
                        dtype           : Optional[np.dtype] = np.float32) -> Callable:
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
    # Unpack the nonlocal and local operators
    _op_f_mod_sites, _op_i_mod_sites, _op_m_mod_sites           = unpack_operator_terms(ns, _op_mod_sites)
    _op_f_mod_nosites, _, _op_m_mod_nosites                     = unpack_operator_terms(ns, _op_mod_nosites)
    _op_f_nmod_sites, _op_i_nmod_sites, _op_m_nmod_sites        = unpack_operator_terms(ns, _op_nmod_sites)
    _op_f_nmod_nosites, _, _op_m_nmod_nosites                   = unpack_operator_terms(ns, _op_nmod_nosites)
    
    # transform the multipliers to numpy arrays
    _op_m_mod_sites_nb = numba.typed.List()
    for m in _op_m_mod_sites:
        _op_m_mod_sites_nb.append(np.asarray(m, dtype=dtype).flatten())
    _op_m_mod_nosites_nb = numba.typed.List()
    for m in _op_m_mod_nosites:
        _op_m_mod_nosites_nb.append(np.asarray(m, dtype=dtype).flatten())
    _op_m_nmod_sites_nb = numba.typed.List()
    for m in _op_m_nmod_sites:
        _op_m_nmod_sites_nb.append(np.asarray(m, dtype=dtype).flatten())
    _op_m_nmod_nosites_nb = numba.typed.List()
    for m in _op_m_nmod_nosites:
        _op_m_nmod_nosites_nb.append(np.asarray(m, dtype=dtype).flatten())
    
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
        
        # get the local lists
        op_m_mod_sites_i      = _op_m_mod_sites_nb[i]
        op_f_mod_sites_i      = _op_f_mod_sites[i] if hasattr(op_m_mod_sites_i, '__len__') else None
        op_i_mod_sites_i      = _op_i_mod_sites[i] if op_f_mod_sites_i is not None else None
        
        op_m_mod_nosites_i    = _op_m_mod_nosites_nb[i]
        op_f_mod_nosites_i    = _op_f_mod_nosites[i] if hasattr(op_m_mod_nosites_i, '__len__') else None
        
        op_m_nmod_sites_i     = _op_m_nmod_sites_nb[i]
        op_f_nmod_sites_i     = _op_f_nmod_sites[i] if hasattr(op_m_nmod_sites_i, '__len__') else None
        op_i_nmod_sites_i     = _op_i_nmod_sites[i] if op_f_mod_sites_i is not None else None
        
        op_m_nmod_nosites_i   = _op_m_nmod_nosites_nb[i]
        op_f_nmod_nosites_i   = _op_f_nmod_nosites[i] if hasattr(op_m_nmod_nosites_i, '__len__') else None
        
        # Define an inner function that takes all parameters as explicit arguments.
        # @numba.njit
        def wrapper_in(
                    k_map,
                    _op_f_mod_sites_i, _op_i_mod_sites_i, _op_m_mod_sites_i,
                    _op_f_mod_nosites_i, _op_m_mod_nosites_i,
                    _op_f_nmod_sites_i, _op_i_nmod_sites_i, _op_m_nmod_sites_i,
                    _op_f_nmod_nosites_i, _op_m_nmod_nosites_i, _dtype):
            return local_energy_int(k_map,
                                    _op_f_mod_sites_i,
                                    _op_i_mod_sites_i,
                                    _op_m_mod_sites_i,
                                    _op_f_mod_nosites_i,
                                    _op_m_mod_nosites_i,
                                    _op_f_nmod_sites_i,
                                    _op_i_nmod_sites_i,
                                    _op_m_nmod_sites_i,
                                    _op_f_nmod_nosites_i,
                                    _op_m_nmod_nosites_i,
                                    _dtype)
        # Call the njitted function passing all required arguments.
        return wrapper_in(k_map,
                        op_f_mod_sites_i, op_i_mod_sites_i, op_m_mod_sites_i,
                        op_f_mod_nosites_i, op_m_mod_nosites_i,
                        op_f_nmod_sites_i, op_i_nmod_sites_i, op_m_nmod_sites_i,
                        op_f_nmod_nosites_i, op_m_nmod_nosites_i, dtype)
    
    return wrapper

################################################################################
#! NUMPY ARRAY REPRESENTATION
################################################################################

# @numba.njit
def process_mod_sites(state         : np.ndarray,
                    sites_args      : Any,
                    multiplier      : float,
                    op_func         : Callable,
                    dtype           : np.dtype) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a state-modyfing operator to the given array state. 
    This implementation needs sites to be passed as a list.
    
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
    if op_func is None:
        return state.reshape(1, -1), np.zeros((1,), dtype=dtype)
    
    new_state, op_value = op_func(state, sites_args)

    if new_state.ndim == 1:
        new_state_2d = new_state.reshape(1, new_state.shape[0])
    else:
        new_state_2d = new_state.reshape(new_state.shape[0], -1)

    # Force op_value to be at least a 1D array.
    num_states  = new_state_2d.shape[0]
    op_value    = np.atleast_1d(np.asarray(op_value).astype(dtype))
    op_value_1d = op_value * multiplier  # Multiply elementwise
    
    # If op_value has a single element but multiple states exist, broadcast it.
    if op_value_1d.shape[0] == 1 and num_states > 1:
        op_values_final = np.full(num_states, op_value_1d[0], dtype=dtype)
    elif op_value_1d.shape[0] == num_states:
        op_values_final = op_value_1d
    else:
        raise ValueError("op_value length does not match number of states in process_mod_nosites")
    
    return new_state_2d, op_values_final.astype(dtype)

# @numba.njit
def process_mod_nosites(state       : np.ndarray,
                        multiplier  : float,
                        op_func     : Callable,
                        dtype       : np.dtype) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a state-modyfing operator to the given array state. 
    This implementation does not require sites to be passed as a list.
    
    Parameters:
        state      : 1D numpy array representing the current state.
        multiplier : Scalar multiplier.
        op_func    : Function that takes (state) and returns (new_state, op_value).
                    new_state may have shape (m, n) with m>=1.
    
    Returns:
        A tuple (new_state, op_value) where new_state is guaranteed to be 2D and
        op_value is scaled by the multiplier.
    """
    if op_func is None:
        return state.reshape(1, -1), np.zeros((1,), dtype=dtype)
    
    new_state, op_value = op_func(state)
    
    if new_state.ndim == 1:
        new_state_2d = new_state.reshape(1, new_state.shape[0])
    else:
        new_state_2d = new_state.reshape(new_state.shape[0], -1)

    # Force op_value to be at least a 1D array.
    num_states  = new_state_2d.shape[0]
    op_value    = np.atleast_1d(np.asarray(op_value).astype(dtype))
    op_value_1d = op_value * multiplier  # Multiply elementwise
    
    # If op_value has a single element but multiple states exist, broadcast it.
    if op_value_1d.shape[0] == 1 and num_states > 1:
        op_values_final = np.full(num_states, op_value_1d[0], dtype=dtype)
    elif op_value_1d.shape[0] == num_states:
        op_values_final = op_value_1d
    else:
        raise ValueError("op_value length does not match number of states in process_mod_nosites")
    
    return new_state_2d, op_values_final.astype(dtype)

# @numba.njit
def process_nmod_sites(state        : np.ndarray,
                    sites_flat      : List[np.ndarray],
                    multipliers     : np.ndarray,
                    op_funcs        : List[Callable],
                    dtype           : np.dtype) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a non-modifying operator to the given array state.
    This implementation needs sites to be passed as a list.

    Parameters:
        state          : 1D numpy array representing the current state.
        sites_flat     : List of site-specific arguments for the operator - this is a list of arrays.
        multipliers    : Scalar multipliers - this is a list of floats or complex numbers.
        op_funcs       : List of functions that take (state, sites_flat) and return (new_state, op_value).
                        new_state may have shape (m, n) with m>=1.
    
    Returns:
        A tuple (new_state, op_value) where new_state is guaranteed to be 2D and
        op_value is scaled by the multipliers.
    """
    
    value   = np.zeros((1,), dtype=dtype)
    if op_funcs is None:
        return value
    
    numfunc = multipliers.shape[0]
    # for ii in numba.prange(numfunc):
    for ii in range(numfunc):
        op_func             = op_funcs[ii]
        sites               = sites_flat[ii]
        multiplier          = multipliers[ii]
        _, op_value         = op_func(state, sites)
        op_value            = multiplier * op_value
        op_value            = np.asarray(op_value, dtype=dtype)
        value               += np.sum(op_value)
    return value
    
# @numba.njit
def process_nmod_nosites(state      : np.ndarray,
                        multipliers : np.ndarray,
                        op_funcs    : List[Callable],
                        dtype       : np.dtype) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a non-modifying operator to the given array state.
    This implementation does not require sites to be passed as a list.
    Parameters:
        state          : 1D numpy array representing the current state.
        multipliers    : Scalar multipliers - this is a list of floats or complex numbers.
        op_funcs       : List of functions that take (state) and return (new_state, op_value).
                        new_state may have shape (m, n) with m>=1.
                        
    Returns:
        A tuple (new_state, op_value) where new_state is guaranteed to be 2D and
        op_value is scaled by the multipliers.
    """
    
    value   = np.zeros((1,), dtype=dtype)
    if op_funcs is None:
        return value
    
    numfunc = multipliers.shape[0]
    # for ii in numba.prange(numfunc):
    for ii in range(numfunc):
        op_func             =   op_funcs[ii]
        multiplier          =   multipliers[ii]
        _, op_value         =   op_func(state)
        op_value            =   multiplier * op_value
        op_value            =   np.asarray(op_value, dtype=dtype)
        value               +=  np.sum(op_value)
    return value

################################################################################

def local_energy_np_wrap(ns                         : int,
                        operator_terms_list         : List,
                        operator_terms_list_ns      : List,
                        operator_terms_list_nmod    : List,
                        operator_terms_list_nmod_ns : List,
                        n_max                       : Optional[int] = 1,
                        dtype                       : Optional[np.dtype] = np.float32) -> Callable:
    """
    Generates a JIT-compiled wrapper function for computing the local energy.
    
    This function processes various operator term lists by unpacking and flattening them, converting the multiplier
    lists to numpy arrays of the specified dtype, and calculating auxiliary parameters required for energy evaluation.
    It then defines and returns a Numba nopython-mode (njit) compiled wrapper function that executes the local energy
    calculation using the prepared data.
    
    Parameters:
        ns (int):
            The number of sites or degrees of freedom.
        operator_terms_list (List):
            List of operator terms for local operators acting on sites.
        operator_terms_list_ns (List):
            List of operator terms for non-local operators acting on sites.
        operator_terms_list_nmod (List):
            List of operator terms for local operators acting on a modified layout.
        operator_terms_list_nmod_ns (List):
            List of operator terms for non-local operators acting on a modified layout.
        n_max (Optional[int], default=1):
            The maximum number of local modifications (how many states can a single operator produce).
        dtype (Optional[np.dtype], default=np.float32):
            The numpy data type to be used for numerical arrays.
    Returns:
        Callable: A Numba JIT-compiled function that takes a numpy array `state` as input and returns a tuple of 
                    numpy arrays (typically representing computed local energies and additional computed metrics).
    Notes:
        - The returned function is optimized using Numba's nopython mode for improved performance.
        - The function internally computes the total number of operator rows (for both local and non-local operators)
            by multiplying the number of terms by `n_max`.
        - This wrapper function interfaces with `local_energy_arr`, passing all necessary preprocessed operator data.
    """

    if not isinstance(dtype, np.dtype):
        dtype = np.dtype(dtype)
        
    # unpack the operator terms
    _op_f_mod_sites, _op_i_mod_sites, _op_m_mod_sites           = unpack_operator_terms(ns, operator_terms_list)
    _op_f_mod_nosites, _op_i_mod_nosites, _op_m_mod_nosites     = unpack_operator_terms(ns, operator_terms_list_ns)
    _op_f_nmod_sites, _op_i_nmod_sites, _op_m_nmod_sites        = unpack_operator_terms(ns, operator_terms_list_nmod)
    _op_f_nmod_nosites, _op_i_nmod_nosites, _op_m_nmod_nosites  = unpack_operator_terms(ns, operator_terms_list_nmod_ns)
        
    # flatten the operator terms for all the operators
    _op_f_mod_sites, _op_i_mod_sites, _op_m_mod_sites           = flatten_operator_terms(_op_f_mod_sites, _op_i_mod_sites, _op_m_mod_sites)
    _op_f_mod_nosites, _, _op_m_mod_nosites                     = flatten_operator_terms(_op_f_mod_nosites, _op_i_mod_nosites, _op_m_mod_nosites)
    _op_f_nmod_sites, _op_i_nmod_sites, _op_m_nmod_sites        = flatten_operator_terms(_op_f_nmod_sites, _op_i_nmod_sites, _op_m_nmod_sites)
    _op_f_nmod_nosites, _, _op_m_nmod_nosites                   = flatten_operator_terms(_op_f_nmod_nosites, _op_i_nmod_nosites, _op_m_nmod_nosites)
    
    # change all multipliers to numpy arrays
    _op_m_mod_sites_np                                          = np.array(_op_m_mod_sites, dtype=dtype)
    _op_m_mod_nosites_np                                        = np.array(_op_m_mod_nosites, dtype=dtype)
    _op_m_nmod_sites_np                                         = np.array(_op_m_nmod_sites, dtype=dtype)
    _op_m_nmod_nosites_np                                       = np.array(_op_m_nmod_nosites, dtype=dtype)
    
    # calculate the number of non-local and local operators
    total_rows_sites                                            = len(_op_f_mod_sites) * n_max
    total_rows_nosites                                          = len(_op_f_mod_nosites) * n_max
    
    # convert the functions to numba lists
    _op_f_mod_sites_py      = tuple(_op_f_mod_sites) if len(_op_f_mod_sites) > 0 else None
    _op_f_mod_nosites_py    = tuple(_op_f_mod_nosites) if len(_op_f_mod_nosites) > 0 else None
    _op_f_nmod_sites_py     = tuple(_op_f_nmod_sites) if len(_op_f_nmod_sites) > 0 else None
    _op_f_nmod_nosites_py   = tuple(_op_f_nmod_nosites) if len(_op_f_nmod_nosites) > 0 else None
    
    # convert the sites to numba lists
    int32_array_type        = numba.types.Array(numba.types.int32, 1, 'A')        # 1D, contiguous
    _op_i_mod_sites_nb      = numba.typed.List.empty_list(int32_array_type)
    if _op_i_mod_sites:
        for s in _op_i_mod_sites:
            _op_i_mod_sites_nb.append(np.array(s, dtype=np.int32))

    _op_i_nmod_sites_nb = numba.typed.List.empty_list(int32_array_type)
    if _op_i_nmod_sites:
        for s in _op_i_nmod_sites:
            _op_i_nmod_sites_nb.append(np.array(s, dtype=np.int32))
    
    _total_rows_mod_sites                                       = total_rows_sites
    _total_rows_mod_nosites                                     = total_rows_nosites
    _nmax                                                       = n_max
    
    #! Create the wrapper function
    # @numba.njit
    def wrapper(state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Create typed lists to accumulate results.
        state_dim                   = state.shape[0]
        
        #! start with local energy
        if _op_m_mod_sites_np.shape[0] > 0 and _op_f_mod_sites_py is not None:
            e1                      = process_nmod_sites(state, _op_i_nmod_sites_nb,
                                                            _op_m_nmod_sites_np, _op_f_nmod_sites_py, dtype)
        else:
            e1                      = np.zeros((1,), dtype=dtype)
        if _op_m_nmod_nosites_np.shape[0] > 0 and _op_f_nmod_nosites_py is not None:
            e2                      = process_nmod_nosites(state, _op_m_nmod_nosites_np,
                                                            _op_f_nmod_nosites_py, dtype)
        else:
            e2                      = np.zeros((1,), dtype=dtype)
        diagonal_value              = e1 + e2
        out_states                  = state.reshape(1, -1)
        out_values                  = diagonal_value
        
        #! continue with modifying operators
        num_mod_sites   = len(_op_m_mod_sites_np)
        if num_mod_sites > 0 and _op_f_mod_sites_py is not None:
            states_list_mod_sites       = np.zeros((_total_rows_mod_sites, state_dim), dtype=dtype)
            values_list_mod_sites       = np.zeros((_total_rows_mod_sites,), dtype=dtype)
            mask_sites                  = np.zeros((_total_rows_mod_sites,), dtype=np.bool_)
            
            # for i in numba.prange(num_mod_sites):
            for i in range(num_mod_sites):
                new_states, op_values   = process_mod_sites(state,
                                                        _op_i_mod_sites_nb[i],
                                                        _op_m_mod_sites_np[i],
                                                        _op_f_mod_sites_py[i],
                                                        dtype)
                # check the size
                new_states_size                             = new_states.shape[0]
                if new_states_size > _nmax:
                    raise ValueError(f"new_states_size ({new_states_size}) > _nmax ({_nmax})")
                start_idx                                   = i * _nmax
                end_idx                                     = i * _nmax + new_states_size
                states_list_mod_sites[start_idx:end_idx, :] = new_states
                values_list_mod_sites[start_idx:end_idx]    = op_values
                mask_sites[start_idx:end_idx]               = True

            # concatenate the results
            states_list_mod_sites       = states_list_mod_sites[mask_sites]
            values_list_mod_sites       = values_list_mod_sites[mask_sites]
            if states_list_mod_sites.shape[0] > 0:
                out_states              = np.concatenate((out_states, states_list_mod_sites), axis=0)
                out_values              = np.concatenate((out_values, values_list_mod_sites), axis=0)
    
        #! continue with non-modifying operators
        num_mod_nosites = len(_op_m_mod_nosites_np)
        if num_mod_nosites > 0 and _op_f_mod_nosites_py is not None:
            states_list_nosites       = np.zeros((_total_rows_mod_nosites, state_dim), dtype=dtype)
            values_list_nosites       = np.zeros((_total_rows_mod_nosites,), dtype=dtype)
            mask_nosites              = np.zeros((_total_rows_mod_nosites,), dtype=np.bool_)
            
            # for i in numba.prange(num_mod_nosites):
            for i in range(num_mod_nosites):
                new_states, op_values   = process_mod_nosites(state,
                                                            _op_m_mod_nosites_np[i],
                                                            _op_f_mod_nosites_py[i],
                                                            dtype)
                # check the size
                new_states_size = new_states.shape[0]
                if new_states_size > _nmax:
                    raise ValueError(f"new_states_size ({new_states_size}) > _nmax ({_nmax})")
                start_idx                                   = i * _nmax
                end_idx                                     = i * _nmax + new_states_size
                states_list_nosites[start_idx:end_idx, :]   = new_states
                values_list_nosites[start_idx:end_idx]      = op_values
                mask_nosites[start_idx:end_idx]             = True
            
            # concatenate the results
            states_list_nosites       = states_list_nosites[mask_nosites]
            values_list_nosites       = values_list_nosites[mask_nosites]
            if states_list_nosites.shape[0] > 0:
                out_states              = np.concatenate((out_states, states_list_nosites), axis=0)
                out_values              = np.concatenate((out_values, values_list_nosites), axis=0)
        return out_states, out_values
    return wrapper

################################################################################