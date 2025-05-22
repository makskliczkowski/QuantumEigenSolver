'''
file    : Algebra/hamil_energy.py
author  :   
'''

################################################################################

import numba
import numpy as np
import numba.typed
from numba.typed import List as NList
from numba import types, int32, int64, float32, float64, complex64, complex128, typeof
from general_python.algebra.utils import JAX_AVAILABLE
from typing import Callable, Tuple, Optional, Any, List
from Algebra.Hamil.hamil_energy_helper import unpack_operator_terms, flatten_operator_terms

if JAX_AVAILABLE:
    from Algebra.Hamil.hamil_energy_jax import local_energy_jax_wrap
    import jax.numpy as jnp
else:
    local_energy_jax_wrap = None
    jax = None
    jnp = np
    
#################################################################################
#! INTEGER REPRESENTATION
#################################################################################

def _dummy_op(sites: bool, dtype_code: int): # Takes integer dtype_code
    """
    Creates a dummy operator function that returns empty arrays, useful as a placeholder.
    ...
    Parameters:
        ...
        dtype_code (int):
            The integer code for the desired data-type for the returned empty values array.
    ...
    """
    actual_dtype = _convert_dtype(dtype_code) # Use the code to get actual numpy dtype
    if sites:
        @numba.njit
        def _dummy_op_with_sites(k_map, *args):
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=actual_dtype)
        return _dummy_op_with_sites
    else:
        @numba.njit
        def _dummy_op_no_sites(k_map):
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=actual_dtype)
        return _dummy_op_no_sites

def _convert_dtype(code):
    if code == 0:
        return np.float32
    elif code == 1:
        return np.float64
    elif code == 2:
        return np.complex64
    elif code == 3:
        return np.complex128
    else:
        return np.float64

def _pad_site_lists(
    funcs:    List[List[Callable]],
    idxs:     List[List[int]],
    muls:     List[List[float]],
    dummy_fn: Callable,
) -> Tuple[List[List[Callable]], List[List[int]], List[List[float]]]:
    """
    Ensure each per-site list has at least one entry by padding
    with a dummy no-op operator (with zero multiplier).
    """
    for i in range(len(funcs)):
        if len(funcs[i]) == 0:
            funcs[i]    = [dummy_fn]
            if idxs is not None:
                idxs[i] = [[i]]
            muls[i]     = [0.0]
    return funcs, idxs, muls

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
                    code                : float) -> Tuple[np.ndarray, np.ndarray]:

    """
    Computes the non-local and local energy contributions for a given state map by applying a set of operator functions.
    Parameters:
        - k_map (np.int64) :
            Current state represented as an integer mapping.
        - operator_terms_func (iterable of function) :
            Functions corresponding to non-local operator terms.
        - operator_terms_site (iterable) :
            Site-specific arguments for the non-local operator terms.
        - operator_terms_mult (iterable) :
            Multipliers for the non-local operator terms.
        - loc_operator_func (iterable of function) :
            Functions corresponding to local operator terms.
        - loc_operator_site (iterable) :
            Site-specific arguments for the local operator terms.
        - loc_operator_mult (iterable) :
            Multipliers for the local operator terms.
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

    states_list = NList()
    values_list = NList()

    # Convert the dtype code to numpy dtype instance for array creation
    # This local_dtype will be used for arrays created inside this function
    if code == 0:
        local_dtype = np.float32
    elif code == 1:
        local_dtype = np.float64
    elif code == 2:
        local_dtype = np.complex64
    elif code == 3:
        local_dtype = np.complex128
    else:
        local_dtype = np.float64 # Default

    # Non-modifying operators
    current_value_sum   = np.zeros((1,), dtype=local_dtype)

    # Non-modifying, site-based
    for ii in range(len(_op_f_nmod_sites)):
        f           = _op_f_nmod_sites[ii]
        sites       = _op_i_nmod_sites[ii] # This is an array of site indices
        mult_arr    = _op_m_nmod_sites[ii] # This is a 1-element array [multiplier]
        
        _, val      = f(k_map, *sites) # val is an array
        if len(val) == 1:
            current_value_sum  += mult_arr[0] * val[0].astype(local_dtype) # Ensure scalar mult and val element
        elif len(val) > 1:
            current_value_sum  += mult_arr[0] * np.sum(val).astype(local_dtype) # Sum over the array

    # Non-modifying, no-site
    for ii in range(len(_op_f_nmod_nosites)):
        f           = _op_f_nmod_nosites[ii]
        mult_arr    = _op_m_nmod_nosites[ii]
        _, val      = f(k_map)
        if len(val) == 1:
            current_value_sum  += mult_arr[0] * val[0].astype(local_dtype)
            has_nondiag_contrib = True
        elif len(val) > 1:
            current_value_sum  += mult_arr[0] * np.sum(val).astype(local_dtype)

    if np.abs(current_value_sum) > 1e-10:
        states_list.append(np.array([k_map], dtype=np.int64))
        values_list.append(current_value_sum)

    # Modifying, site-based
    for ii in range(len(_op_f_mod_sites)):
        f               = _op_f_mod_sites[ii]
        sites           = _op_i_mod_sites[ii]
        mult_arr        = _op_m_mod_sites[ii]
        new_states, val = f(k_map, *sites)
        
        if len(new_states) > 0:
            if len(val) == 1:
                if np.abs(val[0]) > 1e-10:
                    states_list.append(new_states)
                    values_list.append(np.array([mult_arr[0] * val[0]], dtype=local_dtype))
            elif len(val) > 0 and len(val) == len(new_states):
                filtered_vals   = val[np.abs(val) > 1e-10]
                filtered_states = new_states[np.abs(val) > 1e-10]
                if len(filtered_states) > 0:
                    states_list.append(filtered_states)
                    values_list.append(mult_arr[0] * filtered_vals.astype(local_dtype))

    # Modifying, no-site
    for ii in range(len(_op_f_mod_nosites)):
        f               = _op_f_mod_nosites[ii]
        mult_arr        = _op_m_mod_nosites[ii]
        new_states, val = f(k_map)
        if len(new_states) > 0:
            if len(val) == 1:
                if np.abs(val[0]) > 1e-10:
                    states_list.append(new_states)
                    values_list.append(np.array([mult_arr[0] * val[0]], dtype=local_dtype))
            elif len(val) > 0 and len(val) == len(new_states):
                filtered_vals = val[np.abs(val) > 1e-10]
                filtered_states = new_states[np.abs(val) > 1e-10]
                if len(filtered_states) > 0:
                    states_list.append(filtered_states)
                    values_list.append(mult_arr[0] * filtered_vals.astype(local_dtype))

    if len(states_list) > 0:
        # Ensure all arrays in values_list are 1D for concatenation
        # This might involve reshaping if some values became scalar inadvertently
        for i in range(len(values_list)):
            if values_list[i].ndim == 0: # if it's a scalar array
                values_list[i] = values_list[i].reshape(1)
        return np.concatenate(states_list), np.concatenate(values_list)
    else:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=local_dtype)

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
    # 1) Unpack your per‐site operator metadata
    fm_sites,   im_sites,   mm_sites   = unpack_operator_terms(ns, _op_mod_sites)
    fm_nos,     _,          mm_nos     = unpack_operator_terms(ns, _op_mod_nosites)
    fnm_sites,  inm_sites,  mnm_sites  = unpack_operator_terms(ns, _op_nmod_sites)
    fnm_nos,    _,          mnm_nos    = unpack_operator_terms(ns, _op_nmod_nosites)
    
    # 1.5) Create a dummy operator function
    dtype_code_val       = {np.float32:0, np.float64:1, np.complex64:2, np.complex128:3}
    if JAX_AVAILABLE:
        dtype_code_val.update({jnp.float32:0, jnp.float64:1, jnp.complex64:2, jnp.complex128:3})
        
    dtype_code_val       = dtype_code_val[dtype]
    _dummy_op_with_sites = _dummy_op(True, dtype_code_val)
    _dummy_op_no_sites   = _dummy_op(False, dtype_code_val)
    
    # 2) Pad any empty per‐site list with a dummy no-op
    #    (you'll need to define _dummy_op_with_sites / _dummy_op_no_sites elsewhere)
    fm_sites,   im_sites,   mm_sites   = _pad_site_lists(fm_sites,   im_sites,   mm_sites,   _dummy_op_with_sites)
    fm_nos,     _,          mm_nos     = _pad_site_lists(fm_nos,     None,       mm_nos,     _dummy_op_no_sites)
    fnm_sites,  inm_sites,  mnm_sites  = _pad_site_lists(fnm_sites,  inm_sites,  mnm_sites,  _dummy_op_with_sites)
    fnm_nos,    _,          mnm_nos    = _pad_site_lists(fnm_nos,    None,       mnm_nos,    _dummy_op_no_sites)

    # 3) Flatten all four categories across *all* sites
    f_mod_sites_flat    = [f   for site in fm_sites   for f in site]
    i_mod_sites_flat    = [idx for site in im_sites   for idx in site]
    m_mod_sites_flat    = [m   for site in mm_sites   for m in site]

    f_mod_nos_flat      = [f   for site in fm_nos     for f in site]
    m_mod_nos_flat      = [m   for site in mm_nos     for m in site]

    f_nmod_sites_flat   = [f   for site in fnm_sites  for f in site]
    i_nmod_sites_flat   = [idx for site in inm_sites  for idx in site]
    m_nmod_sites_flat   = [m   for site in mnm_sites  for m in site]

    f_nmod_nos_flat     = [f   for site in fnm_nos    for f in site]
    m_nmod_nos_flat     = [m   for site in mnm_nos    for m in site]

    # 4) Convert each Python list into a numba.typed.List with fixed element‐type
    fms_nb              = []
    fmn_nb              = []
    fns_nb              = []
    fnn_nb              = []
    for f in f_mod_sites_flat:   fms_nb.append(f)
    for f in f_mod_nos_flat:     fmn_nb.append(f)
    for f in f_nmod_sites_flat:  fns_nb.append(f)
    for f in f_nmod_nos_flat:    fnn_nb.append(f)

    ims_nb              = []
    inm_nb              = []
    for idx in i_mod_sites_flat:  ims_nb.append(np.asarray(idx, dtype=np.int64))
    for idx in i_nmod_sites_flat: inm_nb.append(np.asarray(idx, dtype=np.int64))

    mms_nb              = []
    mmn_nb              = []
    mns_nb              = []
    mnn_nb              = []
    for m in m_mod_sites_flat:   mms_nb.append(np.array([m], dtype=dtype))
    for m in m_mod_nos_flat:     mmn_nb.append(np.array([m], dtype=dtype))
    for m in m_nmod_sites_flat:  mns_nb.append(np.array([m], dtype=dtype))
    for m in m_nmod_nos_flat:    mnn_nb.append(np.array([m], dtype=dtype))

    # 5) Map dtype → small int code
    code = {np.float32:0, np.float64:1, np.complex64:2, np.complex128:3}[dtype]

    # 6) Compile one flat, nopython entrypoint
    # @numba.njit
    def wrapper(k_map: int64) -> Tuple[np.ndarray, np.ndarray]:
        return local_energy_int(
            k_map,
            fms_nb,   ims_nb,   mms_nb,
            fmn_nb,   mmn_nb,
            fns_nb,   inm_nb,   mns_nb,
            fnn_nb,   mnn_nb,
            code
        )

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
    _op_f_mod_sites_py                                          = tuple(_op_f_mod_sites) if len(_op_f_mod_sites) > 0 else None
    _op_f_mod_nosites_py                                        = tuple(_op_f_mod_nosites) if len(_op_f_mod_nosites) > 0 else None
    _op_f_nmod_sites_py                                         = tuple(_op_f_nmod_sites) if len(_op_f_nmod_sites) > 0 else None
    _op_f_nmod_nosites_py                                       = tuple(_op_f_nmod_nosites) if len(_op_f_nmod_nosites) > 0 else None
    
    def wrap_fun(_op_f, _op_i):
        sites_args = np.asarray(_op_i, dtype=np.int32)
        @numba.njit
        def wrapper(state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            return _op_f(state, sites_args)
        return wrapper
    _op_f_mod_sites_py                                          = tuple(wrap_fun(f, i) for f, i in zip(_op_f_mod_sites, _op_i_mod_sites))
    _op_f_nmod_sites_py                                         = tuple(wrap_fun(f, i) for f, i in zip(_op_f_nmod_sites, _op_i_nmod_sites))
    _total_rows_mod_sites                                       = total_rows_sites
    _total_rows_mod_nosites                                     = total_rows_nosites
    _nmax                                                       = n_max
    
    #! Create the wrapper function
    # @numba.njit
    def wrapper(state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Create typed lists to accumulate results.
        state_dim                   = state.shape[0]
        has_mod_sites               = len(_op_f_mod_sites) > 0 and _op_f_mod_sites_py is not None
        has_mod_nosites             = len(_op_f_mod_nosites) > 0 and _op_f_mod_nosites_py is not None
        #! start with local energy
        
        if has_mod_sites:
            e1                      = process_nmod_nosites(state, _op_m_nmod_sites_np, _op_f_nmod_sites_py, dtype)
        else:
            e1                      = np.zeros((1,), dtype=dtype)
        if has_mod_nosites:
            e2                      = process_nmod_nosites(state, _op_m_nmod_nosites_np,
                                                            _op_f_nmod_nosites_py, dtype)
        else:
            e2                      = np.zeros((1,), dtype=dtype)
        diagonal_value              = e1 + e2
        out_states                  = state.reshape(1, -1)
        out_values                  = diagonal_value
        
        #! continue with modifying operators
        num_mod_sites = len(_op_m_mod_sites_np)
        if num_mod_sites > 0 and _op_f_mod_sites_py is not None:
            states_list_mod_sites       = np.zeros((_total_rows_mod_sites, state_dim), dtype=dtype)
            values_list_mod_sites       = np.zeros((_total_rows_mod_sites,), dtype=dtype)
            mask_sites                  = np.zeros((_total_rows_mod_sites,), dtype=np.bool_)
            
            # for i in numba.prange(num_mod_sites):
            for i in range(num_mod_sites):
                new_states, op_values   = process_mod_nosites(state, _op_m_mod_sites_np[i], _op_f_mod_sites_py[i], dtype)
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