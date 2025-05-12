'''
file: Algebra/Operator/operator_matrix.py
This file contains the functions to create the operator matrix based on the Hilbert space and specified parameters.
'''

import numpy as np
import scipy as sp
from numba import njit, objmode, jit, prange
from typing import Callable, Union

####################################################################################################
from general_python.algebra.utils import JAX_AVAILABLE, get_backend, DEFAULT_NP_INT_TYPE
if JAX_AVAILABLE:
    from general_python.algebra.utils import JIT, DEFAULT_JP_INT_TYPE
    import jax.numpy as jnp
    import jax.scipy as jsp

####################################################################################################
# from Algebra.hilbert import HilbertSpace, set_operator_elem, get_mapping, get_matrix_element
import Algebra.hilbert as Hilbert

####################################################################################################
#! Numpy operator setup

def _operator_create_np_inner_loop_no_hilbert():
    ''' 
    Inner loop to update the matrix elements - go through all basis states
    - when hilbert is not used or not modifies 
    Arguments:
        ranges (np.ndarray) : The ranges for the inner loop.
    Returns:
        _inner_loop (function): The inner loop function.
    '''
    # @njit(fastmath=True)
    def inner_k(k, local_funct, cols, rows, data, data_idx):
        # Get the new rows and data from the local function.
        new_rows, new_data                  = local_funct(k)
        num_new                             = len(new_rows)
        rows[data_idx : data_idx + num_new] = new_rows
        data[data_idx : data_idx + num_new] = new_data
        # Set the column indices for these new elements.
        cols[data_idx : data_idx + num_new] = k
        data_idx                           += num_new
        return data_idx
    # @njit(fastmath=True)
    def _inner_loop(local_funct, nh: int, cols, rows, data, data_idx):
        # Loop over all basis states.
        for k in prange(nh):
            data_idx = inner_k(k, local_funct, cols, rows, data, data_idx)
        return data_idx
    return _inner_loop

def _operator_create_np_sparse_inner_loop(hilbert : Hilbert.HilbertSpace):
    ''' Inner loop to update the matrix elements - go through all basis states '''
    
    if hilbert is not None and hilbert.modifies:
        _mapping = np.array(hilbert.mapping, dtype=DEFAULT_NP_INT_TYPE)
        _norm    = np.array(hilbert.normalization)
        _sym     = hilbert.sym_group if hilbert.sym_group is not None and len(hilbert.sym_group) > 0 else []
        
        # get the representation map if it exists
        _repr    = np.array(hilbert.reprmap, dtype=DEFAULT_NP_INT_TYPE)
        if _repr.ndim == 1:
            _repr = _repr.reshape(-1, 2)

        @njit(fastmath=True)
        def _get_mapping(k):
            return Hilbert.jitted_get_mapping(_mapping, k)

        @jit(forceobj=True, fastmath=True)
        def _get_matrix_elem(k, new_row, k_map):
            # Apply transformation and get the matrix element directly
            return Hilbert.jitted_get_matrix_element(k, new_row, k_map, False, _mapping, _norm, _sym, _repr)
        
        @njit(fastmath=True)
        def _update_data(num_new, new_rows, new_data, k, k_map, rows, data, data_idx):
            for j in range(num_new):
                (new_row, _), sym_eig = _get_matrix_elem(k, new_rows[j], k_map)
                rows[data_idx + j] = new_row
                data[data_idx + j] = new_data[j] * sym_eig
            return data_idx + num_new
        
        @njit(fastmath=True)
        def _inner_k(k, local_funct, cols, rows, data, data_idx):
            # Get the mapping for the current basis state
            k_map                               = _get_mapping(k)
            # Loop over the sites (modes).
            new_rows, new_data                  = local_funct(k_map)
            num_new                             = len(new_rows)
            data_idx                            = _update_data(num_new, new_rows, new_data, k, k_map, rows, data, data_idx)
            # Set the column indices for these new elements.
            cols[data_idx - num_new : data_idx] = k
            return data_idx
            
        # create the inner loop function
        def _inner_loop(local_funct, nh, cols, rows, data, data_idx):
            # Loop over all basis states.
            for k in range(nh):
                data_idx = _inner_k(k, local_funct, cols, rows, data, data_idx)
            return data_idx
        return _inner_loop
    return _operator_create_np_inner_loop_no_hilbert()

def operator_create_np_sparse(  ns                  : int,
                                hilbert             : Hilbert.HilbertSpace,
                                local_fun           : Union[Callable],
                                max_local_changes   : int,
                                start               = 0,
                                dtype               = None) -> sp.sparse.csr_matrix:
    """
    NumPy version: Update the matrix elements, 
    where the matrix is a sparse matrix in a numpy format (scipy.sparse.csr_matrix).

    Parameters:
        ham (np.ndarray)            : The Hamiltonian matrix (will be overwritten).
        ns (int)                    : The number of sites.
        hilbert_space (HilbertSpace): The Hilbert space.
        max_local_changes (int)     : The maximum number of local changes.
        start (int)                 : The starting index for the update (default is 0).
        dtype                       : The data type of the Hamiltonian.
        *args                       : Additional arguments for the local function - it is assumed that
                                        the local function takes the form (k, *args) -> (rows, values).
                                        This means that if the ranges are provided, the local function
                                        shall take a single argument on top of the state k (i.e. the site).  
    Returns:
        np.ndarray                  : The updated Hamiltonian matrix.    
    """
    if hilbert is None:
        raise ValueError('Hilbert space is None!')
    
    nh          = hilbert.Nh                                # The number of basis states
    if ns is not None and ns > 0:
        max_inner   = max_local_changes * (ns - start)      # The maximum number of inner loop iterations
    else:
        max_inner   = max_local_changes
        
    #! The maximum number of non-zero elements
    max_nnz     = nh * max_inner
    dtype       = dtype if dtype is not None else hilbert.dtype
        
    # Pre-allocate arrays with the estimated size
    rows        = np.zeros(max_nnz, dtype=DEFAULT_NP_INT_TYPE)
    cols        = np.zeros(max_nnz, dtype=DEFAULT_NP_INT_TYPE)
    data        = np.zeros(max_nnz, dtype=dtype)
    data_idx    = 0
    
    # Inner loop is now a separate Numba function for clarity and potential reuse
    _inner_loop = _operator_create_np_sparse_inner_loop(hilbert)

    # Call the Numba-accelerated inner loop
    data_idx    = _inner_loop(local_funct=local_fun, nh=nh, cols=cols, rows=rows, data=data, data_idx=data_idx)
    data        = data[:data_idx]
    rows        = rows[:data_idx]
    cols        = cols[:data_idx]
    # Create the sparse matrix from the collected data (outside the jitted function)
    return sp.sparse.csr_matrix((data, (rows, cols)), shape=(nh, nh))#, dtype=dtype, copy=False)

#####################################################################################################
#! Numpy DENSE operator setup
#####################################################################################################

def _operator_create_np_dense_inner_loop_no_hilbert():
    ''' Inner loop to update the matrix elements - go through all basis states
    - when hilbert is not used or not modifies 
    '''
    
    # @njit(fastmath=True)
    def _inner_k(k, local_funct, matrix):
        # Get the new rows and data from the local function.
        new_rows, new_data  = local_funct(k)
        matrix[new_rows, k] += new_data

    # @njit(fastmath=True)
    def _inner_loop(local_funct, nh, matrix):
        # Loop over all basis states.
        for k in range(nh):
            # Loop over the sites (modes).
            _inner_k(k, local_funct, matrix)
        return matrix
    return _inner_loop

def _operator_create_np_dense_inner_loop(hilbert : Hilbert.HilbertSpace):
    ''' Inner loop to update the matrix elements - go through all basis states '''
    
    if hilbert.modifies:
        _mapping = np.array(hilbert.mapping, dtype=DEFAULT_NP_INT_TYPE)
        _norm    = np.array(hilbert.normalization)
        _sym     = hilbert.sym_group if hilbert.sym_group is not None and len(hilbert.sym_group) > 0 else []
        _repr    = np.array(hilbert.reprmap, dtype=DEFAULT_NP_INT_TYPE)
        if _repr.ndim == 1:
            _repr = _repr.reshape(-1, 2)
            
        # @njit(fastmath=True)
        def _get_mapping(k):
            return Hilbert.jitted_get_mapping(_mapping, k)

        # @jit(forceobj=True, fastmath=True)
        def _get_matrix_elem(k, new_row, k_map):
            # Apply transformation and get the matrix element directly
            return Hilbert.jitted_get_matrix_element(k, new_row, k_map, False,
                    _mapping, _norm, _sym, _repr)
        
        # @njit(fastmath=True)
        def _update_data(num_new, new_rows, new_data, k, k_map, matrix):
            for j in range(num_new):
                (new_row, _), sym_eig   =   _get_matrix_elem(k, new_rows[j], k_map)
                matrix[new_row, k]      +=  new_data[j] * sym_eig

        # @njit(fastmath=True)
        def _inner_k(k, local_funct, matrix):
            # Get the mapping for the current basis state
            k_map                   = _get_mapping(k)
            # Loop over the sites (modes).
            for i in ranges:
                # Get the new rows and data from the local function.
                new_rows, new_data  = local_funct(k_map, i)
                num_new             = len(new_rows)
                _update_data(num_new, new_rows, new_data, k, k_map, matrix)
        
        # @njit(fastmath=True)
        def _inner_loop(local_funct, nh, matrix):
            # Loop over all basis states.
            for k in prange(nh):
                _inner_k(k, local_funct, matrix)
            return matrix
        return _inner_loop
    return _operator_create_np_dense_inner_loop_no_hilbert()
        
def operator_create_np_dense(ns                     : int,
                    hilbert_space                   : Hilbert.HilbertSpace,
                    local_fun                       : Callable,
                    start                           = 0,
                    dtype                           = None):
    '''
    Create an operator matrix based on the Hilbert space and specified parameters.

    Parameters:
        ns (int)                        : The number of sites.
        hilbert_space (HilbertSpace)    : The associated Hilbert space object.
        local_fun (Callable)            : Function to define local operator actions. Signature: (k_map, i) -> (rows, values)
        max_local_changes (int)         : The maximum number of allowed local updates.
        is_sparse (bool)                : Whether to create a sparse matrix (True) or dense matrix (False).
        start (int)                     : The starting index for updates (default 0).
        dtype (optional)                : Data type for the matrix (if None, uses Hilbert space dtype).

    Returns:
        Either a sparse or dense matrix representing the operator.
    '''

    if dtype is None:
        dtype = hilbert_space.dtype
    
    # Dense matrix creation (can also be optimized with Numba)
    nh          = hilbert_space.Nh
    matrix      = np.zeros((nh, nh), dtype=dtype)
    _inner_loop = _operator_create_np_dense_inner_loop(hilbert_space)
    _inner_loop(local_funct=local_fun, nh=nh, matrix=matrix)
    return matrix

#####################################################################################################

def operator_create_np( ns                  : Union[int, None],
                        hilbert_space       : Hilbert.HilbertSpace,
                        local_fun           : Union[Callable],
                        max_local_changes   : int,
                        is_sparse           : bool,
                        start               = 0,
                        dtype               = None):
    '''
    In-place operator matrix creation based on the Hilbert space and specified parameters.

    Parameters:
        ns (int)                    : The number of sites.
        hilbert_space (HilbertSpace): The associated Hilbert space object.
        local_fun (Callable)        : Function to define local operator actions.
        max_local_changes (int)     : The maximum number of allowed local updates.
        is_sparse (bool)            : Whether to create a sparse matrix (True) or dense matrix (False).
        start (int)                 : The starting index for updates (default 0).
        dtype (optional)            : Data type for the matrix (if None, uses Hilbert space dtype).        
    '''    
    # call the operator_create_np function
    if is_sparse:
        return operator_create_np_sparse(ns=ns,
                hilbert=hilbert_space, local_fun=local_fun,
                max_local_changes=max_local_changes, start=start, dtype=dtype)
    return operator_create_np_dense(ns=ns, hilbert_space=hilbert_space, local_fun=local_fun, start=start, dtype=dtype)

####################################################################################################

def operator_create(ns                  : int,
                    hilbert_space       : Hilbert.HilbertSpace,
                    local_fun           : Union[Callable],
                    max_local_changes   : int,
                    is_sparse           : bool,
                    start               = 0,
                    dtype               = None,
                    backend             = 'default'):
    '''
    In-place operator matrix creation based on the Hilbert space and specified parameters.
    
    Parameters:
        ns (int)                        : The number of sites.
        hilbert_space (HilbertSpace)    : The associated Hilbert space object.
        local_fun (Callable)            : Function to define local operator actions.
        max_local_changes (int)         : The maximum number of allowed local updates.
        is_sparse (bool)                : Whether to create a sparse matrix (True) or dense matrix (False).
        start (int)                     : The starting index for updates (default 0).
        dtype (optional)                : Data type for the matrix (if None, uses Hilbert space dtype).
        backend (str)                   : The computation backend to use (default is 'default').
    '''

    if dtype is None:
        dtype = hilbert_space.dtype

    backend = get_backend(backend)
    if backend == 'numpy':
        if is_sparse:
            return operator_create_np_sparse(ns, hilbert_space, local_fun, max_local_changes, start, dtype)
        return operator_create_np_dense(ns, hilbert_space, local_fun, start, dtype)
    else:
        raise ValueError(f'Backend {backend} not recognized.')
    
####################################################################################################