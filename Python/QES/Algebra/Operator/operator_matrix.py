'''
file: Algebra/Operator/operator_matrix.py
This file contains the functions to create the operator matrix based on the Hilbert space and specified parameters.
'''

import numpy as np
import scipy as sp
from numba import njit
from typing import Callable, Union

####################################################################################################
from general_python.algebra.utils import _JAX_AVAILABLE, get_backend, JIT, DEFAULT_NP_INT_TYPE, DEFAULT_JP_INT_TYPE
if _JAX_AVAILABLE:
    import jax.numpy as jnp
    import jax.scipy as jsp

####################################################################################################
from Algebra.hilbert import HilbertSpace, set_operator_elem
from Algebra.Operator.operator import OperatorFunction

####################################################################################################
#! Numpy operator setup

def operator_create_np_sparse(ns                    : int,
                            hilbert                 : HilbertSpace,
                            local_fun               : Union[Callable, OperatorFunction],
                            max_local_changes       : int, 
                            start                   = 0,
                            dtype                   = None) -> sp.sparse.csr_matrix:
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
    """
    nh          = hilbert.Nh                            # The number of basis states
    max_inner   = max_local_changes * (ns - start)      # The maximum number of inner loop iterations
    max_nnz     = nh * max_inner                        # Estimated maximum number of non-zero elements
    dtype       = dtype if dtype is not None else hilbert.dtype

    # Pre-allocate arrays with the estimated size
    rows        = np.empty(max_nnz, dtype=DEFAULT_NP_INT_TYPE) 
    cols        = np.empty(max_nnz, dtype=DEFAULT_NP_INT_TYPE)
    data        = np.empty(max_nnz, dtype=dtype)
    data_idx    = 0
    hilbert_mod = hilbert.modifies
    
    # Inner loop is now a separate Numba function for clarity and potential reuse
    @njit(fastmath=True)
    def _inner_loop(hilbert_getitem : Callable,
                    start   : int,
                    ns      : int,
                    nh      : int,
                    rows    : np.ndarray,
                    data    : np.ndarray,
                    data_idx: int):
        # Inner loop to update the matrix elements - go through all basis states
        for k in range(nh):
            k_map           = hilbert_getitem(k)
            # Go through all sites (modes) and update the matrix elements
            for i in range(start, ns):
                new_rows, new_data  = local_fun(k_map, i)
                
                num_new             = new_rows.size
                # transform the new rows and data
                if hilbert_mod:
                    for j in range(num_new):
                        # Apply transformation and get the matrix element directly
                        (new_row, _), sym_eig   = hilbert.get_matrix_element(k, new_rows[j], k_map)
                        rows[data_idx+j]        = new_row
                        data[data_idx+j]        = new_data[j] * sym_eig
                        data_idx                += 1
                else:
                    rows[data_idx:data_idx+num_new] = new_rows
                    data[data_idx:data_idx+num_new] = new_data
                cols[data_idx:data_idx+num_new] = k
                data_idx += num_new
        return data_idx


    # Call the Numba-accelerated inner loop
    data_idx = _inner_loop(hilbert.get_mapping,
            start, ns, nh, rows, data, data_idx)

    # Create the sparse matrix from the collected data (outside the jitted function)
    return sp.sparse.csr_matrix((data[:data_idx], (rows[:data_idx], cols[:data_idx])), shape=(nh, nh))

def operator_create_np_dense(ns                     : int,
                    hilbert_space                   : HilbertSpace,
                    local_fun                       : Callable,
                    start                           = 0,
                    dtype                           = None):
    '''
    Create an operator matrix based on the Hilbert space and specified parameters.

    Parameters:
        ns (int): The number of sites.
        hilbert_space (HilbertSpace): The associated Hilbert space object.
        local_fun (Callable): Function to define local operator actions. Signature: (k_map, i) -> (rows, values)
        max_local_changes (int): The maximum number of allowed local updates.
        is_sparse (bool): Whether to create a sparse matrix (True) or dense matrix (False).
        start (int): The starting index for updates (default 0).
        dtype (optional): Data type for the matrix (if None, uses Hilbert space dtype).

    Returns:
        Either a sparse or dense matrix representing the operator.
    '''

    if dtype is None:
        dtype = hilbert_space.dtype
        
    # Dense matrix creation (can also be optimized with Numba)
    nh      = hilbert_space.Nh
    matrix  = np.zeros((nh, nh), dtype=dtype)

    @njit(fastmath=True)
    def _fill_dense_matrix(matrix, hilbert_get_mapping, start, ns, nh):
        # go through all basis states
        for k in range(nh):
            k_map = hilbert_get_mapping(k)  # Get the basis state
            for i in range(start, ns):      # Go through all sites (modes) and update the matrix elements
                new_rows, new_data          = local_fun(k_map, i)
                for j in range(new_rows.size):
                    set_operator_elem(matrix, hilbert_space, k, new_data[j], new_rows[j])
    _fill_dense_matrix(matrix, hilbert_space.get_mapping, start, ns, nh)
    return matrix

def operator_create_np(ns                 : int,
                    hilbert_space       : HilbertSpace,
                    local_fun           : Union[Callable, OperatorFunction],
                    max_local_changes   : int,
                    is_sparse           : bool,
                    start               = 0,
                    dtype               = None):
    '''
    In-place operator matrix creation based on the Hilbert space and specified parameters.
    

    Parameters:
        ns (int): The number of sites.
        hilbert_space (HilbertSpace): The associated Hilbert space object.
        local_fun (Callable): Function to define local operator actions.
        max_local_changes (int): The maximum number of allowed local updates.
        is_sparse (bool): Whether to create a sparse matrix (True) or dense matrix (False).
        start (int): The starting index for updates (default 0).
        dtype (optional): Data type for the matrix (if None, uses Hilbert space dtype).        
    '''
    if is_sparse:
        return operator_create_np_sparse(ns, hilbert_space, local_fun, max_local_changes, start, dtype)
    return operator_create_np_dense(ns, hilbert_space, local_fun, start, dtype)

####################################################################################################

def operator_create(ns                 : int,
                    hilbert_space       : HilbertSpace, 
                    local_fun           : Union[Callable, OperatorFunction],
                    max_local_changes   : int,
                    is_sparse           : bool,
                    start               = 0,
                    dtype               = None,
                    backend             = 'default'):
    '''
    In-place operator matrix creation based on the Hilbert space and specified parameters.
    
    Parameters:
        ns (int): The number of sites.
        hilbert_space (HilbertSpace): The associated Hilbert space object.
        local_fun (Callable): Function to define local operator actions.
        max_local_changes (int): The maximum number of allowed local updates.
        is_sparse (bool): Whether to create a sparse matrix (True) or dense matrix (False).
        start (int): The starting index for updates (default 0).
        dtype (optional): Data type for the matrix (if None, uses Hilbert space dtype).
        backend (str): The computation backend to use (default is 'default').
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