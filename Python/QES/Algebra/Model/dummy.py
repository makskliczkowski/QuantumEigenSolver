'''
file:   : Model/dummy.py
author: : Maksymilian Kliczkowski
email:  : maksymilian.kliczkowski@pwr.edu.pl
This file defines a dummy Hamiltonian class for testing purposes. 

Description:
        This module implements a dummy Hamiltonian class for testing and development purposes.
        It provides a simplified implementation of quantum Hamiltonian mechanics with integer
        and array representations of quantum states. The class includes support for different
        computational backends including Numba and JAX (when available).
        
        The DummyHamiltonian class inherits from the Hamiltonian base class and implements
        a simple parametric Hamiltonian with diagonal and off-diagonal elements for 
        benchmarking and testing matrix operations within the QES framework.
'''

import numpy as np
import numba

# typing
from typing import Union, Optional

# Assume these are available from the QES package:
from Algebra.hilbert import HilbertSpace
from Algebra.hamil import Hamiltonian

##########################################################################################
import general_python.algebra.linalg as linalg
from general_python.algebra.utils import DEFAULT_NP_INT_TYPE, DEFAULT_NP_FLOAT_TYPE, _JAX_AVAILABLE
from general_python.common import binary as _binary
##########################################################################################

# ----------------------------------------------------------------------------------------
#! Integer states
# ----------------------------------------------------------------------------------------

@numba.njit
def _local_energy_int(k_map     : int,
                    i           : int,
                    param       : Union[float, complex]):
    """
    Calculate the local energy contribution from a given parameter.
    This function simulates the computation of a local energy term by generating
    matrix elements represented by row indices and corresponding values.
    Parameters
    ----------
    k_map : int
        The integer representation of a basis state.
    i : int
        The bit position to flip for the off-diagonal element.
    param : Union[float, complex]
        The parameter used to calculate the matrix elements.
    Returns
    -------
    tuple
        A tuple containing two arrays:
        - new_rows: An array of row indices (int) representing the positions in the matrix.
        - new_vals: An array of values representing the energy contributions at those positions.
        The first element of each array corresponds to the diagonal element (k_map),
        while the second corresponds to the off-diagonal element (k_map with bit i flipped).
    """
    
    new_rows    = np.empty(2, dtype=DEFAULT_NP_INT_TYPE)
    new_vals    = np.empty(2, dtype=type(param))
    
    # simulate diagonal element
    idx, val    = k_map, param / 2.0 * _binary.check_int(k_map, i)
    new_rows[0] = idx
    new_vals[0] = val
    
    # simulate off-diagonal element
    idx, val    = k_map ^ (1 << i), param**2.0
    new_rows[1] = idx
    new_vals[1] = val
    return new_rows, new_vals

def _local_energy_int_wrap(param : Union[float, complex]):
    '''
    Wrapper function to call the local energy calculation function.
    This function is used to ensure compatibility with Numba's JIT compilation.
    '''
    @numba.njit
    def _local_energy_int_numba(k_map       : int,
                                i           : int):
        return _local_energy_int(k_map, i, param)
    return _local_energy_int_numba

# ----------------------------------------------------------------------------------------
#! Array states
# ----------------------------------------------------------------------------------------

@numba.njit
def _local_energy_arr_loop(state, param: Union[float, complex]):
    """
    Calculate the local energy array using a loop-based approach.
    This function computes the local energy for a quantum state with a specific parameter.
    It iterates through each element of the state vector, calculating both diagonal and
    off-diagonal elements of the energy contribution.
    Parameters
    ----------
    state : numpy.ndarray
        The input quantum state vector.
    param : Union[float, complex]
        The parameter used for energy calculation, can be either real or complex.
    Returns
    -------
    tuple
        A tuple containing:
        - new_states : numpy.ndarray
            An array of shape (size+1, size) containing the original state and modified states,
            where each modified state has one element negated.
        - vals : numpy.ndarray
            An array of length (size+1) containing the energy values corresponding to each state.
            The last element is the diagonal energy contribution.
    Notes
    -----
    This function is optimized for parallel execution with numba.prange.
    """
    
    size        = state.shape[0]
    local_value = 0.0
    new_states  = np.empty((size+1, size), dtype=state.dtype)
    vals        = np.empty(size+1, dtype=type(param))
    
    # range through all elements in the state
    for i in numba.prange(size):
        # calculate the diagonal element
        val                 = param / 2.0 * state[i]
        local_value         += val
        
        # off-diagonal element
        new_states[i, :]    = state
        new_states[i, i]    = -state[i]
        vals[i]             = param**2.0
    
    new_states[size, :] = state
    vals[size] = local_value
    return new_states, vals

# ----------------------------------------------------------------------------------------

if _JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
    from jax import jit as jax_jit
    
    @jax_jit
    def _local_energy_arr_lax(state, param: Union[float, complex]):
        """
        Calculate the local energy contribution from a given parameter using JAX.
        This function simulates the computation of a local energy term by generating
        matrix elements represented by row indices and corresponding values.
        Parameters
        ----------
        state : jnp.ndarray
            The integer representation of a basis state.
        param : Union[float, complex]
            The parameter used to calculate the matrix elements.
        Returns
        -------
        tuple
            A tuple containing two arrays:
            - new_rows: An array of row indices (int) representing the positions in the matrix.
            - new_vals: An array of values representing the energy contributions at those positions.
            The first element of each array corresponds to the diagonal element (k_map),
            while the second corresponds to the off-diagonal element (k_map with bit i flipped).
        """
        
        size = state.shape[0]
        
        def scan_fun(local_value, i):
            val         = param / 2.0 * state[i]
            new_state   = state.at[i].set(-state[i])
            new_val     = param**2.0
            return local_value + val, (new_state, new_val)
        
        free_indices = jnp.arange(size)
        local_value, (new_states, new_vals) = jax.lax.scan(scan_fun, 0.0, free_indices)
        
        # concatenate the results
        all_states = jnp.concatenate((new_states, state[None, :]), axis=0)
        all_vals   = jnp.concatenate((new_vals, jnp.array([local_value])), axis=0)
        return all_states, all_vals
    
    @jax_jit
    def _local_energy_arr_jax(state, param: Union[float, complex]):
        """
        Calculate the local energy contribution from a given parameter using JAX.
        This function simulates the computation of a local energy term by generating
        matrix elements represented by row indices and corresponding values.
        Parameters
        ----------
        state : jnp.ndarray
            The integer representation of a basis state.
        param : Union[float, complex]
            The parameter used to calculate the matrix elements.
        Returns
        -------
        tuple
            A tuple containing two arrays:
            - new_rows: An array of row indices (int) representing the positions in the matrix.
            - new_vals: An array of values representing the energy contributions at those positions.
            The first element of each array corresponds to the diagonal element (k_map),
            while the second corresponds to the off-diagonal element (k_map with bit i flipped).
        """
        
        return _local_energy_arr_lax(state, param)
    
def _local_energy_arr_wrap(param: Union[float, complex], use_jax: bool = False):
    """
    Wrapper function to call the local energy calculation function.
    This function is used to ensure compatibility with Numba's JIT compilation.
    Parameters
    ----------
    state : np.ndarray
        The integer representation of a basis state.
    param : Union[float, complex]
        The parameter used to calculate the matrix elements.
    use_jax : bool
        Flag to indicate whether to use JAX for computation.
    Returns
    -------
    tuple
        A tuple containing two arrays:
        - new_rows: An array of row indices (int) representing the positions in the matrix.
        - new_vals: An array of values representing the energy contributions at those positions.
    """
    
    if use_jax and _JAX_AVAILABLE:
        @jax_jit
        def wrapper(state):
            return _local_energy_arr_jax(state, param)
        return wrapper
    
    @numba.njit
    def wrapper(state):
        return _local_energy_arr_loop(state, param)
    return wrapper

# ----------------------------------------------------------------------------------------

class DummyHamiltonian(Hamiltonian):
    '''
    Dummy Hamiltonian class for testing
    '''
    
    def __init__(self, 
                hilbert_space   : HilbertSpace, 
                param           : Union[float, complex] = 1.0, 
                ns              : Optional[int]         = None, 
                backend         : str                   = 'default', 
                dtype                                   = complex):
        '''
        Dummy Hamiltonian class for testing
        Parameters
        ----------
        hilbert_space   : HilbertSpace
            Hilbert space object
        param           : float or complex
            Parameter for the Hamiltonian
        '''
        
        # Initialize the Hamiltonian
        if hilbert_space is None:
            if ns is None:
                raise ValueError(self._ERR_EITHER_HIL_OR_NS)
            hilbert_space = HilbertSpace(ns=ns, backend=backend, dtype=dtype, nhl=2)
        
        super().__init__(hilbert_space=hilbert_space, backend=backend, dtype=dtype)
        
        self._param = param
        if self.dtype == np.float64 or self.dtype == np.float32 or self.dtype == float and isinstance(param, complex):
            self._param = np.complex128(param)
        
        # initialize the functions 
        self._loc_energy_int_fun        = _local_energy_int_wrap(self._param)
        self._loc_energy_np_fun         = _local_energy_arr_wrap(self._param, use_jax=False)
        if _JAX_AVAILABLE:
            self._loc_energy_jax_fun    = _local_energy_arr_wrap(self._param, use_jax=True)       
        self._local_energy_test()
        
    # ------------------------------------------------------------------------------------
    
    def __repr__(self):
        return f'Dummy Hamiltonian with {self.ns} sites and parameter {self._param}'
    
    def __str__(self):
        return f'dummy,{self.ns},p={self._param}'
    
    # ------------------------------------------------------------------------------------
    
    @property
    def param(self):
        return self._param
    
    @param.setter
    def param(self, value):
        self._param = value
        self._loc_energy_int_fun = _local_energy_int_wrap(self._param)
        self._loc_energy_np_fun  = _local_energy_arr_wrap(self._param, use_jax=False)
        if _JAX_AVAILABLE:
            self._loc_energy_jax_fun = _local_energy_arr_wrap(self._param, use_jax=True)
        self._local_energy_test()
        
    # ------------------------------------------------------------------------------------
    
    # ------------------------------------------------------------------------------------
    
# ----------------------------------------------------------------------------------------