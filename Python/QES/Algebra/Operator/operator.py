"""
This module defines a set of classes and functions for
handling general operators in quantum mechanics,
particularly in the context of Hilbert spaces.

Main components of this module include:

Classes:
    - GeneralOperator: Describes a general operator. It can be expanded for more complicated operators acting on Hilbert space or other spaces. It supports various constructors for different initializations and provides methods for setting and getting operator properties.
    - OperatorContainer: Stores elements in a matrix form and provides methods for updating, sampling, and normalizing operator values.

Enumerations:
    - SymGenerators     : Enumerates various implemented symmetry types - used for symmetry analysis.
    - FermionicOperators: Enumerates various implemented fermionic operators.
    
This module is under constant development and is intended to be expanded for more complex operators and functionalities.

Date    : April 2023
Author  : Maksymilian Kliczkowski, WUST, Poland
"""

import copy                  
import time
import numpy as np
import numba
from abc import ABC, abstractmethod
from enum import Enum, auto, unique
from typing import Optional, Callable, Union, Iterable, Any
from typing import Union, Tuple, List               # type hints for the functions and methods
from functools import partial                       # partial function application for operator composition
####################################################################################################
from general_python.algebra.utils import get_backend as get_backend, JAX_AVAILABLE
from general_python.lattices import Lattice
# from Algebra.hilbert import HilbertSpace
####################################################################################################

if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap
    from jax.experimental import sparse

####################################################################################################

class SymmetryGenerators(Enum):
    """
    Available symmetry generators for symmetry analysis. 
    - E                 : Identity symmetry. 
    - Translation-x (Tx): Translation symmetry in the x-direction.
    - Translation-y (Ty): Translation symmetry in the y-direction.
    - Translation-z (Tz): Translation symmetry in the z-direction.
    - Reflecition (R)   : Reflection symmetry.
    - Parity (Px)       : Parity symmetry in the x-direction.
    - Parity (Py)       : Parity symmetry in the y-direction.
    - Parity (Pz)       : Parity symmetry in the z-direction. 
    """
    
    # standard symmetries
    E               = auto()
    Translation_x   = auto()
    Translation_y   = auto()
    Translation_z   = auto()
    Reflection      = auto()
    ParityX         = auto()
    ParityY         = auto()
    ParityZ         = auto()
    
    # other symmetries - fallback
    Other           = auto()

    # -----------
    
    def has_translation(self):
        """
        Check if the symmetry is a translation symmetry.
        """
        return self in [SymmetryGenerators.Translation_x, SymmetryGenerators.Translation_y, SymmetryGenerators.Translation_z]
    
    # -----------

    def has_reflection(self):
        """
        Check if the symmetry is a reflection symmetry.
        """
        return self in [SymmetryGenerators.Reflection]
    
    # -----------

class GlobalSymmetries(Enum):
    """
    Global symmetries for representing different symmetry groups.
    """
    
    U1      = auto()
    Other   = auto()

####################################################################################################
#! Distinguish type of function

def op_func_wrapper(op_func: Callable, *args: Any) -> Callable:
    """
    Wrap an operator function by fixing extra arguments so that the resulting function
    has the signature f(k) -> (rows, values), where op_func is assumed to have signature
    (k, *args) -> (rows, values).

    Parameters:
        op_func (Callable): The operator function.
        *args (Any): Additional arguments to fix in the operator function.

    Returns:
        Callable: A function of one argument (k) that calls op_func(k, *args).
    """
    if op_func.__code__.co_argcount == 1:
        # If op_func takes only one argument, no wrapping is needed.
        return op_func
    else:
        # Otherwise, return a new function that calls op_func with the fixed extra arguments.
        return lambda k: op_func(k, *args)

####################################################################################################

class OperatorFunction:
    '''
    OperatorFunction is a class that represents a mathematical operator that can be applied to a state. 
    The operator can be defined for different backends (integer, NumPy, JAX) and supports various operations 
    such as addition, subtraction, multiplication, and composition.
    The class provides a flexible way to define and apply operators to quantum states, allowing for
    different manipulations and analyses of quantum systems.
    
    Attr:
    ---
    - fun_int (Callable): 
        The function defining the operator for integer states.
    - fun_np (Optional[Callable]): 
        The function defining the operator for NumPy array states.
    - fun_jax (Optional[Callable]): 
        The function defining the operator for JAX array states.
    - modifies_state (bool): 
        A flag indicating whether the operator modifies the state.
    - necessary_args (int): 
        The number of necessary arguments for the operator function.
        
    Methods:
    ---
    - __init__(fun_int, fun_np=None, fun_jax=None, modifies_state=True, necessary_args=0):
        Initializes the OperatorFunction object with the provided functions and attributes.
    - apply(s, *args):
        Applies the operator function to a given state with the specified arguments.
    - __call__(s, *args):
        Calls the operator function on a given state. Equivalent to `apply`.
    - __mul__(other):
        Composes the current operator with another operator or scales it by a scalar.
    - __rmul__(other):
        Reverse composition of the current operator with another operator or scales it by a scalar.
    - __add__(other):
        Adds two operator functions, combining their effects.
    - __sub__(other):
        Subtracts one operator function from another, combining their effects.
    - wrap(*args):
        Wraps the operator function with additional arguments.
    - fun (property):
        Gets or sets the function defining the operator for integer states.
    - np (property):
        Gets or sets the function defining the operator for NumPy array states.
    - jax (property):
        Gets or sets the function defining the operator for JAX array states.
    - modifies_state (property):
        Gets or sets the flag indicating whether the operator modifies the state.
    - necessary_args (property):
        Gets or sets the number of necessary arguments for the operator function.
    '''
    
    _ERR_INVALID_ARG_NUMBER     = "Invalid number of arguments for the operator function. Expected {}, got {}."
    _ERR_WRONG_MULTIPLICATION   = "Invalid multiplication with type {}."
    # -----------
    
    def __init__(self,
                fun_int         : Callable,
                fun_np          : Optional[Callable]    = None,
                fun_jax         : Optional[Callable]    = None,
                modifies_state  : bool                  = True,
                necessary_args  : int                   = 0):
        """
        Initialize the OperatorFunction object.
        
        Params:
        - fun (callable)    : The function that defines the operator - it shall take a state 
            (or a list of states) and return the transformed state (or a list of states). States can be 
            represented as integers or numpy arrays or JAX arrays. This enables the user to define
            any operator that can be applied to the state. The function shall return a list of pairs (state, value).
        """
        self._fun_int           = fun_int
        self._fun_np            = fun_np
        self._fun_jax           = fun_jax
        self._modifies_state    = modifies_state            # flag for the operator that modifies the state
        self._necessary_args    = int(necessary_args)       # number of necessary arguments for the operator function

    # -----------
    
    def _apply_global(self, s: Union[int, np.ndarray]) -> List[Tuple[Optional[Union[int]], Union[float, complex]]]:
        """
        Apply the operator function to a given state.

        Parameters:
            s (int or np.ndarray or jnp.ndarray): 
                The state to which the operator is applied.

        Returns:
            A list of tuples (state, value), where each tuple contains the transformed state 
            (or None if not applicable) and its corresponding value.
        """
        
        # If the state is a JAX array, use the JAX function
        if isinstance(s, jnp.ndarray) and self._fun_jax is not None:
            return self._fun_jax(s)
        # If the state is a NumPy array, use the NumPy function
        elif isinstance(s, (np.ndarray, List)) and self._fun_np is not None:
            return self._fun_np(s)
        # Fallback to the integer function
        return self._fun_int(s)
    
    # -----------
    
    def _apply_local(self, s: Union[int, np.ndarray], i) -> List[Tuple[Optional[Union[int]], Union[float, complex]]]:
        """
        Apply the operator function to a given state.

        Parameters:
            s (int or jnp.ndarray): The state to which the operator is applied.
            i: Additional argument for the operator.

        Returns:
            A list of tuples (state, value), where each tuple contains the transformed state 
            (or None if not applicable) and its corresponding value.
        """
        
        # If the state is a JAX array, use the JAX function
        if isinstance(s, jnp.ndarray) and self._fun_jax is not None:
            return self._fun_jax(s, i)
        # If the state is a NumPy array, use the NumPy function
        elif isinstance(s, np.ndarray) and self._fun_np is not None:
            # If the state is a NumPy array, use the NumPy function
            return self._fun_np(s, i)
        # If the state is an integer, use the integer function
        # Note: The integer function should be able to handle the additional argument
        return self._fun_int(s, i)
    
    # -----------
    
    def _apply_correlation(self, s: Union[int], i, j) -> List[Tuple[Optional[Union[int]], Union[float, complex]]]:
        """
        Apply the operator function to a given state.

        Parameters:
            s (int or jnp.ndarray): The state to which the operator is applied.
            i, j: Additional arguments for the operator.

        Returns:
            A list of tuples (state, value), where each tuple contains the transformed state 
            (or None if not applicable) and its corresponding value.
        """
        # If the state is a JAX array, use the JAX function
        if isinstance(s, jnp.ndarray) and self._fun_jax is not None:
            # If the state is a JAX array, use the JAX function
            return self._fun_jax(s, i, j)
        elif isinstance(s, np.ndarray) and self._fun_np is not None:
            # If the state is a NumPy array, use the NumPy function
            return self._fun_np(s, i, j)
        # If the state is an integer, use the integer function
        # Note: The integer function should be able to handle the additional arguments
        return self._fun_int(s, i, j)
    
    # -----------
    
    def apply(self, s: Union[int, np.ndarray], *args) -> List[Tuple[Optional[Union[int]], Union[float, complex]]]:
        """
        Apply the operator function to a given state.

        Parameters:
            s (int or np.ndarray or jnp.ndarray): 
                The state to which the operator is applied.
            args:
                Additional arguments for the operator. The number of arguments must equal self._necessary_args.

        Returns:
            A list of tuples (state, value), where each tuple contains the transformed state 
            (or None if not applicable) and its corresponding value.

        Raises:
            ValueError: If the number of provided arguments does not equal self._necessary_args,
                        or if the return type from the operator function is not recognized.
        """
        if len(args) != self._necessary_args:
            raise ValueError(self._ERR_INVALID_ARG_NUMBER.format(self._necessary_args, len(args)))
        
        result = None
        
        # apply the operator function based on the number of necessary arguments
        if self._necessary_args == 0:
            result = self._apply_global(s)
        elif self._necessary_args == 1:
            result = self._apply_local(s, args[0])
        elif self._necessary_args == 2:
            result = self._apply_correlation(s, args[0], args[1])
        else:
            if isinstance(s, jnp.ndarray) and self._fun_jax is not None:
                result = self._fun_jax(s, *args)
            elif isinstance(s, np.ndarray) and self._fun_np is not None:
                result = self._fun_np(s, *args)
            else:
                result = self._fun_int(s, *args)

        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], (int, np.ndarray, jnp.ndarray)):
            return result
        elif isinstance(result, list) and all(isinstance(item, tuple) and len(item) == 2 for item in result):
            return result
        raise ValueError("Operator function returned an invalid type. Expected a tuple or a list of (state, value) pairs.")
    
    # -----------
    
    def __call__(self, s: Union[int], *args) -> List[Tuple[Optional[Union[int]], Union[float, complex]]]:
        """
        Apply the operator function to a given state.

        Parameters:
            s (int or jnp.ndarray): The state to which the operator is applied.
            args: Additional arguments for the operator. The number of arguments must equal self._necessary_args.

        Returns:
            A list of tuples (state, value), where each tuple contains the transformed state 
            (or None if not applicable) and its corresponding value.

        Raises:
            ValueError: If the number of provided arguments does not equal self._necessary_args,
                        or if the return type from the operator function is not recognized.
        """
        return self.apply(s, *args)
    
    # -----------
    #! Getters and Setters
    # -----------
    
    @property
    def fun(self):
        ''' Set the function that defines the operator '''
        return self._fun_int
    
    @property
    def npy(self):
        ''' Set the function that defines the operator '''
        return self._fun_np
    
    @property
    def jax(self):
        ''' Set the function that defines the operator '''
        return self._fun_jax
    
    @fun.setter
    def fun(self, val):
        self._fun_int = val
        
    @property
    def modifies_state(self):
        ''' Set the flag for the operator that modifies the state '''
        return self._modifies_state
    
    @modifies_state.setter
    def modifies_state(self, val):
        self._modifies_state = val
        
    @property
    def necessary_args(self):
        ''' Set the number of necessary arguments for the operator function '''
        return self._necessary_args
    
    @necessary_args.setter
    def necessary_args(self, val):
        self._necessary_args = val
    
    # -----------
    #! Composition
    # -----------
    
    @staticmethod
    def _normalize_result_np(states, values):
        """For the NumPy backend: ensure states is 2D and values is 1D."""
        if states.ndim == 1:
            states = states.reshape(1, -1)
            values = np.atleast_1d(values)
        return states, values

    @staticmethod
    def _normalize_result_jax(states, values):
        """For the JAX backend: ensure states is 2D and values is 1D."""
        states = jnp.atleast_2d(states)
        values = jnp.atleast_1d(values)
        return states, values
    
    def _pack_result(self, norm_result):
        """
        Pack a normalized list of (state, value) pairs into a tuple of two NumPy arrays.
        Returns:
            (states, values) where states is a 2D np.array and values is a 1D np.array.
        """
        states_list = [st for st, _ in norm_result]
        values_list = [val for _, val in norm_result]
        return np.array(states_list), np.array(values_list)
    
    # ----------
    
    def _multiply_const(self, constant):
        """
        Multiply the operator by a constant value.
        This method constructs and returns a new OperatorFunction instance where the underlying
        functions for different processing backends (integer, NumPy, and JAX) are modified to multiply
        their computed values by the specified constant. Specifically, it defines three nested functions:
        
        - mul_int:
            Applies the constant multiplication to the result of self._fun_int.
        - mul_np:
            Applies the constant multiplication to the result of self._fun_np and is accelerated using Numba.
        - mul_jax:
            Applies the constant multiplication to the result of self._fun_jax and, if JAX is available,
            is compiled with jax.jit for performance.
            
        Parameters:
                constant (numeric): The constant value to multiply with the operator's result.
        Returns:
                OperatorFunction: A new instance of OperatorFunction encapsulating the modified functions,
                                                    along with metadata about state modifications and necessary arguments.
        """
        def mul_int(s, *args):
            st, val = self._fun_int(s, *args)
            return st, val * constant

        # ------

        # @numba.njit
        def mul_np(s, *args):
            st, val = self._fun_np(s, *args)
            return st, val * constant

        # ------
        
        def mul_jax(s, *args):
            st, val = self._fun_jax(s, *args)
            return st, val * constant
        if JAX_AVAILABLE:
            mul_jax = jax.jit(mul_jax)
            
        return OperatorFunction(mul_int, mul_np, mul_jax,
                        modifies_state=self._modifies_state,
                        necessary_args=self._necessary_args)
        
    # =========================================================================
    # Composition: f * g  (i.e. (f * g)(s) = f(g(s)) )
    # =========================================================================
    
    def __mul__(self, other: Union[int, float, complex, np.int64, np.float64, np.complex128, 'OperatorFunction']):

        if isinstance(other, (int, float, complex, np.int64, np.float64, np.complex128)):
            return self._multiply_const(other)
            
        if not isinstance(other, OperatorFunction):
            raise TypeError(f"Unsupported multiplication with type {type(other)}")
        
        other_modifies  = other.modifies_state
        new_args        = max(self._necessary_args, other._necessary_args)
        
        # Compose as: (f * g)(s) = f(g(s))
        #! Integer version
        def mul_int(s, *args):
            # Get intermediate result from operator 'other'
            s_g, v_g    = other.fun(s, *args)
            s_g, v_g    = OperatorFunction._normalize_result_np(s_g, v_g)
            n           = s_g.shape[0]
            
            # Preallocate result arrays
            res_states  = np.empty((n, s_g.shape[1]), dtype=s.dtype)
            res_values  = np.empty(n, dtype=np.complex128)
            
            # Loop over each intermediate state.
            for i in range(n):
                # Apply self to each intermediate state.
                s_self, v_self = self._fun_int(s_g[i], *args)
                s_self, v_self = OperatorFunction._normalize_result_np(s_self, v_self)
                if self._modifies_state:
                    # Expect a single pair from self.
                    res_states[i, :] = s_g[i]   # keep state from g(s)
                    res_values[i] = v_g[i] * v_self[0]
                else:
                    # If self does not modify state, assume multiple pairs;
                    # here we simply take the first pair for simplicity.
                    res_states[i, :]    = s_self[0]
                    res_values[i]       = v_g[i] * v_self[0]
            return res_states, res_values

        # --- NumPy version ---
        def mul_np(s, *args):
            s_g, v_g = other._fun_np(s, *args)
            s_g, v_g = OperatorFunction._normalize_result_np(s_g, v_g)
            n = s_g.shape[0]
            res_states = []
            res_values = []
            for i in range(n):
                s_self, v_self = self._fun_np(s_g[i], *args)
                s_self, v_self = OperatorFunction._normalize_result_np(s_self, v_self)
                if self._modifies_state:
                    res_states.append(s_g[i])
                    res_values.append(v_g[i] * v_self[0])
                else:
                    for j in range(s_self.shape[0]):
                        res_states.append(s_self[j])
                        res_values.append(v_g[i] * v_self[j])
            return np.array(res_states), np.array(res_values)

        # --- JAX version ---
        def mul_jax(s, *args):
            s_g, v_g    = other._fun_jax(s, *args)
            s_g, v_g    = OperatorFunction._normalize_result_jax(s_g, v_g)
            n = s_g.shape[0]
            # For JAX, we use a Python loop wrapped in jax.lax.map.
            def body(i):
                s_self, v_self = self._fun_jax(s_g[i], args)
                s_self, v_self = OperatorFunction._normalize_result_jax(s_self, v_self)
                if self._modifies_state:
                    return s_g[i], v_g[i] * v_self[0]
                else:
                    return s_self[0], v_g[i] * v_self[0]
            result = jax.lax.map(body, jnp.arange(n))
            # result is a tuple of two arrays; we stack them accordingly.
            res_states = jnp.stack([x[0] for x in result])
            res_values = jnp.stack([x[1] for x in result])
            return res_states, res_values
        # if JAX_AVAILABLE:
            # mul_jax = jax.jit(mul_jax)

        return OperatorFunction(mul_int, mul_np, mul_jax,
                                modifies_state=self._modifies_state or other._modifies_state,
                                necessary_args=new_args)
    # -----------
    
    def __rmul__(self, other: Union[int, float, complex, np.int64, np.float64, np.complex128, 'OperatorFunction']):
        """
        Reverse composition of two operator functions: g * f ≡ g(f(n,...), ...).
        That is, the current operator (self) is applied first, and then the left-hand operator (other)
        is applied to the resulting states.
        
        If other is a scalar, the operator's output values are simply scaled by that scalar.
        
        JIT compilation is applied for the JAX version when available.
        
        Parameters
        ----------
        other : int, float, complex, np.int64, np.float64, np.complex128, or OperatorFunction
            Either a scalar multiplier or an operator function to compose with self in reverse order.
        
        Returns
        -------
        OperatorFunction
            A new operator function representing the reverse composition.
        """
        if isinstance(other, (int, float, complex, np.int64, np.float64, np.complex128)):
            return self._multiply_const(other)
        
        # Ensure that other is an OperatorFunction.
        if not isinstance(other, OperatorFunction):
            raise TypeError("Unsupported reverse multiplication with type {}".format(type(other)))
        
        new_args = max(self._necessary_args, other._necessary_args)

        def rmul_int(s, *args):
            s_self, v_self = self._fun_int(s, *args)
            s_self, v_self = OperatorFunction._normalize_result_np(s_self, v_self)
            n = s_self.shape[0]
            res_states = np.empty((n, s_self.shape[1]), dtype=s.dtype)
            res_values = np.empty(n, dtype=np.complex128)
            for i in range(n):
                s_other, v_other = other._fun_int(s_self[i], *args)
                s_other, v_other = OperatorFunction._normalize_result_np(s_other, v_other)
                if other._modifies_state:
                    res_states[i, :] = s_self[i]
                    res_values[i] = v_self[i] * v_other[0]
                else:
                    res_states[i, :] = s_other[0]
                    res_values[i] = v_self[i] * v_other[0]
            return res_states, res_values

        def rmul_np(s, *args):
            s_self, v_self = self._fun_np(s, *args)
            s_self, v_self = OperatorFunction._normalize_result_np(s_self, v_self)
            n = s_self.shape[0]
            res_states = []
            res_values = []
            for i in range(n):
                s_other, v_other = other._fun_np(s_self[i], *args)
                s_other, v_other = OperatorFunction._normalize_result_np(s_other, v_other)
                if other._modifies_state:
                    res_states.append(s_self[i])
                    res_values.append(v_self[i] * v_other[0])
                else:
                    for j in range(s_other.shape[0]):
                        res_states.append(s_other[j])
                        res_values.append(v_self[i] * v_other[j])
            return np.array(res_states), np.array(res_values)

        def rmul_jax(s, *args):
            s_self, v_self = self._fun_jax(s, *args)
            s_self, v_self = OperatorFunction._normalize_result_jax(s_self, v_self)
            n = s_self.shape[0]
            def body(i):
                s_other, v_other = other._fun_jax(s_self[i], *args)
                s_other, v_other = OperatorFunction._normalize_result_jax(s_other, v_other)
                if other._modifies_state:
                    return s_self[i], v_self[i] * v_other[0]
                else:
                    return s_other[0], v_self[i] * v_other[0]
            result = jax.lax.map(body, jnp.arange(n))
            res_states = jnp.stack([x[0] for x in result])
            res_values = jnp.stack([x[1] for x in result])
            return res_states, res_values
        # if JAX_AVAILABLE:
            # rmul_jax = jax.jit(rmul_jax)

        return OperatorFunction(rmul_int, rmul_np, rmul_jax,
                                modifies_state=self._modifies_state or other._modifies_state,
                                necessary_args=new_args)
    # -----------
    
    def __getitem__(self, other):
        """
        Applies operator to a given - returns the first element of the result.
        """
        return self.apply(other)[0]
    
    # ----------
    
    def __mod__(self, other):
        '''
        Applies operator to a given - returns the second element of the result.
        '''
        return self.apply(other)[1]
        
    # -----------
    #! Addition
    # -----------

    def __add__(self, other: 'OperatorFunction'):
        """
        Add two operator functions. The operator is defined as:
        (f + g)(s, *args) = f(s, *args) + g(s, *args)
        where the sum means that when applied to a state, the outputs (a list of (state, value) pairs)
        are combined by summing the values for duplicate states.
        
        JIT compilation is applied for the JAX version when available.
        """
        # Pure Python / NumPy version.
        def add_int(s, *args):
            result_f            = np.array(self(s, *args))
            result_g            = np.array(other(s, *args))
            # Extract states and values.
            states_f, values_f  = result_f[:, 0], result_f[:, 1]
            states_g, values_g  = result_g[:, 0], result_g[:, 1]
            # Combine results.
            combined_states     = np.concatenate([states_f, states_g])
            combined_values     = np.concatenate([values_f, values_g])
            # Get unique states and sum the corresponding values.
            unique_states, indices  = np.unique(combined_states, return_inverse=True)
            summed_values           = np.zeros_like(unique_states, dtype=complex)
            for idx, val in zip(indices, combined_values):
                summed_values[idx] += val
            return list(zip(unique_states, summed_values))
        
        # For the NumPy version we use the same as add_int.
        def add_np(s, *args):
            return add_int(s, *args)
        
        # JAX version.
        def add_jax(s, *args):
            result_f                = jnp.array(self(s, *args))
            result_g                = jnp.array(other(s, *args))
            states_f, values_f      = result_f[:, 0], result_f[:, 1]
            states_g, values_g      = result_g[:, 0], result_g[:, 1]
            combined_states         = jnp.concatenate([states_f, states_g])
            combined_values         = jnp.concatenate([values_f, values_g])
            unique_states, indices  = jnp.unique(combined_states, return_inverse=True)
            # jnp.zeros returns a jax array; note that we use jnp.complex128 as the dtype.
            summed_values           = jnp.zeros(unique_states.shape, dtype=jnp.complex128).at[indices].add(combined_values)
            return list(zip(unique_states, summed_values))
        
        fun_jax = add_jax
        if JAX_AVAILABLE:
            fun_jax = jax.jit(add_jax)
        
        return OperatorFunction(
            add_int,
            add_np, 
            fun_jax,
            modifies_state=self._modifies_state or other._modifies_state,
            necessary_args=max(self._necessary_args, other._necessary_args)
        )

        
    # -----------
    #! Subtraction
    # -----------
    
    def __sub__(self, other: 'OperatorFunction'):
        """
        Subtract two operator functions. The operator is defined as:
        (f - g)(s, *args) = f(s, *args) - g(s, *args)
        which is implemented by negating the output of g and then combining results.
        
        JIT compilation is applied for the JAX version when available.
        """
        def sub_int(s, *args):
            result_f = np.array(self(s, *args))
            result_g = np.array(other(s, *args))
            states_f, values_f = result_f[:, 0], result_f[:, 1]
            states_g, values_g = result_g[:, 0], result_g[:, 1]
            combined_states = np.concatenate([states_f, states_g])
            combined_values = np.concatenate([values_f, -values_g])  # negate g's values
            unique_states, indices = np.unique(combined_states, return_inverse=True)
            summed_values = np.zeros_like(unique_states, dtype=complex)
            for idx, val in zip(indices, combined_values):
                summed_values[idx] += val
            return list(zip(unique_states, summed_values))
        
        def sub_np(s, *args):
            return sub_int(s, *args)
        
        def sub_jax(s, *args):
            result_f = jnp.array(self(s, *args))
            result_g = jnp.array(other(s, *args))
            states_f, values_f = result_f[:, 0], result_f[:, 1]
            states_g, values_g = result_g[:, 0], result_g[:, 1]
            combined_states = jnp.concatenate([states_f, states_g])
            combined_values = jnp.concatenate([values_f, -values_g])
            unique_states, indices = jnp.unique(combined_states, return_inverse=True)
            summed_values = jnp.zeros(unique_states.shape, dtype=jnp.complex128).at[indices].add(combined_values)
            return list(zip(unique_states, summed_values))
        
        fun_jax = sub_jax
        if JAX_AVAILABLE:
            fun_jax = jax.jit(sub_jax)
        
        return OperatorFunction(
            sub_int,
            sub_np,
            fun_jax,
            modifies_state=self._modifies_state or other.modifies_state,
            necessary_args=max(self._necessary_args, other.necessary_args)
        )
        
    # -----------
    #! Wrapping
    # -----------
    
    def wrap(self, *args):
        """
        Wraps the operator functions with additional fixed arguments, returning a new OperatorFunction
        that encapsulates integer, numpy, and (if available) JAX implementations.
        Parameters:
            *args: Additional arguments to be bound to the underlying operator functions.
        Returns:
            OperatorFunction: An object constructed with three wrapper functions:
                - wrap_int: Calls self._fun_int with the initial fixed arguments (from *args) followed
                    by any additional arguments provided during its invocation.
                - wrap_np: Calls self._fun_np in a similar manner for numpy-compatible operations.
                - wrap_jax: Calls self._fun_jax with the fixed and additional arguments, and, if JAX is available,
                    it is decorated with jax.jit for just-in-time compilation.
        Notes:
            - The inner wrapper functions merge the fixed arguments passed to wrap with any further arguments.
            - If JAX_AVAILABLE is True, the wrap_jax function is optimized using jax.jit.
            - The returned OperatorFunction also propagates metadata such as 'modifies_state' and 'necessary_args'
                from the current operator instance.
        """
        
        
        def wrap_int(s, *more_args):
            return self._fun_int(s, *(args + more_args))
        def wrap_np(s, *more_args):
            return self._fun_np(s, *(args + more_args))
        def wrap_jax(s, *more_args):
            return self._fun_jax(s, *(args + more_args))
        
        if JAX_AVAILABLE:
            wrap_jax = jax.jit(wrap_jax)
        
        return OperatorFunction(wrap_int, wrap_np, wrap_jax,
                                modifies_state=self._modifies_state,
                                necessary_args=self._necessary_args)
    
    # -----------
    
####################################################################################################

@unique
class OperatorTypeActing(Enum):
    """
    Enumerates the types of operators acting on the system.
    """
    
    Global      = auto()    # Global operator - acts on the whole system (does not need additional arguments).
    Local       = auto()    # Local operator - acts on the local physical space (needs additional argument - 1).
    Correlation = auto()    # Correlation operator - acts on the correlation space (needs additional argument - 2).
    # -----------
    
    def is_global(self):
        """
        Check if the operator is a global operator.
        """
        return self == OperatorTypeActing.Global
    
    # -----------
    
    def is_local(self):
        """
        Check if the operator is a local operator.
        """
        return self == OperatorTypeActing.Local
    
    # -----------
    
    def is_correlation(self):
        """
        Check if the operator is a correlation operator.
        """
        return self == OperatorTypeActing.Correlation
    
    # -----------

####################################################################################################

class Operator(ABC):
    """
    A class to represent a general operator acting on a Hilbert space.
    
    Attributes:
    """
    
    _INVALID_OPERATION_TYPE_ERROR = "Invalid type for function. Expected a callable function."
    _INVALID_SYSTEM_SIZE_PROVIDED = "Invalid system size provided. Number of sites or a lattice object must be provided."
    _INVALID_FUNCTION_NONE        = "Invalid number of necessary arguments for the operator function."
    
    #################################
    
    def __init__(self,
                op_fun      : OperatorFunction              = None,
                fun_int     : Callable                      = None,        
                fun_np      : Optional[Callable]            = None,
                fun_jnp     : Optional[Callable]            = None,
                eigval                                      = 1.0,
                lattice     : Optional[Lattice]             = None,
                ns          : Optional[int]                 = None,
                typek       : Optional[SymmetryGenerators]  = SymmetryGenerators.Other,
                name        : str                           = 'Operator',
                modifies    : bool                          = True,
                quadratic   : bool                          = False,
                backend     : str                           = 'default',
                **kwargs):
        """
        Initialize the GeneralOperator object.
        
        Args:
            Ns (int)            : The number of sites in the system.
            Nhl (int)           : The local Hilbert space dimension - 2 for spin-1/2, 4 for spin-1, etc (default is 2).
            Nhint (int)         : The number of modes (fermions, bosons, etc. on each site).
            lattice (Lattice)   : The lattice object.
            name (str)          : The name of the operator.
            type (str)          : The type of the operator.
            eigval              : The eigenvalue of the operator (default is 1.0).
            quadratic (bool)    : Flag for the quadratic operator.
            acton (bool)        : Flag for the action of the operator on the local physical space.
            modifies (bool)     : Flag for the operator that modifies the state.
            backend (str)       : The backend for the operator - for using linear algebra libraries, not integer representation.
            
            Important arguments:
            - fun (Callable | OperatorFunction) : The function that defines the operator - it shall take a state 
                (or a list of states) and return the transformed state (or a list of states). States can be 
                represented as integers or numpy arrays or JAX arrays. This enables the user to define
                any operator that can be applied to the state. The function shall return a list of pairs (state, value).
        """
        
        # handle the system phyisical size dimension and the lattice
        if lattice is None and ns is not None:
            self._ns        = ns
            self._lattice   = lattice
        elif lattice is not None:
            self._lattice   = lattice
            self._ns        = self._lattice.ns
        else:
            raise ValueError(Operator._INVALID_SYSTEM_SIZE_PROVIDED)
        
        # set the backend for the operator
        self._backend_str               = backend
        self._backend, self._backend_sp = get_backend(backend, scipy=True)
        
        # property of the operator itself
        self._eigval        = eigval
        self._name          = name
        self._type          = typek
        if self._type != SymmetryGenerators.Other and self._name == 'Operator':
            self._name      = self._type.name
        
        # property for the behavior of the operator - e.g., quadratic, action, etc.
        self._quadratic     = quadratic                             # flag for the quadratic operator - this enables different matrix representation
        self._acton         = kwargs.get('acton', False)            # flag for the action of the operator on the local physical space
        self._modifies      = modifies                              # flag for the operator that modifies the state
        self._matrix_fun    = None                                  # the function that defines the matrix form of the operator - if not provided, the matrix is generated from the function fun
        self._necessary_args= 0                                     # number of necessary arguments for the operator function
        self._fun           = None                                  # the function that defines the operator - it is set to None if not provided
        self._init_functions(op_fun, fun_int, fun_np, fun_jnp)      # initialize the operator function
    
    def __repr__(self):
        """
        String representation of the operator.
        """
        return f"Operator({self._name}, eigval={self._eigval}, type={self._type.name})"
    
    #################################
    #! Initialize functions
    #################################
    
    def _init_functions(self, op_fun = None, fun_int = None, fun_np = None, fun_jax = None):
        """
        Initializes the operator functions and determines the operator type based on the 
        number of necessary arguments.
        
        ---
        Parameters:
            op_fun (OperatorFunction, optional): 
                An instance of `OperatorFunction`. If provided, 
                it is directly assigned to the operator.
            fun_int (callable, optional): 
                A Python function representing the internal implementation 
                of the operator. Must be provided if `op_fun` is not specified.
            fun_np (callable, optional): 
                A NumPy-compatible implementation of the operator function.
            fun_jax (callable, optional): 
                A JAX-compatible implementation of the operator function.
        Raises:
            ValueError: If both `op_fun` and `fun_int` are `None`.
            NotImplementedError: If the number of necessary arguments exceeds 2.
        Notes:
            - The `necessary_args` attribute is determined based on the number of arguments 
                required by `fun_int`, excluding the first argument (assumed to be `self`).
            - The operator type (`_type_acting`) is set based on the number of necessary arguments:
                - 0 arguments: `OperatorTypeActing.Global`
                - 1 argument: `OperatorTypeActing.Local`
                - 2 arguments: `OperatorTypeActing.Correlation`
        """
        
        if op_fun is not None and isinstance(op_fun, OperatorFunction):
            self._fun               = op_fun
            self._necessary_args    = op_fun.necessary_args
        else:
            if fun_int is None:
                raise ValueError(self._INVALID_FUNCTION_NONE)
            
            # get the necessary args
            self._necessary_args    = fun_int.__code__.co_argcount - 1
            self._fun               = OperatorFunction(fun_int, fun_np, fun_jax,
                                        modifies_state = self._modifies,
                                        necessary_args = self._necessary_args)
        # set the operator function type
        if self._necessary_args == 0:
            self._type_acting = OperatorTypeActing.Global
        elif self._necessary_args == 1:
            self._type_acting = OperatorTypeActing.Local
        elif self._necessary_args == 2:
            self._type_acting = OperatorTypeActing.Correlation
        else:
            raise NotImplementedError()
    
    #################################
    #! Static methods
    #################################
    
    @staticmethod
    def idn(state, *args):
        """
        Identity operator function.
        """
        return state, 1.0
    
    @staticmethod
    def idn_f(state, *args):
        """
        Identity operator function.
        """
        return 1.0
    
    #################################
    #! Copying and cloning
    #################################
    
    def copy(self):
        """
        Create a copy of the operator.
        """
        return copy.deepcopy(self)
    
    def clone(self):
        """
        Clone the operator.
        """
        return Operator(**self.__dict__)
    
    #################################
    #! Operators that modify the operator class itself
    #################################
    
    def __imul__(self, scalar):
        """ *= Operator for a general operator """
        
        self._fun = self._fun * scalar
        return self
    
    # -------------------------------
    
    def __itruediv__(self, scalar):
        """ /= Operator with Division by Zero Check """
        
        if isinstance(scalar, (int, float, complex, np.int64, np.float64, np.complex128)):
            if scalar == 0:
                raise ZeroDivisionError("Division by zero in GeneralOperator.__itruediv__")
            self._fun = self._fun * (1.0 / scalar)
        else:
            raise ValueError(f"Invalid type for multiplication.{type(scalar)}")
        return self
    
    # -------------------------------
    
    def __rmul__(self, other):
        """
        Right multiplication of the operator by another operator or a scalar.
        In particular for an operator O and a scalar s, the operation O * s is defined.
        """
        
        if isinstance(other, Operator):
            name        = f"{self._name} * {other._name}"
            function_m  = self._fun * other._fun
        elif isinstance(other, (int, float, complex, np.int64, np.float64, np.complex128)):
            name = f"{self._name} * {other}"
            function_m  = self._fun * other
        else:
            raise ValueError(f"Invalid type for multiplication.{type(other)}")
        
        return Operator(
            op_fun      = function_m,
            eigval      = self._eigval,
            lattice     = self._lattice,
            ns          = self._ns,
            typek       = self._type,
            name        = name,
            modifies    = self._modifies,
            quadratic   = self._quadratic,
            acton       = self._acton,
            backend     = self._backend_str,
            **self.__dict__)
    
    # -------------------------------
    
    def __mul__(self, other):
        """
        Multiplication of the operator by another operator or a scalar.
        In particular for an operator O and a scalar s, the operation s * O is defined.
        """
        
        if isinstance(other, Operator):
            name            = f"{self._name} * {other._name}"
            function_m      = other._fun * self._fun
        elif isinstance(other, (int, float, complex, np.int64, np.float64, np.complex128)):
            name = f"{self._name} * {other}"
            function_m      = other * self._fun
        else:
            raise ValueError(f"Invalid type for multiplication.{type(other)}")
        
        return Operator(
            op_fun      = function_m,
            eigval      = self._eigval,
            lattice     = self._lattice,
            ns          = self._ns,
            typek       = self._type,
            name        = name,
            modifies    = self._modifies,
            quadratic   = self._quadratic,
            acton       = self._acton,
            backend     = self._backend_str,
            **self.__dict__)
    
    # -------------------------------
    
    def __truediv__(self, scalar):
        """
        Division of the operator by a scalar.
        """
        
        if isinstance(scalar, (int, float, complex, np.int64, np.float64, np.complex128)):
            if scalar == 0:
                raise ZeroDivisionError("Division by zero in GeneralOperator.__truediv__")
            return self._multiply_const(1.0 / scalar)
        elif isinstance(scalar, Operator):
            raise NotImplementedError("Division of two operators is not implemented.")
        else:
            raise ValueError(f"Invalid type for multiplication.{type(scalar)}")
        return None
    
    # -------------------------------
    
    def __rtruediv__(self, scalar):
        """
        Division of a scalar by the operator.
        """
        if isinstance(scalar, (int, float, complex, np.int64, np.float64, np.complex128)):
            if scalar == 0:
                raise ZeroDivisionError("Division by zero in GeneralOperator.__rtruediv__")
            return self._multiply_const(scalar)
        else:
            raise ValueError(f"Invalid type for multiplication.{type(scalar)}")
        return None
    # -------------------------------
    
    #################################
    #! Setters and Getters
    #################################
    
    @property
    def eigval(self):
        '''Eigenvalue of the operator.'''
        return self._eigval
    
    @eigval.setter
    def eigval(self, val):
        self._eigval = val
        
    # -------------------------------
    
    @property
    def name(self):
        '''Name of the operator.'''
        return self._name
    
    @name.setter
    def name(self, val):
        self._name = val
        
    # -------------------------------
    
    @property
    def type(self):
        '''Type of the operator.'''
        return self._type
    
    @type.setter
    def type(self, val):
        self._type = val
        
    # -------------------------------
    
    @property
    def quadratic(self):
        '''Quadratic property of the operator.'''
        return self._quadratic
    
    @quadratic.setter
    def quadratic(self, val):
        self._quadratic = val
        
    # -------------------------------
    
    @property
    def acton(self):
        ''' Flag for the action of the operator on the local physical space. '''
        return self._acton
    
    @acton.setter
    def acton(self, val):
        self._acton = val
    
    # -------------------------------
    
    @property
    def modifies(self):
        ''' Flag for the operator that modifies the state. '''
        return self._modifies
    
    @modifies.setter
    def modifies(self, val):
        self._modifies = val
    
    # -------------------------------
    
    @property
    def type_acting(self):
        ''' Type of the operator acting on the system. '''
        return self._type_acting
    
    def get_acting_type(self):
        """
        Get the acting type of the operator.
        """
        return self._type_acting

    # -------------------------------
    
    @property
    def fun(self):
        ''' Set the function that defines the operator '''
        return self._fun
    
    @fun.setter
    def fun(self, val):
        self._fun = val
    
    @property
    def int(self):
        ''' Set the function that defines the operator '''
        return self._fun.fun
    
    @property
    def npy(self):
        ''' Set the function that defines the operator '''
        return self._fun.npy
    
    @property
    def jax(self):
        ''' Set the function that defines the operator '''
        return self._fun.jax
    
    # -------------------------------
    
    def override_matrix_function(self, function : Callable):
        """
        Override the matrix function of the operator.
        """
        self._matrix_fun = function
        
    #################################
    #! Apply the operator
    #################################
    
    # -------------------------------
    
    def _apply_global(self, states : Union[int, list, np.ndarray | jnp.ndarray]):
        """
        Applies a function to a given state or a collection of states.

        This method processes either a single state or an iterable of states, 
        applying the `_fun` function to each state. If the input is a single 
        state, the result of `_fun` is returned directly. If the input is a 
        collection of states, the method returns two lists: one containing 
        the transformed states and the other containing the corresponding 
        values.

        Args:
            states (Union[int, list, np.ndarray, jnp.ndarray]): 
                A single state or a collection of states to which the `_fun` 
                function will be applied. Can be an integer, list, numpy 
                array, or jax.numpy array.

        Returns:
            Union[tuple[list, list], Any]: 
                - If `states` is a single state, returns the result of `_fun(states)`.
                - If `states` is a collection, returns a tuple of two lists:
                    - The first list contains the transformed states.
                    - The second list contains the corresponding values.
        """
        if (hasattr(states, 'shape') and len(states.shape) == 1) or isinstance(states, (int, np.int8, np.int16, np.int32, np.int64)):
            # if the state is a single state, apply the function directly
            st, val = self._fun(states)
            return st, val * self._eigval
        
        # if the state is a collection of states, apply the function to each state
        results     = [self._fun(state) for state in states]
        out, val    = zip(*results) if results else ([], [])
        return list(out), list(val * self._eigval)
    
    def _apply_local(self, states : Union[int, list, np.ndarray | jnp.ndarray], i):
        """
        Applies a local operation to a given state or a collection of states.
        Parameters:
            states (Union[int, list, np.ndarray, jnp.ndarray]): The input state(s) to which the operation is applied.
                Can be a single integer, a list of integers, or a NumPy/JAX array.
            i (int): The index or parameter used by the local operation.
        Returns:
            tuple: A tuple containing two lists:
                - The first list contains the resulting states after applying the operation.
                - The second list contains the corresponding values or results of the operation.
                If the input is a single state, the result is returned as a single state and value.
        """
        if (hasattr(states, 'shape') and len(states.shape) == 1) or isinstance(states, (int, np.int8, np.int16, np.int32, np.int64)):
            # if the state is a single state, apply the function directly
            st, val = self._fun(states, i)
            return st, val * self._eigval
        results     = [self._fun(state, i) for state in states]
        out, val    = zip(*results) if results else ([], [])
        return list(out), list(val * self._eigval)
    
    def _apply_correlation(self, states : Union[int, list, np.ndarray | jnp.ndarray], i, j):
        """
        Applies a correlation function to a given state or a collection of states.
        Parameters:
            states (Union[int, list, np.ndarray, jnp.ndarray]): The input state(s) to which the correlation function 
                will be applied. Can be a single integer, a list of integers, or a NumPy/JAX array of states.
            i (int): The first index parameter for the correlation function.
            j (int): The second index parameter for the correlation function.
        Returns:
            tuple: If `states` is a single state, returns a tuple `(out, val)` where:
                - `out` is the output state after applying the correlation function.
                - `val` is the value associated with the correlation function.
            If `states` is a collection of states, returns a tuple of lists `(out_list, val_list)` where:
                - `out_list` contains the output states for each input state.
                - `val_list` contains the corresponding values for each input state.
        """
        if (hasattr(states, 'shape') and len(states.shape) == 1) or isinstance(states, (int, np.int8, np.int16, np.int32, np.int64)):
            # if the state is a single state, apply the function directly
            st, val = self._fun(states, i, j)
            return st, val * self._eigval
        
        results     = [self._fun(state, i, j) for state in states]
        out, val    = zip(*results) if results else ([], [])
        return list(out), list(val * self._eigval)
        
    def apply(self, states : list | np.ndarray | jnp.ndarray, *args):
        """
        Apply the operator to the state. 
        
        Args:
            states            : list of states to which the operator is applied.
            args              : Additional arguments for the operator - inform how to act on a state.
                                If there no arguments, the operator acts on the state as a whole - global operator.
                                If there are arguments, the operator acts on the state locally - local operator (e.g., site-dependent).
        """
        if self._type_acting.is_global():
            return self._apply_global(states)
        elif self._type_acting.is_local():
            return self._apply_local(states, *args)
        elif self._type_acting.is_correlation():
            return self._apply_correlation(states, *args)
        else:
            raise NotImplementedError("Invalid operator acting type.")
        return [None], [0.0]
        
    # -------------------------------
    
    def __call__(self, states: list | np.ndarray | jnp.ndarray, *args):
        """
        Apply the operator to the state. 
        
        Args:
            states            : list of states to which the operator is applied.
            args              : Additional arguments for the operator - inform how to act on a state.
                                If there no arguments, the operator acts on the state as a whole - global operator.
                                If there are arguments, the operator acts on the state locally - local operator (e.g., site-dependent).
        """
        return self.apply(states, *args)
    
    def __getitem__(self, states: list | np.ndarray | jnp.ndarray,):
        """
        Apply the operator to the state - returns modified state only.
        
        Args:
            states:
                list of states to which the operator is applied.
        
        Returns:
            list: The first element is the transformed state and the second element is the value - thus only the modified state is returned.
        """
        if isinstance(states, tuple):
            return self.apply(states[0], *states[1:])[0]
        return self.apply(states)[0]
    
    def __mod__(self, other):
        """
        Apply the operator to a given state and return the modified state values.
        """
        if isinstance(other, tuple):
            return self.apply(other[0], *other[1:])[1]
        return self.apply(other)[1]
    
    #################################
    #! Generate matrix form of the operator
    #################################
    
    def _matrix_no_hilbert_np(self, dim: int, is_sparse: bool, wrapped_funct, dtype, max_loc_upd:int = 1):
        """
        Generate the matrix form of the operator without Hilbert space.
        """
        # create a dummy Hilbert space for convenience
        from Algebra.hilbert import HilbertSpace
        from Algebra.Operator.operator_matrix import operator_create_np
        
        dummy_hilbert = HilbertSpace(nh = dim, backend = self._backend)
        dummy_hilbert.log("Calculating the Hamiltonian matrix using NumPy...", lvl = 2, log = 'debug')
        # calculate the time to create the matrix
        t1 = time.time()
        matrix = operator_create_np(ns      = None, 
                        hilbert_space       = dummy_hilbert,
                        local_fun           = wrapped_funct,
                        max_local_changes   = max_loc_upd,
                        is_sparse           = is_sparse,
                        start               = None,
                        dtype               = dtype)
        time_taken = time.time() - t1
        dummy_hilbert.log(f"Time taken to create the matrix {self._name}: {time_taken:.2f} seconds", lvl=2, log = 'debug')
        return matrix
        
    def _matrix_no_hilbert_jax(self, dim: int, is_sparse: bool, wrapped_funct, dtype, max_loc_upd:int = 1):
        """
        Generate the matrix form of the operator without Hilbert space.
        """
        # create a dummy Hilbert space for convenience
        dummy_hilbert = HilbertSpace(nh = dim, backend = self._backend)
        dummy_hilbert.log("Calculating the Hamiltonian matrix using JAX...", lvl = 2)
        #!TODO: Implement the JAX version of the matrix function
        return None
        
    def matrix(self, matrix_type : str, *args, **kwargs) -> np.ndarray | jnp.ndarray | sparse.COO | sparse.CSR | None:
        """
        Generates the matrix representation of the operator.

        Parameters:
        - param dim             : Dimension of the matrix.
        - param matrix_type     : Type of matrix ("dense" or "sparse").
        - param *args           : Additional arguments for the operator function - moved to self._fun.
        - param **kwargs        : Additional keyword arguments for the operator function.
        :return: The matrix representation of the operator.
        """
        
        hilbert_1   = kwargs.get('hilbert_1', None)     # first Hilbert space
        hilbert_2   = kwargs.get('hilbert_2', None)     # second Hilbert space
        
        # check the dimension of the matrix
        dim1, dim2      = None, None
        matrix_hilbert  = 'None'
        if hilbert_1 is not None and hilbert_2 is not None:
            dim1, dim2  = hilbert_1.nh, hilbert_2.nh
            matrix_hilbert = 'double'
        elif hilbert_1 is not None and hilbert_2 is None:
            dim1, dim2  = hilbert_1.nh, hilbert_1.nh
            matrix_hilbert = 'single'
        elif hilbert_1 is None and hilbert_2 is not None:
            hilbert_1   = hilbert_2
            dim1, dim2  = hilbert_2.nh, hilbert_2.nh
            matrix_hilbert = 'single'
        else:
            dim         = kwargs.get('dim', None)
            if dim is None:
                raise ValueError("Dimension or at least one Hilbert space must be provided.")
            dim1, dim2  = dim, dim
            matrix_hilbert = 'None'

        # check if there are functions from the Hilbert space
        use_numpy    = kwargs.get('use_numpy', False)
        jax_maybe_av = JAX_AVAILABLE and self._backend != np
        is_sparse    = (matrix_type == 'sparse')
        
        # check if the matrix function is provided and skips kwargs if unnecessary
        if self._matrix_fun is not None:
            if is_sparse:
                return self._matrix_fun(dim1, matrix_type, *args)
            else:
                return self._backend.asarray(self._matrix_fun(dim1, matrix_type, *args))
        
        # wrap the function
        wrapped_fun  = self._fun.wrap(*args)
        dtype        = kwargs.get('dtype', self._backend.float64 if not use_numpy else np.float64)
        max_loc_upd  = kwargs.get('max_loc_upd', 1)

        # Case1: easiest case - no Hilbert space provided
        if matrix_hilbert == 'None':
            # maximum local updates - how many states does the operator create - for sparse
            if not jax_maybe_av or use_numpy:
                return self._matrix_no_hilbert_np(dim1, is_sparse, wrapped_fun, dtype, max_loc_upd)
            else:
                return self._matrix_no_hilbert_jax(dim1, is_sparse, wrapped_fun, dtype, max_loc_upd)
        # Case2: one Hilbert space provided
        elif matrix_hilbert == 'single':
            pass
        # Case3: two Hilbert spaces provided
        elif matrix_hilbert == 'double':
            pass
        else:
            raise ValueError("Invalid Hilbert space provided.")
        return None
        
    #################################
    
    def standardize_matrix(self, matrix):
        """
        Standardizes the given matrix if the _standarize flag is set to true.
        """
        pass
    
    #################################
    
####################################################################################################

def operator_identity(backend : str = 'default') -> Operator:
    """
    Generate the identity operator.
    Parameters:
    - backend (str)     : The backend for the operator - for using linear algebra libraries, not integer representation.
    Returns:
    - Operator          : The identity operator.
    """
    def identity_fun(state):
        return state, 1.0
    
    return Operator(fun_int = identity_fun,
                    fun_np  = identity_fun,
                    fun_jax = identity_fun, 
                    eigval  = 1.0,
                    ns      = 1, 
                    backend = backend, 
                    name    = SymmetryGenerators.E, 
                    modifies=False, 
                    quadratic=False)

####################################################################################################

def create_operator(type_act        : int | OperatorTypeActing,
                    op_func_int     : Callable,
                    op_func_np      : Callable,
                    op_func_jnp     : Callable,
                    lattice         : Optional[Any]         = None,
                    ns              : Optional[int]         = None,
                    sites           : Optional[List[int]]   = None,
                    extra_args      : Tuple[Any, ...]       = (),
                    name            : Optional[str]         = None,
                    modifies        : bool                  = True) -> Operator:
    """
    Create a general operator that distinguishes the type of operator action (global, local, correlation)
    and wraps the provided operator functions for int, NumPy, and JAX. The operator functions must have
    the signature: (state, *args) -> (new_state, op_value) for global or (state, site, *args) for local,
    and (state, site1, site2, *args) for correlation.
    
    ---
    Note:
        - The operator functions should be defined to handle the specific state representation (int, NumPy, JAX).
        - The operator functions should return a tuple of the new state and the operator value.
        - The global operator should have either a list of sites or act on all sites.
        - The local operator should have a single site argument (additional to the state).
        - The correlation operator should have two site arguments (additional to the state).
        - The operator functions should be able to handle the extra arguments passed to them.
    
    The extra arguments (extra_args) are passed on to the operator functions (apart from the sites,
    which are provided dynamically). For global operators, if no sites are provided the operator is applied
    to all sites (0, 1, ..., ns-1).
    
    ---
    Parameters:
        type_act (int):
            The type of operator acting. Use OperatorTypeActing.Global, .Local, or .Correlation.
        op_func_int (Callable):
            Operator function for integer-based states.
        op_func_np (Callable):
            Operator function for NumPy-based states.
        op_func_jnp (Callable):
            Operator function for JAX-based states.
        lattice (Optional[Any]):
            A lattice object; if provided, ns is set from lattice.ns.
        ns (Optional[int]):
            The number of sites. Required if lattice is None.
        sites (Optional[List[int]]):
            For global operators: a list of sites on which to act. If None, all sites are used.
            (For local or correlation operators, sites are provided when the operator is applied.)
        extra_args (Tuple[Any,...]):
            Extra parameters to be forwarded to the operator functions.
        name (Optional[str]):
            A base name for the operator. If not provided, op_func_int.__name__ is used.
        modifies (bool):
            Whether the operator modifies the state.
    
    Returns:
        Operator:
            An operator object with fun_int, fun_np, fun_jnp methods appropriately wrapped.
    """
    # Ensure we know ns.
    if lattice is not None:
        ns = lattice.ns
    else:
        assert ns is not None, "Either lattice or ns must be provided."
    
    sites_jnp = jnp.array(sites, dtype = jnp.int32) if sites is not None and JAX_AVAILABLE else sites
    
    # Global operator: the operator acts on a specified set of sites (or all if sites is None)
    if type_act == OperatorTypeActing.Global or sites is not None:
        
        # If sites is None, we act on all sites.
        if sites is None or len(sites) == 0:
            sites = list(range(ns))
        sites           = tuple(sites) if isinstance(sites, list) else sites
        sites_np        = np.array(sites, dtype = np.int32)
        if JAX_AVAILABLE:
            sites_jnp   = jnp.array(sites, dtype = jnp.int32)
        else:
            sites_jnp   = sites_np
        
        # @numba.njit
        def fun_int(state):
            return op_func_int(state, ns, sites, *extra_args)
        
        @numba.njit
        def fun_np(state):
            return op_func_np(state, sites_np, *extra_args)
        
        if JAX_AVAILABLE:
            @jax.jit
            def fun_jnp(state):
                return op_func_jnp(state, sites_jnp, *extra_args)
        else:
            def fun_jnp(state):
                return state, 0.0
        
        op_name =   (name if name is not None else op_func_int.__name__) + "/"
        op_name +=  "-".join(str(site) for site in sites)
        return Operator(fun_int = fun_int,
                        fun_np  = fun_np,
                        fun_jnp = fun_jnp,
                        eigval  = 1.0,
                        lattice = lattice,
                        ns      = ns,
                        name    = op_name,
                        typek   = SymmetryGenerators.Other,
                        modifies= modifies)
    
    # Local operator: the operator acts on one specific site. The returned functions expect an extra site argument.
    elif type_act == OperatorTypeActing.Local:
        
        # @numba.njit
        def fun_int(state, i):
            return op_func_int(state, ns, [i], *extra_args)
        
        @numba.njit
        def fun_np(state, i):
            return op_func_np(state, i, *extra_args)
        
        if JAX_AVAILABLE:
            @jax.jit
            def fun_jnp(state, i):
                sites_jnp = jnp.array([i], dtype = jnp.int32)
                return op_func_jnp(state, sites_jnp, *extra_args)
        else:
            def fun_jnp(state, i):
                return state, 0.0
        op_name = (name if name is not None else op_func_int.__name__) + "/L"
        return Operator(fun_int = fun_int,
                        fun_np  = fun_np,
                        fun_jnp = fun_jnp,
                        eigval  = 1.0,
                        lattice = lattice,
                        ns      = ns,
                        name    = op_name,
                        typek   = SymmetryGenerators.Other,
                        modifies= modifies)
    
    # Correlation operator: the operator acts on a pair of sites.
    elif type_act == OperatorTypeActing.Correlation:
        
        # @numba.njit
        def fun_int(state, i, j):
            return op_func_int(state, ns, [i, j], *extra_args)
        
        @numba.njit
        def fun_np(state, i, j):
            return op_func_np(state, (i, j), *extra_args)
        
        if JAX_AVAILABLE:
            @partial(jax.jit)
            def fun_jnp(state, i, j):
                sites_jnp = jnp.array([i, j], dtype = jnp.int32)
                return op_func_jnp(state, sites_jnp, *extra_args)
        else:
            def fun_jnp(state, i, j):
                return state, 0.0
        op_name = (name if name is not None else op_func_int.__name__) + "/C"
        return Operator(fun_int = fun_int,
                        fun_np  = fun_np,
                        fun_jnp = fun_jnp,
                        eigval  = 1.0,
                        lattice = lattice,
                        ns      = ns,
                        name    = op_name,
                        typek   = SymmetryGenerators.Other,
                        modifies= modifies)
    
    else:
        raise ValueError("Invalid OperatorTypeActing")

# Example usage:
# (Assume sigma_x_int_np, sigma_x_np, sigma_x_jnp are defined elsewhere and JAX_AVAILABLE is set.)

# For a global operator:
#   op = create_operator(OperatorTypeActing.Global, sigma_x_int_np, sigma_x_np, sigma_x_jnp,
#                        lattice=my_lattice, sites=[0, 2, 3], extra_args=(spin_value,))
#
# For a local operator:
#   op = create_operator(OperatorTypeActing.Local, sigma_x_int_np, sigma_x_np, sigma_x_jnp,
#                        ns=16, extra_args=(spin_value,))
#
# For a correlation operator:
#   op = create_operator(OperatorTypeActing.Correlation, sigma_x_int_np, sigma_x_np, sigma_x_jnp,
#                        ns=16, extra_args=(spin_value,))

def create_add_operator(operator: Operator, multiplier: Union[float, int, complex], sites = None):
    """
    Create a tuple representing an operator with its associated sites and multiplier.
    This function takes an operator instance along with a multiplier and an optional list of sites,
    and returns a tuple containing the operator, its sites, and the multiplier. If the operator is of
    Global type, the provided sites are ignored and replaced with an empty list.
    Parameters:
        operator (Operator):
            The operator instance to be added.
        multiplier (Union[float, int, complex]):
            The scalar multiplier associated with the operator.
        sites (Optional[List[Any]], optional):
            The list of sites where the operator acts. Defaults to None. If the operator's type is Global, sites will be overridden with an empty list.
    
    ---
    Returns:
        Tuple[Operator, List[Any], Union[float, int, complex]]:
            A tuple containing the operator, the adjusted list of sites, and the multiplier.
    """

    
    # if the operator is of Global type, we don't want to add the states to the argument, pass empty list
    
    if operator.type == OperatorTypeActing.Global:
        return (operator, [], multiplier)
        
    # if sites is None, we pass an empty list
    if sites is None:
        sites_arg = []
    else:
        sites_arg = sites
    # create the operator tuple
    return (operator, sites_arg, multiplier)

####################################################################################################

def test_operators(op,
                state,
                ns              : Optional[int] = None,
                output_format   : str           ='tabs',
                r                               = 5,
                n                               = 5,
                add_args        : Optional[Tuple[Any, ...]] = None):
    """
    Test the operator using three syntaxes:
        - op(state),
        - op[state], 
        - op % state
    and display the results either in separate tabs or as a combined Markdown output.
    
    This function:
        - Shows the initial state.
        - Describes the operator being applied.
        - Measures execution time using %timeit.
        - Displays results (if the operator returns a tuple, it prints both parts).

    It works for any state (e.g., NumPy, JAX, etc.) by displaying the raw state.
    
    Parameters:
        op:
            The operator to be applied.
        state:
            The initial state.
        ns:
            The number of sites in the system.
        r:
            The number of repetitions for %timeit.
        n:
            The number of loops for %timeit.
        output_format:
            'tabs' (default) to display results in ipywidgets tabs,
                    or 'markdown' to display a combined Markdown output.
    Returns:
        pd.DataFrame:
            A DataFrame containing the test results.
    """
    
    import io
    from contextlib import redirect_stdout
    from IPython import get_ipython
    from IPython.display import display, Markdown
    import pandas as pd
    import general_python.common.binary as bin_mod
    
    if ns is None:
        ns = 32
    
    # Get the IPython shell
    ip = get_ipython()
    
    # Define tests for each operator syntax.
    if add_args is None:
        tests = {
            "op(state)"     : lambda: op(state),
            "op[state]"     : lambda: op[state],
            "op[op[state]]" : lambda: op[op[state][0]] if isinstance(state, (int, np.integer)) else op[op[state]],
            "op % state"    : lambda: op % state,
        }
    else:
        # If add_args is provided, we need to adjust the tests accordingly.
        # Note: We assume that add_args is a tuple of additional arguments.
        #       The first argument is the state, and the rest are additional arguments.
        #       The operator should be able to handle these additional arguments.
        tests = {
            f"op(state, {str(*add_args)})"     : lambda: op(state, *add_args),
            f"op[state, {str(*add_args)}]"     : lambda: op[state, *add_args],
            f"op % state, {str(*add_args)}"    : lambda: op % (state, *add_args),
        }
    
    # List to collect the data for each test.
    results_data = []
    
    # Go through each test.
    for method_expr, func in tests.items():
        # Capture the timing output from %timeit.
        f = io.StringIO()
        with redirect_stdout(f):
            try:
                ip.run_line_magic('timeit', f'-r {r} -n {n} {method_expr}')
            except Exception as e:
                print(f"Timeit error: {e}")

        # Get the captured output.
        timing_output = f.getvalue().strip()
        
        # Execute the operator call to get the result.
        try:
            result = func()
        except Exception as e:
            result = f"Error: {e}"
        
        # Check the state type
        if isinstance(state, (int, np.integer)):
            state_str   = f"({state}), which is b{bin_mod.int2binstr(state, ns)}"
            if isinstance(result, tuple):
                state_str_r = f"{[r for r in result[0]]}, which is {['b' + bin_mod.int2binstr(r, ns) for r in result[0]]}"
                val_str     = f"{[v for v in result[1]]}"
            else:
                state_str_r = ""
                val_str     = f"{[v for v in result]}"
        else:
            state_str   = str(state)
            state_str_r = str(result[0]) if isinstance(result, tuple) else str(result)
            val_str     = str(result[1]) if isinstance(result, tuple) else ""
            
        if len(state_str_r) > 0 and len(val_str) > 0:
            result_str = f"{state_str_r} with {val_str}"
        elif len(state_str_r) > 0:
            result_str = f"{state_str_r}"
        elif len(val_str) > 0:
            result_str = f"{val_str}"
        else:
            result_str = f"{result}"
        
        # Add the test information to our list.
        results_data.append({
            "Test Expression"   : method_expr,
            "Initial State"     : state_str,
            "Operator"          : op.name if hasattr(op, "name") else str(op),
            "Result"            : result_str,
            "Time Measurement"  : timing_output,
        })
    
    # Create a DataFrame from the collected results.
    df = pd.DataFrame(results_data)
    
    if output_format == 'markdown':
        # Convert the DataFrame to a Markdown table and display.
        from IPython.display import display, Markdown
        display(Markdown(df.to_markdown(index=False)))
    else:
        # Display the DataFrame directly.
        from IPython.display import display
        display(df)    
    return df

#####################################################################################################