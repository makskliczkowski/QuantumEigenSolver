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
from abc import ABC, abstractmethod
from enum import Enum, auto, unique
from typing import Optional, Callable, Union, Iterable
from typing import Union, Tuple, List               # type hints for the functions and methods
from functools import partial                       # partial function application for operator composition
####################################################################################################
from general_python.algebra.utils import DEFAULT_BACKEND, get_backend as get_backend, _JAX_AVAILABLE
from general_python.lattices import Lattice
# from Algebra.hilbert import HilbertSpace
####################################################################################################

if _JAX_AVAILABLE:
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

def _local_funct_wrapper(local_fun, *args):
    '''
    Basic wrapper for the local function - it is assumed that the local function
    takes the form (k, *args) -> (rows, values).
    
    This allows to remove the arguments from the function and to use the JIT compilation.
    Parameters:
        local_fun (Callable)    : The local function.    
        *args (Any)             : Additional arguments for the local function.
    Returns:
        Callable                : The wrapped local function.
    -----------   
    '''
    # check the type of the local function based on the number of arguments
    if local_fun.__code__.co_argcount == 1:
        # if the local function takes only one argument, return the function itself
        return local_fun
    elif local_fun.__code__.co_argcount == 2 and len(args) == 1:
        # if the local function takes two arguments and one argument is provided, wrap the function - LOCAL
        def local_funct_wrapper_in(k):
            return local_fun(k, args[0])
        return local_funct_wrapper_in
    elif local_fun.__code__.co_argcount > 2 and len(args) == 2:
        # if the local function takes more than two arguments and two arguments are provided, wrap the function - CORRELATION
        def local_funct_wrapper_in(k):
            return local_fun(k, args[0], args[1])
        return local_funct_wrapper_in
    # otherwise, wrap without possibility of JIT compilation
    def local_fun_wrapper_in(k):
        return local_fun(k, *args)
    return local_fun_wrapper_in

####################################################################################################

class OperatorFunction:
    '''
    OperatorFunction is a class that represents a mathematical operator that can be applied to a state. 
    The operator can be defined for different backends (integer, NumPy, JAX) and supports various operations 
    such as addition, subtraction, multiplication, and composition.
    Attributes:
    -----------
    - fun_int (Callable): The function defining the operator for integer states.
    - fun_np (Optional[Callable]): The function defining the operator for NumPy array states.
    - fun_jax (Optional[Callable]): The function defining the operator for JAX array states.
    - modifies_state (bool): A flag indicating whether the operator modifies the state.
    - necessary_args (int): The number of necessary arguments for the operator function.
    Methods:
    --------
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
    
    def _apply_global(self, s: Union[int, jnp.ndarray]) -> List[Tuple[Optional[Union[int, jnp.ndarray]], Union[float, complex]]]:
        """
        Apply the operator function to a given state.

        Parameters:
            s (int or jnp.ndarray): The state to which the operator is applied.

        Returns:
            A list of tuples (state, value), where each tuple contains the transformed state 
            (or None if not applicable) and its corresponding value.
        """
        if isinstance(s, jnp.ndarray):
            # If the state is a JAX array, use the JAX function
            return self._fun_jax(s)
        elif isinstance(s, np.ndarray):
            # If the state is a NumPy array, use the NumPy function
            return self._fun_np(s)
        return self._fun_int(s)
    
    # -----------
    
    def _apply_local(self, s: Union[int, jnp.ndarray], i) -> List[Tuple[Optional[Union[int, jnp.ndarray]], Union[float, complex]]]:
        """
        Apply the operator function to a given state.

        Parameters:
            s (int or jnp.ndarray): The state to which the operator is applied.
            i: Additional argument for the operator.

        Returns:
            A list of tuples (state, value), where each tuple contains the transformed state 
            (or None if not applicable) and its corresponding value.
        """
        if isinstance(s, jnp.ndarray):
            # If the state is a JAX array, use the JAX function
            return self._fun_jax(s, i)
        elif isinstance(s, np.ndarray):
            # If the state is a NumPy array, use the NumPy function
            return self._fun_np(s, i)
        # If the state is an integer, use the integer function
        # Note: The integer function should be able to handle the additional argument
        return self._fun_int(s, i)
    
    # -----------
    
    def _apply_correlation(self, s: Union[int, jnp.ndarray], i, j) -> List[Tuple[Optional[Union[int, jnp.ndarray]], Union[float, complex]]]:
        """
        Apply the operator function to a given state.

        Parameters:
            s (int or jnp.ndarray): The state to which the operator is applied.
            i, j: Additional arguments for the operator.

        Returns:
            A list of tuples (state, value), where each tuple contains the transformed state 
            (or None if not applicable) and its corresponding value.
        """
        if isinstance(s, jnp.ndarray):
            # If the state is a JAX array, use the JAX function
            return self._fun_jax(s, i, j)
        elif isinstance(s, np.ndarray):
            # If the state is a NumPy array, use the NumPy function
            return self._fun_np(s, i, j)
        # If the state is an integer, use the integer function
        # Note: The integer function should be able to handle the additional arguments
        return self._fun_int(s, i, j)
    
    def apply(self, s: Union[int, jnp.ndarray], *args) -> List[Tuple[Optional[Union[int, jnp.ndarray]], Union[float, complex]]]:
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
        if len(args) != self._necessary_args:
            raise ValueError(f"Invalid number of arguments for the operator function. Expected {self._necessary_args}, got {len(args)}.")
        
        result = None
        # apply the operator function based on the number of necessary arguments
        if self._necessary_args == 0:
            result = self._apply_global(s)
        elif self._necessary_args == 1:
            result = self._apply_local(s, *args)
        elif self._necessary_args == 2:
            result = self._apply_correlation(s, *args)
        else:
            if isinstance(s, jnp.ndarray):
                result = self._fun_jax(s, *args)
            elif isinstance(s, np.ndarray):
                result = self._fun_np(s, *args)
            else:
                result = self._fun_int(s, *args)

        # If the result is a tuple representing a single (state, value) pair
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], (int, np.ndarray, jnp.ndarray)):
            return result
        # If the result is already a list of (state, value) pairs
        elif isinstance(result, list) and all(isinstance(item, tuple) and len(item) == 2 for item in result):
            return result
        raise ValueError("Operator function returned an invalid type. Expected a tuple or a list of (state, value) pairs.")
    
    # -----------
    
    def __call__(self, s: Union[int, jnp.ndarray], *args) -> List[Tuple[Optional[Union[int, jnp.ndarray]], Union[float, complex]]]:
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
        
    def __mul__(self, other: Union[int, float, complex, np.int64, np.float64, np.complex128, 'OperatorFunction']):
        """
        Composition of two operator functions.
        Implements the operator composition: f * g ≡ f(g(n, ...), ...) for two operator functions f and g,
        where f is the current operator function and g is the other operator function.
        
        Notes:
        - If the operator functions have a different number of arguments, the maximum number is taken.
        - If the operator functions modify the state, the result is a list of pairs (state, value).
        - The multiplication is performed as f * g, meaning that (f * g)(state) = f(g(state)).
        - JIT compilation is applied to the JAX version when available.
        - If 'other' is a scalar, the operator function is simply scaled by that value.
        
        Parameters
        ----------
        other : Union[int, float, complex, np.int64, np.float64, np.complex128, OperatorFunction]
            Either a scalar (in which case the operator’s output values are multiplied by that scalar)
            or another operator function to be composed with the current operator function.
        
        Returns
        -------
        OperatorFunction
            A new OperatorFunction representing the composition or scaled operator.
        """
        # Handle scalar multiplication.
        if isinstance(other, (int, float, complex, np.int64, np.float64, np.complex128)):
            def fun_int(s, *args):
                return [(state, value * other) for state, value in self(s, *args)]
            def fun_np(s, *args):
                return [(state, value * other) for state, value in self(s, *args)]
            def fun_jax(s, *args):
                return [(state, value * other) for state, value in self(s, *args)]
            if _JAX_AVAILABLE:
                fun_jax = jax.jit(fun_jax)
            return OperatorFunction(fun_int, fun_np, fun_jax,
                                    modifies_state=True,
                                    necessary_args=self._necessary_args)
        
        # Make sure that 'other' is an OperatorFunction.
        if not isinstance(other, OperatorFunction):
            raise TypeError("Unsupported multiplication with type {}".format(type(other)))
        
        # Determine the maximum number of necessary arguments and whether any operator modifies state.
        other_modifies = other.modifies_state
        args_needed = max(self._necessary_args, getattr(other, 'necessary_args', 0))
        
        # Compose the operator functions.
        # Note: by definition (f * g)(s) = f(g(s)); here, g is 'other' and f is 'self'.
        if self._modifies_state:
            # If self modifies the state, assume that self returns a single pair.
            def composed_fun(s, *args):
                intermediate = other(s, *args)
                final_result = []
                for s1, v1 in intermediate:
                    next_result = self(s1, *args)
                    # Since self modifies the state, we assume it produces one pair.
                    # We keep the state from 'other' (s1) and multiply the values.
                    final_result.append((s1, v1 * next_result[0][1]))
                return final_result
        else:
            # Otherwise, self may produce multiple results.
            def composed_fun(s, *args):
                intermediate = other(s, *args)
                final_result = []
                for s1, v1 in intermediate:
                    next_result = self(s1, *args)
                    final_result.extend([(s2, v1 * v2) for s2, v2 in next_result])
                return final_result

        # Prepare the versions for different backends.
        fun_int = composed_fun
        fun_np = composed_fun
        fun_jax = composed_fun
        if _JAX_AVAILABLE:
            fun_jax = jax.jit(composed_fun)
        
        return OperatorFunction(fun_int, fun_np, fun_jax,
                                modifies_state=(self._modifies_state or other_modifies),
                                necessary_args=args_needed)

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
        # Handle scalar multiplication: other * self.
        if isinstance(other, (int, float, complex, np.int64, np.float64, np.complex128)):
            def fun_int(s, *args):
                return [(s, other * v) for s, v in self(s, *args)]
            def fun_np(s, *args):
                return [(s, other * v) for s, v in self(s, *args)]
            def fun_jax(s, *args):
                return [(s, other * v) for s, v in self(s, *args)]
            if _JAX_AVAILABLE:
                fun_jax = jax.jit(fun_jax)
            return OperatorFunction(fun_int, fun_np, fun_jax,
                                    modifies_state=self._modifies_state,
                                    necessary_args=self._necessary_args)
        
        # Ensure that other is an OperatorFunction.
        if not isinstance(other, OperatorFunction):
            raise TypeError("Unsupported reverse multiplication with type {}".format(type(other)))
        
        other_modifies = other.modifies_state
        args_needed = max(self._necessary_args, getattr(other, 'necessary_args', 0))
        
        def composed_fun(s, *args):
            """
            Composed function for reverse operator composition.
            First applies self (f), then applies other (g) on each resulting state.
            """
            # Apply self first.
            intermediate = self(s, *args)
            final_result = []
            # Then apply other to each result.
            for s1, v1 in intermediate:
                next_result = other(s1, *args)
                if other_modifies:
                    final_result.append((s1, v1 * next_result[0][1]))
                else:
                    final_result.extend([(s2, v1 * v2) for s2, v2 in next_result])
            return final_result
        
        fun_int = composed_fun
        fun_np  = composed_fun
        fun_jax = composed_fun
        if _JAX_AVAILABLE:
            fun_jax = jax.jit(composed_fun)
        
        return OperatorFunction(fun_int, fun_np, fun_jax,
                                modifies_state=self._modifies_state or other_modifies,
                                necessary_args=args_needed)

    # -----------
    
    def __mod__(self, other : 'OperatorFunction'):
        '''
        Applies operator to a given
        '''
        pass
        
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
        if _JAX_AVAILABLE:
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
        if _JAX_AVAILABLE:
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
        Wrap the operator function with additional arguments.
        
        Parameters
        ----------
        args : tuple
            Additional arguments that will be passed to the underlying operator function.
        
        Returns
        -------
        OperatorFunction
            A new operator function where the underlying function is partially applied with the provided arguments.
        """
        # For the “int”/NumPy version, we use partial application.
        def wrap_int(s, *more_args):
            return self._fun_int(s, *(args + more_args))
        
        def wrap_np(s, *more_args):
            return self._fun_np(s, *(args + more_args))
        
        def wrap_jax(s, *more_args):
            return self._fun_jax(s, *(args + more_args))
        
        if _JAX_AVAILABLE:
            wrap_jax = jax.jit(wrap_jax)
        
        return OperatorFunction(
            wrap_int,
            wrap_np,
            wrap_jax,
            modifies_state=self._modifies_state,
            necessary_args=self._necessary_args
        )
    
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
                fun_jax     : Optional[Callable]            = None,
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
        self._backend_str   = backend
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
        self._init_functions(op_fun, fun_int, fun_np, fun_jax)      # initialize the operator function
        
    #################################
    #! Initialize functions
    #################################
    
    def _init_functions(self, op_fun = None, fun_int = None, fun_np = None, fun_jax = None):
        """
        Initializes the operator functions and determines the operator type based on the 
        number of necessary arguments.
        Parameters:
            op_fun (OperatorFunction, optional): An instance of `OperatorFunction`. If provided, 
                it is directly assigned to the operator.
            fun_int (callable, optional): A Python function representing the internal implementation 
                of the operator. Must be provided if `op_fun` is not specified.
            fun_np (callable, optional): A NumPy-compatible implementation of the operator function.
            fun_jax (callable, optional): A JAX-compatible implementation of the operator function.
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
        
        if isinstance(scalar, (int, float, complex, np.int64, np.float64, np.complex128)):
            self._eigval *= scalar
        elif isinstance(scalar, Operator):
            #!TODO: Implement the right multiplication of the operator by another operator
            pass
        else:
            raise ValueError(f"Invalid type for multiplication.{type(scalar)}")
        return self
    
    # -------------------------------
    
    def __itruediv__(self, scalar):
        """ /= Operator with Division by Zero Check """
        
        if isinstance(scalar, (int, float, complex, np.int64, np.float64, np.complex128)):
            if scalar == 0:
                raise ZeroDivisionError("Division by zero in GeneralOperator.__itruediv__")
            self._eigval /= scalar
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
            #!TODO: Implement the right multiplication of the operator by another operator
            pass
        elif isinstance(other, (int, float, complex, np.int64, np.float64, np.complex128)):
            # modify the eigenvalue of the operator by the scalar
            out         =   self.copy()
            out._eigval *=  other
            return out
        else:
            raise ValueError(f"Invalid type for multiplication.{type(other)}")
    
    # -------------------------------
    
    def __mul__(self, other):
        """
        Multiplication of the operator by another operator or a scalar.
        In particular for an operator O and a scalar s, the operation s * O is defined.
        """
        
        if isinstance(other, Operator):
            #!TODO: Implement the multiplication of the operator by another operator
            pass
        elif isinstance(other, (int, float, complex, np.int64, np.float64, np.complex128)):
            # modify the eigenvalue of the operator by the scalar
            out         =   self.copy()
            out._eigval *=  other
            return out
        else:
            raise ValueError(f"Invalid type for multiplication.{type(other)}")
    
    # -------------------------------
    
    def __truediv__(self, scalar):
        """
        Division of the operator by a scalar.
        """
        
        if isinstance(scalar, (int, float, complex, np.int64, np.float64, np.complex128)):
            if scalar == 0:
                raise ZeroDivisionError("Division by zero in GeneralOperator.__truediv__")
            out = self.copy()
            out._eigval /= scalar
            return out
        else:
            raise ValueError(f"Invalid type for multiplication.{type(scalar)}")
    
    # -------------------------------
    
    def __rtruediv__(self, scalar):
        """
        Division of a scalar by the operator.
        """
        
        if isinstance(scalar, (int, float, complex, np.int64, np.float64, np.complex128)):
            if self._eigval == 0:
                raise ZeroDivisionError("Division by zero in GeneralOperator.__rtruediv__")
            out = self.copy()
            out._eigval = scalar / out._eigval
            return out
        else:
            raise ValueError(f"Invalid type for multiplication.{type(scalar)}")
    
    # -------------------------------
    
    #################################
    #! Setters and Getters
    #################################
    
    @property
    def eigval(self):
        return self._eigval
    
    @eigval.setter
    def eigval(self, val):
        self._eigval = val
        
    # -------------------------------
    
    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, val):
        self._name = val
        
    # -------------------------------
    
    @property
    def type(self):
        return self._type
    
    @type.setter
    def type(self, val):
        self._type = val
        
    # -------------------------------
    
    @property
    def quadratic(self):
        return self._quadratic
    
    @quadratic.setter
    def quadratic(self, val):
        self._quadratic = val
        
    # -------------------------------
    
    @property
    def acton(self):
        return self._acton
    
    @acton.setter
    def acton(self, val):
        self._acton = val
    
    # -------------------------------
    
    @property
    def modifies(self):
        return self._modifies
    
    @modifies.setter
    def modifies(self, val):
        self._modifies = val
    
    # -------------------------------
    
    @property
    def type_acting(self):
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
        return self._fun.np
    
    @property
    def jax(self):
        ''' Set the function that defines the operator '''
        return self._fun.jax
    
    # -------------------------------
    
    def override_matrix_fuction(self, function : Callable):
        """
        Override the matrix function of the operator.
        """
        self._matrix_fun = function
        
    #################################
    #! Apply the operator
    #################################
    
    # def chi(self, state):
    #     """
    #     Apply the operator to the state. Representative eigenvalue calculator for the operator.
    #     """
    #     return self._fun(state)
    
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
        if not isinstance(states, Iterable) or (hasattr(states, 'ndim') and states.ndim == 0):
            return self._fun(states)
        results     = [self._fun(state) for state in states]
        out, val    = zip(*results) if results else ([], [])
        return list(out), list(val)
    
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
        if not isinstance(states, Iterable) or (hasattr(states, 'ndim') and states.ndim == 0):
            return self._fun(states, i)
        results     = [self._fun(state, i) for state in states]
        out, val    = zip(*results) if results else ([], [])
        return list(out), list(val)
    
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
        if not isinstance(states, Iterable) or (hasattr(states, 'ndim') and states.ndim == 0):
            return self._fun(states, i, j)
        
        results     = [self._fun(state, i, j) for state in states]
        out, val    = zip(*results) if results else ([], [])
        return list(out), list(val)
        
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
        jax_maybe_av = _JAX_AVAILABLE and self._backend != np
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