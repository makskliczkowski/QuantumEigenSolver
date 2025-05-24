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
import numbers

#####################################################################################################
from abc import ABC
from enum import Enum, auto, unique
from typing import Optional, Callable, Union, Sequence, Any
from typing import Union, Tuple, List               # type hints for the functions and methods
from functools import partial                       # partial function application for operator composition
####################################################################################################
from general_python.algebra.utils import get_backend as get_backend, JAX_AVAILABLE, Array
from general_python.lattices import Lattice
# from Algebra.hilbert import HilbertSpace
####################################################################################################

if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
    from jax.experimental import sparse
    
    def make_jax_operator_closure(
        op_func         : Callable,
        sites           : Sequence[int],
        *static_args    : Any,
        **static_kwargs : Any) -> Callable:
        """
        Create a JAX-compiled closure with fixed `sites`, args, and kwargs.
        All additional parameters are assumed to be static (compile-time constants).

        Parameters
        ----------
        op_func : Callable
            The operator function to compile. Must accept (state, sites, *args, **kwargs).
        sites : Sequence[int]
            List or tuple of site indices to be treated as static.
        *static_args : Any
            Additional positional arguments to fix.
        **static_kwargs : Any
            Additional keyword arguments to fix.

        Returns
        -------
        Callable[[jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]
            A JIT-compiled function accepting only `state`.
        """
        static_sites = tuple(int(i) for i in sites)
        kwarg_keys   = tuple(sorted(static_kwargs))
        kwarg_vals   = tuple(static_kwargs[k] for k in kwarg_keys)

        @partial(jax.jit, static_argnums=range(1, 1 + 1 + len(static_args) + len(kwarg_vals)))
        def op_func_jax(state, static_sites_, *args_and_kwargs):
            args        = args_and_kwargs[:len(static_args)]
            kwargs_vals = args_and_kwargs[len(static_args):]
            kwargs      = dict(zip(kwarg_keys, kwargs_vals))
            return op_func(state, static_sites_, *args, **kwargs)

        #! Precompile the full closure with static args fixed
        def compiled_op(state):
            return op_func_jax(state, static_sites, *static_args, *kwarg_vals)

        return jax.jit(compiled_op)
    
else:
    jax                         = None
    jnp                         = np
    sparse                      = None
    make_jax_operator_closure   = lambda op_func, sites, *args, **kwargs: op_func

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
####################################################################################################

def op_func_wrapper(op_func: Callable, *args: Any) -> Callable:
    '''
    Wraps the operator function to handle different argument counts.
    Parameters:
        op_func (Callable):
            The operator function to be wrapped.
        *args (Any):
            Additional arguments to be passed to the operator function.
    '''
    if op_func.__code__.co_argcount == 1:
        return op_func
    else:
        return lambda k: op_func(k, *args)

####################################################################################################
#! Constants
####################################################################################################

_PYTHON_SCALARS = (int, float, complex, np.number)

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
        
    def is_local(self):
        """
        Check if the operator is a local operator.
        """
        return self == OperatorTypeActing.Local
        
    def is_correlation(self):
        """
        Check if the operator is a correlation operator.
        """
        return self == OperatorTypeActing.Correlation

    def __eq__(self, other):
        """
        Compare OperatorTypeActing with another OperatorTypeActing or with an integer.
        """
        if isinstance(other, OperatorTypeActing):
            return super().__eq__(other)
        elif isinstance(other, int):
            return self.value == other
        return NotImplemented

####################################################################################################
#! OperatorFunction
####################################################################################################

@numba.njit(cache=True)
def _empty_result_int():
    return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

####################################################################################################

def _make_mul_int_njit(outer_op_fun, inner_op_fun, allocator_m=2):
    """
    Creates a Numba-jitted function that composes two operator functions acting on integer-based quantum states.
    This function returns a compiled implementation that, given a quantum state and additional arguments, applies
    `inner_op_fun` to the state, then applies `outer_op_fun` to each resulting state, combining the coefficients
    appropriately. The result is a tuple of arrays containing the resulting states and their corresponding coefficients.
    If Numba compilation fails, a fallback implementation returning an empty result is provided.
    Args:
        outer_op_fun (callable):
            A function that takes a state and additional arguments, returning a tuple of
            (states, coefficients) as 1D numpy arrays.
        inner_op_fun (callable): 
            A function with the same signature as `outer_op_fun`, applied first.
    Returns:
        callable:
            A function with signature (state, *args) -> (states, coefficients), where `states` is a 1D numpy
            array of int64 and `coefficients` is a 1D numpy array of float64.
    """
    try:
        # if not isinstance(outer_op_fun, numba.core.registry.CPUDispatcher) or not isinstance(inner_op_fun, numba.core.registry.CPUDispatcher):
            # raise TypeError("Both outer_op_fun and inner_op_fun must be @numba.njit-compiled functions")
        
        @numba.njit(cache=True)
        def mul_int_impl(state, *args):
            g_states, g_coeffs = inner_op_fun(state, *args)
            if g_states.shape[0] == 0:
                return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

            total_estimate  = g_states.shape[0] * allocator_m
            res_states      = np.empty(total_estimate, dtype=np.int64)
            res_coeffs      = np.empty(total_estimate, dtype=g_coeffs.dtype)
            count           = 0

            for k in range(g_states.shape[0]):
                g_state_k   = g_states[k]
                g_coeff_k   = g_coeffs[k]
                f_states_k, f_coeffs_k = outer_op_fun(g_state_k, *args)

                for l in range(f_states_k.shape[0]):
                    if count >= res_states.shape[0]:
                        new_size = res_states.shape[0] * 2
                        tmp_s = np.empty(new_size, dtype=np.int64)
                        tmp_c = np.empty(new_size, dtype=np.float64)
                        tmp_s[:count] = res_states[:count]
                        tmp_c[:count] = res_coeffs[:count]
                        res_states = tmp_s
                        res_coeffs = tmp_c

                    res_states[count] = f_states_k[l]
                    res_coeffs[count] = g_coeff_k * f_coeffs_k[l]
                    count += 1

            return res_states[:count], res_coeffs[:count]

        return mul_int_impl
    except Exception as e: # Catch Numba errors
        print(f"Numba compilation failed for _make_mul_int_njit: {e}. Falling back to Python.")
        
        def mul_int_fallback(state, *args):
            return _empty_result_int()
        return mul_int_fallback

def _make_mul_np_njit(outer_op_fun, inner_op_fun, allocator_m=2):
    """
    Creates a function that applies two operator functions in sequence to a given state,
    efficiently handling the allocation of output arrays for states and coefficients.
    Parameters
    ----------
    outer_op_fun : callable
        A function that takes a state and additional arguments, and returns a tuple of
        (states, coefficients) as NumPy arrays. This function is applied after `inner_op_fun`.
    inner_op_fun : callable
        A function that takes a state and additional arguments, and returns a tuple of
        (states, coefficients) as NumPy arrays. This function is applied first.
    allocator_m : int, optional
        A multiplier used to estimate the maximum number of output states for pre-allocation.
        Default is 2.
    Returns
    -------
    mul_np_impl : callable
        A function that takes a state (as a NumPy array) and additional arguments, applies
        `inner_op_fun` followed by `outer_op_fun` to each resulting state, and returns a tuple:
        (resulting_states, resulting_coefficients), both as NumPy arrays.
    Notes
    -----
    - If the number of resulting states exceeds the pre-allocated size, the function will
      stop processing further states to avoid overflow.
    - If Numba compilation is intended but fails, the function falls back to a pure NumPy implementation.
    - The function assumes that the input state is a 1D NumPy array.
    """
    
    def mul_np_impl(state_np, *args_np):
        g_states, g_coeffs = inner_op_fun(state_np, *args_np)  # (M, D), (M,)
        if g_states.shape[0] == 0:
            return np.empty((0, state_np.shape[0]), dtype=state_np.dtype), np.empty((0,), dtype=g_coeffs.dtype)

        max_est     = g_states.shape[0] * allocator_m
        state_dim   = state_np.shape[0]  # assumes 1D input
        res_states  = np.empty((max_est, state_dim), dtype=state_np.dtype)
        res_coeffs  = np.empty((max_est,), dtype=g_coeffs.dtype)
        count       = 0

        for k in range(g_states.shape[0]):
            f_states_k, f_coeffs_k = outer_op_fun(g_states[k], *args_np)
            for j in range(f_states_k.shape[0]):
                if count >= max_est:
                    break
                res_states[count] = f_states_k[j]
                res_coeffs[count] = g_coeffs[k] * f_coeffs_k[j]
                count            += 1
        return res_states[:count], res_coeffs[:count]
    
    #! Attempt Numba compilation; if it fails, use the pure Python/NumPy version.
    try:
        return mul_np_impl
    except Exception as e:
        print("Numba compilation failed for _make_mul_np_njit. Falling back to pure NumPy version: ", e)
        return lambda state_np, *args_np: np.empty((0, state_np.shape[0]), dtype=state_np.dtype), np.empty((0,), dtype=np.float64)

def _make_mul_jax_vmap(outer_op_fun_jax, inner_op_fun_jax):

    if not JAX_AVAILABLE:
        return None 
        
    if outer_op_fun_jax is None or inner_op_fun_jax is None:
        return None

    @partial(jax.jit, static_argnums=())
    def mul_jax_impl(state_initial, *args):
        g_states, g_coeffs = inner_op_fun_jax(state_initial, *args)  # (M, D), (M,)

        def apply_outer(state, coeff):
            f_states, f_coeffs = outer_op_fun_jax(state, *args)  # (K, D), (K,)
            return f_states, coeff * f_coeffs

        # g_states: (M, D), g_coeffs: (M,)
        f_states, f_coeffs = jax.vmap(apply_outer, in_axes=(0, 0))(g_states, g_coeffs)

        s_dim = f_states.shape[-1]
        return f_states.reshape(-1, s_dim), f_coeffs.reshape(-1)
    
    try:
        # Example call to force compilation and catch errors early (optional)
        # This requires knowing typical shapes and dtypes for s_initial and args_for_ops
        # For instance, if s_initial is (D,) and args_for_ops is empty for global ops:
        # test_state = jnp.zeros((1,), dtype=jnp.float32) # Or a more representative state
        # if necessary_args == 0: mul_jax_impl.lower(test_state).compile()
        # elif necessary_args == 1: mul_jax_impl.lower(test_state, 0).compile()
        # This is complex to generalize here. JAX will compile on first actual call.
        pass
    except jax.errors.JAXTypeError as e:
        print(f"JAX compilation failed for _make_mul_jax_vmap: {e}. JAX operations will not be available for this composed operator.")
        return None
    return mul_jax_impl

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
        self._fun_int           =   fun_int
        self._fun_np            =   fun_np
        self._fun_jax           =   fun_jax
        self._modifies_state    =   modifies_state            # flag for the operator that modifies the state
        self._necessary_args    =   int(necessary_args)       # number of necessary arguments for the operator function
        self._acting_type       =   OperatorTypeActing.Global       if self._necessary_args == 0 else \
                                    OperatorTypeActing.Local        if self._necessary_args == 1 else \
                                    OperatorTypeActing.Correlation  if self._necessary_args == 2 else OperatorTypeActing.Global
        self._dispatch          =   self._apply_local if self._necessary_args == 1 else \
                                    self._apply_correlation if self._necessary_args == 2 else \
                                    self._apply_global if self._necessary_args == 0 else self._apply_any
    
    # -----------
    
    def _choose_apply(self, s: Union[int, np.ndarray]) -> Callable:
        """
        Choose the appropriate function to apply based on the type of state and arguments.
        
        Parameters:
            s (int or np.ndarray):
                The state to which the operator is applied.
        Returns:
            A callable function that applies the operator to the state.
        """
        if isinstance(s, jnp.ndarray):
            return self._fun_jax
        elif isinstance(s, np.ndarray):
            return self._fun_np
        else:
            return self._fun_int
    
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
        if JAX_AVAILABLE and isinstance(s, jnp.ndarray) and self._fun_jax is not None:
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
        if JAX_AVAILABLE and isinstance(s, jnp.ndarray) and self._fun_jax is not None:
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
        if JAX_AVAILABLE and isinstance(s, jnp.ndarray) and self._fun_jax is not None:
            # If the state is a JAX array, use the JAX function
            return self._fun_jax(s, i, j)
        elif isinstance(s, np.ndarray) and self._fun_np is not None:
            # If the state is a NumPy array, use the NumPy function
            return self._fun_np(s, i, j)
        # If the state is an integer, use the integer function
        # Note: The integer function should be able to handle the additional arguments
        return self._fun_int(s, i, j)
    
    # -----------
    
    def _apply_any(self, s: Union[int, np.ndarray], *args) -> List[Tuple[Optional[Union[int]], Union[float, complex]]]:
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
        """
        if isinstance(s, jnp.ndarray) and self._fun_jax is not None:
            result = self._fun_jax(s, *args)
        elif isinstance(s, np.ndarray) and self._fun_np is not None:
            result = self._fun_np(s, *args)
        else:
            result = self._fun_int(s, *args)
        return result
    
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
        
        # apply the operator function based on the number of necessary arguments
        result = self._dispatch(s, *args)

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
        
        if JAX_AVAILABLE and isinstance(constant, jax.Array):
            pass
        elif isinstance(constant, np.ndarray):  # Numpy array scalar
            constant = constant.item()          # Convert to Python scalar
        
        def mul_int(s, *args):
            st, val = self._fun_int(s, *args)
            return st, np.array(val) * constant # Ensure val is array for broadcasting

        @numba.njit(cache=True)
        def mul_np(s, *args):
            st, val = self._fun_np(s, *args)    # Assume fun_np returns (np.array, np.array)
            return st, val * constant 
        
        mul_jax_defined = False
        if JAX_AVAILABLE and self._fun_jax is not None:
            
            @jax.jit
            def mul_jax_impl(s_jax, *args_jax):
                st_jax, val_jax = self._fun_jax(s_jax, *args_jax)
                return st_jax, val_jax * constant
            mul_jax         = jax.jit(mul_jax_impl)
            mul_jax_defined = True
        else:
            mul_jax         = None

        return OperatorFunction(mul_int, mul_np, mul_jax if mul_jax_defined else None,
                        modifies_state=self._modifies_state,
                        necessary_args=self._necessary_args)
        
    # =========================================================================
    #! Composition: f * g  (i.e. (f * g)(s) = f(g(s)) )
    # =========================================================================
    
    def __mul__(self, other: Union[float, 'OperatorFunction']) -> 'OperatorFunction':

        is_python_scalar    = isinstance(other, _PYTHON_SCALARS)
        is_jax_scalar       = False
        if JAX_AVAILABLE:
            is_jax_scalar   = isinstance(other, jax.Array) and other.ndim == 0
        
        if is_python_scalar or is_jax_scalar:
            return self._multiply_const(other)
        
        if not isinstance(other, OperatorFunction):
            return NotImplementedError("Incompatible operator function")

        #! Type consistency check for operator composition f * g (self is f, other is g)
        if self._acting_type != other._acting_type:
            raise ValueError(f"Operator acting type mismatch for composition: {self._acting_type} (f) vs {other._acting_type} (g)")
        
        # The composed operator (f*g) modifies state if f modifies g's output, or if g modifies initial state.
        # If f is diagonal (modifies_state=False), it doesn't change g's state structure.
        # If f can change state (modifies_state=True), it can change g's output state.
        composed_modifies_state = self._modifies_state or other._modifies_state
        
        # Number of arguments for the composed operator
        # This is correct due to the acting_type check ensuring they are compatible.
        composed_necessary_args = self._necessary_args 
        #? or max(self._necessary_args, other._necessary_args)

        # Create composed functions: (f * g)(s) = f(g(s))
        # self._fun_xxx is f, other._fun_xxx is g
        
        # Note on branching: Current _make_mul_int/np_njit take f_coeff[0], f_state[0].
        # This means if f (self) branches, only its first branch is propagated.
        # The new _make_mul_jax_vmap handles branching in f.
        
        composed_fun_int = _make_mul_int_njit(self._fun_int, other._fun_int)
        composed_fun_np  = _make_mul_np_njit(self._fun_np, other._fun_np)
        composed_fun_jax = _make_mul_jax_vmap(self._fun_jax, other._fun_jax) if JAX_AVAILABLE else None
        
        return OperatorFunction(
            composed_fun_int,
            composed_fun_np,
            composed_fun_jax,
            modifies_state=composed_modifies_state,
            necessary_args=composed_necessary_args
        )
            
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
        is_python_scalar    = isinstance(other, _PYTHON_SCALARS)
        is_jax_scalar       = False
        if JAX_AVAILABLE:
            is_jax_scalar   = isinstance(other, jax.Array) and other.ndim == 0

        if is_python_scalar or is_jax_scalar:
            return self._multiply_const(other) 
        
        if not isinstance(other, OperatorFunction):
            return NotImplementedError("Incompatible operator function")

        # For other * self (g * f), where g is `other`, f is `self`
        if other._acting_type != self._acting_type:
            raise ValueError(f"Operator acting type mismatch for composition: {other._acting_type} (g) vs {self._acting_type} (f)")
        
        composed_modifies_state = other._modifies_state or self._modifies_state
        composed_necessary_args = self._necessary_args # From check, same as other._necessary_args

        # Create composed functions: (g * f)(s) = g(f(s))
        # other._fun_xxx is g, self._fun_xxx is f
        composed_fun_int = _make_mul_int_njit(other._fun_int, self._fun_int)
        composed_fun_np  = _make_mul_np_njit(other._fun_np, self._fun_np)
        composed_fun_jax = _make_mul_jax_vmap(other._fun_jax, self._fun_jax) if JAX_AVAILABLE else None

        return OperatorFunction(
            composed_fun_int,
            composed_fun_np,
            composed_fun_jax,
            modifies_state=composed_modifies_state,
            necessary_args=composed_necessary_args
        )

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
            result_f                = np.array(self(s, *args))
            result_g                = np.array(other(s, *args))
            # Extract states and values.
            states_f, values_f      = result_f[:, 0], result_f[:, 1]
            states_g, values_g      = result_g[:, 0], result_g[:, 1]
            # Combine results.
            combined_states         = np.concatenate([states_f, states_g])
            combined_values         = np.concatenate([values_f, values_g])
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
    
    def wrap(self, *fixed_args):
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
        
        new_necessary_args = self._necessary_args - len(fixed_args)
        if new_necessary_args < 0:
            raise ValueError("Too many arguments provided for wrapping.")

        def wrap_int(s, *runtime_args):
            return self._fun_int(s, *(fixed_args + runtime_args))
        
        def wrap_np(s, *runtime_args):
            return self._fun_np(s, *(fixed_args + runtime_args))
        
        wrapped_fun_jax = None
        if JAX_AVAILABLE and self._fun_jax is not None:
            def wrap_jax_impl(s_jax, *runtime_args_jax):
                return self._fun_jax(s_jax, *(fixed_args + runtime_args_jax))
            wrapped_fun_jax = jax.jit(wrap_jax_impl)

        return OperatorFunction(wrap_int, 
                                wrap_np,
                                wrapped_fun_jax,
                                modifies_state  =   self._modifies_state,
                                necessary_args  =   new_necessary_args)

    # -----------

####################################################################################################

class Operator(ABC):
    """
    A class to represent a general operator acting on a Hilbert space.
    
    Attributes:
        - op_fun (OperatorFunction):
            The operator function that defines the operator.
        - fun_int (Callable):
            The function defining the operator for integer states.
        - fun_np (Optional[Callable]):
            The function defining the operator for NumPy array states.
        - fun_jnp (Optional[Callable]):
            The function defining the operator for JAX array states.
        - eigval (float):
            The eigenvalue of the operator.
        - lattice (Optional[Lattice]):
            The lattice object representing the physical system.
        - ns (Optional[int]):
            The number of sites in the system.
        - typek (Optional[SymmetryGenerators]):
            The symmetry generators of the operator.
        - name (str):
            The name of the operator.
        - modifies (bool):
            Flag for the operator that modifies the state.
        - quadratic (bool):
            Flag for the quadratic operator.
        - backend (str):
            The backend for the operator (default is 'default').
        - backend_sp (str):
            The backend for the operator (default is 'default').
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
            Ns (int) :
                The number of sites in the system.
            Nhl (int) :
                The local Hilbert space dimension - 2 for spin-1/2, 4 for spin-1, etc (default is 2).
            Nhint (int) :
                The number of modes (fermions, bosons, etc. on each site).
            lattice (Lattice)   :
                The lattice object.
            name (str)          :
                The name of the operator.
            type (str)          :
                The type of the operator.
            eigval              :
                The eigenvalue of the operator (default is 1.0).
            quadratic (bool)    :
                Flag for the quadratic operator.
            acton (bool)        :
                Flag for the action of the operator on the local physical space.
            modifies (bool)     :
                Flag for the operator that modifies the state.
            backend (str)       :
                The backend for the operator (default is 'default').
            fun_int (callable)  :
                The function that defines the operator - it shall take a state (or a list of states) 
                and return the transformed state (or a list of states). States can be represented as 
                integers or numpy arrays or JAX arrays. This enables the user to define any operator 
                that can be applied to the state. The function shall return a list of pairs (state, value).
            fun_np (callable)   :
                The function that defines the operator for NumPy arrays.
            fun_jnp (callable)  :
                The function that defines the operator for JAX arrays.
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
        self._necessary_args= kwargs.get("necessary_args", 0)       # number of necessary arguments for the operator function
        self._fun           = None                                  # the function that defines the operator - it is set to None if not provided
        self._init_functions(op_fun, fun_int, fun_np, fun_jnp)      # initialize the operator function
    
    def __repr__(self):
        """
        String representation of the operator.
        """
        return f"Operator({self._name}, type_acting={self.type_acting.name}, eigval={self._eigval}, type={self._type.name})"
        
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
    
    def __mul__(self, other):
        new_kwargs = self.__dict__.copy()
        new_kwargs.pop('_fun', None)    # Remove _fun, it will be new
        new_kwargs.pop('_name', None)   # Remove _name, it will be new
        new_kwargs.pop('_eigval', None) # Eigval handled separately or combined if meaningful

        if isinstance(other, Operator):
            new_fun                 = self._fun * other._fun
            #! If eigvals are simple scalars, product makes sense.
            new_eigval              = self._eigval * other._eigval 
            new_name                = f"({self._name} * {other._name})"
            # Modifies state is handled by OperatorFunction composition
            # Quadratic, acton etc. might need rules for combining
            # For now, inherit from self, or define combination rules
            new_kwargs['quadratic'] = self._quadratic or other._quadratic #? what to do here?
        elif isinstance(other, _PYTHON_SCALARS) or (JAX_AVAILABLE and isinstance(other, jax.Array) and other.ndim == 0):
            new_fun                 = self._fun * other
            new_eigval              = self._eigval * other
            new_name                = f"({self._name} * {other})"
        else:
            return NotImplementedError("Incompatible operator function")
        return Operator(op_fun=new_fun, name=new_name, eigval=new_eigval,
                        ns=new_kwargs['_ns'],
                        lattice=new_kwargs['_lattice'], modifies=new_kwargs['_modifies'],
                        backend=new_kwargs['_backend'], **new_kwargs)
        

    def __rmul__(self, other):
        new_kwargs = self.__dict__.copy()
        new_kwargs.pop('_fun', None)
        new_kwargs.pop('_name', None)
        new_kwargs.pop('_eigval', None)

        if isinstance(other, _PYTHON_SCALARS) or (JAX_AVAILABLE and isinstance(other, jax.Array) and other.ndim == 0):
            new_fun     = other * self._fun # OperatorFunction scalar rmul (same as mul)
            new_eigval  = other * self._eigval
            new_name    = f"({other} * {self._name})"
        elif isinstance(other, Operator):
            new_fun     = other._fun * self._fun
            #! If eigvals are simple scalars, product makes sense.
            new_eigval  = other._eigval * self._eigval
            new_name    = f"({other._name} * {self._name})"
        else:
            return NotImplementedError("Incompatible operator function")
        
        return Operator(op_fun=new_fun, name=new_name, eigval=new_eigval, **new_kwargs)
    
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
    
    def _apply_global(self, states : Union[int, list, Array]):
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
    
    def _apply_local(self, states : Union[int, list, Array], i):
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
    
    def _apply_correlation(self, states : Union[int, list, Array], i, j):
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
        
    def apply(self, states : list | Array, *args):
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
    
    def __call__(self, states: list | Array, *args):
        """
        Apply the operator to the state. 
        
        Args:
            states            : list of states to which the operator is applied.
            args              : Additional arguments for the operator - inform how to act on a state.
                                If there no arguments, the operator acts on the state as a whole - global operator.
                                If there are arguments, the operator acts on the state locally - local operator (e.g., site-dependent).
        """
        return self.apply(states, *args)
    
    def __getitem__(self, states: list | Array,):
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
    
    def _matrix_no_hilbert_np(self, 
                            dim             : int, 
                            is_sparse       : bool, 
                            wrapped_funct, 
                            dtype, 
                            max_loc_upd     : int = 1,
                            verbose         : bool = False,
                            **kwargs) -> np.ndarray:
        """
        Generate the matrix form of the operator without Hilbert space.
        """
        # create a dummy Hilbert space for convenience
        from Algebra.hilbert import HilbertSpace
        from Algebra.Operator.operator_matrix import operator_create_np
        
        dummy_hilbert   = HilbertSpace(nh = dim, backend = self._backend)
        if verbose:
            dummy_hilbert.log("Calculating the Hamiltonian matrix using NumPy...", lvl = 2)

        # calculate the time to create the matrix
        t1              = time.time()
        matrix          = operator_create_np(ns                 = None, 
                                            hilbert_space       = dummy_hilbert,
                                            local_fun           = wrapped_funct,
                                            max_local_changes   = max_loc_upd,
                                            is_sparse           = is_sparse,
                                            start               = None,
                                            dtype               = dtype)
        time_taken      = time.time() - t1
        if verbose:
            dummy_hilbert.log(f"Time taken to create the matrix {self._name}: {time_taken:.2e} seconds", lvl=2)
        return matrix
        
    def _matrix_no_hilbert_jax(self, dim: int, is_sparse: bool, wrapped_funct, dtype, max_loc_upd:int = 1):
        """
        Generate the matrix form of the operator without Hilbert space.
        """
        from Algebra.hilbert import HilbertSpace
        
        # create a dummy Hilbert space for convenience
        dummy_hilbert = HilbertSpace(nh = dim, backend = self._backend)
        dummy_hilbert.log("Calculating the Hamiltonian matrix using JAX...", lvl = 2)
        #!TODO: Implement the JAX version of the matrix function
        return None
        
    def matrix(self, *args, dim = None, matrix_type = 'sparse',
            hilbert_1 = None, hilbert_2 = None, use_numpy: bool = True, **kwargs) -> Array | None:
        """
        Generates the matrix representation of the operator.

        Parameters:
        - param dim             : Dimension of the matrix.
        - param matrix_type     : Type of matrix ("dense" or "sparse").
        - param *args           : Additional arguments for the operator function - moved to self._fun.
        - param **kwargs        : Additional keyword arguments for the operator function.
        :return: The matrix representation of the operator.
        """
        
        # check the dimension of the matrix
        dim1, dim2          = None, None
        matrix_hilbert      = 'None'
        if hilbert_1 is not None and hilbert_2 is not None:
            dim1, dim2      = hilbert_1.nh, hilbert_2.nh
            matrix_hilbert  = 'double'
        elif hilbert_1 is not None and hilbert_2 is None:
            dim1, dim2      = hilbert_1.nh, hilbert_1.nh
            matrix_hilbert  = 'single'
        elif hilbert_1 is None and hilbert_2 is not None:
            hilbert_1       = hilbert_2
            dim1, dim2      = hilbert_2.nh, hilbert_2.nh
            matrix_hilbert  = 'single'
        else:
            if dim is None:
                raise ValueError("Dimension or at least one Hilbert space must be provided.")
            dim1, dim2      = dim, dim
            matrix_hilbert  = 'None'

        verbose      = kwargs.get('verbose', False)
        
        # check if there are functions from the Hilbert space
        jax_maybe_av = JAX_AVAILABLE and self._backend != np
        is_sparse    = (matrix_type == 'sparse')
        use_numpy    = use_numpy or (not jax_maybe_av)
        
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
                return self._matrix_no_hilbert_np(dim1, is_sparse, wrapped_fun, dtype, max_loc_upd, verbose, **kwargs)
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
    the signature:
        - (state, *args) -> (new_state, op_value) for global
        - or (state, site, *args) for local,
        - and (state, site1, site2, *args) for correlation.
    
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
    
    # Ensure we know ns - the number of sites or modes for the operator
    if lattice is not None:
        ns = lattice.ns
    else:
        assert ns is not None, "Either lattice or ns must be provided."
    
    #! Global operator: the operator acts on a specified set of sites (or all if sites is None)
    if type_act == OperatorTypeActing.Global.value or sites is not None:
        
        # If sites is None, we act on all sites.
        if sites is None or len(sites) == 0:
            sites = list(range(ns))
        sites           = tuple(sites) if isinstance(sites, list) else sites
        sites_np        = np.array(sites, dtype = np.int32)
        
        @numba.njit
        def fun_int(state):
            return op_func_int(state, ns, sites, *extra_args)
        
        @numba.njit
        def fun_np(state):
            return op_func_np(state, sites_np, *extra_args)

        if JAX_AVAILABLE:
            fun_jnp = make_jax_operator_closure(op_func_jnp, sites, *extra_args)
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
    
    #! Local operator: the operator acts on one specific site. The returned functions expect an extra site argument.
    elif type_act == OperatorTypeActing.Local.value:
        
        @numba.njit
        def fun_int(state, i):
            sites_1 = np.array([i], dtype=np.int32)
            return op_func_int(state, ns, sites_1, *extra_args)
        
        @numba.njit
        def fun_np(state, i):
            sites_1 = np.array([i], dtype=np.int32)
            return op_func_np(state, sites_1, *extra_args)
        
        if JAX_AVAILABLE:
            @jax.jit
            def fun_jnp(state, i):
                sites_jnp = jnp.array([i], dtype = jnp.int32)
                return op_func_jnp(state, sites_jnp, *extra_args)
                # sites_jnp = tuple([i])
                # op = make_op_jnp([i])
                # return op(state)
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
    
    #! Correlation operator: the operator acts on a pair of sites.
    elif type_act == OperatorTypeActing.Correlation.value:
        
        @numba.njit
        def fun_int(state, i, j):
            sites_2 = np.array([i, j], dtype=np.int32)
            return op_func_int(state, ns, sites_2, *extra_args)
        
        @numba.njit
        def fun_np(state, i, j):
            sites_2 = np.array([i, j], dtype=np.int32)
            return op_func_np(state, sites_2, *extra_args)
        
        if JAX_AVAILABLE:
            @jax.jit
            def fun_jnp(state, i, j):
                sites_jnp = jnp.array([i, j], dtype = jnp.int32)
                return op_func_jnp(state, sites_jnp, *extra_args)
                # op = make_op_jnp([i, j])
                # return op(state)
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
#! Operator shape
####################################################################################################

@numba.njit
def ensure_operator_output_shape_numba(state_out: np.ndarray, coeff_out: np.ndarray):
    """
    Ensure output is consistently 2D for states and 1D for coefficients.
    Works only when inputs already have correct shape or are known to be 1D scalars.
    """
    # if isinstance(coeff_out, (int, float, complex, np.int8, np.int16, np.int32, np.int64, np.float32, np.float64, np.complex64, np.complex128)):
    #     coeff_out = np.array([coeff_out])
    
    if state_out.ndim == 1 and coeff_out.ndim == 1:
        out_state = state_out.reshape(1, state_out.shape[0])
        out_coeff = coeff_out
    elif state_out.ndim == 1 and coeff_out.ndim == 0:
        out_state = state_out.reshape(1, state_out.shape[0])
        out_coeff = np.empty(1, dtype=coeff_out.dtype)
        out_coeff[0] = coeff_out
    elif state_out.ndim == 2 and coeff_out.ndim == 1:
        out_state = state_out
        out_coeff = coeff_out
    else:
        raise ValueError("Unsupported shape combination in ensure_operator_output_shape_numba")

    return out_state, out_coeff

if JAX_AVAILABLE:
    @jax.jit
    def ensure_operator_output_shape_jax(state_out  : jnp.ndarray,
                                        coeff_out   : jnp.ndarray):
        r"""
        Ensure (state, coeff) output is 2D-batched: (N, L), (N,)
        where N is the number of states and L is the dimension of the state.
        Using JAX for JIT compilation.

        Parameters:
            state_out (jnp.ndarray):
            The output state array.
            coeff_out (jnp.ndarray):
            The output coefficient array.
        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]:
            A tuple containing the reshaped state and coefficient arrays.
            With the first dimension being the number of states that the operator returns
            and the second dimension being the dimension of the state.
            This corresponds to matrix elements of the operator:
            .. math::
                \{\, |s'\rangle,\ \langle s'|O|s\rangle\,\}
        """
        # Convert scalars to arrays
        state_out = jnp.atleast_1d(state_out)
        coeff_out = jnp.atleast_1d(coeff_out)

        # Handle scalar state (single integer)
        if state_out.ndim == 1 and coeff_out.shape[0] == 1:
            state_out = state_out.reshape(1, -1)  # (1, 1) or (1, L)
        elif state_out.ndim == 2:
            # already batched: do nothing
            pass
        else:
            raise ValueError("Unsupported state_out shape for JAX.")

        return state_out, coeff_out
else:
    def ensure_operator_output_shape_jax(state_out  : np.ndarray,
                                        coeff_out   : np.ndarray):
        r"""
        Ensure (state, coeff) output is 2D-batched: (N, L), (N,)
        where N is the number of states and L is the dimension of the state.
        Using JAX for JIT compilation.

        Parameters:
            state_out (jnp.ndarray):
            The output state array.
            coeff_out (jnp.ndarray):
            The output coefficient array.
        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]:
            A tuple containing the reshaped state and coefficient arrays.
            With the first dimension being the number of states that the operator returns
            and the second dimension being the dimension of the state.
            This corresponds to matrix elements of the operator:
            .. math::
                \{\, |s'\rangle,\ \langle s'|O|s\rangle\,\}
        """
        return ensure_operator_output_shape_numba(state_out, coeff_out)

# ##################################################################################################

def initial_states(ns       : int,
                display     : bool                  = False, 
                int_state   : Optional[int]         = None,
                np_state    : Optional[np.ndarray]  = None) -> tuple:
    '''
    Create initial states for testing the operator. It generates:
        - int_state: a random integer state (int) in the range [0, 2**ns)
        - np_state: a random NumPy state (np.ndarray) of size ns
        - jnp_state: a random JAX state (jnp.ndarray) of size ns
    The function also checks if the integer state is out of bounds and generates a random state if necessary.
    The function returns the generated states as a tuple.
    Parameters:
        ns (int):
            The number of sites in the system.
        display (bool):
            Whether to display the generated states.
        int_state (Optional[int]):
            The integer state to be used. If None, a random state is generated.
        np_state (Optional[np.ndarray]):
            The NumPy state to be used. If None, a random state is generated.
    Returns:
        tuple:
            A tuple containing the generated states:
            - int_state: a random integer state (int) in the range [0, 2**ns)
            - np_state: a random NumPy state (np.ndarray) of size ns
            - jnp_state: a random JAX state (jnp.ndarray) of size ns
    '''
    import QES.general_python.common.binary as _bin_mod
    from general_python.common.display import display_state
    
    #! Take the integer state as input
    int_state = np.random.randint(0, 2**(ns%64), dtype=np.int32) if int_state is None else int_state
    
    #! Check the integer state whether it is out of bounds
    if np_state is None:
        if ns >= 64:
            np_state  = np.random.randint(0, 2, size=ns).astype(np.float32)
        else:
            np_state  = _bin_mod.int2base_np(int_state, size = ns, value_true=1, value_false=0).astype(np.float32)
    
    if JAX_AVAILABLE:
        jnp_state = jnp.array(np_state, dtype=jnp.float32) if np_state is not None else None
    else:
        jnp_state = None
    
    if display:
        display_state(int_state, ns,    label = "Integer state")
        display_state(np_state,  ns,    label = "NumPy state")
        display_state(jnp_state, ns,    label = "JAX state")
        
    return int_state, np_state, jnp_state

####################################################################################################
#! Test the operator
####################################################################################################

def _dispatch(op    : Operator,
            state   : Union[int, np.ndarray],
            lat     : Lattice,
            is_int  : bool,
            to_bin  : Optional[Callable[[int, int], str]] = None,
            lab     : Optional[str] = None,
            *, 
            i=None, j=None, sites=None, just_time=False):
    """
    Call *op* with the right signature and send its first output to
    `display_operator_action`.
    
    Parameters
    ----------
    op : Operator
        The operator acting on the state.
    state : int | np.ndarray
        The state on which the operator acts.
    lat : Lattice 
        The lattice object providing the number of sites.
    is_int : bool
        Whether the state is an integer or not.
    to_bin : Callable[[int, int], str], optional
        Function to convert integer to binary string.
    lab : str, optional
        The label for the operator.
    i : int, optional
        The first site index for the operator.
    j : int, optional
        The second site index for the operator.
    just_time : bool, default = False
        If True, only measure the time taken for the operation without
        displaying the results. This is useful for benchmarking.
    Returns
    -------
    Tuple[Union[int, np.ndarray], float]
        The new state and the coefficient after applying the operator.
    Notes
    -----
    * The function dispatches the operator call based on the type of operator
        (global, local, or correlation) and the type of state (integer or array).
    * The function also handles the display of the operator action using
        `display_operator_action` if `just_time` is False.
    * The function returns the new state and the coefficient after applying
        the operator.
    * The function uses `numba` for JIT compilation to speed up the operator
        application.
    * The function handles both NumPy and JAX arrays for the state.
    * The function uses `jnp` for JAX arrays if available.
    """
    if not just_time:
        from QES.general_python.common.display import display_operator_action
    
    state_act = state
    # call signature depends on OperatorTypeActing
    if i is None:                        # Global operator
        st_out, coeff = op(state_act)
    elif j is None:                      # Local operator
        if is_int:
            st_out, coeff = op(state_act, i)
        elif isinstance(state_act, (np.ndarray)):
            sites         = np.array([i])
            st_out, coeff = op(state_act, *sites)
        elif isinstance(state_act, (jnp.ndarray)):
            sites         = jnp.array([i])
            st_out, coeff = op(state_act, sites[0])
    else:                                # Correlation operator
        if is_int:
            st_out, coeff = op(state_act, i, j)
        elif isinstance(state_act, (np.ndarray)):
            sites         = np.array([i, j])
            st_out, coeff = op(state_act, *sites)
        elif isinstance(state_act, (jnp.ndarray)):
            sites         = jnp.array([i, j])
            st_out, coeff = op(state_act, sites[0], sites[1])
            
    #! choose what to show depending on state representation
    new_state = st_out
    new_coeff = coeff
    if not just_time:
        site_lbl = ''
        if i is not None:
            site_lbl = str(i)
        if j is not None:
            site_lbl += f",{j}"
        if sites is not None:
            site_lbl = ",".join(map(str, sites))

        display_operator_action(f"\\quad \\quad {lab}",
                                site_lbl,
                                state,
                                lat.ns,
                                new_state,
                                new_coeff,
                                to_bin = to_bin)
    return new_state, new_coeff

def test_operator_on_state(op           : Union[Operator, Sequence[Operator]],
                        lat             : Lattice,
                        state           : Union[int, Array],
                        *,
                        ns              : Optional[int] = None,
                        op_acting       : "OperatorTypeActing" = OperatorTypeActing.Local,
                        op_label        = None,
                        to_bin          = None,
                        just_time       = False,
                        sites           : Optional[List[int]] = None,
                        ) -> None:
    r"""
    Pretty-print the action of *one or several* lattice operators
    on a basis state or wave-function.

    Parameters
    ----------
    op : Operator or sequence[Operator]
        The operator(s) Ô acting on 0, 1, or 2 sites.
    lat : Lattice
        Provides the number of sites ``lat.ns`` = :math:`N_s`.
    ns : int, optional
        Number of sites.  If *None*, uses ``lat.ns``.
    state : int | np.ndarray | jax.numpy.ndarray
        *Basis state* (integer encoding) or *wave-function* :math:`|\psi\rangle`.
    op_acting : OperatorTypeActing, default = ``Local``
        How Ô acts: Local (Ôᵢ), Correlation (Ôᵢⱼ), or Global (Ô).
    op_label : str | sequence[str], optional
        LaTeX label(s).  If *None*, uses ``op.name`` for every operator.
    to_bin : Callable[[int, int], str], optional
        Integer → binary-string formatter.  Defaults to
        ``lambda k,L: format(k, f'0{L}b')``.
    just_time : bool, default = False
        If True, only measure the time taken for the operation without
        displaying the results.  This is useful for benchmarking.

    Notes
    -----
    * For **integer states** we reproduce the coefficient table you had before.
    * For **array states** (NumPy / JAX) we show only the *first* non-zero
        coefficient returned by the operator.  Adjust if you need more detail.
    """
    
    from QES.general_python.common.timer import Timer
    if not just_time:
        from IPython.display import Math, display
        from QES.general_python.common.display import (
            display_state,
            prepare_labels
        )

    # ------------------------------------------------------------------
    ops      = (op,) if not isinstance(op, Sequence) else tuple(op)
    is_int   = isinstance(state, (numbers.Integral, int, np.integer)) 
    ns       = lat.ns if ns is None else ns
    labels   = prepare_labels(ops, op_label) if not just_time else [''] * len(ops)
    
    # ------------------------------------------------------------------
    
    if not just_time:
        display_state(state,
                    ns,
                    label   = f"Initial integer state (Ns={ns})",
                    to_bin  = to_bin,
                    verbose = not just_time)

    # ------------------------------------------------------------------
    with Timer(verbose=True, precision='us', name="Operator action"):
        for cur_op, lab in zip(ops, labels):
            if not just_time:
                display(Math(fr"\text{{Operator: }} {lab}, \text{{typeacting}}: {op_acting}"))
            
            if op_acting == OperatorTypeActing.Local.value:
                for i in range(ns):
                    if not just_time: 
                        display(Math(fr"\quad \text{{Site index: }} {i}"))
                    s, c = _dispatch(cur_op, state, lat, is_int, to_bin, lab, i=i, just_time=just_time)

            elif op_acting == OperatorTypeActing.Correlation.value:
                for i in range(ns):
                    for j in range(ns):
                        if not just_time:
                            display(Math(fr"\text{{Site indices: }} {i}, {j}"))
                        s, c = _dispatch(cur_op, state, lat, is_int, to_bin, lab, i=i, j=j, just_time=just_time)

            elif op_acting == OperatorTypeActing.Global.value:
                s, c = _dispatch(cur_op, state, lat, is_int, to_bin, lab, just_time=just_time, sites=sites)
            else:
                raise ValueError(f"Operator acting type {op_acting!r} not supported.")

# --------------------------------------------------------------------------------------------------

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
        params_str = ", ".join(map(str, add_args))
        tests = {
            f"op(state, {params_str})"     : lambda: op(state, *add_args),
            # f"op[state, {params_str}]"     : lambda: op[state, *add_args],
            f"op % state, {params_str}"    : lambda: op % (state, *add_args),
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

####################################################################################################
#! End of file
####################################################################################################