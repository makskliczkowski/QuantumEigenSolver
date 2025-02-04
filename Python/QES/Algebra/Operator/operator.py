"""
This module defines a set of classes and functions for handling general operators in quantum mechanics, 
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

from abc import ABC, abstractmethod
from enum import Enum, auto
import numpy as np
import jax.numpy as jnp
from jax import jit
from jax.experimental import sparse
from typing import Union, Callable, Tuple, List
import copy

# Hilbert space
from ..hilbert import HilbertSpace

####################################################################################################

class SymmetryGenerators(Enum):
    """
    Available symmetry generators for symmetry analysis. 
    - E                 : Identity symmetry. 
    - Translation (Tx)  : Translation symmetry in the x-direction.
    - Reflecition (R)   : Reflection symmetry.
    - Parity (Px)       : Parity symmetry in the x-direction.
    - Parity (Py)       : Parity symmetry in the y-direction.
    - Parity (Pz)       : Parity symmetry in the z-direction. 
    """
    
    E           = auto()
    Translation = auto()
    Reflection  = auto()
    ParityX     = auto()
    ParityY     = auto()
    ParityZ     = auto()
    Other       = auto()
    
    # -----------
    
    def __str__(self):
        return self.name
    
    # -----------

####################################################################################################

class OperatorFunction:
    """
    Imitates the operator function behavior. The function shall act on the state (or list of states) and return the transformed 
    state(s) with the corresponding value(s). States can be represented as integers or numpy arrays or JAX arrays. This enables
    the user to define any operator that can be applied to the state. The function shall return a list of pairs (state, value). 
    """
    
    def __init__(self, fun : Callable, modifies_state : bool = True, necessary_args : int = 0):
        """
        Initialize the OperatorFunction object.
        
        Params:
        - fun (callable)    : The function that defines the operator - it shall take a state 
            (or a list of states) and return the transformed state (or a list of states). States can be 
            represented as integers or numpy arrays or JAX arrays. This enables the user to define
            any operator that can be applied to the state. The function shall return a list of pairs (state, value).
        """
        self._fun               = fun
        self._modifies_state    = modifies_state
        self._necessary_args    = int(necessary_args)

    # -----------
    
    def __call__(self, s : Union[int, jnp.array], *args) -> List[Tuple[Union[int, jnp.array], Union[float, complex]]]:
        """
        Apply the operator to the state. It returns the transformed state with the corresponding value. The 
        returned value is a list of pairs (state, value). 
        Params:
        - s (int, jnp.array) : The state to which the operator is applied.
        - args               : Additional arguments for the operator - inform how to act on a state.
        """
        if len(args) != self._necessary_args:
            raise ValueError(f"Invalid number of arguments for the operator function. Expected {self._necessary_args} arguments.")
        
        result = self._fun(s, *args)
        
        if isinstance(result, tuple) and isinstance(result[0], (int, jnp.ndarray)):
            if self._modifies_state:
                return [None, result[1]]
            else:
                return [result]
        elif isinstance(result, list):
            return result
        raise ValueError("Invalid return type from the operator function.")           

    # -----------
    
    # Getters and Setters
    
    # -----------
    
    @property
    def fun(self):
        ''' Set the function that defines the operator '''
        return self._fun
    
    @fun.setter
    def fun(self, val):
        self._fun = val
        
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
    
    # Composition
    
    # -----------
        
    def __mul__(self, other):
        """
        Composition of two operator functions.
        Implements the operator composition: f * g ≡ f(g(n,...),...) for two operator functions f and g,
        where f is the current operator function and g is the other operator function.
        
        Params:
        - other (OperatorFunction) : The operator function to be composed with the current operator function - those 
        may have different number of arguments so it is important to combine them properly. 
        """
        other_modifies  = True
        args_needed     = 0
        if isinstance(other, OperatorFunction):
            other_modifies = other.modifies_state
            args_needed    = other.necessary_args
        elif isinstance(other, Callable):
            # check the number of arguments for the function
            args_needed = other.__code__.co_argcount - 1
        else:
            raise ValueError("Invalid type for operator function composition.")
        
        def composed_fun(s: Union[int, jnp.ndarray], *args):
            """
            Composed function for the operator function.
            First, apply the function on the right side (other), then apply the function on the left side (self).

            Params:
            - s (int, jnp.ndarray): The state to which the operator is applied.
            - args: Additional arguments for the operator.

            Returns:
            - List of (state, value) pairs after applying the composed operator function.
            """
            # First apply the right-side operator
            intermediate = other(s, *args)

            if other_modifies:
                next_result = self(s, *args)
                if self._modifies_state:
                    return [(None, intermediate[0][1] * next_result[0][1])]
                return [(si, intermediate[0][1] * next_result_i[1]) for si, next_result_i in next_result]
            else:
                final_result = []
                for s1, v1 in intermediate:
                    # Apply the left operator function
                    next_result = self(s1, *args)
                    if self._modifies_state:
                        final_result.append((s1, v1 * next_result[0][1]))
                    else:
                        final_result.extend([(s2, v1 * v2) for s2, v2 in next_result])
                return final_result

        # return the composed operator function
        return OperatorFunction(composed_fun, modifies_state = self._modifies_state or other_modifies, necessary_args = max(self._necessary_args, args_needed))
        
    # -----------
    
    def __rmul__(self, other):
        """
        Reverse composition of two operator functions: g * f ≡ g(f(n,...),...).
        Applies `f` first, then `g` on the result.
        
        :param other: The operator function to compose with `self` in reversed order.
        :return: A new composed operator function.
        """
        other_modifies  = isinstance(other, OperatorFunction) and other.modifies_state
        args_needed     = other.necessary_args if isinstance(other, OperatorFunction) else other.__code__.co_argcount - 1

        def composed_fun(s: Union[int, jnp.ndarray], *args):
            intermediate = self(s, *args)
            if self.modifies_state:
                next_result = other(s, *args)
                if other_modifies:
                    return [(None, intermediate[0][1] * next_result[0][1])]
                return [(si, intermediate[0][1] * next_result_i[1]) for si, next_result_i in next_result]
            else:
                final_result = []
                for s1, v1 in intermediate:
                    next_result = other(s1, *args)
                    if other_modifies:
                        final_result.append((s1, v1 * next_result[0][1]))
                    else:
                        final_result.extend([(s2, v1 * v2) for s2, v2 in next_result])
                return final_result

        return OperatorFunction(composed_fun, modifies_state=self._modifies_state or other_modifies, necessary_args=max(self._necessary_args, args_needed))

    # -----------
    
    # Addition
    
    # -----------

    @jit
    def __add__(self, other):
        """
        Add two operator functions. The operator takes a maximum of two arguments. 
        The operator function is defined as f'(s, *args) = f(s, *args[:max_F]) + g(s, *args[:max_g]).
        Note:
        - If the operator functions have different number of arguments, the maximum number of arguments is taken. 
        - If the operator functions modify the state, the result is a list of pairs (state, value).
        - The addition is performed as f + g.
        - JIT compilation is used for the addition operation when the state is a JAX array.
        
        Params:
        - other (OperatorFunction) : The operator function to be added to the current operator function.
        """

        def adding(s : Union[int, jnp.ndarray], *args):
            """
            Add the operator functions.
            
            Params:
            - s (int, jnp.ndarray)  : The state to which the operator is applied.
            - args                  : Additional arguments for the operator.
            """
            if isinstance(s, jnp.ndarray):
                result_f = jnp.array(self(s, *args))
                result_g = jnp.array(other(s, *args))

                # Extract states and values
                states_f, values_f = result_f[:, 0], result_f[:, 1]
                states_g, values_g = result_g[:, 0], result_g[:, 1]
                
                # Combine both results
                combined_states = jnp.concatenate([states_f, states_g])
                combined_values = jnp.concatenate([values_f, values_g])

                # Get unique states and sum corresponding values
                unique_states, indices = jnp.unique(combined_states, return_inverse=True)
                summed_values = jnp.zeros_like(unique_states, dtype=complex).at[indices].add(combined_values)

                return list(zip(unique_states, summed_values))
            
            # otherwise - apply the operator to the state
            
            result_f    = self(s, *args)
            result_g    = other(s, *args)
            combinded   = {}
            
            # combine the results
            if self._modifies_state:
                for state, value in result_f:
                    if state not in combinded:
                        combinded[state] = value
                    else:
                        combinded[state] += value
            else:
                combinded[None] = result_f[0][1]
            
            if other.modifies_state:
                for state, value in result_g:
                    if state not in combinded:
                        combinded[state] = value
                    else:
                        combinded[state] += value
            else:
                combinded[None] += result_g[0][1]
            
            if len(combinded) == 1 and None in combinded:
                return [(None, combinded[None])]
            elif None in combinded:
                return [(state, value) for state, value in combinded.items() if state is not None] + [(s, combinded[None])]
            return list(combinded.items())
        return OperatorFunction(adding, modifies_state = self._modifies_state or other._modifies_state, necessary_args = max(self._necessary_args, other._necessary_args))    
        
        
    # -----------
    
    # Subtraction
    
    # -----------
    
    @jit
    def __sub__(self, other):
        """
        Substract two operator functions. The operator takes a maximum of two arguments. 
        The operator function is defined as f'(s, *args) = f(s, *args[:max_F]) - g(s, *args[:max_g]).
        Note:
        - If the operator functions have different number of arguments, the maximum number of arguments is taken. 
        - If the operator functions modify the state, the result is a list of pairs (state, value).
        - The subtraction is performed as f - g.
        - JIT compilation is used for the subtraction operation when the state is a JAX array.
        
        Params:
        - other (OperatorFunction) : The operator function to be added to the current operator function.
        """

        def substract(s : Union[int, jnp.ndarray], *args):
            """
            Substract the operator functions. When the operator is applied to a state, the function returns the
            transformed state with the corresponding value. The returned value is a list of pairs (state, value).
            
            Note:
            - If the operator functions have different number of arguments, the maximum number of arguments is taken.
            - If the operator functions modify the state, the result is a list of pairs (state, value).
            
            Params:
            - s (int, jnp.ndarray)  : The state to which the operator is applied.
            - args                  : Additional arguments for the operator.
            """
            if isinstance(s, jnp.ndarray):
                # Get results from both operators
                result_f = jnp.array(self(s, *args))
                result_g = jnp.array(other(s, *args))

                # Extract states and values
                states_f, values_f = result_f[:, 0], result_f[:, 1]
                states_g, values_g = result_g[:, 0], result_g[:, 1]

                # Combine results with signs for subtraction
                combined_states = jnp.concatenate([states_f, states_g])
                combined_values = jnp.concatenate([values_f, -values_g])  # Negate `g` values for subtraction

                # Get unique states and sum corresponding values
                unique_states, indices = jnp.unique(combined_states, return_inverse=True)
                summed_values = jnp.zeros_like(unique_states, dtype=complex).at[indices].add(combined_values)

                return list(zip(unique_states, summed_values))
            
            result_f    = self(s, *args)
            result_g    = other(s, *args)
            combinded   = {}
            
            # combine the results
            if self._modifies_state:
                for state, value in result_f:
                    if state not in combinded:
                        combinded[state] = value
                    else:
                        combinded[state] += value
            else:
                combinded[None] = result_f[0][1]
            
            if other.modifies_state:
                for state, value in result_g:
                    if state not in combinded:
                        combinded[state] = -value
                    else:
                        combinded[state] -= value
            else:
                combinded[None] -= result_g[0][1]
            
            if len(combinded) == 1 and None in combinded:
                return [(None, combinded[None])]
            elif None in combinded:
                return [(state, value) for state, value in combinded.items() if state is not None] + [(s, combinded[None])]
            return [(state, value) for state, value in combinded.items()]
        
        return OperatorFunction(substract, modifies_state = self._modifies_state or other._modifies_state, necessary_args = max(self._necessary_args, other._necessary_args))    

####################################################################################################

class Operator(ABC):
    """
    A class to represent a general operator acting on a Hilbert space.
    
    Attributes:
    """
    
    #################################
    
    def __init__(self, **kwargs):
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
            
            Important arguments:
            - fun (Callable | OperatorFunction) : The function that defines the operator - it shall take a state 
                (or a list of states) and return the transformed state (or a list of states). States can be 
                represented as integers or numpy arrays or JAX arrays. This enables the user to define
                any operator that can be applied to the state. The function shall return a list of pairs (state, value).
        """
        
        # handle the system phyisical size dimension and the lattice
        if "Ns" in kwargs and "lattice" not in kwargs:
            self._Ns        = kwargs.get('Ns', 1)
            self._lattice   = None
        elif "lattice" in kwargs:
            self._lattice   = kwargs.get('lattice')
            self._Ns        = self.lattice.get_Ns()
        else:
            raise ValueError("Either 'Ns' or 'lattice' must be provided.")            
        
        # property of the operator itself
        self._eigval        = kwargs.get('eigval', 1.0)
        self._name          = kwargs.get('name', 'Operator')
        self._type          = SymmetryGenerators(kwargs.get('type', 'E'))
        if self._type != SymmetryGenerators.Other:
            self._name      = self._type.name
        
        
        # property for the behavior of the operator - e.g., quadratic, action, etc.
        self._quadratic     = kwargs.get('quadratic', False)        # flag for the quadratic operator - this enables different matrix representation
        self._acton         = kwargs.get('acton', False)            # flag for the action of the operator on the local physical space
        self._modifies      = kwargs.get('modifies', True)          # flag for the operator that modifies the state
        
        if isinstance(kwargs.get('fun'), OperatorFunction):
            self._fun       = kwargs.get('fun')                     # the function that defines the operator already through the OperatorFunction object
        elif isinstance(kwargs.get('fun'), Callable):
            self._fun       = OperatorFunction(kwargs.get('fun'), modifies_state = self._modifies)
        else:
            raise ValueError("Invalid operator function type.")
        self._matrix_fun    = None                                  # the function that defines the matrix form of the operator - if not provided, the matrix is generated from the function fun
    
    #################################
    
    # Static methods
    
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
    
    # Copying and cloning
    
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
    
    # Operators that modify the operator class itself
    
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
    
    # Setters and Getters
    
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
    def fun(self):
        ''' Set the function that defines the operator '''
        return self._fun
    
    @fun.setter
    def fun(self, val):
        self._fun = val
        
    # -------------------------------
    
    def override_matrix_fuction(self, function : Callable):
        """
        Override the matrix function of the operator.
        """
        self._matrix_fun = function
        
        
    #################################
    
    # Apply the operator
    
    #################################
    
    # def chi(self, state):
    #     """
    #     Apply the operator to the state. Representative eigenvalue calculator for the operator.
    #     """
    #     return self._fun(state)
    
    # -------------------------------
    
    def apply(self, states : list | np.ndarray | jnp.ndarray, *args):
        """
        Apply the operator to the state. 
        
        Args:
            states            : list of states to which the operator is applied.
            args              : Additional arguments for the operator - inform how to act on a state.
                                If there no arguments, the operator acts on the state as a whole - global operator.
                                If there are arguments, the operator acts on the state locally - local operator (e.g., site-dependent).
        """
        return self._fun(states, *args)
    
    # -------------------------------
    
    def __call__(self, states : list | np.ndarray | jnp.ndarray, *args):
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
    
    # Generate matrix form of the operator
    
    #################################
    
    def matrix(self, dim : int, matrix_type : str = 'dense', *args) -> np.ndarray | jnp.ndarray | sparse.COO | sparse.CSR | None:
        """
        Generates the matrix representation of the operator.

        Parameters:
        - param dim             : Dimension of the matrix.
        - param matrix_type     : Type of matrix ("dense" or "sparse").
        :return: The matrix representation of the operator.
        """
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError("Dimension must be a positive integer.")
        
        # check if the matrix function is provided
        if self._matrix_fun is not None:
            if matrix_type == 'sparse':
                return self._matrix_fun(dim, matrix_type, *args)
            else:
                return jnp.asarray(self._matrix_fun(dim, matrix_type, *args))
        else:
            # generate the matrix from the function fun
            pass
    
    #################################
    
    def standardize_matrix(self, matrix):
        """
        Standardizes the given matrix if the _standarize flag is set to true.
        """
        pass