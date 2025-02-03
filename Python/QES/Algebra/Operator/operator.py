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
import copy

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

class Operator(ABC):
    """
    A class to represent a general operator acting on a Hilbert space.
    
    Attributes:
    """
    
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
            - fun (callable)    : The function that defines the operator - it shall take a state 
                (or a list of states) and return the transformed state (or a list of states). States can be 
                represented as integers or numpy arrays or JAX arrays. This enables the user to define
                any operator that can be applied to the state. The function shall return a list of pairs (state, value).
        """
        
        # handle the system phyisical size dimension
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
        
        self._fun           = kwargs.get('fun', lambda x: x)        # the function that defines the operator
    
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
    
    def matrix(self, hilbert : Hilb)