'''
file: Algebra/global.py
This file contains the global imports and definitions for the Algebra module.
'''

# Import the necessary modules
import math
import numpy as np
import time
from functools import partial
from typing import Callable, Union, Tuple, List, Optional
from abc import ABC, abstractmethod

# operator module for operator overloading
from Algebra.Operator.operator import Operator, GlobalSymmetries

# from general Python modules
from general_python.lattices.lattice import Lattice, LatticeBC, LatticeDirection
from general_python.common.binary import rotate_left, rotate_right, flip_all, rev, rotate_left_ax, popcount
from general_python.algebra.utils import get_backend

class GlobalSymmetry(ABC):
    """
    GlobalSymmetry represents a global symmetry check on a state.
    It stores:
        - a lattice (lat) (if needed),
        - a symmetry value (val),
        - a symmetry name (an element of GlobalSymmetries),
        - and a checking function (check) that takes (state, val) and returns a bool (True if the state satisfies the symmetry).
    
    The symmetry check supports integer states as well as NumPy or JAX arrays through the backend.
    """
    
    def __init__(self,
                lat     : Optional[Lattice]     = None,
                ns      : Optional[int]         = None,
                val     : float                 = 0.0,
                name    : GlobalSymmetries      = GlobalSymmetries.Other,
                backend : str                   = 'default'):
        '''
        Initialize the GlobalSymmetry object.
        Parameters:
        - lat : Optional[Lattice]     : The lattice associated with the symmetry.
        - ns  : Optional[int]         : The number of states (if needed).
        - val : float                 : The symmetry value (default is 0.0).
        - name: GlobalSymmetries      : The name of the symmetry (default is GlobalSymmetries.Other).
        '''
        if lat is not None:
            self._lat   = lat
            ns          = lat.ns
            self._ns    = ns
        elif ns is None:
            raise ValueError("Either the lattice or the number of states must be provided!")
        
        if isinstance(backend, str):
            self._backend_str   = backend
            self._backend       = get_backend(backend)
        else:
            self._backend_str   = 'np' if backend == np else 'jax'
            self._backend       = backend
        
        self.ns     = ns
        self.val    = val
        self.name   = name
        self.check  : Optional[Callable[[Union[int, np.ndarray], float], bool]] = None

    # ---------- SETTERS -----------
    
    def set_fun(self, fun: Callable[[Union[int, np.ndarray], float], bool]) -> None:
        """Set the checking function."""
        self.check = fun

    def set_name(self, name: GlobalSymmetries) -> None:
        """Set the name of the symmetry."""
        self.name = name

    # ---------- GETTERS -----------
    def get_name(self) -> GlobalSymmetries:
        """Return the symmetry name (enum element)."""
        return self.name

    def get_name_str(self) -> str:
        """Return the string representation of the symmetry name."""
        return self.name.name

    def get_val(self) -> float:
        """Return the symmetry value."""
        return self.val
    
    @property
    def lat(self) -> Lattice:
        """Return the lattice associated with the symmetry."""
        return self._lat
    
    @property
    def ns(self) -> int:
        """Return the number of states."""
        return self._ns
    
    @ns.setter
    def ns(self, ns: int) -> None:
        """Set the number of states."""
        self._ns = ns
        
    @property
    def backend(self) -> str:
        """Return the backend used for the symmetry check."""
        return self._backend
    
    @backend.setter
    def backend(self, backend: str) -> None:
        """Set the backend used for the symmetry check."""
        self._backend = backend

    # ---------- CHECKER -----------
    def __call__(self, state: Union[int, np.ndarray]) -> bool:
        """
        When the object is called with a state, the checking function is applied.
        Raises a ValueError if no check function is set.
        """
        if self.check is None:
            raise ValueError("No symmetry check function has been set!")
        return self.check(state, self.val)

    def check_state(self, state: Union[int, np.ndarray], out_cond: bool) -> bool:
        """
        Returns True if the state satisfies the symmetry and the additional condition outCond.
        """
        return self(state) and out_cond

# ---------------------------
# Global U(1) Symmetry
# ---------------------------

def u1_sym(state: Union[int, np.ndarray], val: float) -> bool:
    """
    Global U(1) symmetry check.
    
    For a given state, returns True if the popcount (number of 1-bits or up spins)
    equals the given value 'val'. This works for both integer states and array-like states.
    """
    return popcount(state) == val

def get_u1_sym(lat: Lattice, val: float) -> GlobalSymmetry:
    """
    Factory function that creates a U(1) global symmetry object.
    
    Parameters:
        lat: Lattice on which the symmetry is defined.
        val: The symmetry value (typically the required number of 1-bits).
    
    Returns:
        An instance of GlobalSymmetry with name U1, value val, and the checking function set to U1_sym.
    """
    sym = GlobalSymmetry(lat, val, GlobalSymmetries.U1)
    sym.set_fun(u1_sym)
    return sym