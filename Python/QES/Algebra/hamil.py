
from abc import ABC, abstractmethod

from typing import List, Tuple, Union
import jax.numpy as jnp
from jax import jit
import numpy as np
from functools import partial

####################################################################################################

from .hilbert import HilbertSpace

####################################################################################################

class Hamiltonian(ABC):
    '''
    A general Hamiltonian class. This class is used to define the Hamiltonian of a system. It may be 
    either a Many-Body Quantum Mechanics Hamiltonian or a non-interacting system Hamiltonian. It may 
    generate a Hamiltonian matrix but in addition it defines how an operator acts on a state. The
    Hamiltonian class is an abstract class and is not meant to be instantiated. It is meant to be
    inherited by other classes.
    '''
    
    def __init__(self, hilbert_space : HilbertSpace, is_sparse : bool = True, **kwargs):
        '''
        Initialize the Hamiltonian class.
        
        Args:
            hilbert_space (HilbertSpace)    : The Hilbert space of the system.
            is_sparse (bool)                : A flag to indicate whether the Hamiltonian is sparse or not.
            **kwargs                        : Additional arguments.
        '''
        
        self._hilbert_space = hilbert_space
        self._is_sparse     = is_sparse
        
        if self._hilbert_space is None:
            raise ValueError("The Hilbert space must be provided.")
        else:
            self._lattice   = self._hilbert_space.get_lattice()
            self._Nh        = self._hilbert_space.get_Nh()
            
            
    # ----------------------------------------------------------------------------------------------