import numpy as np
import scipy as sp
from numba import jit, njit, prange
from typing import Union, Tuple, Union, Callable, Optional

# for the abstract class
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto, unique

# from algebra
from general_python.algebra.utils import _JAX_AVAILABLE, get_backend
from general_python.algebra.ran_wrapper import choice, randint, uniform
from general_python.common.directories import Directories
import general_python.common.binary as Binary

# from hilbert
from Algebra.hilbert import HilbertSpace

# JAX imports
if _JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
    import jax.random as random
    from jax import vmap

#########################################

from Solver.MonteCarlo.montecarlo import MonteCarloSolver, McsTrain

#########################################

class NQS(MonteCarloSolver):
    '''
    Neural Quantum State (NQS) Solver.
    Implements a Monte Carlo-based training method for optimizing NQS models.
    Supports both NumPy and JAX backends for efficiency and flexibility.
    '''
    
    def __init__(self,
                seed    : Optional[int]             = None,
                beta    : float                     = 1,
                replica : int                       = 1,
                backend : str                       = 'default',
                size    : int                       = 1,
                hilbert : Optional[HilbertSpace]    = None,
                modes   : int                       = 2,
                dir     : Optional[str]             = None,
                **kwargs):
        '''
        Initialize the NQS solver.
        '''
        super().__init__(seed, beta, replica, backend, size, hilbert, modes, dir, **kwargs)
        
        # look for the architecture_parameters
        self._arch_params = kwargs.get('architecture_parameters', None)