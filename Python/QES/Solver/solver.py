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

# from hilbert
from Algebra.hilbert import HilbertSpace

# precodintioners etc.
from general_python.algebra.preconditioners import Preconditioner

# JAX imports
if _JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
    import jax.random as random
    from jax import vmap

#######################################

class Solver(ABC):
    """
    Abstract base class for the solvers in many body physics. This class is used to define the basic structure of the solvers.
    """
    
    ###################################
    @dataclass
    class SolverLastLoss:
        """
        Class to store the last loss.
        """
        last            : float = None
        std             : float = None
        mean            : float = None
        max             : float = None
        min             : float = None
        current         : float = None
        best            : float = None
    
    ###################################
    defdir     = "./data"       # default directory for saving the data

    ###################################
    NOT_IMPLEMENTED_ERROR       = "The state is not implemented for the given modes."
    NOT_IMPLEMENTED_ERROR_SAVE  = "The saving is not implemented for the given modes."
    NOT_GIVEN_SIZE_ERROR        = "The size or shape of the system is not defined."
    NOT_A_VALID_STATE_STRING    = "The state is not a valid string, check SolverInitState."
    NOT_A_VALID_STATE_DISTING   = "The state must be an integer, a jnp.ndarray, or a valid string representing an initial state."
    ###################################
    
    def __init__(self,
                shape       : Union[int, Tuple[int, ...]]   = (1,),
                modes       : int                           = 2,
                seed        : Optional[int]                 = None,
                hilbert     : Optional[HilbertSpace]        = None,
                directory   : Union[str, Directories]       = defdir,
                nthreads    : int                           = 1,
                backend     : str                           = 'default',
                **kwargs):
        '''
        Initialize the solver.
        
        Parameters:
        - size          : size of the configuration (like lattice sites etc.)
        - modes         : number of modes for the binary representation
        - hilbert       : Hilbert space representation
        - directory     : directory for saving the data (potentially)
        - seed          : seed for the random number generator
        - backend       : backend for the calculations (default is 'default')
        '''
        self._shape         = shape                                                         # size of the configuration (like lattice sites etc.)
                                                                                            # This can be either 1D chain, 2D lattice, transformed
                                                                                            # lattice, etc.
        self._size          = np.prod(shape) if isinstance(shape, tuple) else shape         # size of the configuration (like lattice sites etc.)
        self._modes         = modes                                                         # number of modes for the binary representation
        
        # Handle the Hilbert space representation
        self._hilbert       = hilbert                                                       # Hilbert space representation
        if self._hilbert is not None:
            self._size  = hilbert.ns
            self._modes = hilbert.modes
        elif hilbert is None and shape is not None:
            self._hilbert   = HilbertSpace(ns=self._size, modes=self._modes)
        elif hilbert is None and shape is None:
            raise ValueError(Solver.NOT_GIVEN_SIZE_ERROR)
        
        # directory creation
        self._dir           = directory                                                     # directory for saving the data (potentially)
        if not isinstance(directory, Directories):
            self._dir       = Directories(directory)
        self._dir.create_folder(False)
        
        # check the backend
        self._backend, self._backend_sp, (self._rng, self._rng_key) = self.obtain_backend(backend, seed)
        
        # set the precision
        if self._backend == np:
            self._eps   = np.finfo(np.float64).eps
            self._prec  = np.float32
        else:
            self._eps   = jnp.finfo(jnp.float64).eps
            self._prec  = jnp.float32
        
        # statistical
        self._lastloss      = Solver.SolverLastLoss()
        self._replica_idx   = 1
        
        # initialize threads
        self._nthreads      = nthreads
        
        # allow for preconditioner and scheduler
        self._precond       = kwargs.get('preconditioner', None)
        self._scheduler     = kwargs.get('scheduler', None)
        self._solver        = kwargs.get('solver', None)
        self._optimizer     = kwargs.get('optimizer', None)
        self._early_stop    = kwargs.get('early_stop', None)
        self._arch_params   = kwargs.get('architecture_parameters', None)

    #####################################
    #! PROPERTIES AND GETTERS
    #####################################
    
    @property
    def size(self):
        '''Return the size of the configuration.'''
        return self._size
    
    @property
    def modes(self):
        '''Return the number of modes for the binary representation.'''
        return self._modes
    
    @property
    def hilbert(self):
        '''Return the Hilbert space representation.'''
        return self._hilbert
    
    @property
    def lastloss(self):
        '''Return the last loss.'''
        return self._lastloss.last
    
    @property
    def lastloss_std(self):
        '''Return the last loss standard deviation.'''
        return self._lastloss.std
    
    @property
    def lastloss_mean(self):
        '''Return the last loss mean.'''
        return self._lastloss.mean
    
    @property
    def lastloss_max(self):
        '''Return the last loss maximum.'''
        return self._lastloss.max
    
    @property
    def lastloss_min(self):
        '''Return the last loss minimum.'''
        return self._lastloss.min
    
    @property
    def currentloss(self):
        '''Return the current loss.'''
        return self._lastloss.current
    
    @property
    def replica_idx(self):
        '''Return the replica index.'''
        return self._replica_idx
    
    #####################################
    #! BACKEND
    #####################################
    
    @property
    def backend(self):
        '''Return the backend used for calculations.'''
        return self._backend
    
    @property
    def backend_sp(self):
        '''Return the backend (SciPy) used for calculations.'''
        return self._backend_sp
    
    @property
    def rng(self):
        '''Return the random number generator.'''
        return self._rng
    
    @property
    def rng_key(self):
        '''Return the random number generator key.'''
        return self._rng_key
    
    @property
    def random(self):
        '''Return random number'''
        return uniform(shape=(1,), backend=self._backend, rng=self._rng, rng_k=self._rng_key)[0]
    
    # ----------------------------------
    
    def reset_backend(self, backend: str = 'default', seed: Optional[int] = None):
        '''
        Reset the backend for the calculations.
        Parameters:
        - backend       : backend for the calculations (default is 'default')
        - seed          : seed for the random number generator
        '''
        self._backend, self._backend_sp, (self._rng, self._rng_key) = self.obtain_backend(backend, seed)
        return self._backend, self._backend_sp, (self._rng, self._rng_key)
    
    @staticmethod
    def obtain_backend(backend: str, seed: Optional[int]):
        '''
        Set the backend for the calculations.
        Parameters:
        - backend       : backend for the calculations (default is 'default')
        - seed          : seed for the random number generator
        '''
        if isinstance(backend, str):
            bck = get_backend(backend, scipy=True, random=True, seed=seed)
            if isinstance(bck, tuple):
                _backend, _backend_sp = bck[0], bck[1]
                if isinstance(bck[2], tuple):
                    _rng, _rng_k = bck[2][0], bck[2][1]
                else:
                    _rng, _rng_k = bck[2], None
            else:
                _backend, _backend_sp = bck, None
                _rng, _rng_k = None, None
            return _backend, _backend_sp, (_rng, _rng_k), backend
        _backendstr = 'np' if (backend is None or (backend == 'default' and not _JAX_AVAILABLE) or backend == np) else 'jax'
        return Solver.obtain_backend(_backendstr, seed)
    
    ###################################
    #! Setters
    ###################################
    
    def set_replica_idx(self, idx: int):
        '''
        Set the replica index.
        '''
        self._replica_idx = idx
        return self._replica_idx
    
    def set_early_stopping(self, *args, **kwargs):
        '''
        Set the early stopping criteria.
        '''
        #!TODO: implement the early stopping criteria
        self._early_stop = kwargs.get('early_stop', None)
        return self._early_stop
    
    def set_optimizer(self, *args, **kwargs):
        '''
        Set the optimizer.
        '''
        #!TODO: implement the optimizer
        self._optimizer = kwargs.get('optimizer', None)
        return self._optimizer
    
    def set_preconditioner(self, *args, **kwargs):
        '''
        Set the preconditioner.
        '''
        #!TODO: implement the preconditioner
        self._preconditioner = kwargs.get('preconditioner', None)
        return self._preconditioner
    
    def set_scheduler(self, *args, **kwargs):
        '''
        Set the scheduler.
        '''
        #!TODO: implement the scheduler        
        self._scheduler = kwargs.get('scheduler', None)
        return self._scheduler
    
    def set_solver(self, *args, **kwargs):
        '''
        Set the solver.
        '''
        #!TODO: implement the solver
        self._solver = kwargs.get('solver', None)
        return self._solver
    
    ###################################
    #! ABSTRACT METHODS
    ###################################
    
    @abstractmethod
    def clone(self):
        '''
        Clone the solver.
        '''
        pass
    
    @abstractmethod
    def swap(self, other):
        '''
        Swap the state of the solver with another solver.
        '''
        pass
    
########################################